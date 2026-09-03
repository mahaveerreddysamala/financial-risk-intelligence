"""Optional Confluent Kafka producer and consumer adapters."""
from __future__ import annotations

import logging
from collections.abc import Iterator
from typing import Any

from financial_risk.streaming.events import EventEnvelope


logger = logging.getLogger(__name__)


def _require_kafka() -> Any:
    """Import the Kafka client lazily so core workflows stay dependency-light."""
    try:
        from confluent_kafka import Consumer, Producer
    except ImportError as exc:  # pragma: no cover - exercised through public error path
        raise RuntimeError(
            "confluent-kafka is not installed. Install requirements-streaming.txt to use Kafka."
        ) from exc
    return Producer, Consumer


class KafkaEventProducer:
    """Small producer adapter with JSON event serialization."""

    def __init__(self, bootstrap_servers: str, client_id: str = "financial-risk-producer") -> None:
        if not bootstrap_servers.strip():
            raise ValueError("bootstrap_servers must not be empty")
        self._producer_cls, _ = _require_kafka()
        self._producer = self._producer_cls(
            {"bootstrap.servers": bootstrap_servers, "client.id": client_id}
        )

    def publish(self, topic: str, event: EventEnvelope) -> None:
        """Publish an envelope using its event ID as the Kafka key."""
        if not topic.strip():
            raise ValueError("topic must not be empty")
        self._producer.produce(
            topic=topic,
            key=event.event_id,
            value=event.to_json(),
        )
        self._producer.poll(0)

    def flush(self, timeout: float = 10.0) -> int:
        """Wait for queued messages and return the remaining queue size."""
        if timeout < 0:
            raise ValueError("timeout must be non-negative")
        return int(self._producer.flush(timeout))


class KafkaEventConsumer:
    """Iterator-style consumer adapter that yields validated event envelopes."""

    def __init__(
        self,
        bootstrap_servers: str,
        group_id: str,
        topic: str,
        *,
        auto_offset_reset: str = "earliest",
    ) -> None:
        for name, value in {
            "bootstrap_servers": bootstrap_servers,
            "group_id": group_id,
            "topic": topic,
        }.items():
            if not value.strip():
                raise ValueError(f"{name} must not be empty")
        _, self._consumer_cls = _require_kafka()
        self._consumer = self._consumer_cls(
            {
                "bootstrap.servers": bootstrap_servers,
                "group.id": group_id,
                "auto.offset.reset": auto_offset_reset,
                "enable.auto.commit": False,
            }
        )
        self._topic = topic
        self._consumer.subscribe([topic])

    def poll(self, timeout: float = 1.0) -> EventEnvelope | None:
        """Poll one message and deserialize it, returning None for invalid records."""
        if timeout < 0:
            raise ValueError("timeout must be non-negative")
        message = self._consumer.poll(timeout)
        if message is None:
            return None
        error = message.error()
        if error is not None:
            raise RuntimeError(str(error))
        value = message.value()
        if value is None:
            logger.warning("Skipping Kafka tombstone/null-value record on topic=%s", self._topic)
            return None
        try:
            return EventEnvelope.from_json(value)
        except (TypeError, ValueError) as exc:
            logger.warning(
                "Skipping malformed Kafka record on topic=%s: %s", self._topic, exc
            )
            return None

    def consume(self, *, timeout: float = 1.0) -> Iterator[EventEnvelope]:
        """Yield validated events until the caller stops iteration."""
        while True:
            event = self.poll(timeout=timeout)
            if event is not None:
                yield event

    def close(self) -> None:
        """Close the underlying consumer and release broker resources."""
        self._consumer.close()
