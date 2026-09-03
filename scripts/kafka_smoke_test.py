"""End-to-end Kafka smoke test for the financial-risk streaming path."""
from __future__ import annotations

import os
import time
import uuid

from financial_risk.streaming.events import EventEnvelope, serialize_event
from financial_risk.streaming.kafka import KafkaEventConsumer, KafkaEventProducer
from financial_risk.streaming.worker import process_event


def _wait_for_broker(bootstrap_servers: str, attempts: int = 30) -> None:
    """Wait until Kafka accepts connections and has no queued producer messages."""
    last_error: Exception | None = None
    for _ in range(attempts):
        producer = None
        try:
            producer = KafkaEventProducer(bootstrap_servers)
            remaining = producer.flush(1.0)
            if remaining == 0:
                return
            last_error = RuntimeError(
                f"Kafka producer still has {remaining} queued messages while probing broker"
            )
        except (OSError, RuntimeError, ValueError) as exc:  # pragma: no cover - integration environment only
            last_error = exc
        finally:
            if producer is not None:
                producer.flush(0.0)
        time.sleep(1)
    raise RuntimeError(f"Kafka broker did not become ready: {last_error}")


def main() -> None:
    bootstrap_servers = os.getenv("KAFKA_BOOTSTRAP_SERVERS", "localhost:9092")
    topic = os.getenv("KAFKA_TOPIC", "financial-risk-events")
    group_id = f"financial-risk-smoke-{uuid.uuid4().hex[:8]}"

    _wait_for_broker(bootstrap_servers)
    producer = KafkaEventProducer(bootstrap_servers, client_id="financial-risk-smoke-producer")
    consumer = KafkaEventConsumer(bootstrap_servers, group_id, topic, auto_offset_reset="earliest")

    event = EventEnvelope(
        event_id=f"smoke-{uuid.uuid4().hex}",
        event_type="transaction.created",
        schema_version=1,
        occurred_at="2026-09-02T00:00:00+00:00",
        payload={
            "transaction_id": "SMOKE-TXN-1",
            "fraud_probability": 0.90,
            "anomaly_score": 0.80,
            "network_risk": 0.70,
            "velocity_risk": 0.60,
        },
    )

    producer.publish(topic, event)
    remaining = producer.flush(10.0)
    if remaining != 0:
        raise RuntimeError(f"Kafka producer still has {remaining} queued messages")

    received = None
    deadline = time.time() + 20
    while time.time() < deadline:
        received = consumer.poll(1.0)
        if received is not None:
            break
    consumer.close()

    if received is None:
        raise RuntimeError("Kafka smoke test timed out waiting for the event")
    if received.event_id != event.event_id:
        raise RuntimeError("Kafka smoke test received an unexpected event")

    result = process_event(received, publish=None)
    if result.risk_band != "CRITICAL" or result.action != "hold_and_investigate":
        raise RuntimeError("Kafka smoke test produced an unexpected risk decision")

    # Re-serialize once to verify the result boundary remains JSON-compatible.
    serialize_event(
        result.event_id,
        "transaction.risk_scored",
        result.to_dict(),
        occurred_at=received_occurred_at(received),
    )
    print("KAFKA E2E SMOKE TEST: PASS")


def received_occurred_at(event: EventEnvelope):
    """Return the received event timestamp as an aware datetime."""
    from datetime import datetime

    return datetime.fromisoformat(event.occurred_at)


if __name__ == "__main__":
    main()
