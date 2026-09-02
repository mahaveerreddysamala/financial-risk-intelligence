from __future__ import annotations

import json
from datetime import UTC, datetime

import pytest

from financial_risk.streaming.events import EventEnvelope, serialize_event
from financial_risk.streaming.kafka import KafkaEventConsumer, KafkaEventProducer


def test_serialize_event_round_trip() -> None:
    occurred_at = datetime(2026, 9, 2, 12, 0, tzinfo=UTC)
    raw = serialize_event(
        "TXN123",
        "transaction.created",
        {"amount": 125.5, "is_fraud": 0},
        occurred_at=occurred_at,
    )
    envelope = EventEnvelope.from_json(raw)
    assert envelope.event_id == "TXN123"
    assert envelope.event_type == "transaction.created"
    assert envelope.schema_version == 1
    assert envelope.occurred_at == occurred_at.isoformat()
    assert envelope.payload["amount"] == 125.5


def test_event_deserialization_validates_required_fields() -> None:
    raw = json.dumps({"event_id": "TXN123", "payload": {}})
    with pytest.raises(ValueError, match="missing fields"):
        EventEnvelope.from_json(raw)


def test_serialize_event_requires_timezone_aware_timestamp() -> None:
    naive_timestamp = datetime.fromtimestamp(0, UTC).replace(tzinfo=None)
    with pytest.raises(ValueError, match="timezone-aware"):
        serialize_event("TXN123", "transaction.created", {}, occurred_at=naive_timestamp)


def test_kafka_adapters_validate_configuration() -> None:
    with pytest.raises(ValueError, match="bootstrap_servers"):
        KafkaEventProducer(" ")
    with pytest.raises(ValueError, match="group_id"):
        KafkaEventConsumer("localhost:9092", " ", "transactions")
    with pytest.raises(ValueError, match="topic"):
        KafkaEventConsumer("localhost:9092", "risk-group", " ")


def test_kafka_adapters_report_optional_dependency(monkeypatch: pytest.MonkeyPatch) -> None:
    def missing_kafka() -> object:
        raise RuntimeError("confluent-kafka is not installed")

    monkeypatch.setattr("financial_risk.streaming.kafka._require_kafka", missing_kafka)
    with pytest.raises(RuntimeError, match="confluent-kafka"):
        KafkaEventProducer("localhost:9092")


def test_kafka_producer_publishes_json(monkeypatch: pytest.MonkeyPatch) -> None:
    captured: dict[str, object] = {}

    class FakeProducer:
        def __init__(self, config: dict[str, object]) -> None:
            captured["config"] = config

        def produce(self, **kwargs: object) -> None:
            captured["message"] = kwargs

        def poll(self, timeout: float) -> None:
            captured["poll"] = timeout

        def flush(self, timeout: float) -> int:
            captured["flush"] = timeout
            return 0

    monkeypatch.setattr(
        "financial_risk.streaming.kafka._require_kafka",
        lambda: (FakeProducer, object),
    )
    producer = KafkaEventProducer("localhost:9092", client_id="test-producer")
    event = EventEnvelope("TXN1", "transaction.created", 1, "2026-09-02T00:00:00+00:00", {"amount": 5})
    producer.publish("transactions", event)
    remaining = producer.flush()

    assert captured["config"] == {"bootstrap.servers": "localhost:9092", "client.id": "test-producer"}
    assert captured["message"]["topic"] == "transactions"
    assert captured["message"]["key"] == "TXN1"
    assert json.loads(captured["message"]["value"])["event_id"] == "TXN1"
    assert captured["poll"] == 0
    assert remaining == 0


def test_kafka_consumer_deserializes_message(monkeypatch: pytest.MonkeyPatch) -> None:
    class FakeMessage:
        def error(self) -> None:
            return None

        def value(self) -> bytes:
            return serialize_event("TXN2", "transaction.created", {"amount": 10})

    class FakeConsumer:
        def __init__(self, config: dict[str, object]) -> None:
            self.config = config
            self.subscribed: list[str] = []

        def subscribe(self, topics: list[str]) -> None:
            self.subscribed = topics

        def poll(self, timeout: float) -> FakeMessage:
            return FakeMessage()

        def close(self) -> None:
            return None

    monkeypatch.setattr(
        "financial_risk.streaming.kafka._require_kafka",
        lambda: (object, FakeConsumer),
    )
    consumer = KafkaEventConsumer("localhost:9092", "risk-group", "transactions")
    event = consumer.poll()

    assert event is not None
    assert event.event_id == "TXN2"
    assert event.payload["amount"] == 10
    consumer.close()
