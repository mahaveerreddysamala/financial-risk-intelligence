from __future__ import annotations

from pathlib import Path


ROOT = Path(__file__).parents[1]


def test_kafka_compose_declares_broker_and_smoke_service() -> None:
    source = (ROOT / "docker-compose.kafka.yml").read_text(encoding="utf-8")
    assert "services:" in source
    assert "kafka:" in source
    assert "kafka-smoke:" in source
    assert "condition: service_healthy" in source
    assert "KAFKA_BOOTSTRAP_SERVERS: kafka:9092" in source


def test_kafka_smoke_script_has_expected_result_boundary() -> None:
    source = (ROOT / "scripts" / "kafka_smoke_test.py").read_text(encoding="utf-8")
    assert "KafkaEventProducer" in source
    assert "KafkaEventConsumer" in source
    assert "process_event" in source
    assert "KAFKA E2E SMOKE TEST: PASS" in source
