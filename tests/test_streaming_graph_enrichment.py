import pytest

from financial_risk.streaming.events import EventEnvelope
from financial_risk.streaming.graph_enrichment import StreamingGraphEnricher


def transaction(transaction_id: str, customer_id: str, device_id: str = "dev-1") -> dict[str, str]:
    return {
        "transaction_id": transaction_id,
        "customer_id": customer_id,
        "account_id": f"acct-{customer_id}",
        "device_id": device_id,
        "ip_id": "ip-1",
        "merchant_id": "merchant-1",
    }


def test_streaming_enrichment_builds_connected_community() -> None:
    enricher = StreamingGraphEnricher()
    first = enricher.enrich_transaction(
        transaction("txn-1", "cust-1"),
        fraud_probability=0.6,
        anomaly_score=0.2,
        velocity_score=0.1,
        event_id="evt-1",
    )
    second = enricher.enrich_transaction(
        transaction("txn-2", "cust-2"),
        fraud_probability=0.6,
        anomaly_score=0.2,
        velocity_score=0.1,
        event_id="evt-2",
    )

    assert first.community_customer_count == 1
    assert second.community_customer_count == 2
    assert second.community_entity_count > first.community_entity_count
    assert second.community_risk > first.community_risk


def test_shared_entity_creates_direct_network_risk() -> None:
    enricher = StreamingGraphEnricher()
    enricher.enrich_transaction(
        transaction("txn-1", "cust-1"),
        fraud_probability=0.2,
        anomaly_score=0.1,
        velocity_score=0.1,
        event_id="evt-1",
    )
    result = enricher.enrich_transaction(
        transaction("txn-2", "cust-2"),
        fraud_probability=0.2,
        anomaly_score=0.1,
        velocity_score=0.1,
        event_id="evt-2",
    )

    assert result.network_risk == 1.0
    assert result.graph_risk.adjusted_score > result.graph_risk.base_score


def test_duplicate_events_are_rejected() -> None:
    enricher = StreamingGraphEnricher()
    kwargs = {
        "fraud_probability": 0.2,
        "anomaly_score": 0.1,
        "velocity_score": 0.1,
        "event_id": "evt-1",
    }
    enricher.enrich_transaction(transaction("txn-1", "cust-1"), **kwargs)
    with pytest.raises(ValueError, match="duplicate event_id"):
        enricher.enrich_transaction(transaction("txn-2", "cust-2"), **kwargs)


def test_invalid_graph_transaction_is_rejected() -> None:
    enricher = StreamingGraphEnricher()
    with pytest.raises(ValueError, match="Missing required graph fields"):
        enricher.enrich_transaction(
            {"transaction_id": "txn-1", "customer_id": "cust-1"},
            fraud_probability=0.2,
            anomaly_score=0.1,
            velocity_score=0.1,
            event_id="evt-1",
        )


def test_invalid_risk_signal_is_rejected() -> None:
    enricher = StreamingGraphEnricher()
    with pytest.raises(ValueError, match="fraud_probability must be between 0 and 1"):
        enricher.enrich_transaction(
            transaction("txn-1", "cust-1"),
            fraud_probability=1.2,
            anomaly_score=0.1,
            velocity_score=0.1,
            event_id="evt-1",
        )


def test_event_enrichment_preserves_payload_and_changes_event_type() -> None:
    event = EventEnvelope(
        event_id="evt-1",
        event_type="transaction.created",
        schema_version=1,
        occurred_at="2026-09-04T12:00:00+00:00",
        payload={
            **transaction("txn-1", "cust-1"),
            "fraud_probability": 0.6,
            "anomaly_score": 0.2,
            "velocity_risk": 0.1,
            "source": "test",
        },
    )
    result = StreamingGraphEnricher().enrich_event(event)

    assert result.event_type == "transaction.graph_enriched"
    assert result.payload["source"] == "test"
    assert result.payload["graph_enrichment"]["transaction_id"] == "txn-1"
    assert "community_risk" in result.payload["graph_enrichment"]


def test_non_transaction_event_is_rejected() -> None:
    event = EventEnvelope(
        event_id="evt-1",
        event_type="transaction.risk_scored",
        schema_version=1,
        occurred_at="2026-09-04T12:00:00+00:00",
        payload={},
    )
    with pytest.raises(ValueError, match="unsupported event_type"):
        StreamingGraphEnricher().enrich_event(event)
