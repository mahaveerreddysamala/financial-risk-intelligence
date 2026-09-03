from financial_risk.streaming.events import EventEnvelope
from financial_risk.streaming.kafka import routing_key


def test_transaction_events_route_by_customer_id() -> None:
    event = EventEnvelope(
        event_id="event-1",
        event_type="transaction.created",
        schema_version=1,
        occurred_at="2026-09-03T06:00:00+00:00",
        payload={"customer_id": "C-123", "transaction_id": "T-1"},
    )

    assert routing_key(event) == "C-123"


def test_non_transaction_events_route_by_event_id() -> None:
    event = EventEnvelope(
        event_id="event-2",
        event_type="transaction.risk_scored",
        schema_version=1,
        occurred_at="2026-09-03T06:00:00+00:00",
        payload={"customer_id": "C-123"},
    )

    assert routing_key(event) == "event-2"


def test_transaction_without_customer_id_falls_back_to_event_id() -> None:
    event = EventEnvelope(
        event_id="event-3",
        event_type="transaction.created",
        schema_version=1,
        occurred_at="2026-09-03T06:00:00+00:00",
        payload={},
    )

    assert routing_key(event) == "event-3"
