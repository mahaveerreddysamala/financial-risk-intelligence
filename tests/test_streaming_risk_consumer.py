from __future__ import annotations

import pytest

from financial_risk.streaming.events import EventEnvelope
from financial_risk.streaming.risk_consumer import score_transaction_event, scoring_result_event


def _event(**overrides: object) -> EventEnvelope:
    payload = {
        "transaction_id": "TXN-1",
        "amount": 125.0,
        "country": "US",
        "channel": "ecommerce",
        "payment_method": "credit",
        "device_id": "D1",
        "ip_id": "IP1",
        "merchant_id": "M1",
        "shared_device_account_count": 2,
        "fraud_probability": 0.90,
        "anomaly_score": 0.80,
        "network_risk": 0.70,
        "velocity_risk": 0.60,
    }
    payload.update(overrides)
    return EventEnvelope(
        event_id="evt-1",
        event_type="transaction.created",
        schema_version=1,
        occurred_at="2026-09-02T12:00:00+00:00",
        payload=payload,
    )


def test_score_transaction_event_reuses_ensemble_decisioning() -> None:
    result = score_transaction_event(_event())
    assert result.event_id == "evt-1"
    assert result.transaction_id == "TXN-1"
    assert result.occurred_at == "2026-09-02T12:00:00+00:00"
    assert result.risk_score == pytest.approx(0.81)
    assert result.risk_band == "CRITICAL"
    assert result.action == "hold_and_investigate"
    assert result.investigation_case is not None
    assert result.investigation_case["transaction_id"] == "TXN-1"


def test_score_transaction_event_skips_case_for_noncritical_result() -> None:
    result = score_transaction_event(
        _event(
            fraud_probability=0.20,
            anomaly_score=0.10,
            network_risk=0.10,
            velocity_risk=0.10,
        )
    )
    assert result.risk_score == pytest.approx(0.15)
    assert result.risk_band == "LOW"
    assert result.action == "approve"
    assert result.investigation_case is None


@pytest.mark.parametrize(
    "field",
    ["fraud_probability", "anomaly_score", "network_risk", "velocity_risk"],
)
def test_score_transaction_event_requires_all_signals(field: str) -> None:
    payload = _event().payload
    payload.pop(field)
    event = EventEnvelope("evt-1", "transaction.created", 1, _event().occurred_at, payload)
    with pytest.raises(ValueError, match="missing risk signal"):
        score_transaction_event(event)


def test_score_transaction_event_rejects_unsupported_event_type() -> None:
    event = _event()
    invalid = EventEnvelope(
        event.event_id,
        "account.updated",
        event.schema_version,
        event.occurred_at,
        event.payload,
    )
    with pytest.raises(ValueError, match="unsupported event_type"):
        score_transaction_event(invalid)


def test_score_transaction_event_rejects_out_of_range_signal() -> None:
    with pytest.raises(ValueError, match="between 0 and 1"):
        score_transaction_event(_event(network_risk=1.2))


def test_scoring_result_event_preserves_traceability() -> None:
    result = score_transaction_event(_event(), build_case=False)
    event = scoring_result_event(result)
    assert event.event_id == result.event_id
    assert event.event_type == "transaction.risk_scored"
    assert event.occurred_at == result.occurred_at
    assert event.payload["risk_band"] == "CRITICAL"
    assert event.payload["transaction_id"] == "TXN-1"
