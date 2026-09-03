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
    assert result.feature_telemetry == {"inference_source": "precomputed_signal"}


def test_score_transaction_event_with_persisted_model_service() -> None:
    class DummyPersistedService:
        def predict(self, features):
            return type(
                "Prediction",
                (),
                {
                    "fraud_probability": 0.95,
                    "model_name": "financial-fraud-xgboost",
                    "model_version": "1.0.0",
                    "feature_contract_version": "1.0",
                },
            )()

    model_features = {
        "amount": 250.0,
        "is_international": 0,
        "is_night": 1,
        "shared_device_account_count": 2,
        "customer_txn_count_7d": 3,
        "customer_avg_amount_30d": 100.0,
        "customer_std_amount_30d": 25.0,
        "customer_unique_merchants_7d": 2,
        "customer_unique_devices_30d": 1,
        "customer_international_rate_30d": 0.1,
        "customer_night_txn_rate_30d": 0.2,
        "amount_vs_customer_avg": 2.5,
        "amount_zscore": 6.0,
        "txn_count_5m": 2,
        "txn_count_1h": 3,
        "txn_count_24h": 4,
        "merchant_category": "electronics",
        "payment_method": "credit",
        "channel": "ecommerce",
        "country": "US",
    }
    result = score_transaction_event(
        _event(
            model_features=model_features,
            anomaly_score=0.80,
            network_risk=0.70,
            velocity_risk=0.60,
        ),
        model_service=DummyPersistedService(),
    )
    assert result.fraud_probability == pytest.approx(0.95)
    assert result.model_name == "financial-fraud-xgboost"
    assert result.model_version == "1.0.0"
    assert result.feature_contract_version == "1.0"
    assert result.risk_score == pytest.approx(0.835)
    assert result.risk_band == "CRITICAL"
    assert result.feature_telemetry == {
        "inference_source": "persisted_model_artifact",
        "feature_count": 20,
        "customer_txn_count_7d": 3,
        "customer_avg_amount_30d": 100.0,
        "amount_vs_customer_avg": 2.5,
        "txn_count_5m": 2,
        "txn_count_1h": 3,
        "txn_count_24h": 4,
    }


def test_model_features_require_persisted_service() -> None:
    with pytest.raises(ValueError, match="persisted model service"):
        score_transaction_event(_event(model_features={"amount": 250.0}))


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


def test_scoring_result_event_preserves_traceability_and_telemetry() -> None:
    result = score_transaction_event(_event(), build_case=False)
    event = scoring_result_event(result)
    assert event.event_id == result.event_id
    assert event.event_type == "transaction.risk_scored"
    assert event.occurred_at == result.occurred_at
    assert event.payload["risk_band"] == "CRITICAL"
    assert event.payload["transaction_id"] == "TXN-1"
    assert event.payload["feature_telemetry"] == {"inference_source": "precomputed_signal"}
