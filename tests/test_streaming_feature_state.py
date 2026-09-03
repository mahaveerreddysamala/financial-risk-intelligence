from __future__ import annotations

import pytest

from financial_risk.streaming.feature_state import MODEL_FEATURE_COLUMNS, StreamingFeatureService


def _payload(transaction_id: str, amount: float = 100.0) -> dict[str, object]:
    return {
        "transaction_id": transaction_id,
        "customer_id": "C-1",
        "amount": amount,
        "merchant_id": "M-1",
        "device_id": "D-1",
        "country": "US",
        "shared_device_account_count": 1,
        "merchant_category": "electronics",
        "payment_method": "credit",
        "channel": "ecommerce",
    }


def test_prepare_uses_prior_history_only() -> None:
    service = StreamingFeatureService()
    service.commit(_payload("TXN-1", 100.0), "2026-09-03T03:00:00+00:00")

    features = service.prepare(_payload("TXN-2", 200.0), "2026-09-03T03:02:00+00:00")

    assert list(features) == MODEL_FEATURE_COLUMNS
    assert features["customer_txn_count_7d"] == 1
    assert features["customer_avg_amount_30d"] == pytest.approx(100.0)
    assert features["amount_vs_customer_avg"] == pytest.approx(2.0)
    assert features["txn_count_5m"] == 1


def test_prepare_does_not_mutate_history_before_commit() -> None:
    service = StreamingFeatureService()
    first = service.prepare(_payload("TXN-1"), "2026-09-03T03:00:00+00:00")
    assert first["customer_txn_count_7d"] == 0

    second = service.prepare(_payload("TXN-2"), "2026-09-03T03:02:00+00:00")
    assert second["customer_txn_count_7d"] == 0


def test_prepare_requires_real_time_transaction_fields() -> None:
    service = StreamingFeatureService()
    with pytest.raises(ValueError, match="missing fields"):
        service.prepare({"transaction_id": "TXN-1"}, "2026-09-03T03:00:00+00:00")


def test_commit_expires_history_after_window() -> None:
    service = StreamingFeatureService(history_days=30)
    service.commit(_payload("TXN-1"), "2026-01-01T00:00:00+00:00")
    service.commit(_payload("TXN-2"), "2026-02-02T00:00:00+00:00")

    features = service.prepare(_payload("TXN-3"), "2026-02-02T00:01:00+00:00")
    assert features["customer_txn_count_7d"] == 0
    assert features["customer_avg_amount_30d"] == pytest.approx(100.0)
