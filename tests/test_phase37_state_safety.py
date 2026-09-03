from __future__ import annotations

from financial_risk.streaming.events import EventEnvelope
from financial_risk.streaming.feature_state import StreamingFeatureService
from financial_risk.streaming.risk_consumer import RiskScoringResult
from financial_risk.streaming.runtime import InMemoryIdempotencyStore, StreamingRuntime


def _event(event_id: str = "evt-1") -> EventEnvelope:
    return EventEnvelope(
        event_id=event_id,
        event_type="transaction.created",
        schema_version=1,
        occurred_at="2026-09-03T06:00:00+00:00",
        payload={"transaction_id": event_id},
    )


def _result() -> RiskScoringResult:
    return RiskScoringResult(
        event_id="evt-1",
        transaction_id="TXN-1",
        occurred_at="2026-09-03T06:00:00+00:00",
        risk_score=0.2,
        risk_band="LOW",
        action="approve",
        fraud_probability=0.1,
        anomaly_score=0.1,
        network_risk=0.1,
        velocity_risk=0.1,
    )


def test_idempotency_claim_blocks_concurrent_duplicate_claims() -> None:
    store = InMemoryIdempotencyStore()

    assert store.claim("evt-1") is True
    assert store.claim("evt-1") is False
    store.release("evt-1")
    assert store.claim("evt-1") is True


def test_successful_mark_clears_claim_and_blocks_future_claim() -> None:
    store = InMemoryIdempotencyStore()

    assert store.claim("evt-1") is True
    store.mark("evt-1")

    assert store.contains("evt-1") is True
    assert store.claim("evt-1") is False


def test_runtime_releases_claim_after_failed_processing() -> None:
    store = InMemoryIdempotencyStore()
    runtime = StreamingRuntime(
        consumer=type("Consumer", (), {"poll": lambda self, timeout=1.0: None, "close": lambda self: None})(),
        processor=lambda event: (_ for _ in ()).throw(ValueError("bad payload")),
        idempotency=store,
        sleep=lambda _: None,
    )

    assert runtime.process_event(_event()) is False
    assert store.claim("evt-1") is True


def test_feature_service_uses_atomic_append_history_when_available() -> None:
    class FakeStateStore:
        def __init__(self) -> None:
            self.history_calls: list[tuple[str, dict[str, object], str]] = []

        def get_history(self, customer_id: str) -> list[dict[str, object]]:
            return []

        def append_history(
            self,
            customer_id: str,
            transaction: dict[str, object],
            *,
            cutoff_timestamp: str,
        ) -> list[dict[str, object]]:
            self.history_calls.append((customer_id, transaction, cutoff_timestamp))
            return [transaction]

    store = FakeStateStore()
    service = StreamingFeatureService(state_store=store)
    payload = {
        "transaction_id": "TXN-1",
        "customer_id": "C-1",
        "amount": 100.0,
        "merchant_id": "M-1",
        "device_id": "D-1",
        "country": "US",
        "shared_device_account_count": 1,
        "merchant_category": "electronics",
        "payment_method": "credit",
        "channel": "ecommerce",
    }

    service.commit(payload, "2026-09-03T06:00:00+00:00")

    assert len(store.history_calls) == 1
    assert store.history_calls[0][0] == "C-1"
    assert store.history_calls[0][1]["transaction_id"] == "TXN-1"
