from __future__ import annotations

from financial_risk.streaming.events import EventEnvelope
from financial_risk.streaming.risk_consumer import RiskScoringResult
from financial_risk.streaming.runtime import (
    DeadLetterRecord,
    InMemoryIdempotencyStore,
    RetryPolicy,
    StreamingRuntime,
)


def _event(event_id: str = "evt-1") -> EventEnvelope:
    return EventEnvelope(
        event_id=event_id,
        event_type="transaction.created",
        schema_version=1,
        occurred_at="2026-09-02T12:00:00+00:00",
        payload={"transaction_id": "TXN-1"},
    )


def _result() -> RiskScoringResult:
    return RiskScoringResult(
        event_id="evt-1",
        transaction_id="TXN-1",
        occurred_at="2026-09-02T12:00:00+00:00",
        risk_score=0.81,
        risk_band="CRITICAL",
        action="hold_and_investigate",
        fraud_probability=0.9,
        anomaly_score=0.8,
        network_risk=0.7,
        velocity_risk=0.6,
    )


def _runtime(processor, **kwargs) -> StreamingRuntime:
    class FakeConsumer:
        def poll(self, timeout: float = 1.0):
            return None

        def close(self) -> None:
            return None

    return StreamingRuntime(consumer=FakeConsumer(), processor=processor, **kwargs)


def test_runtime_succeeds_and_marks_idempotency() -> None:
    calls = []
    store = InMemoryIdempotencyStore()
    runtime = _runtime(lambda event: calls.append(event.event_id) or _result(), idempotency=store)

    assert runtime.process_event(_event()) is True
    assert runtime.process_event(_event()) is False
    assert calls == ["evt-1"]
    assert runtime.stats.received == 2
    assert runtime.stats.succeeded == 1
    assert runtime.stats.duplicates == 1


def test_runtime_retries_then_succeeds() -> None:
    attempts = []

    def processor(event):
        attempts.append(event.event_id)
        if len(attempts) < 3:
            raise RuntimeError("temporary failure")
        return _result()

    sleeps = []
    runtime = _runtime(
        processor,
        retry_policy=RetryPolicy(max_attempts=3, backoff_seconds=0.25),
        sleep=sleeps.append,
    )

    assert runtime.process_event(_event()) is True
    assert len(attempts) == 3
    assert sleeps == [0.25, 0.25]
    assert runtime.stats.retried == 2


def test_runtime_dead_letters_after_exhausting_retries() -> None:
    failures: list[DeadLetterRecord] = []

    def processor(event):
        raise ValueError("invalid event payload")

    runtime = _runtime(
        processor,
        retry_policy=RetryPolicy(max_attempts=2, backoff_seconds=0),
        sleep=lambda _: None,
        dead_letter=failures.append,
    )

    assert runtime.process_event(_event()) is False
    assert runtime.stats.retried == 1
    assert runtime.stats.dead_lettered == 1
    assert len(failures) == 1
    assert failures[0].attempts == 2
    assert failures[0].error_type == "ValueError"
    assert failures[0].event.event_id == "evt-1"


def test_runtime_publishes_only_after_processing_succeeds() -> None:
    published: list[RiskScoringResult] = []
    runtime = _runtime(
        lambda event: _result(),
        publisher=published.append,
    )

    assert runtime.process_event(_event()) is True
    assert published == [_result()]


def test_retry_policy_rejects_invalid_values() -> None:
    for policy in (
        (0, 1.0),
        (1, -1.0),
    ):
        try:
            RetryPolicy(max_attempts=policy[0], backoff_seconds=policy[1])
        except ValueError:
            pass
        else:
            raise AssertionError("RetryPolicy accepted invalid configuration")
