"""Transport-agnostic real-time risk-scoring worker helpers."""
from __future__ import annotations

from collections.abc import Callable

from financial_risk.streaming.events import EventEnvelope
from financial_risk.streaming.risk_consumer import RiskScoringResult, score_transaction_event


def process_event(
    event: EventEnvelope,
    *,
    publish: Callable[[RiskScoringResult], None] | None = None,
    build_case: bool = True,
) -> RiskScoringResult:
    """Score one event and optionally publish the structured result.

    The worker deliberately contains no Kafka-specific behavior. A Kafka adapter,
    batch replay job, or test harness can supply the publisher callback.
    """
    if publish is not None and not callable(publish):
        raise TypeError("publish must be callable when provided")
    result = score_transaction_event(event, build_case=build_case)
    if publish is not None:
        publish(result)
    return result
