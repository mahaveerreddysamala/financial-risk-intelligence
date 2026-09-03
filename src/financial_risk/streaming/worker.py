"""Transport-agnostic real-time risk-scoring worker helpers."""
from __future__ import annotations

from collections.abc import Callable

from financial_risk.models.artifact import PersistedModelService
from financial_risk.streaming.events import EventEnvelope
from financial_risk.streaming.risk_consumer import RiskScoringResult, score_transaction_event


def process_event(
    event: EventEnvelope,
    *,
    publish: Callable[[RiskScoringResult], None] | None = None,
    build_case: bool = True,
    model_service: PersistedModelService | None = None,
) -> RiskScoringResult:
    """Score one event and optionally publish the structured result.

    The worker deliberately contains no Kafka-specific behavior. A Kafka adapter,
    batch replay job, or test harness can supply the publisher callback. When a
    persisted model service is supplied, events containing ``model_features`` use
    the persisted XGBoost artifact for fraud probability.
    """
    if publish is not None and not callable(publish):
        raise TypeError("publish must be callable when provided")
    result = score_transaction_event(
        event,
        build_case=build_case,
        model_service=model_service,
    )
    if publish is not None:
        publish(result)
    return result
