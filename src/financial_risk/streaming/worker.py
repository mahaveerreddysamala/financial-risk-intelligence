"""Transport-agnostic real-time risk-scoring worker helpers."""
from __future__ import annotations

from collections.abc import Callable

from financial_risk.models.artifact import PersistedModelService
from financial_risk.streaming.events import EventEnvelope
from financial_risk.streaming.feature_state import StreamingFeatureService
from financial_risk.streaming.risk_consumer import RiskScoringResult, score_transaction_event


def process_event(
    event: EventEnvelope,
    *,
    publish: Callable[[RiskScoringResult], None] | None = None,
    build_case: bool = True,
    model_service: PersistedModelService | None = None,
    feature_service: StreamingFeatureService | None = None,
) -> RiskScoringResult:
    """Score one event and optionally publish the structured result.

    When a feature service is supplied, raw transaction fields are converted into
    prior-only behavioral and velocity model features before persisted-model
    inference. Feature history is committed only after scoring succeeds.
    """
    if publish is not None and not callable(publish):
        raise TypeError("publish must be callable when provided")

    scoring_event = event
    if feature_service is not None:
        payload = dict(event.payload)
        payload["model_features"] = feature_service.prepare(payload, event.occurred_at)
        scoring_event = EventEnvelope(
            event_id=event.event_id,
            event_type=event.event_type,
            schema_version=event.schema_version,
            occurred_at=event.occurred_at,
            payload=payload,
        )

    result = score_transaction_event(
        scoring_event,
        build_case=build_case,
        model_service=model_service,
    )

    if feature_service is not None:
        feature_service.commit(event.payload, event.occurred_at)
    if publish is not None:
        publish(result)
    return result
