"""Real-time transaction risk scoring over the streaming event contract."""
from __future__ import annotations

from dataclasses import asdict, dataclass
from typing import Any

from financial_risk.investigation.case_builder import build_investigation_case, case_to_dict
from financial_risk.models.risk_score import RiskDecision, combine_risk_signals, decision_from_score
from financial_risk.streaming.events import EventEnvelope


@dataclass(frozen=True)
class RiskScoringResult:
    """Structured streaming inference result for one transaction event."""

    event_id: str
    transaction_id: str
    occurred_at: str
    risk_score: float
    risk_band: str
    action: str
    fraud_probability: float
    anomaly_score: float
    network_risk: float
    velocity_risk: float
    investigation_case: dict[str, Any] | None = None

    def to_dict(self) -> dict[str, Any]:
        """Return a JSON-serializable scoring result."""
        return asdict(self)


def _signal(payload: dict[str, Any], name: str) -> float:
    """Read and validate one normalized risk signal from an event payload."""
    value = payload.get(name)
    if value is None:
        raise ValueError(f"transaction event missing risk signal: {name}")
    try:
        normalized = float(value)
    except (TypeError, ValueError) as exc:
        raise ValueError(f"risk signal {name} must be numeric") from exc
    if not 0.0 <= normalized <= 1.0:
        raise ValueError(f"risk signal {name} must be between 0 and 1")
    return normalized


def score_transaction_event(
    event: EventEnvelope,
    *,
    build_case: bool = True,
) -> RiskScoringResult:
    """Score one validated transaction event using the ensemble risk engine.

    The streaming boundary expects upstream feature generation/model inference to
    supply normalized ``fraud_probability``, ``anomaly_score``, ``network_risk``,
    and ``velocity_risk`` values in the event payload. This keeps the Kafka consumer
    transport-focused while reusing the established decisioning and case layers.
    """
    if event.event_type != "transaction.created":
        raise ValueError(f"unsupported event_type: {event.event_type}")

    payload = event.payload
    transaction_id = str(payload.get("transaction_id") or event.event_id)
    fraud_probability = _signal(payload, "fraud_probability")
    anomaly_score = _signal(payload, "anomaly_score")
    network_risk = _signal(payload, "network_risk")
    velocity_risk = _signal(payload, "velocity_risk")

    score = combine_risk_signals(
        fraud_probability,
        anomaly_score,
        network_risk,
        velocity_risk,
    )
    decision: RiskDecision = decision_from_score(score)

    investigation_case = None
    if build_case and decision.action == "hold_and_investigate":
        case = build_investigation_case(
            payload,
            fraud_probability=fraud_probability,
            anomaly_score=anomaly_score,
            network_risk=network_risk,
            velocity_risk=velocity_risk,
            risk_score=decision.score,
            risk_band=decision.level,
            action=decision.action,
        )
        investigation_case = case_to_dict(case)

    return RiskScoringResult(
        event_id=event.event_id,
        transaction_id=transaction_id,
        occurred_at=event.occurred_at,
        risk_score=decision.score,
        risk_band=decision.level,
        action=decision.action,
        fraud_probability=fraud_probability,
        anomaly_score=anomaly_score,
        network_risk=network_risk,
        velocity_risk=velocity_risk,
        investigation_case=investigation_case,
    )


def scoring_result_event(result: RiskScoringResult) -> EventEnvelope:
    """Build a downstream event envelope for risk-scoring results."""
    return EventEnvelope(
        event_id=result.event_id,
        event_type="transaction.risk_scored",
        schema_version=1,
        occurred_at=result.occurred_at,
        payload=result.to_dict(),
    )
