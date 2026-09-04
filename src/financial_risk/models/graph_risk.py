"""Graph-risk integration into the final transaction risk decision."""
from __future__ import annotations

from dataclasses import dataclass

from financial_risk.models.risk_score import (
    RiskDecision,
    combine_risk_signals,
    decision_from_score,
)


@dataclass(frozen=True)
class GraphRiskAdjustment:
    """Auditable graph-risk contribution applied to an existing risk score."""

    base_score: float
    network_score: float
    community_score: float
    adjusted_score: float
    decision: RiskDecision


def integrate_graph_risk(
    fraud_probability: float,
    anomaly_score: float,
    network_score: float,
    velocity_score: float,
    community_risk: float = 0.0,
    *,
    fraud_weight: float = 0.50,
    anomaly_weight: float = 0.30,
    network_weight: float = 0.10,
    velocity_weight: float = 0.10,
    community_weight: float = 0.05,
) -> GraphRiskAdjustment:
    """Blend core risk signals, then apply a bounded community-risk uplift.

    Graph-risk integration uses fraud as the dominant signal while giving
    additional weight to anomaly detection. Community risk provides a small,
    bounded uplift to the final score.

    The four core signal weights must sum to 1.0. Community weight is bounded
    to prevent graph intelligence from overwhelming the core risk ensemble.
    """
    signals = [
        fraud_probability,
        anomaly_score,
        network_score,
        velocity_score,
        community_risk,
    ]
    if any(not 0.0 <= value <= 1.0 for value in signals):
        raise ValueError("All graph risk signals must be between 0 and 1")

    core_weights = [
        fraud_weight,
        anomaly_weight,
        network_weight,
        velocity_weight,
    ]
    if any(weight < 0.0 for weight in core_weights) or abs(
        sum(core_weights) - 1.0
    ) > 1e-9:
        raise ValueError("Core graph risk weights must be non-negative and sum to 1")

    if not 0.0 <= community_weight <= 0.10:
        raise ValueError("community_weight must be between 0 and 0.10")

    base_score = combine_risk_signals(
        fraud_probability,
        anomaly_score,
        network_score,
        velocity_score,
        fraud_weight=fraud_weight,
        anomaly_weight=anomaly_weight,
        network_weight=network_weight,
        velocity_weight=velocity_weight,
    )

    adjusted_score = float(base_score + community_risk * community_weight)
    adjusted_score = max(0.0, min(1.0, adjusted_score))

    return GraphRiskAdjustment(
        base_score=base_score,
        network_score=network_score,
        community_score=community_risk,
        adjusted_score=adjusted_score,
        decision=decision_from_score(adjusted_score),
    )
