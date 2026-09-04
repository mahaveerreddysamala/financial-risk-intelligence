"""Graph-risk integration into the final transaction risk decision."""
from __future__ import annotations

from dataclasses import dataclass

from financial_risk.models.risk_score import RiskDecision, combine_risk_signals, decision_from_score


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
    anomaly_weight: float = 0.20,
    network_weight: float = 0.15,
    velocity_weight: float = 0.10,
    community_weight: float = 0.05,
) -> GraphRiskAdjustment:
    """Blend network and community signals into the operational risk score.

    Community risk is an externally derived graph signal normalized to [0, 1].
    The default weights preserve fraud-model dominance while making connected
    entity and community intelligence part of the final decision.
    """
    signals = [fraud_probability, anomaly_score, network_score, velocity_score, community_risk]
    if any(not 0.0 <= value <= 1.0 for value in signals):
        raise ValueError("All graph risk signals must be between 0 and 1")
    weights = [fraud_weight, anomaly_weight, network_weight, velocity_weight, community_weight]
    if any(weight < 0.0 for weight in weights) or abs(sum(weights) - 1.0) > 1e-9:
        raise ValueError("Graph risk weights must be non-negative and sum to 1")

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
    adjusted_score = float(
        fraud_probability * fraud_weight
        + anomaly_score * anomaly_weight
        + network_score * network_weight
        + velocity_score * velocity_weight
        + community_risk * community_weight
    )
    adjusted_score = max(0.0, min(1.0, adjusted_score))
    return GraphRiskAdjustment(
        base_score=base_score,
        network_score=network_score,
        community_score=community_risk,
        adjusted_score=adjusted_score,
        decision=decision_from_score(adjusted_score),
    )
