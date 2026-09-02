"""Ensemble risk scoring and operational decisioning utilities."""
from __future__ import annotations

from dataclasses import dataclass


def combine_risk_signals(
    fraud_probability: float,
    anomaly_score: float,
    network_score: float,
    velocity_score: float,
    *,
    fraud_weight: float = 0.50,
    anomaly_weight: float = 0.20,
    network_weight: float = 0.20,
    velocity_weight: float = 0.10,
) -> float:
    """Combine normalized risk signals into a bounded score."""
    signals = [fraud_probability, anomaly_score, network_score, velocity_score]
    if any(not 0.0 <= value <= 1.0 for value in signals):
        raise ValueError("All risk signals must be between 0 and 1")
    weights = [fraud_weight, anomaly_weight, network_weight, velocity_weight]
    if any(weight < 0.0 for weight in weights) or abs(sum(weights) - 1.0) > 1e-9:
        raise ValueError("Risk weights must be non-negative and sum to 1")
    return float(
        fraud_probability * fraud_weight
        + anomaly_score * anomaly_weight
        + network_score * network_weight
        + velocity_score * velocity_weight
    )


@dataclass(frozen=True)
class RiskDecision:
    score: float
    level: str
    action: str


def decision_from_score(score: float) -> RiskDecision:
    """Map a normalized risk score to an operational action."""
    if not 0.0 <= score <= 1.0:
        raise ValueError("score must be between 0 and 1")
    if score >= 0.80:
        return RiskDecision(score, "CRITICAL", "hold_and_investigate")
    if score >= 0.60:
        return RiskDecision(score, "HIGH", "step_up_verification")
    if score >= 0.30:
        return RiskDecision(score, "MEDIUM", "monitor")
    return RiskDecision(score, "LOW", "approve")
