"""Model-serving boundary for the production risk API."""
from __future__ import annotations

from dataclasses import dataclass

from financial_risk.models.risk_score import combine_risk_signals, decision_from_score


@dataclass(frozen=True)
class RiskModelMetadata:
    """Stable serving metadata exposed with each risk decision."""

    model_name: str = "financial-risk-ensemble"
    model_version: str = "1.0.0"
    feature_contract_version: str = "1.0"


@dataclass(frozen=True)
class RiskPrediction:
    """Risk decision returned by the serving boundary."""

    risk_score: float
    risk_band: str
    action: str
    model_name: str
    model_version: str
    feature_contract_version: str


class RiskModelService:
    """Encapsulate model inference behind a deployable service contract."""

    def __init__(self, metadata: RiskModelMetadata | None = None) -> None:
        self.metadata = metadata or RiskModelMetadata()

    def predict(
        self,
        *,
        fraud_probability: float,
        anomaly_score: float,
        network_score: float,
        velocity_score: float,
    ) -> RiskPrediction:
        values = {
            "fraud_probability": fraud_probability,
            "anomaly_score": anomaly_score,
            "network_score": network_score,
            "velocity_score": velocity_score,
        }
        if any(value < 0 or value > 1 for value in values.values()):
            raise ValueError("all model signals must be between 0 and 1")

        risk_score = combine_risk_signals(
            fraud_probability=fraud_probability,
            anomaly_score=anomaly_score,
            network_score=network_score,
            velocity_score=velocity_score,
        )
        decision = decision_from_score(risk_score)
        return RiskPrediction(
            risk_score=risk_score,
            risk_band=decision.level,
            action=decision.action,
            model_name=self.metadata.model_name,
            model_version=self.metadata.model_version,
            feature_contract_version=self.metadata.feature_contract_version,
        )
