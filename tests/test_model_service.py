import pytest

from financial_risk.models.service import RiskModelMetadata, RiskModelService


def test_predict_returns_versioned_serving_metadata():
    service = RiskModelService(
        RiskModelMetadata(
            model_name="risk-model",
            model_version="2.1.0",
            feature_contract_version="3.0",
        )
    )

    prediction = service.predict(
        fraud_probability=0.9,
        anomaly_score=0.8,
        network_score=0.7,
        velocity_score=0.6,
    )

    assert prediction.risk_score == pytest.approx(0.81)
    assert prediction.risk_band == "CRITICAL"
    assert prediction.action == "hold_and_investigate"
    assert prediction.model_name == "risk-model"
    assert prediction.model_version == "2.1.0"
    assert prediction.feature_contract_version == "3.0"


def test_predict_rejects_out_of_range_signal():
    service = RiskModelService()

    with pytest.raises(ValueError, match="between 0 and 1"):
        service.predict(
            fraud_probability=1.1,
            anomaly_score=0.8,
            network_score=0.7,
            velocity_score=0.6,
        )
