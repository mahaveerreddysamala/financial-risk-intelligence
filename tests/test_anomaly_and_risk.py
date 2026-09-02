import numpy as np
import pytest

from financial_risk.data_generation.generator import generate_transactions
from financial_risk.features.pipeline import build_feature_table
from financial_risk.models.anomaly import fit_anomaly_detector, score_anomalies
from financial_risk.models.risk_score import combine_risk_signals, decision_from_score


def test_anomaly_detector_returns_bounded_scores() -> None:
    data = build_feature_table(generate_transactions(1_500, seed=42))
    train = data.iloc[:1_000].copy()
    test = data.iloc[1_000:].copy()

    scaler, detector = fit_anomaly_detector(train)
    result = score_anomalies(test, scaler, detector)

    assert len(result.scores) == len(test)
    assert len(result.flags) == len(test)
    assert np.all(result.scores >= 0.0)
    assert np.all(result.scores <= 1.0)
    assert result.flags.dtype == bool


def test_risk_score_and_decisions() -> None:
    score = combine_risk_signals(0.9, 0.8, 0.7, 0.6)
    assert score == pytest.approx(0.81)
    decision = decision_from_score(score)
    assert decision.level == "CRITICAL"
    assert decision.action == "hold_and_investigate"

    assert decision_from_score(0.10).action == "approve"
    assert decision_from_score(0.40).action == "monitor"
    assert decision_from_score(0.65).action == "step_up_verification"


def test_risk_weights_must_sum_to_one() -> None:
    with pytest.raises(ValueError, match="sum to 1"):
        combine_risk_signals(0.5, 0.5, 0.5, 0.5, fraud_weight=0.9)
