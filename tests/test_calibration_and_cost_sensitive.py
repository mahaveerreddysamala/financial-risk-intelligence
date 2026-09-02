import numpy as np
import pandas as pd
import pytest

from financial_risk.data_generation.generator import generate_transactions
from financial_risk.features.pipeline import build_feature_table
from financial_risk.models.baseline import CATEGORICAL_FEATURES, NUMERIC_FEATURES
from financial_risk.models.calibration import calibration_bins, evaluate_calibration
from financial_risk.models.cost_sensitive import (
    CostPolicy,
    choose_cost_sensitive_action,
    expected_costs,
)
from financial_risk.models.xgboost_model import build_xgboost_model


def test_expected_costs_and_actions():
    policy = CostPolicy(false_positive_cost=5, false_negative_cost=100, review_cost=3)
    costs = expected_costs(np.array([0.01, 0.50, 0.99]), policy)

    assert list(costs.columns) == [
        "fraud_probability",
        "approve_expected_cost",
        "review_expected_cost",
        "hold_expected_cost",
    ]
    assert choose_cost_sensitive_action(0.01, policy) == "approve"
    assert choose_cost_sensitive_action(0.50, policy) == "review"
    assert choose_cost_sensitive_action(0.99, policy) == "hold"


def test_cost_policy_validates_costs():
    with pytest.raises(ValueError, match="greater than zero"):
        CostPolicy(false_negative_cost=0)
    with pytest.raises(ValueError, match="non-negative"):
        CostPolicy(review_cost=-1)
    with pytest.raises(ValueError, match="fraud probabilities"):
        expected_costs(np.array([1.1]))


def test_calibration_metrics_and_bins():
    raw = generate_transactions(rows=1000, seed=7)
    features = build_feature_table(raw)
    model = build_xgboost_model(scale_pos_weight=5.0)
    model.fit(features[NUMERIC_FEATURES + CATEGORICAL_FEATURES], features["is_fraud"])

    result = evaluate_calibration(model, features.tail(250))
    bins = calibration_bins(model, features.tail(250), bins=5)

    assert 0 <= result.brier_score <= 1
    assert 0 <= result.mean_predicted_probability <= 1
    assert 0 <= result.observed_fraud_rate <= 1
    assert len(bins) <= 5
    assert bins["samples"].sum() == 250


def test_calibration_bins_validate_bin_count():
    with pytest.raises(ValueError, match="at least 2"):
        calibration_bins(object(), pd.DataFrame(), bins=1)
