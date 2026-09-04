"""Tests for Phase 41 challenger operating-point analysis."""
from __future__ import annotations

import numpy as np
import pandas as pd

from financial_risk.models.baseline import CATEGORICAL_FEATURES, NUMERIC_FEATURES
from financial_risk.models.challengers import build_random_forest
from financial_risk.models.cost_sensitive import CostPolicy
from financial_risk.models.operating_point import (
    analyze_model,
    realized_cost,
    select_f1_threshold,
)

FEATURE_COLUMNS = NUMERIC_FEATURES + CATEGORICAL_FEATURES


def _frame(rows: int = 40) -> pd.DataFrame:
    frame = pd.DataFrame({feature: 0.0 for feature in NUMERIC_FEATURES}, index=range(rows))
    for feature in CATEGORICAL_FEATURES:
        frame[feature] = "unknown"
    frame["amount"] = np.arange(1, rows + 1, dtype=float)
    frame["is_fraud"] = (frame["amount"] > rows * 0.7).astype(int)
    return frame


def test_select_f1_threshold_uses_validation_scores() -> None:
    y_true = pd.Series([0, 0, 1, 1])
    probabilities = np.array([0.10, 0.40, 0.60, 0.90])
    assert select_f1_threshold(y_true, probabilities) == 0.4


def test_realized_cost_counts_transaction_decisions() -> None:
    y_true = np.array([1, 0, 1])
    probabilities = np.array([0.95, 0.10, 0.50])
    policy = CostPolicy(false_positive_cost=5.0, false_negative_cost=100.0, review_cost=3.0)
    assert realized_cost(y_true, probabilities, policy) == 3.0


def test_operating_point_analysis_returns_common_metrics() -> None:
    frame = _frame()
    train = frame.iloc[:28].copy()
    validation = frame.iloc[28:34].copy()
    test = frame.iloc[34:].copy()
    model = build_random_forest(scale_pos_weight=2.0)
    model.fit(train[FEATURE_COLUMNS], train["is_fraud"])
    result = analyze_model(
        "Random Forest", model, validation, test, top_k=3, policy=CostPolicy()
    )
    assert result.model == "Random Forest"
    assert 0.0 <= result.selected_threshold <= 1.0
    assert result.review_volume >= 0
    assert result.fraud_captured >= 0
    assert result.precision_at_500 >= 0.0
    assert result.recall_at_500 >= 0.0
    assert result.lift_at_500 >= 0.0
    assert result.brier_score >= 0.0
    assert result.realized_cost >= 0.0
