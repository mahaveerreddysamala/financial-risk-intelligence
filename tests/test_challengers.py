"""Tests for Phase 40 fraud-model challenger pipelines."""
from __future__ import annotations

import pandas as pd

from financial_risk.models.baseline import CATEGORICAL_FEATURES, NUMERIC_FEATURES
from financial_risk.models.challengers import (
    build_lightgbm,
    build_random_forest,
    evaluate_challenger,
)


FEATURE_COLUMNS = NUMERIC_FEATURES + CATEGORICAL_FEATURES


def _frame() -> pd.DataFrame:
    rows = []
    for index in range(24):
        row = {feature: 0.0 for feature in NUMERIC_FEATURES}
        row.update({feature: "unknown" for feature in CATEGORICAL_FEATURES})
        row["amount"] = float(index + 1)
        row["is_fraud"] = int(index % 4 == 0)
        rows.append(row)
    return pd.DataFrame(rows)


def test_random_forest_uses_common_feature_contract() -> None:
    model = build_random_forest(scale_pos_weight=3.0)
    frame = _frame()
    model.fit(frame[FEATURE_COLUMNS], frame["is_fraud"])
    result = evaluate_challenger(model, frame, "Random Forest")
    assert result.model == "Random Forest"
    assert 0.0 <= result.roc_auc <= 1.0
    assert 0.0 <= result.pr_auc <= 1.0


def test_lightgbm_uses_common_feature_contract() -> None:
    model = build_lightgbm(scale_pos_weight=3.0)
    frame = _frame()
    model.fit(frame[FEATURE_COLUMNS], frame["is_fraud"])
    result = evaluate_challenger(model, frame, "LightGBM")
    assert result.model == "LightGBM"
    assert 0.0 <= result.roc_auc <= 1.0
    assert 0.0 <= result.pr_auc <= 1.0


def test_challenger_evaluation_respects_threshold() -> None:
    frame = _frame()
    model = build_random_forest()
    model.fit(frame[FEATURE_COLUMNS], frame["is_fraud"])
    low = evaluate_challenger(model, frame, "Random Forest", threshold=0.1)
    high = evaluate_challenger(model, frame, "Random Forest", threshold=0.9)
    assert low.recall >= high.recall
