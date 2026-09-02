"""Reproducible comparison of logistic regression and XGBoost models."""
from __future__ import annotations

from dataclasses import dataclass

import pandas as pd

from financial_risk.models.baseline import (
    CATEGORICAL_FEATURES,
    NUMERIC_FEATURES,
    EvaluationResult,
    build_logistic_baseline,
    evaluate_binary_classifier,
)
from financial_risk.models.xgboost_model import build_xgboost_model, evaluate_xgboost

FEATURE_COLUMNS = NUMERIC_FEATURES + CATEGORICAL_FEATURES


@dataclass(frozen=True)
class ModelComparison:
    logistic: EvaluationResult
    xgboost: EvaluationResult


def fit_and_compare(
    train: pd.DataFrame, test: pd.DataFrame
) -> ModelComparison:
    """Fit both models using the same feature contract and compare test metrics."""
    y_train = train["is_fraud"].astype(int)
    negative = int((y_train == 0).sum())
    positive = int((y_train == 1).sum())
    scale_pos_weight = negative / max(positive, 1)

    logistic = build_logistic_baseline()
    logistic.fit(train[FEATURE_COLUMNS], y_train)

    xgboost = build_xgboost_model(scale_pos_weight=scale_pos_weight)
    xgboost.fit(train[FEATURE_COLUMNS], y_train)

    return ModelComparison(
        logistic=evaluate_binary_classifier(logistic, test),
        xgboost=evaluate_xgboost(xgboost, test),
    )
