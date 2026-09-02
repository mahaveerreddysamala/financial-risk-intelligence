"""XGBoost fraud model with imbalance-aware training and evaluation."""
from __future__ import annotations

from dataclasses import dataclass

import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.metrics import (
    average_precision_score,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
)
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder
from xgboost import XGBClassifier

from financial_risk.models.baseline import CATEGORICAL_FEATURES, NUMERIC_FEATURES, TARGET


@dataclass(frozen=True)
class EvaluationResult:
    roc_auc: float
    pr_auc: float
    precision: float
    recall: float
    f1: float


def build_xgboost_model(scale_pos_weight: float = 1.0) -> Pipeline:
    """Create an XGBoost pipeline with one-hot encoded categorical features."""
    preprocessor = ColumnTransformer(
        transformers=[
            ("numeric", "passthrough", NUMERIC_FEATURES),
            ("categorical", OneHotEncoder(handle_unknown="ignore"), CATEGORICAL_FEATURES),
        ],
        remainder="drop",
    )
    model = XGBClassifier(
        n_estimators=300,
        max_depth=6,
        learning_rate=0.05,
        subsample=0.9,
        colsample_bytree=0.9,
        min_child_weight=3,
        objective="binary:logistic",
        eval_metric="logloss",
        tree_method="hist",
        scale_pos_weight=scale_pos_weight,
        random_state=42,
        n_jobs=4,
    )
    return Pipeline([("preprocess", preprocessor), ("model", model)])


def evaluate_xgboost(model: Pipeline, test: pd.DataFrame, threshold: float = 0.5) -> EvaluationResult:
    """Evaluate a fitted model using probability metrics and a configurable threshold."""
    y_true = test[TARGET].astype(int)
    feature_columns = NUMERIC_FEATURES + CATEGORICAL_FEATURES
    probabilities = model.predict_proba(test[feature_columns])[:, 1]
    predictions = (probabilities >= threshold).astype(int)
    return EvaluationResult(
        roc_auc=float(roc_auc_score(y_true, probabilities)),
        pr_auc=float(average_precision_score(y_true, probabilities)),
        precision=float(precision_score(y_true, predictions, zero_division=0)),
        recall=float(recall_score(y_true, predictions, zero_division=0)),
        f1=float(f1_score(y_true, predictions, zero_division=0)),
    )
