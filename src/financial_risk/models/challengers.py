"""Random Forest and LightGBM challenger models for fraud benchmarking."""
from __future__ import annotations

from dataclasses import dataclass

import pandas as pd
from lightgbm import LGBMClassifier
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (
    average_precision_score,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
)
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder

from financial_risk.models.baseline import CATEGORICAL_FEATURES, NUMERIC_FEATURES, TARGET

FEATURE_COLUMNS = NUMERIC_FEATURES + CATEGORICAL_FEATURES


@dataclass(frozen=True)
class ChallengerEvaluation:
    model: str
    roc_auc: float
    pr_auc: float
    precision: float
    recall: float
    f1: float


def _preprocessor() -> ColumnTransformer:
    return ColumnTransformer(
        transformers=[
            ("numeric", "passthrough", NUMERIC_FEATURES),
            (
                "categorical",
                OneHotEncoder(handle_unknown="ignore", sparse_output=False, dtype="float32"),
                CATEGORICAL_FEATURES,
            ),
        ],
        remainder="drop",
    )


def build_random_forest(scale_pos_weight: float = 1.0) -> Pipeline:
    """Build a reproducible imbalance-aware Random Forest pipeline."""
    model = RandomForestClassifier(
        n_estimators=300,
        max_depth=12,
        min_samples_leaf=2,
        class_weight={0: 1.0, 1: max(scale_pos_weight, 1.0)},
        random_state=42,
        n_jobs=4,
    )
    return Pipeline([("preprocess", _preprocessor()), ("model", model)])


def build_lightgbm(scale_pos_weight: float = 1.0) -> Pipeline:
    """Build a reproducible imbalance-aware LightGBM pipeline."""
    model = LGBMClassifier(
        n_estimators=300,
        learning_rate=0.05,
        num_leaves=31,
        max_depth=-1,
        min_child_samples=20,
        subsample=0.9,
        colsample_bytree=0.9,
        reg_lambda=1.0,
        objective="binary",
        scale_pos_weight=max(scale_pos_weight, 1.0),
        random_state=42,
        n_jobs=1,
        verbosity=-1,
        force_col_wise=True,
    )
    return Pipeline([("preprocess", _preprocessor()), ("model", model)])


def evaluate_challenger(
    model: Pipeline,
    test: pd.DataFrame,
    name: str,
    threshold: float = 0.5,
) -> ChallengerEvaluation:
    """Evaluate a fitted challenger using the common benchmark contract."""
    y_true = test[TARGET].astype(int)
    probabilities = model.predict_proba(test[FEATURE_COLUMNS])[:, 1]
    predictions = (probabilities >= threshold).astype(int)
    return ChallengerEvaluation(
        model=name,
        roc_auc=float(roc_auc_score(y_true, probabilities)),
        pr_auc=float(average_precision_score(y_true, probabilities)),
        precision=float(precision_score(y_true, predictions, zero_division=0)),
        recall=float(recall_score(y_true, predictions, zero_division=0)),
        f1=float(f1_score(y_true, predictions, zero_division=0)),
    )
