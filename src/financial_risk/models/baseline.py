"""Baseline fraud-model training and evaluation utilities."""
from __future__ import annotations

from dataclasses import dataclass

import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import average_precision_score, f1_score, precision_score, recall_score, roc_auc_score
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline

TARGET = "is_fraud"
NUMERIC_FEATURES = [
    "amount",
    "is_international",
    "is_night",
    "shared_device_account_count",
    "customer_txn_count_7d",
    "customer_avg_amount_30d",
    "customer_std_amount_30d",
    "customer_unique_merchants_7d",
    "customer_unique_devices_30d",
    "customer_international_rate_30d",
    "customer_night_txn_rate_30d",
    "amount_vs_customer_avg",
    "amount_zscore",
    "txn_count_5m",
    "txn_count_1h",
    "txn_count_24h",
]
CATEGORICAL_FEATURES = [
    "merchant_category",
    "payment_method",
    "channel",
    "country",
]


@dataclass(frozen=True)
class EvaluationResult:
    roc_auc: float
    pr_auc: float
    precision: float
    recall: float
    f1: float


def build_logistic_baseline() -> Pipeline:
    """Create a reproducible logistic-regression baseline pipeline."""
    preprocessor = ColumnTransformer(
        transformers=[
            ("numeric", StandardScaler(), NUMERIC_FEATURES),
            ("categorical", OneHotEncoder(handle_unknown="ignore"), CATEGORICAL_FEATURES),
        ],
        remainder="drop",
    )
    return Pipeline(
        steps=[
            ("preprocess", preprocessor),
            ("model", LogisticRegression(max_iter=1000, class_weight="balanced", random_state=42)),
        ]
    )


def evaluate_binary_classifier(model: Pipeline, test: pd.DataFrame) -> EvaluationResult:
    """Evaluate a fitted binary classifier on a labeled test frame."""
    y_true = test[TARGET].astype(int)
    probabilities = model.predict_proba(test[NUMERIC_FEATURES + CATEGORICAL_FEATURES])[:, 1]
    predictions = (probabilities >= 0.5).astype(int)
    return EvaluationResult(
        roc_auc=float(roc_auc_score(y_true, probabilities)),
        pr_auc=float(average_precision_score(y_true, probabilities)),
        precision=float(precision_score(y_true, predictions, zero_division=0)),
        recall=float(recall_score(y_true, predictions, zero_division=0)),
        f1=float(f1_score(y_true, predictions, zero_division=0)),
    )
