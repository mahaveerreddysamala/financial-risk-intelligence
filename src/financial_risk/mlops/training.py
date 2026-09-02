"""Train the fraud model and persist a reproducible MLflow model artifact."""
from __future__ import annotations

from pathlib import Path
from typing import Any

import pandas as pd

from financial_risk.models.baseline import CATEGORICAL_FEATURES, NUMERIC_FEATURES, TARGET
from financial_risk.models.xgboost_model import build_xgboost_model, evaluate_xgboost
from financial_risk.mlops.tracking import MLflowRunResult, log_sklearn_run

DEFAULT_FEATURE_COLUMNS = NUMERIC_FEATURES + CATEGORICAL_FEATURES


def train_and_log_xgboost(
    train: pd.DataFrame,
    validation: pd.DataFrame,
    test: pd.DataFrame,
    *,
    experiment_name: str = "financial-fraud-detection",
    model_name: str = "financial-fraud-xgboost",
    registered_model_name: str | None = None,
    tracking_uri: str | None = None,
    artifact_name: str = "xgboost-model",
    threshold: float = 0.85,
    artifact_root: str | Path | None = None,
) -> tuple[Any, MLflowRunResult, dict[str, float]]:
    """Fit XGBoost chronologically, evaluate it, and log the fitted pipeline to MLflow.

    The caller is responsible for creating the temporal train/validation/test splits.
    ``artifact_root`` is retained as explicit metadata for future artifact-store
    integration; the MLflow tracking URI controls where the model is stored.
    """
    if not 0.0 < threshold < 1.0:
        raise ValueError("threshold must be between zero and one")

    feature_columns = DEFAULT_FEATURE_COLUMNS
    missing = [column for column in [*feature_columns, TARGET] if column not in train.columns]
    if missing:
        raise ValueError(f"Training data is missing required columns: {sorted(missing)}")

    positive = int(train[TARGET].sum())
    negative = int(len(train) - positive)
    if positive == 0:
        raise ValueError("Training data must contain at least one positive fraud label")

    scale_pos_weight = negative / positive
    model = build_xgboost_model(scale_pos_weight=scale_pos_weight)
    model.fit(train[feature_columns], train[TARGET].astype(int))

    validation_result = evaluate_xgboost(model, validation, threshold=threshold)
    test_result = evaluate_xgboost(model, test, threshold=threshold)
    metrics = {
        "validation_roc_auc": validation_result.roc_auc,
        "validation_pr_auc": validation_result.pr_auc,
        "validation_precision": validation_result.precision,
        "validation_recall": validation_result.recall,
        "validation_f1": validation_result.f1,
        "test_roc_auc": test_result.roc_auc,
        "test_pr_auc": test_result.pr_auc,
        "test_precision": test_result.precision,
        "test_recall": test_result.recall,
        "test_f1": test_result.f1,
    }
    parameters = {
        "model_type": "xgboost",
        "n_estimators": 300,
        "max_depth": 6,
        "learning_rate": 0.05,
        "subsample": 0.9,
        "colsample_bytree": 0.9,
        "min_child_weight": 3,
        "scale_pos_weight": scale_pos_weight,
        "threshold": threshold,
        "train_rows": len(train),
        "validation_rows": len(validation),
        "test_rows": len(test),
        "feature_count": len(feature_columns),
    }
    tags = {
        "task": "fraud_detection",
        "validation_strategy": "temporal",
        "data_domain": "synthetic_financial",
    }
    if artifact_root is not None:
        tags["artifact_root"] = str(Path(artifact_root))

    result = log_sklearn_run(
        model,
        model_name=model_name,
        experiment_name=experiment_name,
        parameters=parameters,
        metrics=metrics,
        tags=tags,
        tracking_uri=tracking_uri,
        registered_model_name=registered_model_name,
        artifact_name=artifact_name,
    )
    return model, result, metrics
