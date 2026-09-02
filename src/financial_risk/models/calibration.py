"""Probability calibration and reliability diagnostics for fraud models."""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd
from sklearn.calibration import CalibratedClassifierCV
from sklearn.metrics import brier_score_loss
from sklearn.pipeline import Pipeline

from financial_risk.models.baseline import NUMERIC_FEATURES, CATEGORICAL_FEATURES, TARGET


@dataclass(frozen=True)
class CalibrationResult:
    brier_score: float
    mean_predicted_probability: float
    observed_fraud_rate: float


def calibrate_classifier(
    estimator: Pipeline,
    calibration_data: pd.DataFrame,
    method: str = "sigmoid",
    cv: int = 3,
) -> CalibratedClassifierCV:
    """Fit a probability-calibrated wrapper using a labeled calibration set."""
    if method not in {"sigmoid", "isotonic"}:
        raise ValueError("method must be 'sigmoid' or 'isotonic'")
    if cv < 2:
        raise ValueError("cv must be at least 2")
    missing = [c for c in NUMERIC_FEATURES + CATEGORICAL_FEATURES + [TARGET] if c not in calibration_data.columns]
    if missing:
        raise ValueError(f"Missing calibration columns: {missing}")
    if calibration_data[TARGET].nunique() < 2:
        raise ValueError("calibration data must contain both classes")

    features = calibration_data[NUMERIC_FEATURES + CATEGORICAL_FEATURES]
    target = calibration_data[TARGET].astype(int)
    calibrated = CalibratedClassifierCV(estimator=estimator, method=method, cv=cv)
    calibrated.fit(features, target)
    return calibrated


def evaluate_calibration(
    model,
    test: pd.DataFrame,
) -> CalibrationResult:
    """Evaluate probability quality on a held-out labeled test frame."""
    required = NUMERIC_FEATURES + CATEGORICAL_FEATURES + [TARGET]
    missing = [c for c in required if c not in test.columns]
    if missing:
        raise ValueError(f"Missing calibration columns: {missing}")
    probabilities = np.asarray(
        model.predict_proba(test[NUMERIC_FEATURES + CATEGORICAL_FEATURES])[:, 1],
        dtype=float,
    )
    target = test[TARGET].astype(int).to_numpy()
    return CalibrationResult(
        brier_score=float(brier_score_loss(target, probabilities)),
        mean_predicted_probability=float(probabilities.mean()),
        observed_fraud_rate=float(target.mean()),
    )


def calibration_bins(
    model,
    test: pd.DataFrame,
    bins: int = 10,
) -> pd.DataFrame:
    """Return reliability bins with prediction counts and observed fraud rates."""
    if bins < 2:
        raise ValueError("bins must be at least 2")
    probabilities = np.asarray(
        model.predict_proba(test[NUMERIC_FEATURES + CATEGORICAL_FEATURES])[:, 1],
        dtype=float,
    )
    labels = pd.cut(
        probabilities,
        bins=np.linspace(0.0, 1.0, bins + 1),
        include_lowest=True,
        duplicates="drop",
    )
    frame = pd.DataFrame(
        {
            "probability": probabilities,
            TARGET: test[TARGET].astype(int).to_numpy(),
            "bin": labels,
        }
    )
    summary = (
        frame.groupby("bin", observed=False)
        .agg(
            samples=("probability", "size"),
            mean_predicted_probability=("probability", "mean"),
            observed_fraud_rate=(TARGET, "mean"),
        )
        .reset_index()
    )
    return summary
