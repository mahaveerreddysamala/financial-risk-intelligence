"""Common operating-point analysis for fraud-model challengers."""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd

from financial_risk.models.baseline import CATEGORICAL_FEATURES, NUMERIC_FEATURES
from financial_risk.models.calibration import evaluate_calibration
from financial_risk.models.cost_sensitive import CostPolicy, choose_cost_sensitive_action
from financial_risk.models.threshold import evaluate_thresholds, precision_recall_at_k

FEATURE_COLUMNS = NUMERIC_FEATURES + CATEGORICAL_FEATURES


@dataclass(frozen=True)
class OperatingPointResult:
    model: str
    selected_threshold: float
    precision: float
    recall: float
    f1: float
    review_volume: int
    fraud_captured: int
    lift: float
    precision_at_500: float
    recall_at_500: float
    lift_at_500: float
    brier_score: float
    realized_cost: float


def select_f1_threshold(y_true: pd.Series, probabilities: np.ndarray) -> float:
    """Select a threshold on validation data only, maximizing F1."""
    rows = evaluate_thresholds(y_true.to_numpy(), probabilities)
    return max(rows, key=lambda row: (row.f1, row.precision, -row.threshold)).threshold


def realized_cost(
    y_true: np.ndarray,
    probabilities: np.ndarray,
    policy: CostPolicy = CostPolicy(),
) -> float:
    """Compute realized transaction-level decision cost from model probabilities."""
    targets = np.asarray(y_true).astype(int)
    probabilities = np.asarray(probabilities, dtype=float)
    if len(targets) != len(probabilities):
        raise ValueError("y_true and probabilities must have the same length")
    total = 0.0
    for target, probability in zip(targets, probabilities, strict=True):
        action = choose_cost_sensitive_action(float(probability), policy)
        if action == "approve":
            total += policy.false_negative_cost if target == 1 else 0.0
        elif action == "hold":
            total += policy.false_positive_cost if target == 0 else 0.0
        else:
            total += policy.review_cost
    return float(total)


def analyze_model(
    model_name: str,
    model,
    validation: pd.DataFrame,
    test: pd.DataFrame,
    top_k: int = 500,
    policy: CostPolicy = CostPolicy(),
) -> OperatingPointResult:
    """Fit-agnostic operating-point analysis using a validation/test boundary."""
    validation_probabilities = np.asarray(
        model.predict_proba(validation[FEATURE_COLUMNS])[:, 1], dtype=float
    )
    selected_threshold = select_f1_threshold(
        validation["is_fraud"], validation_probabilities
    )
    test_probabilities = np.asarray(
        model.predict_proba(test[FEATURE_COLUMNS])[:, 1], dtype=float
    )
    threshold_row = next(
        row
        for row in evaluate_thresholds(
            test["is_fraud"].to_numpy(),
            test_probabilities,
            thresholds=(selected_threshold,),
        )
        if row.threshold == selected_threshold
    )
    precision_at_k, recall_at_k = precision_recall_at_k(
        test["is_fraud"].to_numpy(), test_probabilities, top_k
    )
    prevalence = float(test["is_fraud"].mean())
    calibration = evaluate_calibration(model, test)
    return OperatingPointResult(
        model=model_name,
        selected_threshold=selected_threshold,
        precision=threshold_row.precision,
        recall=threshold_row.recall,
        f1=threshold_row.f1,
        review_volume=threshold_row.review_volume,
        fraud_captured=threshold_row.fraud_captured,
        lift=threshold_row.lift,
        precision_at_500=precision_at_k,
        recall_at_500=recall_at_k,
        lift_at_500=precision_at_k / prevalence if prevalence > 0 else 0.0,
        brier_score=calibration.brier_score,
        realized_cost=realized_cost(
            test["is_fraud"].to_numpy(), test_probabilities, policy
        ),
    )
