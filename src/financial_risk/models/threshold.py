"""Decision-threshold and top-K evaluation utilities for fraud models."""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from sklearn.metrics import f1_score, precision_score, recall_score


@dataclass(frozen=True)
class ThresholdResult:
    threshold: float
    precision: float
    recall: float
    f1: float
    review_volume: int
    fraud_captured: int
    lift: float


def evaluate_thresholds(
    y_true: np.ndarray,
    probabilities: np.ndarray,
    thresholds: tuple[float, ...] = (0.30, 0.50, 0.70, 0.85),
) -> list[ThresholdResult]:
    """Evaluate operating thresholds and investigator tradeoffs."""
    if len(y_true) != len(probabilities):
        raise ValueError("y_true and probabilities must have the same length")
    if len(y_true) == 0:
        raise ValueError("y_true and probabilities must not be empty")

    y_true = np.asarray(y_true).astype(int)
    probabilities = np.asarray(probabilities, dtype=float)
    total_positives = int(y_true.sum())
    prevalence = total_positives / len(y_true)

    results: list[ThresholdResult] = []
    for threshold in dict.fromkeys(float(value) for value in thresholds):
        predictions = (probabilities >= threshold).astype(int)
        captured = int(np.logical_and(predictions == 1, y_true == 1).sum())
        precision = float(precision_score(y_true, predictions, zero_division=0))
        recall = float(recall_score(y_true, predictions, zero_division=0))
        results.append(
            ThresholdResult(
                threshold=threshold,
                precision=precision,
                recall=recall,
                f1=float(f1_score(y_true, predictions, zero_division=0)),
                review_volume=int(predictions.sum()),
                fraud_captured=captured,
                lift=precision / prevalence if prevalence > 0 else 0.0,
            )
        )
    return results


def precision_recall_at_k(
    y_true: np.ndarray, probabilities: np.ndarray, k: int
) -> tuple[float, float]:
    """Measure precision and recall when investigators review the top-K scores."""
    if len(y_true) != len(probabilities):
        raise ValueError("y_true and probabilities must have the same length")
    if k <= 0:
        raise ValueError("k must be greater than zero")

    y_true = np.asarray(y_true).astype(int)
    probabilities = np.asarray(probabilities, dtype=float)
    k = min(k, len(y_true))
    top_indices = np.argsort(-probabilities, kind="stable")[:k]
    positives = int(y_true[top_indices].sum())
    total_positives = int(y_true.sum())
    precision = positives / k if k else 0.0
    recall = positives / total_positives if total_positives else 0.0
    return float(precision), float(recall)
