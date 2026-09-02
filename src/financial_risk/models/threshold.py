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


def evaluate_thresholds(
    y_true: np.ndarray,
    probabilities: np.ndarray,
    thresholds: tuple[float, ...] = (0.30, 0.50, 0.70, 0.85),
) -> list[ThresholdResult]:
    """Evaluate fixed operating thresholds for investigation-volume tradeoffs."""
    if len(y_true) != len(probabilities):
        raise ValueError("y_true and probabilities must have the same length")

    results: list[ThresholdResult] = []
    for threshold in thresholds:
        predictions = (probabilities >= threshold).astype(int)
        results.append(
            ThresholdResult(
                threshold=float(threshold),
                precision=float(precision_score(y_true, predictions, zero_division=0)),
                recall=float(recall_score(y_true, predictions, zero_division=0)),
                f1=float(f1_score(y_true, predictions, zero_division=0)),
                review_volume=int(predictions.sum()),
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

    k = min(k, len(y_true))
    top_indices = np.argsort(-probabilities, kind="stable")[:k]
    positives = int(np.asarray(y_true)[top_indices].sum())
    total_positives = int(np.asarray(y_true).sum())
    precision = positives / k if k else 0.0
    recall = positives / total_positives if total_positives else 0.0
    return float(precision), float(recall)
