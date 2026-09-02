import numpy as np

from financial_risk.models.threshold import evaluate_thresholds, precision_recall_at_k


def test_threshold_evaluation_returns_expected_structure() -> None:
    y_true = np.array([0, 1, 0, 1, 1])
    probabilities = np.array([0.1, 0.9, 0.2, 0.8, 0.3])

    results = evaluate_thresholds(y_true, probabilities, thresholds=(0.5, 0.8))

    assert [result.threshold for result in results] == [0.5, 0.8]
    assert results[0].review_volume == 2
    assert 0.0 <= results[0].precision <= 1.0
    assert 0.0 <= results[0].recall <= 1.0
    assert 0.0 <= results[0].f1 <= 1.0


def test_precision_recall_at_k() -> None:
    y_true = np.array([0, 1, 1, 0, 1])
    probabilities = np.array([0.10, 0.90, 0.80, 0.70, 0.60])

    precision, recall = precision_recall_at_k(y_true, probabilities, k=3)

    assert precision == 2 / 3
    assert recall == 2 / 3
