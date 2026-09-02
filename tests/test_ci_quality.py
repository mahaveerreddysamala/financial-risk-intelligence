from __future__ import annotations

import pytest

from financial_risk.mlops.ci_quality import QualityGateConfig, evaluate_quality_gates


def test_quality_gates_pass() -> None:
    result = evaluate_quality_gates(
        {"test_pr_auc": 0.08, "test_recall": 0.10, "test_precision": 0.04},
    )
    assert result.passed is True
    assert result.failures == ()


def test_quality_gates_fail_below_thresholds() -> None:
    result = evaluate_quality_gates(
        {"test_pr_auc": 0.01, "test_recall": 0.02, "test_precision": 0.01},
    )
    assert result.passed is False
    assert len(result.failures) == 3


def test_quality_gates_require_psi_when_configured() -> None:
    result = evaluate_quality_gates(
        {"test_pr_auc": 0.08, "test_recall": 0.10, "test_precision": 0.04},
        config=QualityGateConfig(max_psi=0.2),
    )
    assert result.passed is False
    assert result.failures == ("missing metric: max_psi",)


def test_quality_gates_fail_on_high_psi() -> None:
    result = evaluate_quality_gates(
        {
            "test_pr_auc": 0.08,
            "test_recall": 0.10,
            "test_precision": 0.04,
            "max_psi": 0.35,
        },
        config=QualityGateConfig(max_psi=0.2),
    )
    assert result.passed is False
    assert "above max_psi" in result.failures[0]


@pytest.mark.parametrize(
    ("metric", "value"),
    [("test_pr_auc", 0.049), ("test_recall", 0.049), ("test_precision", 0.019)],
)
def test_quality_gates_enforce_boundary(metric: str, value: float) -> None:
    metrics = {"test_pr_auc": 0.08, "test_recall": 0.10, "test_precision": 0.04}
    metrics[metric] = value
    result = evaluate_quality_gates(metrics)
    assert result.passed is False
