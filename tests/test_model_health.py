import pandas as pd
import pytest

from financial_risk.monitoring.model_health import (
    build_monitoring_status,
    evaluate_calibration_health,
    evaluate_performance,
    monitor_model_window,
)


def test_performance_report_flags_large_pr_auc_drop() -> None:
    result = evaluate_performance(
        {"pr_auc": 0.10, "precision": 0.08},
        {"pr_auc": 0.06, "precision": 0.07},
    )
    assert result.loc[result["metric"] == "pr_auc", "degraded"].item() is True
    assert result.loc[result["metric"] == "precision", "degraded"].item() is True


def test_performance_report_handles_missing_metric() -> None:
    result = evaluate_performance({"pr_auc": 0.10}, {"recall": 0.20})
    assert len(result) == 2
    assert result["degraded"].all()


def test_calibration_health_flags_brier_increase() -> None:
    result = evaluate_calibration_health(0.02, 0.03)
    assert result["degraded"] is True
    assert result["absolute_change"] == pytest.approx(0.01)


def test_healthy_status() -> None:
    status = build_monitoring_status()
    assert status.status == "healthy"
    assert status.severity == "info"
    assert status.reasons == ()


def test_drift_status() -> None:
    report = pd.DataFrame([{"feature": "amount", "drift_detected": True}])
    status = build_monitoring_status(drift_report_frame=report)
    assert status.status == "drift"
    assert status.severity == "warning"
    assert "feature_drift_detected" in status.reasons


def test_multiple_issues_are_critical() -> None:
    drift = pd.DataFrame([{"feature": "amount", "drift_detected": True}])
    performance = pd.DataFrame([{"metric": "pr_auc", "degraded": True}])
    status = build_monitoring_status(drift_report_frame=drift, performance_report=performance)
    assert status.status == "degraded"
    assert status.severity == "critical"


def test_monitor_model_window_returns_all_sections() -> None:
    reference = pd.DataFrame({"amount": [10, 20, 30, 40, 50]})
    current = pd.DataFrame({"amount": [11, 21, 31, 41, 51]})
    snapshot = monitor_model_window(
        reference,
        current,
        ["amount"],
        reference_performance={"pr_auc": 0.10},
        current_performance={"pr_auc": 0.10},
        reference_brier=0.02,
        current_brier=0.02,
    )
    assert set(snapshot) == {"status", "feature_drift", "performance", "calibration"}
    assert snapshot["status"].status == "healthy"


def test_invalid_thresholds_are_rejected() -> None:
    with pytest.raises(ValueError):
        evaluate_performance({"pr_auc": 0.1}, {"pr_auc": 0.1}, min_pr_auc=1.1)
    with pytest.raises(ValueError):
        evaluate_calibration_health(0.1, 0.1, max_relative_increase=-0.1)
