"""Unified model-health monitoring for drift, performance, and calibration."""
from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import pandas as pd

from financial_risk.monitoring.drift import drift_report, population_stability_index


@dataclass(frozen=True)
class MonitoringStatus:
    """Aggregated monitoring outcome with deterministic severity."""

    status: str
    severity: str
    drift_detected: bool
    performance_degraded: bool
    calibration_degraded: bool
    reasons: tuple[str, ...]


def evaluate_performance(
    reference: dict[str, float],
    current: dict[str, float],
    *,
    min_pr_auc: float = 0.05,
    max_relative_drop: float = 0.10,
) -> pd.DataFrame:
    """Compare current model metrics with a reference baseline."""
    if min_pr_auc < 0 or min_pr_auc > 1:
        raise ValueError("min_pr_auc must be between zero and one")
    if max_relative_drop < 0:
        raise ValueError("max_relative_drop must be non-negative")

    rows: list[dict[str, Any]] = []
    for metric in sorted(set(reference) | set(current)):
        ref = reference.get(metric)
        cur = current.get(metric)
        if ref is None or cur is None:
            rows.append(
                {
                    "metric": metric,
                    "reference": ref,
                    "current": cur,
                    "absolute_change": None,
                    "relative_change": None,
                    "degraded": True,
                    "reason": "missing_metric",
                }
            )
            continue
        ref_value = float(ref)
        cur_value = float(cur)
        absolute = cur_value - ref_value
        relative = absolute / abs(ref_value) if ref_value != 0 else (0.0 if absolute == 0 else float("inf"))
        degraded = (
            metric in {"pr_auc", "roc_auc", "precision", "recall", "f1"}
            and (cur_value < ref_value * (1.0 - max_relative_drop) or (metric == "pr_auc" and cur_value < min_pr_auc))
        )
        rows.append(
            {
                "metric": metric,
                "reference": ref_value,
                "current": cur_value,
                "absolute_change": absolute,
                "relative_change": relative,
                "degraded": degraded,
                "reason": "threshold_breach" if degraded else "within_tolerance",
            }
        )
    return pd.DataFrame(rows)


def evaluate_calibration_health(
    reference_brier: float,
    current_brier: float,
    *,
    max_relative_increase: float = 0.20,
) -> dict[str, float | bool | str]:
    """Detect degradation in Brier-score calibration quality."""
    if reference_brier < 0 or current_brier < 0:
        raise ValueError("Brier scores must be non-negative")
    if max_relative_increase < 0:
        raise ValueError("max_relative_increase must be non-negative")
    absolute_change = float(current_brier - reference_brier)
    relative_change = (
        absolute_change / reference_brier
        if reference_brier != 0
        else (0.0 if absolute_change == 0 else float("inf"))
    )
    degraded = current_brier > reference_brier * (1.0 + max_relative_increase)
    return {
        "reference_brier": float(reference_brier),
        "current_brier": float(current_brier),
        "absolute_change": absolute_change,
        "relative_change": relative_change,
        "degraded": bool(degraded),
        "reason": "threshold_breach" if degraded else "within_tolerance",
    }


def build_monitoring_status(
    *,
    drift_report_frame: pd.DataFrame | None = None,
    performance_report: pd.DataFrame | None = None,
    calibration_report: dict[str, float | bool | str] | None = None,
    min_drift_sample_size: int = 30,
) -> MonitoringStatus:
    """Aggregate monitoring evidence into an alert-ready status."""
    if min_drift_sample_size < 2:
        raise ValueError("min_drift_sample_size must be at least 2")

    drift_detected = False
    if drift_report_frame is not None and not drift_report_frame.empty:
        if "drift_detected" not in drift_report_frame.columns:
            raise ValueError("drift_report_frame must contain drift_detected")
        reference_ok = "reference_size" not in drift_report_frame.columns or (
            pd.to_numeric(drift_report_frame["reference_size"], errors="coerce") >= min_drift_sample_size
        ).all()
        current_ok = "current_size" not in drift_report_frame.columns or (
            pd.to_numeric(drift_report_frame["current_size"], errors="coerce") >= min_drift_sample_size
        ).all()
        drift_detected = bool(
            reference_ok
            and current_ok
            and drift_report_frame["drift_detected"].astype(bool).any()
        )

    performance_degraded = bool(
        performance_report is not None
        and not performance_report.empty
        and performance_report["degraded"].astype(bool).any()
    )
    calibration_degraded = bool(calibration_report and calibration_report.get("degraded", False))

    reasons: list[str] = []
    if drift_detected:
        reasons.append("feature_drift_detected")
    if performance_degraded:
        reasons.append("performance_degradation_detected")
    if calibration_degraded:
        reasons.append("calibration_degradation_detected")

    issues = sum((drift_detected, performance_degraded, calibration_degraded))
    if issues == 0:
        return MonitoringStatus("healthy", "info", False, False, False, ())
    if performance_degraded or calibration_degraded:
        return MonitoringStatus(
            "degraded",
            "critical" if issues >= 2 else "warning",
            drift_detected,
            performance_degraded,
            calibration_degraded,
            tuple(reasons),
        )
    return MonitoringStatus("drift", "warning", True, False, False, tuple(reasons))


def monitor_model_window(
    reference_features: pd.DataFrame,
    current_features: pd.DataFrame,
    numeric_features: list[str],
    *,
    reference_performance: dict[str, float] | None = None,
    current_performance: dict[str, float] | None = None,
    reference_brier: float | None = None,
    current_brier: float | None = None,
    psi_threshold: float = 0.20,
    min_drift_sample_size: int = 30,
    **_: Any,
) -> dict[str, Any]:
    """Produce one monitoring snapshot suitable for batch jobs or alerting."""
    if min_drift_sample_size < 2:
        raise ValueError("min_drift_sample_size must be at least 2")
    features = drift_report(reference_features, current_features, numeric_features, psi_threshold=psi_threshold)
    performance = None
    if reference_performance is not None and current_performance is not None:
        performance = evaluate_performance(reference_performance, current_performance)
    calibration = None
    if reference_brier is not None and current_brier is not None:
        calibration = evaluate_calibration_health(reference_brier, current_brier)
    status = build_monitoring_status(
        drift_report_frame=features,
        performance_report=performance,
        calibration_report=calibration,
        min_drift_sample_size=min_drift_sample_size,
    )
    return {
        "status": status,
        "feature_drift": features,
        "performance": performance,
        "calibration": calibration,
    }


__all__ = [
    "MonitoringStatus",
    "build_monitoring_status",
    "evaluate_calibration_health",
    "evaluate_performance",
    "monitor_model_window",
    "population_stability_index",
]
