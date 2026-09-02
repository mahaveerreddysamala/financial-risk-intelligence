"""Model quality gates and promotion controls for the fraud platform."""
from __future__ import annotations

from dataclasses import dataclass
from typing import Any

from financial_risk.mlops.tracking import register_model_version


@dataclass(frozen=True)
class GateResult:
    """Result for one promotion criterion."""

    name: str
    passed: bool
    observed: float | None
    threshold: float | None
    message: str


@dataclass(frozen=True)
class QualityGateReport:
    """Aggregate quality-gate decision for a candidate model."""

    passed: bool
    gates: tuple[GateResult, ...]

    @property
    def failed_gates(self) -> tuple[GateResult, ...]:
        """Return only failed gates."""
        return tuple(gate for gate in self.gates if not gate.passed)


def _metric_gate(
    name: str,
    metrics: dict[str, float],
    key: str,
    threshold: float,
) -> GateResult:
    observed = metrics.get(key)
    if observed is None:
        return GateResult(name, False, None, threshold, f"Missing metric: {key}")
    passed = float(observed) >= threshold
    message = f"{key}={float(observed):.6f} {'meets' if passed else 'is below'} {threshold:.6f}"
    return GateResult(name, passed, float(observed), threshold, message)


def run_quality_gates(
    metrics: dict[str, float],
    *,
    min_pr_auc: float = 0.05,
    min_recall: float = 0.05,
    min_precision: float = 0.02,
    drift_report: dict[str, Any] | None = None,
    max_psi: float | None = 0.20,
) -> QualityGateReport:
    """Evaluate candidate-model metrics and optional drift evidence before promotion."""
    if not 0 <= min_pr_auc <= 1 or not 0 <= min_recall <= 1 or not 0 <= min_precision <= 1:
        raise ValueError("metric thresholds must be between zero and one")
    if max_psi is not None and max_psi < 0:
        raise ValueError("max_psi must be non-negative")

    gates = [
        _metric_gate("test_pr_auc", metrics, "test_pr_auc", min_pr_auc),
        _metric_gate("test_recall", metrics, "test_recall", min_recall),
        _metric_gate("test_precision", metrics, "test_precision", min_precision),
    ]

    if drift_report is not None and max_psi is not None:
        psi_values = [
            float(value)
            for value in drift_report.get("feature_psi", {}).values()
            if value is not None
        ]
        worst_psi = max(psi_values, default=0.0)
        passed = worst_psi <= max_psi
        gates.append(
            GateResult(
                "max_feature_psi",
                passed,
                worst_psi,
                max_psi,
                f"worst_feature_psi={worst_psi:.6f} {'within' if passed else 'exceeds'} {max_psi:.6f}",
            )
        )

    return QualityGateReport(
        passed=all(gate.passed for gate in gates),
        gates=tuple(gates),
    )


def promote_if_approved(
    report: QualityGateReport,
    model_uri: str,
    *,
    registered_model_name: str,
    alias: str = "champion",
    tracking_uri: str | None = None,
) -> str:
    """Register and alias a model only when every quality gate has passed."""
    if not report.passed:
        failed = ", ".join(gate.name for gate in report.failed_gates)
        raise ValueError(f"Model failed quality gates: {failed}")
    return register_model_version(
        model_uri,
        registered_model_name=registered_model_name,
        alias=alias,
        tracking_uri=tracking_uri,
    )
