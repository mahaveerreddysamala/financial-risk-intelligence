"""Deterministic CI model-quality gate helpers."""
from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class QualityGateConfig:
    """Portfolio operating thresholds for model-promotion checks."""

    min_pr_auc: float = 0.05
    min_recall: float = 0.05
    min_precision: float = 0.02
    max_psi: float | None = None


@dataclass(frozen=True)
class QualityGateResult:
    """Outcome of model-quality checks."""

    passed: bool
    failures: tuple[str, ...]


def evaluate_quality_gates(
    metrics: dict[str, float],
    *,
    config: QualityGateConfig | None = None,
) -> QualityGateResult:
    """Evaluate model metrics against explicit promotion thresholds."""
    if config is None:
        config = QualityGateConfig()

    failures: list[str] = []
    checks = {
        "test_pr_auc": (config.min_pr_auc, "min_pr_auc"),
        "test_recall": (config.min_recall, "min_recall"),
        "test_precision": (config.min_precision, "min_precision"),
    }
    for metric_name, (minimum, label) in checks.items():
        value = metrics.get(metric_name)
        if value is None:
            failures.append(f"missing metric: {metric_name}")
        elif float(value) < minimum:
            failures.append(f"{metric_name}={float(value):.6f} below {label}={minimum:.6f}")

    if config.max_psi is not None:
        psi = metrics.get("max_psi")
        if psi is None:
            failures.append("missing metric: max_psi")
        elif float(psi) > config.max_psi:
            failures.append(f"max_psi={float(psi):.6f} above max_psi={config.max_psi:.6f}")

    return QualityGateResult(passed=not failures, failures=tuple(failures))
