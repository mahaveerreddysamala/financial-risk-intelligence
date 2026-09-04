"""Automated model-promotion workflow built on champion selection and quality gates."""
from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Callable

import pandas as pd

from financial_risk.mlops.quality_gates import QualityGateReport, run_quality_gates
from financial_risk.models.champion_selection import ChampionSelection, select_champion


@dataclass(frozen=True)
class PromotionDecision:
    """Auditable promotion decision for a candidate champion."""

    champion_selection: ChampionSelection
    quality_gate_report: QualityGateReport
    promoted: bool
    registered_model_version: str | None
    reason: str


def evaluate_promotion(
    operating_points: pd.DataFrame,
    champion_metrics: dict[str, float],
    *,
    drift_report: dict[str, Any] | None = None,
    min_pr_auc: float = 0.05,
    min_recall: float = 0.05,
    min_precision: float = 0.02,
    max_psi: float | None = 0.20,
) -> tuple[ChampionSelection, QualityGateReport]:
    """Select a champion from operating points and evaluate promotion gates."""
    selection = select_champion(operating_points)
    report = run_quality_gates(
        champion_metrics,
        min_pr_auc=min_pr_auc,
        min_recall=min_recall,
        min_precision=min_precision,
        drift_report=drift_report,
        max_psi=max_psi,
    )
    return selection, report


def promote_candidate(
    operating_points: pd.DataFrame,
    champion_metrics: dict[str, float],
    *,
    model_uri: str | None = None,
    registered_model_name: str | None = None,
    alias: str = "champion",
    register: Callable[..., str] | None = None,
    drift_report: dict[str, Any] | None = None,
    min_pr_auc: float = 0.05,
    min_recall: float = 0.05,
    min_precision: float = 0.02,
    max_psi: float | None = 0.20,
) -> PromotionDecision:
    """Gate and optionally register a candidate model.

    Registration is deliberately dependency-injected so CI can exercise the
    governance workflow without requiring a live MLflow service.
    """
    selection, report = evaluate_promotion(
        operating_points,
        champion_metrics,
        drift_report=drift_report,
        min_pr_auc=min_pr_auc,
        min_recall=min_recall,
        min_precision=min_precision,
        max_psi=max_psi,
    )

    if not report.passed:
        failed = ", ".join(gate.name for gate in report.failed_gates)
        return PromotionDecision(
            champion_selection=selection,
            quality_gate_report=report,
            promoted=False,
            registered_model_version=None,
            reason=f"Promotion blocked by quality gates: {failed}",
        )

    if model_uri is None:
        return PromotionDecision(
            champion_selection=selection,
            quality_gate_report=report,
            promoted=False,
            registered_model_version=None,
            reason="Quality gates passed; model registration is pending because model_uri was not supplied.",
        )

    if not registered_model_name:
        raise ValueError("registered_model_name is required when model_uri is supplied")
    if register is None:
        raise ValueError("register callback is required for non-dry-run promotion")

    version = register(
        model_uri,
        registered_model_name=registered_model_name,
        alias=alias,
    )
    return PromotionDecision(
        champion_selection=selection,
        quality_gate_report=report,
        promoted=True,
        registered_model_version=str(version),
        reason="Quality gates passed and candidate was registered with the champion alias.",
    )
