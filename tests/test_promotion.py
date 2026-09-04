"""Tests for automated champion promotion governance."""
from __future__ import annotations

import pandas as pd

from financial_risk.mlops.promotion import evaluate_promotion, promote_candidate


OPERATING_POINTS = pd.DataFrame(
    [
        {
            "model": "XGBoost",
            "f1": 0.0769,
            "precision": 0.1538,
            "recall": 0.0513,
            "lift": 12.87,
            "brier_score": 0.0318,
            "realized_cost": 5977.0,
        },
        {
            "model": "LightGBM",
            "f1": 0.0968,
            "precision": 0.1304,
            "recall": 0.0769,
            "lift": 10.91,
            "brier_score": 0.0213,
            "realized_cost": 4141.0,
        },
    ]
)


def _metrics() -> dict[str, float]:
    return {"test_pr_auc": 0.0825, "test_recall": 0.0769, "test_precision": 0.1304}


def test_evaluate_promotion_selects_lowest_cost_champion() -> None:
    selection, report = evaluate_promotion(OPERATING_POINTS, _metrics())
    assert selection.champion == "LightGBM"
    assert selection.ranked_models == ("LightGBM", "XGBoost")
    assert report.passed is True


def test_promotion_blocks_when_quality_gate_fails() -> None:
    blocked = promote_candidate(
        OPERATING_POINTS,
        {"test_pr_auc": 0.04, "test_recall": 0.0769, "test_precision": 0.1304},
        model_uri="runs:/example/model",
        registered_model_name="financial-fraud",
        register=lambda *_args, **_kwargs: "99",
    )
    assert blocked.promoted is False
    assert blocked.registered_model_version is None
    assert "test_pr_auc" in blocked.reason


def test_promotion_supports_dry_run() -> None:
    decision = promote_candidate(OPERATING_POINTS, _metrics())
    assert decision.promoted is False
    assert decision.registered_model_version is None
    assert "pending" in decision.reason


def test_successful_promotion_uses_injected_registration() -> None:
    captured: dict[str, object] = {}

    def fake_register(model_uri: str, **kwargs: object) -> str:
        captured["model_uri"] = model_uri
        captured.update(kwargs)
        return "7"

    decision = promote_candidate(
        OPERATING_POINTS,
        _metrics(),
        model_uri="runs:/example/model",
        registered_model_name="financial-fraud",
        register=fake_register,
    )
    assert decision.promoted is True
    assert decision.registered_model_version == "7"
    assert decision.champion_selection.champion == "LightGBM"
    assert captured == {
        "model_uri": "runs:/example/model",
        "registered_model_name": "financial-fraud",
        "alias": "champion",
    }
