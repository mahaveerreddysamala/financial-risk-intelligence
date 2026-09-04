"""Tests for Phase 42 champion selection."""
from __future__ import annotations

import pandas as pd
import pytest

from financial_risk.models.champion_selection import select_champion


def test_select_champion_prefers_lowest_realized_cost() -> None:
    results = pd.DataFrame(
        [
            {"model": "XGBoost", "f1": 0.077, "precision": 0.154, "recall": 0.051, "lift": 12.87, "brier_score": 0.032, "realized_cost": 5977.0},
            {"model": "Random Forest", "f1": 0.071, "precision": 0.118, "recall": 0.051, "lift": 9.84, "brier_score": 0.049, "realized_cost": 9871.0},
            {"model": "LightGBM", "f1": 0.097, "precision": 0.130, "recall": 0.077, "lift": 10.91, "brier_score": 0.021, "realized_cost": 4141.0},
        ]
    )
    selection = select_champion(results)
    assert selection.champion == "LightGBM"
    assert selection.ranked_models == ("LightGBM", "XGBoost", "Random Forest")


def test_select_champion_uses_deterministic_tie_breakers() -> None:
    results = pd.DataFrame(
        [
            {"model": "A", "f1": 0.20, "precision": 0.3, "recall": 0.4, "lift": 3.0, "brier_score": 0.10, "realized_cost": 10.0},
            {"model": "B", "f1": 0.20, "precision": 0.3, "recall": 0.4, "lift": 3.0, "brier_score": 0.10, "realized_cost": 10.0},
        ]
    )
    selection = select_champion(results)
    assert selection.champion == "A"


def test_select_champion_validates_schema() -> None:
    with pytest.raises(ValueError, match="Missing champion-selection columns"):
        select_champion(pd.DataFrame({"model": ["XGBoost"]}))
