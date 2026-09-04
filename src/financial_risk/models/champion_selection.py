"""Champion selection policy for fraud-model operating points."""
from __future__ import annotations

from dataclasses import dataclass

import pandas as pd


@dataclass(frozen=True)
class ChampionSelection:
    champion: str
    rationale: str
    ranked_models: tuple[str, ...]


def select_champion(results: pd.DataFrame) -> ChampionSelection:
    """Select a champion from operating-point results using business-first ranking."""
    required = {
        "model",
        "f1",
        "precision",
        "recall",
        "lift",
        "brier_score",
        "realized_cost",
    }
    missing = sorted(required - set(results.columns))
    if missing:
        raise ValueError(f"Missing champion-selection columns: {missing}")
    if results.empty:
        raise ValueError("results must not be empty")

    ranked = results.sort_values(
        by=["realized_cost", "brier_score", "f1", "lift", "precision", "recall"],
        ascending=[True, True, False, False, False, False],
        kind="stable",
    )
    champion_row = ranked.iloc[0]
    champion = str(champion_row["model"])
    rationale = (
        "Champion selected by lowest realized decision cost, then best calibration "
        "(Brier score), then F1, lift, precision, and recall as deterministic tie-breakers."
    )
    return ChampionSelection(
        champion=champion,
        rationale=rationale,
        ranked_models=tuple(ranked["model"].astype(str).tolist()),
    )
