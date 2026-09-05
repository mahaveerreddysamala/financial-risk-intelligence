"""Capacity-aware evaluation for ranked financial-risk queues."""
from __future__ import annotations

from math import ceil

import pandas as pd


def operating_point_summary(
    transactions: pd.DataFrame, review_rate: float
) -> dict[str, float | int]:
    """Evaluate the top-risk queue at a fixed investigation capacity."""
    if not 0.0 < review_rate <= 1.0:
        raise ValueError("review_rate must be greater than zero and at most one")
    required = {"risk_score", "is_fraud"}
    if missing := sorted(required.difference(transactions.columns)):
        raise ValueError(f"Operating-point transactions missing columns: {missing}")
    if transactions.empty:
        raise ValueError("transactions must not be empty")

    ranked = transactions.sort_values("risk_score", ascending=False, kind="stable")
    review_count = min(len(ranked), max(1, ceil(len(ranked) * review_rate)))
    reviewed = ranked.head(review_count)
    total_fraud = int(ranked["is_fraud"].astype(int).sum())
    captured_fraud = int(reviewed["is_fraud"].astype(int).sum())
    prevalence = total_fraud / len(ranked)
    precision = captured_fraud / review_count
    recall = captured_fraud / total_fraud if total_fraud else 0.0
    lift = precision / prevalence if prevalence else 0.0

    return {
        "review_rate": review_count / len(ranked),
        "review_count": review_count,
        "captured_fraud": captured_fraud,
        "precision": precision,
        "recall": recall,
        "lift": lift,
    }


def build_operating_point_curve(
    transactions: pd.DataFrame,
    review_rates: tuple[float, ...] = (0.01, 0.02, 0.05, 0.10, 0.20),
) -> pd.DataFrame:
    """Build a compact capacity trade-off table for the scored queue."""
    if not review_rates:
        raise ValueError("review_rates must not be empty")
    points = [operating_point_summary(transactions, rate) for rate in review_rates]
    curve = pd.DataFrame(points)
    curve["review_rate_pct"] = curve["review_rate"] * 100.0
    return curve
