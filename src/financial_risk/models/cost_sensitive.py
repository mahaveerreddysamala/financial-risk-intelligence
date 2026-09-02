"""Cost-sensitive fraud decisions based on expected loss."""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd


@dataclass(frozen=True)
class CostPolicy:
    """Business costs used to compare approve, review, and hold decisions."""

    false_positive_cost: float = 5.0
    false_negative_cost: float = 100.0
    review_cost: float = 3.0

    def __post_init__(self) -> None:
        if min(self.false_positive_cost, self.false_negative_cost, self.review_cost) < 0:
            raise ValueError("decision costs must be non-negative")
        if self.false_negative_cost == 0:
            raise ValueError("false_negative_cost must be greater than zero")


def expected_costs(
    fraud_probability: float | np.ndarray,
    policy: CostPolicy = CostPolicy(),
) -> pd.DataFrame:
    """Return expected costs for approve, review, and hold decisions."""
    probabilities = np.asarray(fraud_probability, dtype=float)
    if np.any(~np.isfinite(probabilities)) or np.any((probabilities < 0) | (probabilities > 1)):
        raise ValueError("fraud probabilities must be finite values in [0, 1]")

    approve = probabilities * policy.false_negative_cost
    review = np.full_like(probabilities, policy.review_cost, dtype=float)
    hold = (1.0 - probabilities) * policy.false_positive_cost
    return pd.DataFrame(
        {
            "fraud_probability": probabilities,
            "approve_expected_cost": approve,
            "review_expected_cost": review,
            "hold_expected_cost": hold,
        }
    )


def choose_cost_sensitive_action(
    fraud_probability: float,
    policy: CostPolicy = CostPolicy(),
) -> str:
    """Choose the minimum-expected-cost action for one transaction."""
    costs = expected_costs(np.array([fraud_probability]), policy).iloc[0]
    actions = {
        "approve": float(costs["approve_expected_cost"]),
        "review": float(costs["review_expected_cost"]),
        "hold": float(costs["hold_expected_cost"]),
    }
    return min(actions, key=actions.get)
