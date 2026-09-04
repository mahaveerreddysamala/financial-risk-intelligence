from __future__ import annotations

import pandas as pd
import pytest

from financial_risk.models.backtesting import build_temporal_folds, render_backtest_report


def _temporal_frame() -> pd.DataFrame:
    dates = pd.date_range("2025-01-01", "2025-12-31", freq="7D")
    return pd.DataFrame(
        {
            "timestamp": dates,
            "is_fraud": [index % 2 for index in range(len(dates))],
        }
    )


def test_temporal_folds_are_expanding_and_out_of_time() -> None:
    folds = build_temporal_folds(
        _temporal_frame(), ["2025-05-01", "2025-08-01"], test_months=2
    )

    assert len(folds) == 2
    assert len(folds[1].train) > len(folds[0].train)
    for fold in folds:
        assert fold.train["timestamp"].max() < fold.train_end
        assert fold.test["timestamp"].min() >= fold.train_end
        assert fold.test["timestamp"].max() < fold.test_end


def test_temporal_folds_reject_unordered_cutoffs() -> None:
    with pytest.raises(ValueError, match="unique and ordered"):
        build_temporal_folds(_temporal_frame(), ["2025-08-01", "2025-05-01"])


def test_backtest_report_summarizes_stability() -> None:
    results = pd.DataFrame(
        [
            {
                "fold": 1,
                "train_end": "2025-07-01",
                "test_end": "2025-09-01",
                "test_rows": 1000,
                "test_prevalence": 0.01,
                "roc_auc": 0.70,
                "pr_auc": 0.08,
                "top_k": 100,
                "precision_at_k": 0.04,
                "recall_at_k": 0.40,
                "lift_at_k": 4.0,
            },
            {
                "fold": 2,
                "train_end": "2025-09-01",
                "test_end": "2025-11-01",
                "test_rows": 1000,
                "test_prevalence": 0.02,
                "roc_auc": 0.66,
                "pr_auc": 0.06,
                "top_k": 100,
                "precision_at_k": 0.05,
                "recall_at_k": 0.25,
                "lift_at_k": 2.5,
            },
        ]
    )

    report = render_backtest_report(results, rows=20_000, seed=42)

    assert "Mean out-of-time PR-AUC: **0.0700**" in report
    assert "Minimum out-of-time PR-AUC: **0.0600**" in report
    assert "Mean top-K lift: **3.25x**" in report
    assert "leakage-safe stability testing" in report
