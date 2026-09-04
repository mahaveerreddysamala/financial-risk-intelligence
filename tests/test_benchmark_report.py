from __future__ import annotations

import pandas as pd

from financial_risk.models.benchmark import render_markdown_report


def test_benchmark_report_documents_leakage_safe_evaluation() -> None:
    benchmark = pd.DataFrame(
        [
            {
                "model": "XGBoost",
                "roc_auc": 0.72,
                "pr_auc": 0.11,
                "precision": 0.08,
                "recall": 0.40,
                "f1": 0.13,
            }
        ]
    )
    thresholds = pd.DataFrame(
        [
            {
                "threshold": 0.42,
                "precision": 0.08,
                "recall": 0.40,
                "f1": 0.13,
                "selected_on_validation": True,
            }
        ]
    )
    top_k = pd.DataFrame(
        [{"k": 250, "precision_at_k": 0.05, "recall_at_k": 0.30, "lift_at_k": 4.0}]
    )
    top_k.attrs["selected_validation_threshold"] = 0.42
    top_k.attrs["test_prevalence"] = 0.0125

    report = render_markdown_report(benchmark, thresholds, top_k, rows=20_000, seed=42)

    assert "chronological train/validation/test" in report
    assert "Held-out test prevalence: **1.25%**" in report
    assert "| 250 | 5.00% | 30.00% | 4.00x |" in report
    assert "not claims about production fraud performance" in report
