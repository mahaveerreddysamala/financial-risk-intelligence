"""Run the deterministic model-quality gate against a benchmark output."""
from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd

from financial_risk.mlops.ci_quality import QualityGateConfig, evaluate_quality_gates


def main() -> int:
    parser = argparse.ArgumentParser(description="Evaluate model-quality promotion gates.")
    parser.add_argument("--benchmark", default="artifacts/model-benchmark.csv")
    parser.add_argument("--min-pr-auc", type=float, default=0.05)
    parser.add_argument("--min-recall", type=float, default=0.05)
    parser.add_argument("--min-precision", type=float, default=0.02)
    args = parser.parse_args()

    path = Path(args.benchmark)
    if not path.exists():
        print(f"Quality gate input not found: {path}")
        return 2

    frame = pd.read_csv(path)
    xgboost = frame.loc[frame["model"].eq("XGBoost")]
    if xgboost.empty:
        print("Quality gate input does not contain an XGBoost row")
        return 2

    row = xgboost.iloc[0]
    metrics = {
        "test_pr_auc": float(row["pr_auc"]),
        "test_recall": float(row["recall"]),
        "test_precision": float(row["precision"]),
    }
    result = evaluate_quality_gates(
        metrics,
        config=QualityGateConfig(
            min_pr_auc=args.min_pr_auc,
            min_recall=args.min_recall,
            min_precision=args.min_precision,
        ),
    )

    print("Model quality gate")
    print(f"  PR-AUC:     {metrics['test_pr_auc']:.6f} (min {args.min_pr_auc:.6f})")
    print(f"  Recall:     {metrics['test_recall']:.6f} (min {args.min_recall:.6f})")
    print(f"  Precision:  {metrics['test_precision']:.6f} (min {args.min_precision:.6f})")

    if result.passed:
        print("QUALITY GATE: PASS")
        return 0

    print("QUALITY GATE: FAIL")
    for failure in result.failures:
        print(f"  - {failure}")
    return 1


if __name__ == "__main__":
    raise SystemExit(main())
