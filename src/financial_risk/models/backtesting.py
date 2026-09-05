"""Walk-forward temporal backtesting for fraud-model stability analysis."""
from __future__ import annotations

import argparse
from collections.abc import Sequence
from dataclasses import asdict, dataclass
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.metrics import (
    average_precision_score,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
)

from financial_risk.data_generation.generator import generate_transactions
from financial_risk.features.pipeline import build_feature_table
from financial_risk.models.baseline import CATEGORICAL_FEATURES, NUMERIC_FEATURES
from financial_risk.models.threshold import precision_recall_at_k
from financial_risk.models.xgboost_model import build_xgboost_model

FEATURE_COLUMNS = NUMERIC_FEATURES + CATEGORICAL_FEATURES
DEFAULT_CUTOFFS = ("2025-07-01", "2025-09-01", "2025-11-01")


@dataclass(frozen=True)
class TemporalFold:
    """An expanding training window and its strictly later test window."""

    fold: int
    train: pd.DataFrame
    test: pd.DataFrame
    train_end: pd.Timestamp
    test_end: pd.Timestamp


@dataclass(frozen=True)
class BacktestResult:
    fold: int
    train_end: str
    test_end: str
    train_rows: int
    test_rows: int
    test_prevalence: float
    roc_auc: float
    pr_auc: float
    precision: float
    recall: float
    f1: float
    top_k: int
    precision_at_k: float
    recall_at_k: float
    lift_at_k: float


def build_temporal_folds(
    data: pd.DataFrame,
    cutoffs: Sequence[str] = DEFAULT_CUTOFFS,
    *,
    test_months: int = 2,
    time_column: str = "timestamp",
) -> list[TemporalFold]:
    """Create expanding-window folds without sharing future rows with training."""
    if time_column not in data.columns:
        raise ValueError(f"Missing time column: {time_column}")
    if test_months <= 0:
        raise ValueError("test_months must be positive")

    boundaries = [pd.Timestamp(value) for value in cutoffs]
    if not boundaries or boundaries != sorted(set(boundaries)):
        raise ValueError("cutoffs must be unique and ordered chronologically")

    frame = data.copy()
    frame[time_column] = pd.to_datetime(frame[time_column], errors="raise")
    folds = []
    for index, train_end in enumerate(boundaries, start=1):
        test_end = train_end + pd.DateOffset(months=test_months)
        train = frame[frame[time_column] < train_end].copy()
        test = frame[
            (frame[time_column] >= train_end) & (frame[time_column] < test_end)
        ].copy()
        if train.empty or test.empty:
            raise ValueError(f"Fold {index} must contain non-empty train and test sets")
        if train["is_fraud"].nunique() < 2 or test["is_fraud"].nunique() < 2:
            raise ValueError(f"Fold {index} must contain both target classes")
        folds.append(TemporalFold(index, train, test, train_end, test_end))
    return folds


def run_backtest(
    rows: int = 20_000,
    seed: int = 42,
    cutoffs: Sequence[str] = DEFAULT_CUTOFFS,
    test_months: int = 2,
    top_k: int = 100,
) -> pd.DataFrame:
    """Train an XGBoost model on each expanding window and evaluate future periods."""
    if rows <= 0 or top_k <= 0:
        raise ValueError("rows and top_k must be positive")

    data = build_feature_table(generate_transactions(rows, seed=seed))
    results = []
    for fold in build_temporal_folds(data, cutoffs, test_months=test_months):
        y_train = fold.train["is_fraud"].astype(int)
        positive = int(y_train.sum())
        negative = len(y_train) - positive
        model = build_xgboost_model(scale_pos_weight=negative / positive)
        model.fit(fold.train[FEATURE_COLUMNS], y_train)

        y_test = fold.test["is_fraud"].astype(int).to_numpy()
        probabilities = np.asarray(
            model.predict_proba(fold.test[FEATURE_COLUMNS])[:, 1], dtype=float
        )
        predictions = (probabilities >= 0.5).astype(int)
        fold_top_k = min(top_k, len(fold.test))
        precision_at_k, recall_at_k = precision_recall_at_k(
            y_test, probabilities, fold_top_k
        )
        prevalence = float(np.mean(y_test))
        results.append(
            asdict(
                BacktestResult(
                    fold=fold.fold,
                    train_end=fold.train_end.date().isoformat(),
                    test_end=fold.test_end.date().isoformat(),
                    train_rows=len(fold.train),
                    test_rows=len(fold.test),
                    test_prevalence=prevalence,
                    roc_auc=float(roc_auc_score(y_test, probabilities)),
                    pr_auc=float(average_precision_score(y_test, probabilities)),
                    precision=float(precision_score(y_test, predictions, zero_division=0)),
                    recall=float(recall_score(y_test, predictions, zero_division=0)),
                    f1=float(f1_score(y_test, predictions, zero_division=0)),
                    top_k=fold_top_k,
                    precision_at_k=precision_at_k,
                    recall_at_k=recall_at_k,
                    lift_at_k=precision_at_k / prevalence,
                )
            )
        )
    return pd.DataFrame(results)


def render_backtest_report(results: pd.DataFrame, *, rows: int, seed: int) -> str:
    """Summarize temporal stability across all out-of-time folds."""
    required = {
        "fold",
        "train_end",
        "test_end",
        "test_rows",
        "test_prevalence",
        "roc_auc",
        "pr_auc",
        "top_k",
        "precision_at_k",
        "recall_at_k",
        "lift_at_k",
    }
    missing = sorted(required.difference(results.columns))
    if missing or results.empty:
        raise ValueError(f"Backtest results are empty or missing columns: {missing}")

    lines = [
        "# Financial Risk Temporal Backtest",
        "",
        f"Walk-forward evaluation over **{rows:,}** synthetic transactions (seed **{seed}**).",
        "Each model is trained only on records preceding its out-of-time test window.",
        "",
        "| Fold | Train before | Test before | Test rows | Prevalence | ROC-AUC | PR-AUC | Top-K precision | Top-K recall | Lift |",
        "|---:|---|---|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for _, row in results.iterrows():
        lines.append(
            f'| {int(row["fold"])} | {row["train_end"]} | {row["test_end"]} | '
            f'{int(row["test_rows"]):,} | {float(row["test_prevalence"]):.2%} | '
            f'{float(row["roc_auc"]):.4f} | {float(row["pr_auc"]):.4f} | '
            f'{float(row["precision_at_k"]):.2%} | '
            f'{float(row["recall_at_k"]):.2%} | {float(row["lift_at_k"]):.2f}x |'
        )

    lines.extend(
        [
            "",
            "## Stability summary",
            "",
            f'- Mean out-of-time PR-AUC: **{results["pr_auc"].mean():.4f}**',
            f'- Minimum out-of-time PR-AUC: **{results["pr_auc"].min():.4f}**',
            f'- Mean top-K lift: **{results["lift_at_k"].mean():.2f}x**',
            f'- ROC-AUC range: **{results["roc_auc"].min():.4f}–{results["roc_auc"].max():.4f}**',
            "",
            (
                "> Synthetic-data results demonstrate leakage-safe stability testing; they are "
                "not production fraud-performance claims."
            ),
            "",
        ]
    )
    return "\n".join(lines)


def main() -> None:
    parser = argparse.ArgumentParser(description="Run walk-forward fraud-model backtesting.")
    parser.add_argument("--rows", type=int, default=20_000)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--test-months", type=int, default=2)
    parser.add_argument("--top-k", type=int, default=100)
    parser.add_argument("--cutoffs", nargs="+", default=list(DEFAULT_CUTOFFS))
    parser.add_argument("--output", default="artifacts/temporal-backtest.csv")
    parser.add_argument("--report-output", default="artifacts/temporal-backtest-report.md")
    args = parser.parse_args()

    results = run_backtest(args.rows, args.seed, args.cutoffs, args.test_months, args.top_k)
    output = Path(args.output)
    report_output = Path(args.report_output)
    for path in (output, report_output):
        path.parent.mkdir(parents=True, exist_ok=True)
    results.to_csv(output, index=False)
    report_output.write_text(
        render_backtest_report(results, rows=args.rows, seed=args.seed), encoding="utf-8"
    )
    print(results.to_string(index=False))
    print(f"Backtest results written to: {output}")
    print(f"Backtest report written to: {report_output}")


if __name__ == "__main__":
    main()
