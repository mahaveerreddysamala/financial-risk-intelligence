"""Reproducible fraud-model benchmark with temporal validation and threshold selection."""
from __future__ import annotations

import argparse
from dataclasses import asdict, dataclass
from pathlib import Path

import pandas as pd

from financial_risk.data_generation.generator import generate_transactions
from financial_risk.features.pipeline import build_feature_table
from financial_risk.models.baseline import (
    CATEGORICAL_FEATURES,
    NUMERIC_FEATURES,
    build_logistic_baseline,
    evaluate_binary_classifier,
)
from financial_risk.models.split import temporal_split
from financial_risk.models.threshold import evaluate_thresholds, precision_recall_at_k
from financial_risk.models.xgboost_model import build_xgboost_model, evaluate_xgboost

FEATURE_COLUMNS = NUMERIC_FEATURES + CATEGORICAL_FEATURES


@dataclass(frozen=True)
class ModelBenchmarkRow:
    model: str
    roc_auc: float
    pr_auc: float
    precision: float
    recall: float
    f1: float


def _fit_models(train: pd.DataFrame) -> tuple[object, object]:
    """Fit Logistic Regression and XGBoost using the common feature contract."""
    y_train = train["is_fraud"].astype(int)
    logistic = build_logistic_baseline()
    logistic.fit(train[FEATURE_COLUMNS], y_train)

    positive = int(y_train.sum())
    negative = len(y_train) - positive
    scale_pos_weight = negative / max(positive, 1)
    xgboost = build_xgboost_model(scale_pos_weight=scale_pos_weight)
    xgboost.fit(train[FEATURE_COLUMNS], y_train)
    return logistic, xgboost


def _select_f1_threshold(y_true: pd.Series, probabilities: object) -> float:
    """Select a threshold on validation data only, maximizing validation F1."""
    rows = evaluate_thresholds(y_true.to_numpy(), probabilities)
    return max(rows, key=lambda row: (row.f1, row.precision, -row.threshold)).threshold


def run_benchmark(rows: int = 20_000, seed: int = 42) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """Train, tune, and evaluate fraud models with a chronological split."""
    data = build_feature_table(generate_transactions(rows, seed=seed))
    train, validation, test = temporal_split(data)
    logistic, xgboost = _fit_models(train)

    logistic_result = evaluate_binary_classifier(logistic, test)
    xgboost_result = evaluate_xgboost(xgboost, test)
    benchmark = pd.DataFrame(
        [
            asdict(ModelBenchmarkRow("Logistic Regression", **asdict(logistic_result))),
            asdict(ModelBenchmarkRow("XGBoost", **asdict(xgboost_result))),
        ]
    )

    validation_probabilities = xgboost.predict_proba(validation[FEATURE_COLUMNS])[:, 1]
    selected_threshold = _select_f1_threshold(validation["is_fraud"], validation_probabilities)
    test_probabilities = xgboost.predict_proba(test[FEATURE_COLUMNS])[:, 1]
    threshold_values = (0.30, 0.50, 0.70, 0.85, selected_threshold)
    threshold_rows = evaluate_thresholds(
        test["is_fraud"].to_numpy(),
        test_probabilities,
        thresholds=threshold_values,
    )
    threshold_table = pd.DataFrame([asdict(row) for row in threshold_rows])
    threshold_table["model"] = "XGBoost"
    threshold_table["split"] = "test"
    threshold_table["selected_on_validation"] = threshold_table["threshold"].eq(selected_threshold)

    top_k_values = [100, 250, 500, 1000]
    top_k_rows = []
    prevalence = float(test["is_fraud"].mean())
    for k in top_k_values:
        precision, recall = precision_recall_at_k(
            test["is_fraud"].to_numpy(), test_probabilities, k
        )
        top_k_rows.append(
            {
                "model": "XGBoost",
                "split": "test",
                "k": min(k, len(test)),
                "precision_at_k": precision,
                "recall_at_k": recall,
                "lift_at_k": precision / prevalence if prevalence > 0 else 0.0,
            }
        )
    top_k_table = pd.DataFrame(top_k_rows)
    top_k_table.attrs["selected_validation_threshold"] = selected_threshold
    top_k_table.attrs["test_prevalence"] = prevalence
    return benchmark, threshold_table, top_k_table


def main() -> None:
    parser = argparse.ArgumentParser(description="Run the financial fraud model benchmark.")
    parser.add_argument("--rows", type=int, default=20_000)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--output", default="artifacts/model-benchmark.csv")
    parser.add_argument("--threshold-output", default="artifacts/xgboost-thresholds.csv")
    parser.add_argument("--top-k-output", default="artifacts/xgboost-top-k.csv")
    args = parser.parse_args()

    benchmark, thresholds, top_k = run_benchmark(args.rows, args.seed)
    output = Path(args.output)
    threshold_output = Path(args.threshold_output)
    top_k_output = Path(args.top_k_output)
    for path in (output, threshold_output, top_k_output):
        path.parent.mkdir(parents=True, exist_ok=True)

    benchmark.to_csv(output, index=False)
    thresholds.to_csv(threshold_output, index=False)
    top_k.to_csv(top_k_output, index=False)

    selected_threshold = top_k.attrs["selected_validation_threshold"]
    test_prevalence = top_k.attrs["test_prevalence"]
    print("Model benchmark")
    print(benchmark.to_string(index=False))
    print(f"\nTest fraud prevalence: {test_prevalence:.4%}")
    print("\nXGBoost test threshold analysis")
    print(thresholds.to_string(index=False))
    print("\nXGBoost test top-K analysis")
    print(top_k.to_string(index=False))
    print(f"\nSelected validation threshold: {selected_threshold:.4f}")
    print(f"Model results written to: {output}")
    print(f"Threshold results written to: {threshold_output}")
    print(f"Top-K results written to: {top_k_output}")


if __name__ == "__main__":
    main()
