"""Reproducible model-comparison benchmark for financial fraud detection."""
from __future__ import annotations

import argparse
from dataclasses import asdict, dataclass
from pathlib import Path

import pandas as pd

from financial_risk.data_generation.generator import generate_transactions
from financial_risk.features.pipeline import build_feature_table
from financial_risk.models.baseline import build_logistic_baseline, evaluate_binary_classifier
from financial_risk.models.comparison import compare_models
from financial_risk.models.split import temporal_split
from financial_risk.models.threshold import threshold_metrics
from financial_risk.models.xgboost_model import build_xgboost_model, evaluate_xgboost


@dataclass(frozen=True)
class ModelBenchmarkRow:
    model: str
    roc_auc: float
    pr_auc: float
    precision: float
    recall: float
    f1: float


def _feature_columns() -> list[str]:
    return [
        "amount",
        "is_international",
        "is_night",
        "shared_device_account_count",
        "customer_txn_count_7d",
        "customer_avg_amount_30d",
        "customer_std_amount_30d",
        "customer_unique_merchants_7d",
        "customer_unique_devices_30d",
        "customer_international_rate_30d",
        "customer_night_txn_rate_30d",
        "amount_vs_customer_avg",
        "amount_zscore",
        "txn_count_5m",
        "txn_count_1h",
        "txn_count_24h",
        "merchant_category",
        "payment_method",
        "channel",
        "country",
    ]


def run_benchmark(rows: int = 20_000, seed: int = 42) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Train baseline and XGBoost models with temporal evaluation."""
    data = build_feature_table(generate_transactions(rows, seed=seed))
    train, validation, test = temporal_split(data)
    features = _feature_columns()

    logistic = build_logistic_baseline()
    logistic.fit(train[features], train["is_fraud"].astype(int))
    logistic_result = evaluate_binary_classifier(logistic, test)

    positives = int(train["is_fraud"].sum())
    negatives = len(train) - positives
    xgb = build_xgboost_model(scale_pos_weight=negatives / max(positives, 1))
    xgb.fit(train[features], train["is_fraud"].astype(int))
    xgb_result = evaluate_xgboost(xgb, test)

    rows_out = pd.DataFrame(
        [
            asdict(ModelBenchmarkRow("Logistic Regression", **asdict(logistic_result))),
            asdict(ModelBenchmarkRow("XGBoost", **asdict(xgb_result))),
        ]
    )

    comparison = compare_models(logistic_result, xgb_result)

    validation_probabilities = xgb.predict_proba(validation[features])[:, 1]
    threshold_rows = threshold_metrics(validation["is_fraud"].astype(int).to_numpy(), validation_probabilities)
    threshold_table = pd.DataFrame(threshold_rows)
    threshold_table["model"] = "XGBoost"
    threshold_table["split"] = "validation"
    threshold_table["rows"] = len(validation)
    threshold_table.attrs["comparison"] = comparison

    return rows_out, threshold_table


def main() -> None:
    parser = argparse.ArgumentParser(description="Run the financial fraud model benchmark.")
    parser.add_argument("--rows", type=int, default=20_000)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--output", default="artifacts/model-benchmark.csv")
    parser.add_argument("--threshold-output", default="artifacts/xgboost-thresholds.csv")
    args = parser.parse_args()

    results, thresholds = run_benchmark(args.rows, args.seed)
    output = Path(args.output)
    threshold_output = Path(args.threshold_output)
    output.parent.mkdir(parents=True, exist_ok=True)
    threshold_output.parent.mkdir(parents=True, exist_ok=True)
    results.to_csv(output, index=False)
    thresholds.to_csv(threshold_output, index=False)

    print("Model benchmark")
    print(results.to_string(index=False))
    print("\nValidation threshold analysis")
    print(thresholds.to_string(index=False))
    print(f"\nModel results written to: {output}")
    print(f"Threshold results written to: {threshold_output}")


if __name__ == "__main__":
    main()
