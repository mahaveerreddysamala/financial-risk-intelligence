"""Run the Phase 40 Random Forest / LightGBM challenger benchmark."""
from __future__ import annotations

import argparse
from dataclasses import asdict
from pathlib import Path

import pandas as pd

from financial_risk.data_generation.generator import generate_transactions
from financial_risk.features.pipeline import build_feature_table
from financial_risk.models.baseline import CATEGORICAL_FEATURES, NUMERIC_FEATURES
from financial_risk.models.challengers import (
    build_lightgbm,
    build_random_forest,
    evaluate_challenger,
)
from financial_risk.models.split import temporal_split
from financial_risk.models.xgboost_model import build_xgboost_model, evaluate_xgboost

FEATURE_COLUMNS = NUMERIC_FEATURES + CATEGORICAL_FEATURES


def run_benchmark(rows: int = 20_000, seed: int = 42) -> pd.DataFrame:
    """Train XGBoost, Random Forest, and LightGBM on the same temporal split."""
    data = build_feature_table(generate_transactions(rows, seed=seed))
    train, _, test = temporal_split(data)
    y_train = train["is_fraud"].astype(int)
    negative = int((y_train == 0).sum())
    positive = int((y_train == 1).sum())
    scale_pos_weight = negative / max(positive, 1)

    models = [
        ("XGBoost", build_xgboost_model(scale_pos_weight=scale_pos_weight)),
        ("Random Forest", build_random_forest(scale_pos_weight=scale_pos_weight)),
        ("LightGBM", build_lightgbm(scale_pos_weight=scale_pos_weight)),
    ]
    results = []
    for name, model in models:
        model.fit(train[FEATURE_COLUMNS], y_train)
        if name == "XGBoost":
            evaluation = evaluate_xgboost(model, test)
        else:
            evaluation = evaluate_challenger(model, test, name)
        results.append(asdict(evaluation) if name != "XGBoost" else {"model": name, **asdict(evaluation)})
    return pd.DataFrame(results)


def main() -> None:
    parser = argparse.ArgumentParser(description="Run fraud-model challenger benchmark.")
    parser.add_argument("--rows", type=int, default=20_000)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--output", default="artifacts/challenger-benchmark.csv")
    args = parser.parse_args()

    results = run_benchmark(args.rows, args.seed)
    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    results.to_csv(output, index=False)
    print("Model challenger benchmark")
    print(results.to_string(index=False))
    print(f"\nResults written to: {output}")


if __name__ == "__main__":
    main()
