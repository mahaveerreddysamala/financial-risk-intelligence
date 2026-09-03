"""Train and persist the reproducible XGBoost serving artifact."""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import joblib

from financial_risk.data_generation.generator import generate_transactions
from financial_risk.features.pipeline import build_feature_table
from financial_risk.models.xgboost_model import build_xgboost_model
from financial_risk.models.split import temporal_split
from financial_risk.models.baseline import CATEGORICAL_FEATURES, NUMERIC_FEATURES, TARGET

FEATURE_COLUMNS = NUMERIC_FEATURES + CATEGORICAL_FEATURES


def train_and_save(rows: int, seed: int, output: str, model_version: str) -> Path:
    """Generate data, build features, train XGBoost, and save an inference artifact."""
    if rows < 1_000:
        raise ValueError("rows must be at least 1000")
    if not model_version.strip():
        raise ValueError("model_version must not be empty")

    raw = generate_transactions(rows=rows, seed=seed)
    features = build_feature_table(raw)
    train, _, _ = temporal_split(features)
    positive = int(train[TARGET].sum())
    negative = int(len(train) - positive)
    if positive == 0:
        raise ValueError("Generated training data contains no positive fraud labels")

    model = build_xgboost_model(scale_pos_weight=negative / positive)
    model.fit(train[FEATURE_COLUMNS], train[TARGET].astype(int))

    destination = Path(output)
    destination.parent.mkdir(parents=True, exist_ok=True)
    metadata = {
        "model_name": "financial-fraud-xgboost",
        "model_version": model_version,
        "feature_contract_version": "1.0",
        "feature_columns": FEATURE_COLUMNS,
        "training_rows": len(train),
        "seed": seed,
    }
    joblib.dump({"model": model, "metadata": metadata}, destination)
    metadata_path = destination.with_suffix(".json")
    metadata_path.write_text(json.dumps(metadata, indent=2), encoding="utf-8")
    return destination


def main() -> None:
    parser = argparse.ArgumentParser(description="Build the persisted fraud model serving artifact.")
    parser.add_argument("--rows", type=int, default=20_000)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--output", default="artifacts/financial-fraud-xgboost.joblib")
    parser.add_argument("--model-version", default="1.0.0")
    args = parser.parse_args()

    output = train_and_save(args.rows, args.seed, args.output, args.model_version)
    print(f"Saved serving artifact: {output}")


if __name__ == "__main__":
    main()
