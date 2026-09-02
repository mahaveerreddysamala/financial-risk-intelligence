import pandas as pd

from financial_risk.data_generation.generator import generate_transactions
from financial_risk.features.pipeline import build_feature_table
from financial_risk.models.split import temporal_split
from financial_risk.models.xgboost_model import build_xgboost_model, evaluate_xgboost


def test_xgboost_pipeline_trains_and_returns_probability_metrics() -> None:
    df = build_feature_table(generate_transactions(2_000, seed=42))
    train, validation, test = temporal_split(
        df,
        train_end="2025-09-01",
        validation_end="2025-11-01",
    )

    train_positive = int(train["is_fraud"].sum())
    train_negative = len(train) - train_positive
    scale_pos_weight = train_negative / max(train_positive, 1)

    model = build_xgboost_model(scale_pos_weight=scale_pos_weight)
    feature_columns = model.named_steps["preprocess"].transformers[0][2] + model.named_steps["preprocess"].transformers[1][2]
    model.fit(train[feature_columns], train["is_fraud"].astype(int))

    result = evaluate_xgboost(model, test)
    assert 0.0 <= result.roc_auc <= 1.0
    assert 0.0 <= result.pr_auc <= 1.0
    assert 0.0 <= result.precision <= 1.0
    assert 0.0 <= result.recall <= 1.0
    assert 0.0 <= result.f1 <= 1.0
    assert not validation.empty
