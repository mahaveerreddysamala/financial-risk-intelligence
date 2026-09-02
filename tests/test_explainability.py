import pandas as pd
import pytest

from financial_risk.data_generation.generator import generate_transactions
from financial_risk.features.pipeline import build_feature_table
from financial_risk.models.baseline import CATEGORICAL_FEATURES, NUMERIC_FEATURES
from financial_risk.models.explainability import ReasonCode, explain_xgboost, reason_code_table
from financial_risk.models.xgboost_model import build_xgboost_model

shap = pytest.importorskip("shap")

MODEL_FEATURES = NUMERIC_FEATURES + CATEGORICAL_FEATURES


def test_xgboost_shap_reason_codes():
    raw = generate_transactions(rows=800, seed=42)
    features = build_feature_table(raw)
    model = build_xgboost_model(scale_pos_weight=5.0)
    model.fit(features[MODEL_FEATURES], features["is_fraud"])

    explanations = explain_xgboost(model, features.head(4), top_n=3)

    assert len(explanations) == 4
    assert all(1 <= len(items) <= 3 for items in explanations)
    assert all(isinstance(item, ReasonCode) for item in explanations[0])
    table = reason_code_table(explanations[0])
    assert list(table.columns) == ["feature", "reason", "shap_value"]
    assert table["shap_value"].notna().all()
    assert table["feature"].notna().all()
    assert table["reason"].str.len().gt(0).all()


def test_explainability_validates_top_n():
    with pytest.raises(ValueError, match="top_n"):
        explain_xgboost(object(), pd.DataFrame(), top_n=0)


def test_explainability_validates_missing_features():
    with pytest.raises(ValueError, match="Missing model features"):
        explain_xgboost(object(), pd.DataFrame(), top_n=1)
