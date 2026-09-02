import pytest

shap = pytest.importorskip("shap")

from financial_risk.data_generation.generator import generate_transactions
from financial_risk.features.pipeline import build_feature_table
from financial_risk.models.explainability import ReasonCode, explain_xgboost, reason_code_table
from financial_risk.models.xgboost_model import build_xgboost_model


def test_xgboost_shap_reason_codes():
    raw = generate_transactions(n_transactions=800, seed=42)
    features = build_feature_table(raw)
    model = build_xgboost_model(scale_pos_weight=5.0)
    feature_columns = model.named_steps["preprocess"].feature_names_in_
    model.fit(features[list(feature_columns)], features["is_fraud"])

    explanations = explain_xgboost(model, features.head(4), top_n=3)

    assert len(explanations) == 4
    assert all(1 <= len(items) <= 3 for items in explanations)
    assert all(isinstance(item, ReasonCode) for items in explanations[0])
    table = reason_code_table(explanations[0])
    assert list(table.columns) == ["feature", "reason", "shap_value"]
    assert table["shap_value"].notna().all()


def test_explainability_validates_top_n():
    with pytest.raises(ValueError, match="top_n"):
        explain_xgboost(object(), [], top_n=0)
