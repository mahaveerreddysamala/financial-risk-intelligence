import json

import pytest

from financial_risk.monitoring.model_registry import (
    build_run_id,
    load_model_run,
    save_model_run,
)


def test_run_id_is_deterministic():
    parameters = {"max_depth": 6, "learning_rate": 0.05}
    metrics = {"roc_auc": 0.67, "pr_auc": 0.08}
    first = build_run_id("xgboost", parameters, metrics)
    second = build_run_id("xgboost", parameters, metrics)
    assert first == second
    assert len(first) == 12


def test_save_and_load_model_run(tmp_path):
    path = save_model_run(
        tmp_path,
        "xgboost",
        {"max_depth": 6},
        {"roc_auc": 0.67, "pr_auc": 0.08},
        20,
        "artifacts/models/xgboost.joblib",
    )
    loaded = load_model_run(path)

    assert loaded.model_name == "xgboost"
    assert loaded.parameters["max_depth"] == 6
    assert loaded.metrics["roc_auc"] == pytest.approx(0.67)
    assert loaded.feature_count == 20
    assert loaded.artifact_path.endswith("xgboost.joblib")

    payload = json.loads(path.read_text(encoding="utf-8"))
    assert payload["run_id"] == loaded.run_id


def test_model_run_validates_inputs(tmp_path):
    with pytest.raises(ValueError, match="feature_count"):
        save_model_run(tmp_path, "xgboost", {}, {}, 0, "model.joblib")
    with pytest.raises(ValueError, match="model_name"):
        save_model_run(tmp_path, "", {}, {}, 10, "model.joblib")
    with pytest.raises(ValueError, match="artifact_path"):
        save_model_run(tmp_path, "xgboost", {}, {}, 10, "")


def test_load_model_run_requires_fields(tmp_path):
    path = tmp_path / "bad.json"
    path.write_text('{"model_name": "xgboost"}', encoding="utf-8")
    with pytest.raises(ValueError, match="Missing model-run fields"):
        load_model_run(path)
