from __future__ import annotations

import pandas as pd
import pytest

from financial_risk.mlops.training import train_and_log_xgboost


FEATURES = [
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
]

CATEGORICAL_FEATURES = ["merchant_category", "payment_method", "channel", "country"]


def _frame(rows: int = 8) -> pd.DataFrame:
    data = {column: [float(index + 1) for index in range(rows)] for column in FEATURES}
    data["merchant_category"] = ["retail", "travel"] * (rows // 2) + ["retail"] * (rows % 2)
    data["payment_method"] = ["card", "wallet"] * (rows // 2) + ["card"] * (rows % 2)
    data["channel"] = ["online", "pos"] * (rows // 2) + ["online"] * (rows % 2)
    data["country"] = ["US", "CA"] * (rows // 2) + ["US"] * (rows % 2)
    data["is_fraud"] = [0, 1] * (rows // 2) + [0] * (rows % 2)
    return pd.DataFrame(data)


def test_training_logs_model_with_metadata(monkeypatch: pytest.MonkeyPatch) -> None:
    train = _frame(20)
    validation = _frame(10)
    test = _frame(10)
    captured: dict[str, object] = {}

    class _Result:
        run_id = "run-123"
        experiment_id = "42"
        model_uri = "runs:/run-123/xgboost-model"
        registered_model_name = "financial-fraud-xgboost"
        registered_model_version = "3"

    class _Model:
        def fit(self, *_args: object, **_kwargs: object) -> None:
            captured["fit"] = True

        def predict_proba(self, frame: pd.DataFrame) -> list[list[float]]:
            captured["prediction_rows"] = len(frame)
            return [[0.2, 0.8] for _ in range(len(frame))]

    def fake_builder(scale_pos_weight: float) -> _Model:
        captured["scale_pos_weight"] = scale_pos_weight
        return _Model()

    def fake_evaluate(model: _Model, frame: pd.DataFrame, threshold: float) -> object:
        del model
        captured.setdefault("thresholds", []).append(threshold)
        assert not frame.empty
        return type(
            "Eval",
            (),
            {"roc_auc": 0.7, "pr_auc": 0.08, "precision": 0.1, "recall": 0.2, "f1": 0.13},
        )()

    def fake_log(*args: object, **kwargs: object) -> _Result:
        captured["model"] = args[0]
        captured["parameters"] = kwargs["parameters"]
        captured["metrics"] = kwargs["metrics"]
        captured["tags"] = kwargs["tags"]
        return _Result()

    monkeypatch.setattr("financial_risk.mlops.training.build_xgboost_model", fake_builder)
    monkeypatch.setattr("financial_risk.mlops.training.evaluate_xgboost", fake_evaluate)
    monkeypatch.setattr("financial_risk.mlops.training.log_sklearn_run", fake_log)

    model, result, metrics = train_and_log_xgboost(
        train,
        validation,
        test,
        registered_model_name="financial-fraud-xgboost",
        threshold=0.85,
        artifact_root="artifacts/models",
    )

    assert captured["fit"] is True
    assert captured["scale_pos_weight"] == pytest.approx(1.0)
    assert captured["thresholds"] == [0.85, 0.85]
    assert captured["parameters"]["feature_count"] == len(FEATURES) + len(CATEGORICAL_FEATURES)
    assert captured["parameters"]["threshold"] == 0.85
    assert str(captured["tags"]["artifact_root"]).replace("\\", "/") == "artifacts/models"
    assert result.run_id == "run-123"
    assert model is captured["model"]
    assert metrics["test_pr_auc"] == 0.08


def test_training_rejects_invalid_threshold() -> None:
    frame = _frame(8)
    with pytest.raises(ValueError, match="threshold"):
        train_and_log_xgboost(frame, frame, frame, threshold=1.0)


def test_training_rejects_missing_feature_column() -> None:
    frame = _frame(8).drop(columns=["amount"])
    with pytest.raises(ValueError, match="missing required columns"):
        train_and_log_xgboost(frame, _frame(8), _frame(8))


def test_training_rejects_training_data_without_fraud() -> None:
    frame = _frame(8)
    frame["is_fraud"] = 0
    with pytest.raises(ValueError, match="positive fraud label"):
        train_and_log_xgboost(frame, _frame(8), _frame(8))
