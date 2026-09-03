from pathlib import Path

import joblib
import pytest
from fastapi.testclient import TestClient

from financial_risk.models.artifact import PersistedModelService
from financial_risk.models.artifact import FEATURE_COLUMNS
from financial_risk.api.app import app


class DummyModel:
    def predict_proba(self, frame):
        assert list(frame.columns) == FEATURE_COLUMNS
        return [[0.15, 0.85]]


def _features() -> dict[str, object]:
    numeric = {
        "amount": 250.0,
        "is_international": 0,
        "is_night": 0,
        "shared_device_account_count": 1,
        "customer_txn_count_7d": 2,
        "customer_avg_amount_30d": 100.0,
        "customer_std_amount_30d": 20.0,
        "customer_unique_merchants_7d": 2,
        "customer_unique_devices_30d": 1,
        "customer_international_rate_30d": 0.0,
        "customer_night_txn_rate_30d": 0.0,
        "amount_vs_customer_avg": 2.5,
        "amount_zscore": 3.0,
        "txn_count_5m": 1,
        "txn_count_1h": 2,
        "txn_count_24h": 3,
    }
    categorical = {
        "merchant_category": "electronics",
        "payment_method": "credit",
        "channel": "ecommerce",
        "country": "US",
    }
    return {**numeric, **categorical}


def test_persisted_model_service_loads_artifact(tmp_path: Path):
    artifact = tmp_path / "model.joblib"
    joblib.dump(
        {
            "model": DummyModel(),
            "metadata": {
                "model_name": "test-model",
                "model_version": "2.0.0",
                "feature_contract_version": "1.0",
            },
        },
        artifact,
    )

    prediction = PersistedModelService(artifact).predict(_features())
    assert prediction.fraud_probability == pytest.approx(0.85)
    assert prediction.model_name == "test-model"
    assert prediction.model_version == "2.0.0"


def test_persisted_model_service_requires_all_features(tmp_path: Path):
    artifact = tmp_path / "model.joblib"
    joblib.dump(
        {
            "model": DummyModel(),
            "metadata": {
                "model_name": "test-model",
                "model_version": "1.0.0",
                "feature_contract_version": "1.0",
            },
        },
        artifact,
    )

    with pytest.raises(ValueError, match="missing required features"):
        PersistedModelService(artifact).predict({})


def test_model_endpoint_returns_503_without_artifact():
    client = TestClient(app)
    response = client.post("/v1/model/score", json={"features": _features()})
    assert response.status_code == 503
