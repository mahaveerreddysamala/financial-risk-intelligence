import pandas as pd
import pytest
from fastapi.testclient import TestClient

from financial_risk.api.app import app


client = TestClient(app)


def test_health():
    response = client.get("/health")
    assert response.status_code == 200
    assert response.json() == {"status": "ok"}


def test_risk_score_endpoint():
    response = client.post(
        "/v1/risk/score",
        json={
            "fraud_probability": 0.9,
            "anomaly_score": 0.8,
            "network_score": 0.7,
            "velocity_score": 0.6,
        },
    )
    assert response.status_code == 200
    payload = response.json()
    assert payload["risk_score"] == pytest.approx(0.81)
    assert payload["risk_band"] == "CRITICAL"
    assert payload["action"] == "hold_and_investigate"


def test_investigation_case_endpoint():
    response = client.post(
        "/v1/investigations/cases",
        json={
            "fraud_probability": 0.9,
            "anomaly_score": 0.8,
            "network_score": 0.7,
            "velocity_score": 0.6,
            "transaction": {"transaction_id": "TXN123", "amount": 250.0, "country": "US"},
        },
    )
    assert response.status_code == 200
    payload = response.json()
    assert payload["transaction_id"] == "TXN123"
    assert payload["risk_band"] == "CRITICAL"
    assert any(item["field"] == "fraud_probability" for item in payload["evidence"])
    assert any(item["field"] == "amount" for item in payload["evidence"])


def test_copilot_prompt_endpoint():
    response = client.post(
        "/v1/copilot/prompt",
        json={
            "case_id": "CASE123",
            "evidence": [
                {"field": "txn_count_1h", "value": 9, "signal": "velocity_risk", "severity": "high"}
            ],
            "references": [
                {"document_id": "VEL-001", "score": 0.91, "text": "Velocity fraud involves rapid transactions."}
            ],
        },
    )
    assert response.status_code == 200
    prompt = response.json()["grounded_prompt"]
    assert "CASE123" in prompt
    assert "Do not invent facts" in prompt
    assert "VEL-001" in prompt


def test_risk_validation():
    response = client.post(
        "/v1/risk/score",
        json={
            "fraud_probability": 1.5,
            "anomaly_score": 0.1,
            "network_score": 0.1,
            "velocity_score": 0.1,
        },
    )
    assert response.status_code == 422
