from fastapi.testclient import TestClient

from financial_risk.api.app import app


client = TestClient(app)


def test_metrics_endpoint_exposes_prometheus_metrics():
    health = client.get("/health")
    assert health.status_code == 200

    response = client.get("/metrics")

    assert response.status_code == 200
    assert response.headers["content-type"].startswith("text/plain; version=0.0.4")
    assert "financial_risk_api_api_requests_total" in response.text
    assert "financial_risk_api_api_status_200" in response.text
    assert "financial_risk_api_processing_latency_ms_count" in response.text
