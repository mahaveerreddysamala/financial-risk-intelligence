from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
COMPOSE = ROOT / "docker-compose.yml"
PROMETHEUS = ROOT / "monitoring" / "prometheus" / "prometheus.yml"
DATASOURCE = ROOT / "monitoring" / "grafana" / "provisioning" / "datasources" / "prometheus.yml"
DASHBOARD = ROOT / "monitoring" / "grafana" / "dashboards" / "financial-risk-api.json"


def test_monitoring_services_are_declared():
    content = COMPOSE.read_text(encoding="utf-8")
    assert "  prometheus:" in content
    assert "  grafana:" in content
    assert "prom/prometheus:v2.55.1" in content
    assert "grafana/grafana:11.4.0" in content


def test_prometheus_scrapes_api_metrics():
    content = PROMETHEUS.read_text(encoding="utf-8")
    assert "job_name: financial-risk-api" in content
    assert "metrics_path: /metrics" in content
    assert "api:8000" in content


def test_grafana_is_provisioned_from_prometheus():
    datasource = DATASOURCE.read_text(encoding="utf-8")
    dashboard = DASHBOARD.read_text(encoding="utf-8")
    assert "url: http://prometheus:9090" in datasource
    assert "Financial Risk API" in dashboard
    assert "financial_risk_api_api_requests_total" in dashboard
    assert "financial_risk_api_processing_latency_ms_p95" in dashboard
