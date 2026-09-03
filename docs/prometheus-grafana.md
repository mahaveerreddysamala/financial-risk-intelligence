# Prometheus + Grafana Observability

Phase 29 connects the FastAPI `/metrics` endpoint to a local Prometheus collector and a provisioned Grafana dashboard.

## Flow

```text
FastAPI /metrics
       |
       v
  Prometheus
       |
       v
    Grafana
```

## Local validation

Start the stack:

```powershell
docker compose up -d --build
```

Verify services:

```powershell
docker compose ps
```

Open the local interfaces:

- Prometheus: http://localhost:9090
- Grafana: http://localhost:3000

The local Grafana credentials are `admin` / `admin`. This credential is intentionally for reproducible portfolio validation only and must be replaced by a secret-managed credential in a real deployment.

## Dashboard

Grafana provisions the `Financial Risk API` dashboard automatically. It includes API request volume, failed requests, request rate, processing latency, and HTTP status trends.

## Production boundary

This phase demonstrates collector and dashboard integration locally. Prometheus and Grafana persistence, authentication, TLS, alert routing, high availability, and secret management are deployment concerns and are not claimed as production infrastructure by this repository.
