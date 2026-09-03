# Phase 30: Model Serving Contract & Deployment Hardening

## Purpose

Phase 30 adds a stable model-serving boundary between the FastAPI transport layer and the risk decision engine. The service returns the risk decision together with explicit model and feature-contract metadata so downstream systems can trace which serving contract produced a decision.

## Serving contract

`financial_risk.models.service.RiskModelService` wraps the existing ensemble risk calculation and returns:

- `risk_score`
- `risk_band`
- `action`
- `model_name`
- `model_version`
- `feature_contract_version`

The default portfolio metadata is `financial-risk-ensemble`, model version `1.0.0`, and feature-contract version `1.0`. Values are environment-configurable through `MODEL_NAME`, `MODEL_VERSION`, and `FEATURE_CONTRACT_VERSION`.

The API `/v1/risk/score` and investigation endpoint both use this serving boundary. `/version` exposes the same metadata for deployment diagnostics.

## Container hardening

The API and streaming worker images now create a dedicated UID 10001 user and switch to that user before runtime execution. Application files are owned by the non-root user. This reduces the impact of a container compromise compared with running the application as root.

## Validation

Run the full test and lint suite locally:

```powershell
ruff check src tests scripts
python -m pytest -q
```

Build both application images and verify that the containers start successfully:

```powershell
docker compose up -d --build
docker compose ps
```

Verify the serving metadata:

```powershell
Invoke-WebRequest http://localhost:8000/version
```

A production deployment would additionally replace the portfolio risk-engine implementation with a persisted trained-model artifact, externalize model registry and feature-store dependencies, use authenticated/TLS-protected service endpoints, and deploy immutable image tags rather than `latest`.

## Production boundary

This phase demonstrates the application-level serving contract and non-root container hardening. It does not claim a managed model-serving platform, external model registry, autoscaling inference service, or production authentication/TLS configuration.
