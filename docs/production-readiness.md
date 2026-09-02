# Production Deployment Readiness

Phase 15 hardens the FastAPI service for containerized deployment and operational checks.

## Configuration

Runtime settings are read from environment variables with safe defaults:

- `APP_ENV` — runtime environment name; defaults to `development`.
- `LOG_LEVEL` — application log level; defaults to `INFO`.
- `MODEL_ARTIFACT_PATH` — model artifact directory/path; defaults to `artifacts`.
- `APP_VERSION` — API version surfaced by `/version`; defaults to `0.1.0`.

No credentials or secrets are embedded in application code.

## Observability

The API emits compact JSON logs for completed and failed requests. Logged request metadata includes method, path, HTTP status, and duration. Request bodies and transaction payloads are intentionally excluded from application logs.

## Runtime endpoints

```text
GET /health   -> liveness-style service check
GET /ready    -> readiness check for orchestration
GET /version  -> application version and environment
```

## Container validation

The Docker image defines a container `HEALTHCHECK` against `/health`. GitHub Actions builds the image, starts the container, waits for `/ready`, then verifies `/health` and `/version` before cleanup.

## Deployment boundary

This phase establishes a deployable service contract. It does not claim a production cloud deployment, managed secrets integration, or model artifact download until those are implemented and measured.
