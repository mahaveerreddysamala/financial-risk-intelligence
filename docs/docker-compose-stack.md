# Docker Compose Production-Style Stack

Phase 28 composes the FastAPI service, Kafka broker, Kafka topic initialization, and the continuous streaming risk-scoring worker into one local deployment.

## Services

| Service | Purpose | Port |
|---|---|---:|
| `kafka` | Kafka broker for financial transaction events | 9092 |
| `kafka-init` | Creates input, scored-output, and dead-letter topics | — |
| `api` | FastAPI risk, investigation, readiness, and metrics endpoints | 8000 |
| `worker` | Continuous Kafka consumer and real-time risk scorer | — |

## Start the stack

```powershell
docker compose up -d --build
```

Check service state:

```powershell
docker compose ps
```

Verify the API:

```powershell
Invoke-WebRequest http://localhost:8000/health
Invoke-WebRequest http://localhost:8000/ready
Invoke-WebRequest http://localhost:8000/metrics
```

Inspect worker logs:

```powershell
docker compose logs worker
```

Inspect Kafka logs:

```powershell
docker compose logs kafka
```

Stop the stack:

```powershell
docker compose down
```

## Runtime boundary

This is a production-style local deployment, not a production Kafka deployment. The stack uses one Kafka broker, plaintext listeners, one partition per application topic, and the current in-memory idempotency implementation. External persistent state, secured Kafka, multi-broker availability, schema management, and durable metrics remain deployment extensions.
