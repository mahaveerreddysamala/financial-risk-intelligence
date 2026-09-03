# Kafka End-to-End Smoke Test

Phase 23 adds an isolated Docker Compose environment that verifies a real Kafka broker can transport the project's versioned financial-risk events.

## Flow

```text
Kafka producer
     |
     v
Apache Kafka broker
     |
     v
Kafka consumer
     |
     v
process_event()
     |
     v
CRITICAL risk decision
```

## Run

The smoke environment is intentionally separate from the normal CI/runtime dependency set.

```powershell
docker compose -f docker-compose.kafka.yml up --build --abort-on-container-exit --exit-code-from kafka-smoke
```

A successful run ends with:

```text
KAFKA E2E SMOKE TEST: PASS
```

Clean up with:

```powershell
docker compose -f docker-compose.kafka.yml down -v
```

## Scope and boundary

This verifies broker connectivity, producer delivery, consumer retrieval, event deserialization, and integration with the real-time risk worker. It is an integration smoke test, not a production deployment. Authentication, TLS, Schema Registry, multi-broker replication, dead-letter topics, observability, and exactly-once guarantees remain deployment concerns for a managed Kafka environment.
