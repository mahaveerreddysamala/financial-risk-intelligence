# Phase 50 — Cloud-Ready Architecture

## Design goals

The platform is designed as a stateless risk-inference service surrounded by durable event, state, artifact, and observability infrastructure. Containers remain portable between local Docker Compose and a managed cloud runtime.

## Reference architecture

```text
                    +----------------------+
                    | Client / Analyst UI  |
                    +----------+-----------+
                               |
                               v
                    +----------------------+
                    | ALB / API Gateway    |
                    +----------+-----------+
                               |
                 +-------------+-------------+
                 |                           |
                 v                           v
        +------------------+        +------------------+
        | API service      |        | Worker service   |
        | ECS/Fargate      |        | ECS/Fargate      |
        +--------+---------+        +---------+--------+
                 |                            |
                 +------------+---------------+
                              |
               +--------------+--------------+
               |              |              |
               v              v              v
          Managed Kafka   Redis-compatible     S3
          event stream   durable state       artifacts
               |              |              |
               +--------------+--------------+
                              |
                              v
                    CloudWatch / Prometheus

                    Investigation / RAG
                              |
                              v
                    External LLM provider
```

## Service responsibilities

| Component | Responsibility | Scaling model |
|---|---|---|
| API | synchronous scoring, investigation APIs, health/readiness | horizontal |
| Worker | Kafka consumption and risk scoring | consumer-group replicas |
| Kafka | ordered transaction event transport | partitions + brokers |
| Redis | feature history and idempotency state | managed replication/sharding |
| S3 | model artifacts and benchmark outputs | object storage |
| CloudWatch/Prometheus | logs, metrics, alerts | managed/centralized |

## Security model

- Put application workloads in private subnets where the deployment environment supports them.
- Expose only the load balancer/API ingress required by clients.
- Use IAM task roles instead of long-lived cloud credentials in containers.
- Store secrets in a managed secret store and inject them at runtime.
- Encrypt object storage, event transport, and durable state with platform-managed keys.
- Keep model artifacts read-only at inference time.
- Preserve the application's existing non-root container execution boundary.
- Do not place transaction payloads, credentials, API keys, or LLM secrets in logs.

## Reliability model

The application already defines idempotency, retry, dead-letter, durable state, health/readiness, and observability boundaries. Cloud deployment adds infrastructure-level durability and scaling rather than changing those application contracts.

For Kafka workers, scale replicas against partition count and consumer lag. Redis remains the shared state boundary so replicas do not create independent customer histories. API replicas remain stateless.

## Model lifecycle

```text
Training / evaluation
        |
        v
Quality gates + champion selection
        |
        v
Versioned artifact
        |
        v
Encrypted object storage
        |
        v
Controlled deployment
        |
        v
Persisted model serving
        |
        v
Monitoring + rollback
```

The model registry and promotion logic remain application-level controls; cloud storage and compute provide the runtime substrate.

## Cost-conscious portfolio deployment

A complete production deployment is intentionally not required to demonstrate the architecture. Local Docker Compose validates the end-to-end contracts. Terraform provides reproducible cloud foundations. A real cloud environment should be provisioned only when credentials, budgets, networking, managed-service configuration, and operational ownership are available.
