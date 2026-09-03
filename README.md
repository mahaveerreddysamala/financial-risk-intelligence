# Financial Crime & Risk Intelligence Platform

A production-oriented financial AI/ML platform for transaction fraud detection, anomaly detection, graph-based risk intelligence, explainable AI, MLOps, real-time streaming, and GenAI-assisted investigations.

## Project Vision

This project is built as a senior-level Data Scientist / AI-ML portfolio system rather than a single fraud-classification notebook. The platform combines behavioral machine learning, unsupervised anomaly detection, network risk signals, graph/community intelligence, cost-sensitive decisioning, explainability, MLOps, real-time event processing, persisted model serving, durable distributed state, and a grounded investigation copilot.

## Architecture

```text
Financial Transactions
        |
        +--> Batch / Kafka Streaming Ingestion
        |
        v
Data Quality + Contracts
        |
        v
Leakage-Aware Feature Engineering
        |
        +-------------------+-------------------+
        |                   |                   |
        v                   v                   v
Supervised ML       Anomaly Detection    Graph Risk
XGBoost             Isolation Forest      Network + Communities
        |                   |                   |
        +-------------------+-------------------+
                            |
                            v
                  Risk Scoring / Decisioning
                            |
               +------------+------------+
               |                         |
               v                         v
            Approve                 Review / Hold
                                         |
                                         v
                               SHAP + Reason Codes
                                         |
                                         v
                              Investigation Case
                                         |
                                         v
                              GenAI Copilot + RAG
                                         |
                                         v
                                  FastAPI Service
                                         |
                             +-----------+-----------+
                             |                       |
                             v                       v
                       Prometheus              Grafana

Kafka transaction path:

transaction.created
        |
        v
Kafka partition key = customer_id
        |
        v
Consumer Group (scalable workers)
        |
        v
Durable Streaming State
(Redis customer history + idempotency)
        |
        v
Prior-only Feature Generation
        |
        v
Persisted XGBoost Artifact
        |
        v
Fraud Probability
        |
        +--> Anomaly + Network + Velocity Signals
        |
        v
Ensemble Risk Score
        |
        v
Inference Telemetry
        |
        v
transaction.risk_scored
```

## Phases 1–20

The repository includes deterministic synthetic data generation, data contracts, leakage-aware feature engineering, fraud modeling, anomaly detection, graph intelligence, calibration and cost-sensitive decisioning, monitoring, model registry, investigation cases, grounded GenAI/RAG, FastAPI service endpoints, containerization, production-readiness checks, MLflow lifecycle integration, executable MLflow training, model-quality gates, CI enforcement, and Airflow orchestration.

## Phase 21: Kafka Streaming Event Layer

- Versioned, transport-neutral event envelope for financial transaction events
- Stable event IDs suitable for Kafka keys and downstream idempotency strategies
- Explicit event type, schema version, UTC occurrence time, and JSON payload
- Optional Confluent Kafka producer and consumer adapters loaded lazily
- Consumer configuration with manual offset commits for process-then-commit workflows
- Streaming dependency isolated from the core CI/runtime dependency set

See [`docs/kafka-streaming.md`](docs/kafka-streaming.md) for the streaming architecture and reliability contract.

## Phase 22: Real-Time Risk Scoring Consumer

- Reuses the established ensemble fraud/anomaly/network/velocity risk engine
- Validates normalized streaming risk signals before decisioning
- Produces operational LOW, MEDIUM, HIGH, and CRITICAL outcomes
- Automatically assembles an evidence-grounded investigation case for CRITICAL events
- Preserves event ID, transaction ID, and event timestamp for traceability
- Produces a `transaction.risk_scored` downstream event envelope
- Adds a transport-neutral `process_event()` worker boundary for Kafka consumers, replay jobs, and test harnesses
- Explicitly avoids claims of a live broker, exactly-once semantics, or persisted model serving until those components are deployed and verified

See [`docs/real-time-risk-scoring.md`](docs/real-time-risk-scoring.md) for the streaming inference contract.

## Phase 23: Kafka End-to-End Integration Smoke Test

- Docker Compose environment with a real Apache Kafka broker
- Isolated Kafka smoke-test container using the optional streaming dependency
- Producer → broker → consumer verification using the project's event envelope
- Real-time `process_event()` risk-scoring verification after broker delivery
- Container health check and dependency ordering before the smoke test starts
- Repeatable command for local broker integration validation
- Explicit integration-test boundary: no claim of secured production Kafka, Schema Registry, or multi-broker HA deployment

See [`docs/kafka-e2e-smoke.md`](docs/kafka-e2e-smoke.md) for the runbook and production boundary.

## Phase 24: Production-Style Streaming Runtime Reliability

- Bounded retries with configurable retry budget and backoff
- Structured dead-letter records for events that exhaust retries
- Idempotency boundary that marks events only after successful downstream publication
- In-process operational counters for received, succeeded, retried, duplicate, and dead-lettered events
- Structured logging for successful processing, retries, duplicate suppression, and DLQ routing
- Continuous Kafka worker wiring with configurable input, output, DLQ, and consumer-group topics
- Transport-agnostic runtime so the same reliability boundary can support Kafka, replay jobs, or other event transports
- Explicit production boundary: the default idempotency store is in-memory, and exactly-once processing still depends on external state and messaging guarantees

See [`docs/streaming-runtime.md`](docs/streaming-runtime.md) for the reliability model and worker runbook.

## Phase 25: Streaming Observability & Operational Metrics

- Thread-safe in-process counters for streaming throughput and outcomes
- Processing-latency timing with count, average, maximum, and p95 summaries
- Risk-band counters for LOW, MEDIUM, HIGH, and CRITICAL decisions
- Poll-timeout, failure, retry, duplicate, success, and dead-letter counters
- Prometheus-compatible text exposition for integration with a metrics collector
- Metrics embedded in the transport-agnostic runtime without changing the event contract
- FastAPI `/metrics` endpoint exposing Prometheus-compatible application metrics
- Explicit production boundary: the current metrics backend is in-process; persistent metrics storage and dashboards remain deployment concerns

See [`docs/streaming-observability.md`](docs/streaming-observability.md) and [`docs/api-operational-metrics.md`](docs/api-operational-metrics.md).

## Phase 28: Docker Compose Production-Style Stack

- Multi-service local stack combining FastAPI, Kafka, topic initialization, and the streaming risk worker
- Dedicated worker image with the optional Confluent Kafka dependency isolated from the API image
- Health-gated Kafka topic initialization before API and worker startup
- Input, scored-output, and dead-letter Kafka topics wired through environment configuration
- API container with Docker health check and `/health`, `/ready`, `/version`, and `/metrics` endpoints
- Restart policies for long-running Kafka, API, and worker services
- Verified local event path from `transaction.created` through the worker to the `transaction.risk_scored` output
- Explicit deployment boundary: one local Kafka broker, plaintext listeners, and reproducible local validation are used for the portfolio stack

See [`docs/docker-compose-stack.md`](docs/docker-compose-stack.md).

## Phase 29: Prometheus + Grafana Observability

- Prometheus collector scraping the FastAPI `/metrics` endpoint
- Grafana provisioned with a Prometheus datasource and dashboard
- Pre-provisioned `Financial Risk API` dashboard for request volume, failures, request rate, latency, and HTTP status trends
- Monitoring services integrated into the Docker Compose stack
- Live validation of API health, application metrics, Prometheus readiness, Grafana health, and Kafka worker activity
- Explicit production boundary: local dashboards use reproducible demo credentials; persistence, TLS, authentication, alerting, and highly available monitoring remain deployment concerns

See [`docs/prometheus-grafana.md`](docs/prometheus-grafana.md).

## Phase 30: Model Serving Contract & Deployment Hardening

- Dedicated `RiskModelService` boundary between the FastAPI transport layer and the risk decision engine
- Versioned serving metadata for model name, model version, and feature-contract version
- `/version` exposes deployment and model metadata for diagnostics
- API and streaming worker containers run as non-root UID 10001 users
- Container files are owned by the runtime user before process startup
- Explicit production boundary: persisted artifact loading was introduced separately in Phase 31; managed registry, authenticated/TLS service mesh, and autoscaling inference remain deployment extensions

See [`docs/model-serving-deployment.md`](docs/model-serving-deployment.md).

## Phase 31: Persisted XGBoost Model Artifact Serving

- Persisted `financial-fraud-xgboost.joblib` serving artifact
- Dedicated `PersistedModelService` with explicit artifact and metadata validation
- Exact model feature contract enforced before inference
- Model provenance returned with every persisted-model prediction
- FastAPI `POST /v1/model/score` endpoint for persisted-model inference
- Docker Compose mounts the artifact directory read-only into the API and streaming worker
- `scikit-learn==1.5.2` pinned to match the serialized artifact and prevent estimator deserialization incompatibility
- Verified inside the container with scikit-learn 1.5.2 and successful persisted-model inference
- Explicit security boundary: joblib/pickle artifacts are treated as trusted internal model artifacts

## Phase 32: Persisted Model Integrated into Kafka Streaming

- Streaming worker loads the persisted XGBoost artifact at startup
- `transaction.created` events can provide the model feature vector to the persisted model service
- Fraud probability from persisted XGBoost is combined with anomaly, network, and velocity signals through the established ensemble decision engine
- Model name, version, and feature-contract version propagate into downstream `transaction.risk_scored` events
- Kafka end-to-end validation completed using a real broker and the persisted model
- Verified example: `phase32-model-test-002` produced `fraud_probability=0.0355707`, `risk_score=0.3777854`, `risk_band=MEDIUM`, `action=monitor`
- Kafka boundary hardened so null/tombstone and malformed JSON records are skipped without crashing the long-running worker

## Phase 33: Stateful Real-Time Feature Generation

- Streaming feature state generates the persisted model's required behavioral and velocity features from raw transaction events
- Prior-only customer history is used so the current transaction does not leak into its own features
- Rolling customer windows cover 7-day behavioral history and 30-day customer statistics
- Short-horizon transaction velocity features cover 5-minute, 1-hour, and 24-hour windows
- `prepare()` computes features without mutating history; `commit()` records the transaction only after successful processing
- Feature generation remains compatible with in-memory state for deterministic tests
- Docker validation completed with raw transactions for the same customer; generated history was reflected in downstream telemetry

## Phase 34: Streaming Inference Telemetry

- Downstream `transaction.risk_scored` events include compact `feature_telemetry`
- Telemetry identifies whether inference used the persisted model artifact or a precomputed compatibility signal
- Persisted-model events expose feature count and selected prior-history/velocity features without publishing the entire feature vector
- Telemetry preserves model name, model version, and feature-contract version alongside the scoring result
- Tests cover telemetry serialization and persisted-model inference metadata
- Docker validation demonstrated generated telemetry for raw Kafka transactions
- Explicit production boundary: telemetry is currently carried in the event payload; centralized streaming metrics, long-term audit storage, and full distributed tracing remain deployment extensions

## Phase 35: Durable Distributed Streaming State

- Redis-backed customer feature history for state that survives worker restarts
- Redis-backed event idempotency keys with bounded retention
- Stateful feature generation keeps the existing prepare-before-commit leakage boundary while using shared state
- Worker enables Redis automatically when `REDIS_URL` is configured
- Docker Compose adds a health-checked Redis service and health-gated worker startup
- In-memory state implementations remain available for deterministic unit tests and transport-agnostic runs
- Verified restart persistence: customer history retained transactions after worker restart, and the restarted worker recognized a replayed event as a duplicate
- Explicit production boundary: the local stack does not yet configure Redis authentication, TLS, replication, backups, high availability, or operational persistence policies

See [`docs/durable-streaming-state.md`](docs/durable-streaming-state.md).

## Phase 36: Kafka Partition-Aware Horizontal Scaling

- `transaction.created` events now use `customer_id` as the Kafka routing key
- Same-customer transactions therefore map to the same partition, preserving per-customer ordering within the Kafka partition model
- `financial-risk-events` is configured with three local partitions for scale-out validation
- Worker replicas use the same Kafka consumer group so Kafka can distribute partitions across instances
- Worker `container_name` is removed so Docker Compose can create multiple worker replicas
- Redis remains the shared feature-history and idempotency state boundary across replicas
- Added unit coverage for customer-based routing-key behavior and fallback routing
- Added a reproducible runbook for 2-worker / 3-partition validation
- Explicit production boundary: the local validation uses one Kafka broker; production requires replicated brokers, capacity planning, partition sizing, and operational rebalance tuning

See [`docs/kafka-horizontal-scaling.md`](docs/kafka-horizontal-scaling.md).

## Phase 37: Redis Atomicity & Distributed State Safety

- Atomic event-claim semantics prevent two horizontally scaled workers from processing the same event concurrently
- Claim leases expire so a crashed worker does not permanently strand an event
- Successful processing marks the event complete; failed processing releases the claim for retry
- Customer transaction history uses an atomic append operation instead of an unsafe read-modify-write sequence
- Shared Redis state remains durable across worker restarts and replicas
- In-memory idempotency behavior remains available for deterministic unit tests
- Added dedicated Phase 37 state-safety tests covering claim, release, completion, and atomic-history integration
- Verified end-to-end with two workers: the first copy of `phase37-duplicate-002` succeeded and the duplicate was suppressed
- Verified Redis persisted exactly one customer transaction and a bounded idempotency key TTL
- Explicit production boundary: Redis authentication, TLS, replication, failover, backups, and managed operational controls remain deployment concerns

See [`docs/redis-atomicity.md`](docs/redis-atomicity.md) for the state-safety design and validation runbook.

## Phase 38: Graph Community Detection

- Adds deterministic heterogeneous entity-graph construction across customers, accounts, devices, IPs, and merchants
- Uses weighted entity relationships so repeated reuse strengthens graph edges
- Detects communities through modularity optimization for fraud-ring style network segmentation
- Produces customer-level `community_id`, `community_customer_count`, weighted network degree, and `community_risk_signal` features
- Keeps community analysis as an explainable feature layer that can complement supervised fraud probability and existing network-risk signals
- Adds automated tests for graph construction, deterministic community assignment, derived features, and input validation
- Uses NetworkX as a focused graph-analysis dependency; graph storage remains in-process for reproducible portfolio validation
- Explicit production boundary: community detection is currently batch/in-process; distributed graph processing and online incremental community updates remain future deployment extensions

See [`docs/graph-community-detection.md`](docs/graph-community-detection.md).

## Verified 20K Fraud Benchmark

| Model | ROC-AUC | PR-AUC | Precision | Recall | F1 |
|---|---:|---:|---:|---:|---:|
| Logistic Regression | 0.6131 | 0.0573 | 2.45% | 51.28% | 0.0468 |
| XGBoost | **0.6667** | **0.0802** | **4.44%** | 10.26% | **0.0620** |

XGBoost achieved a **12.87x lift** at the validation-selected `0.85` operating threshold on the synthetic test workload. These values are reproducible portfolio benchmarks, not production fraud-performance claims.

## Planned ML / Engineering Extensions

- Advanced graph analytics and online community updates
- Random Forest and LightGBM model comparisons
- Managed Kafka, object storage, managed MLflow, and cloud deployment integrations
- Production alerting, autoscaling, deeper operational monitoring, and distributed tracing
- External LLM provider integration for the grounded investigation copilot
- Broader benchmark and scale testing at 100K, 1M, 10M, and 50M synthetic transactions

## Repository Status

**Current phase:** Phase 38 — Graph community detection.

**Validation status:** Phase 37 distributed idempotency and durable customer-state behavior has been validated end-to-end. Phase 38 implementation and automated tests are now added and ready for local validation.
