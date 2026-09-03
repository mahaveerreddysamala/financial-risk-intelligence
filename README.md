# Financial Crime & Risk Intelligence Platform

A production-oriented financial AI/ML platform for transaction fraud detection, anomaly detection, graph-based risk intelligence, explainable AI, MLOps, real-time streaming, and GenAI-assisted investigations.

## Project Vision

This project is built as a senior-level Data Scientist / AI-ML portfolio system rather than a single fraud-classification notebook. The platform combines behavioral machine learning, unsupervised anomaly detection, network risk signals, cost-sensitive decisioning, explainability, MLOps, real-time event processing, persisted model serving, and a grounded investigation copilot.

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
XGBoost             Isolation Forest      Network Features
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
Streaming Feature State
(prior-only customer history)
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
- Verified local event path from `transaction.created` through the worker to `transaction.risk_scored`
- Explicit deployment boundary: one local Kafka broker, plaintext listeners, single-partition application topics, and in-memory idempotency are used for reproducible portfolio validation

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
- Feature generation remains in-memory for reproducible local validation
- CI includes explicit history-window expiry coverage
- Production boundary: distributed feature state, durable state storage, partition-aware scaling, and external state recovery remain future deployment extensions

## Verified 20K Fraud Benchmark

| Model | ROC-AUC | PR-AUC | Precision | Recall | F1 |
|---|---:|---:|---:|---:|---:|
| Logistic Regression | 0.6131 | 0.0573 | 2.45% | 51.28% | 0.0468 |
| XGBoost | **0.6667** | **0.0802** | **4.44%** | 10.26% | **0.0620** |

XGBoost achieved a **12.87x lift** at the validation-selected `0.85` operating threshold on the synthetic test workload. These values are reproducible portfolio benchmarks, not production fraud-performance claims.

## Planned ML / Engineering Extensions

- Random Forest and LightGBM model comparisons
- Advanced graph/community detection
- Durable/distributed streaming feature state and external idempotency storage
- Managed Kafka, object storage, managed MLflow, and cloud deployment integrations
- Production alerting, autoscaling, TLS/authentication, and deeper operational monitoring
- External LLM provider integration for the grounded investigation copilot
- Broader benchmark and scale testing at 100K, 1M, 10M, and 50M synthetic transactions

## Repository Status

**Current phase:** Phase 33 — stateful real-time feature generation and persisted-model Kafka integration.

**Validation status:** Local lint/tests and Docker-based Kafka/model-serving paths are verified. GitHub Actions continues to enforce the test suite and linting; CI fixes are committed as issues are identified.
