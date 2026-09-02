# Financial Crime & Risk Intelligence Platform

A production-oriented financial AI/ML platform for transaction fraud detection, anomaly detection, graph-based risk intelligence, explainable AI, and GenAI-assisted investigations.

## Project Vision

This project is being built as a senior-level Data Scientist / AI-ML portfolio system rather than a single fraud-classification notebook. The platform combines behavioral machine learning, unsupervised anomaly detection, network risk signals, cost-sensitive decisioning, explainability, MLOps, and a grounded investigation copilot.

## Planned Architecture

```text
Financial Events
      |
      +--> Batch / Streaming Ingestion
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
XGBoost/LightGBM    Isolation Forest      Network Features
      |                   |                   |
      +-------------------+-------------------+
                          |
                          v
                Risk Scoring / Decisioning
                          |
                 +--------+--------+
                 |                 |
                 v                 v
              Approve           Review
                                   |
                                   v
                         SHAP + Reason Codes
                                   |
                                   v
                     Investigation Case + Evidence
                                   |
                                   v
                         GenAI Investigation
                              Copilot + RAG
                                   |
                                   v
                         FastAPI + Container
```

## Phase 1: Foundation

- Deterministic synthetic financial ecosystem generator
- Customers, accounts, merchants, devices, locations, and transactions
- Configurable fraud typologies
- Reproducible Parquet output
- Data-generation configuration
- Unit tests and validation

## Phase 2: Data Quality & Contracts

- Canonical transaction data contract
- Required-field and identifier validation
- Timestamp and numeric-domain checks
- Fraud-label consistency checks
- Fail-fast validation API for batch and future streaming ingestion
- Contract tests for valid and intentionally corrupted datasets

See [`docs/data-contract.md`](docs/data-contract.md) for the field-level contract and validation behavior.

## Phase 3: Leakage-Aware Feature Engineering

- Customer behavioral baselines and rolling activity windows
- Transaction velocity features across 5-minute, 1-hour, and 24-hour windows
- Geographic and temporal deviation signals
- Prior-only device, IP, merchant, and customer/device reuse features
- Amount deviation and customer-normalized risk signals
- Deterministic feature pipeline with explicit feature tests

See [`docs/feature-engineering.md`](docs/feature-engineering.md) for feature definitions, leakage controls, and the PySpark scaling path.

## Phase 4: Fraud Modeling

- Class-balanced Logistic Regression baseline
- XGBoost fraud classifier with training-set class weighting
- Chronological train/validation/test evaluation
- PR-AUC, ROC-AUC, precision, recall, and F1
- Decision-threshold analysis for investigation capacity
- Precision@K and Recall@K for investigator prioritization
- Reproducible model-comparison utilities
- Validation-selected operating threshold
- Fraud prevalence and lift reporting

Measured 20K benchmark:

| Model | ROC-AUC | PR-AUC | Precision | Recall | F1 |
|---|---:|---:|---:|---:|---:|
| Logistic Regression | 0.6131 | 0.0573 | 2.45% | 51.28% | 0.0468 |
| XGBoost | **0.6667** | **0.0802** | **4.44%** | 10.26% | **0.0620** |

XGBoost achieved a **12.87x lift** at the validation-selected `0.85` operating threshold, with 15.38% test precision and 5.13% test recall on the 20K synthetic test workload. These values are a reproducible synthetic-data benchmark, not a claim of production fraud performance.

See [`docs/model-benchmark.md`](docs/model-benchmark.md) for the evaluation protocol.

## Phase 5: Anomaly Detection & Ensemble Risk Scoring

- Isolation Forest trained on historical/training observations only
- Normalized anomaly scoring from behavioral and transaction features
- Ensemble risk score combining supervised fraud probability, anomaly, network, and velocity signals
- Operational risk bands: LOW, MEDIUM, HIGH, CRITICAL
- Decisioning actions for approval, monitoring, step-up verification, and investigation

See [`docs/anomaly-and-risk-scoring.md`](docs/anomaly-and-risk-scoring.md) for the scoring design and operating rules.

## Phase 6: SHAP Explainability & Reason Codes

- Transaction-level SHAP explanations for the XGBoost model
- Ranked positive and negative model contributions
- Human-readable reason codes for investigators
- Analyst-facing reason-code table for downstream APIs and investigation workflows
- Explicit separation between model explanation and a claim of fraud causality

See [`docs/explainability.md`](docs/explainability.md) for the explainability design, interpretation guidance, and limitations.

## Phase 7: Graph-Based Financial Crime Intelligence

- Entity relationships across customers, accounts, devices, IPs, and merchants
- Shared-device and shared-IP account counts
- Customer device/IP degrees and merchant-customer connectivity
- Composite network entity-degree signal
- Log-damped network risk score for investigator prioritization
- Database-free graph feature layer with a clear path to graph engines for larger-scale traversal and community detection

See [`docs/graph-intelligence.md`](docs/graph-intelligence.md) for the graph model, interpretation guidance, and scaling path.

## Phase 8: Probability Calibration & Cost-Sensitive Decisioning

- Reliability diagnostics using Brier score and observed-versus-predicted fraud rates
- Probability calibration support using sigmoid or isotonic calibration
- Reliability-bin summaries for model monitoring
- Expected-loss decisioning for approve, review, and hold actions
- Explicit separation between calibrated model probability and business action policy
- Configurable false-positive, false-negative, and review costs

See [`docs/calibration-and-cost-sensitive-decisioning.md`](docs/calibration-and-cost-sensitive-decisioning.md) for the decision framework and portfolio assumptions.

## Phase 9: Model Monitoring & Drift Detection

- Population Stability Index (PSI) for numeric feature-distribution drift
- Configurable drift thresholds and feature-level monitoring reports
- Observed fraud-rate shift monitoring across scoring periods
- Reference-versus-current sample tracking
- Monitoring outputs designed for scheduled jobs, dashboards, and alerting systems
- Clear production extension path for missingness, categorical drift, delayed labels, and observability integration

See [`docs/model-monitoring.md`](docs/model-monitoring.md) for monitoring design, interpretation guidance, and production extension points.

## Phase 10: Model Run Registry & Experiment Tracking

- Deterministic model-run identifiers derived from experiment metadata
- Versioned JSON records for model parameters and measured metrics
- Feature-count and artifact-path tracking
- Dependency-light local registry contract with a clear MLflow/managed-registry migration path
- Reproducible experiment metadata suitable for model promotion workflows

See [`docs/model-registry.md`](docs/model-registry.md) for the registry contract and production extension path.

## Phase 11: Investigation Cases & Evidence Aggregation

- Structured investigation cases tied to transaction IDs and risk decisions
- Aggregation of supervised, anomaly, network, and velocity signals
- Capture of observed transaction attributes as contextual evidence
- Serializable evidence payloads for analyst APIs and future GenAI workflows
- Grounding boundary that prevents downstream narrative generation from inventing unsupported facts

See [`docs/investigation-cases.md`](docs/investigation-cases.md) for the case schema and GenAI integration path.

## Phase 12: Grounded GenAI Investigation Copilot

- Lightweight retrieval over financial policy and fraud-typology reference documents
- Investigation-context assembly from structured case evidence
- Grounded prompt construction for downstream LLM adapters
- Provider-neutral adapter contract for approved LLM services
- Deterministic evidence-only fallback when no external LLM is configured
- Explicit evidence-only boundary that requires the model to distinguish observation from interpretation
- Production extension path to embedding/vector retrieval and enterprise LLM gateways

See [`docs/investigation-copilot.md`](docs/investigation-copilot.md) and [`docs/grounded-copilot-adapter.md`](docs/grounded-copilot-adapter.md) for the retrieval, grounding, and provider-integration design.

## Phase 13: FastAPI Risk & Investigation Service

- REST API for transaction risk scoring
- Investigation-case creation endpoint
- Grounded GenAI copilot prompt endpoint
- Health endpoint for service checks
- Pydantic request validation with bounded risk-score inputs
- Automated API tests using FastAPI TestClient
- Provider-neutral service layer without hard-coded LLM credentials

Endpoints:

```text
GET  /health
POST /v1/risk/score
POST /v1/investigations/cases
POST /v1/copilot/prompt
```

## Phase 14: Containerization & CI Validation

- Production-oriented Python 3.11 slim container for the FastAPI service
- Uvicorn ASGI server configuration
- Docker build context exclusions for tests, caches, datasets, and local artifacts
- GitHub Actions workflow for automated Ruff linting and pytest execution
- Dependency pin ranges that keep local and CI environments aligned
- Container-ready service boundary suitable for future AWS/ECS, Kubernetes, or managed container deployment

The container exposes port `8000` and starts the API with Uvicorn:

```text
uvicorn financial_risk.api.app:app --host 0.0.0.0 --port 8000
```

## Planned ML Capabilities

- Random Forest comparison
- LightGBM comparison
- Advanced graph/community detection

## Planned GenAI Capabilities

- Embedding-based RAG over financial investigation policies and fraud typologies
- Evidence-grounded investigation summaries
- Transaction explanation assistant
- Controlled investigation tools / agent workflow

## Planned Engineering

- Python, SQL, PySpark
- Kafka / batch ingestion
- AWS S3 / EC2 / ECS
- MLflow
- FastAPI
- Docker
- Airflow
- GitHub Actions
- Terraform

## Scale Targets

The platform will be benchmarked at progressively larger synthetic workloads, including 100K, 1M, 10M, and 50M transactions. Performance figures will be added only after they are measured and verified.

## Repository Status

**Current phase:** Phase 14 — containerization and CI validation.
