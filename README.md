# Financial Crime & Risk Intelligence Platform

A real-time financial risk decisioning platform for detecting suspicious transactions, enriching risk with behavioral and network signals, and supporting investigator workflows.

[![CI](https://github.com/mahaveerreddysamala/financial-risk-intelligence/actions/workflows/ci.yml/badge.svg)](https://github.com/mahaveerreddysamala/financial-risk-intelligence/actions)

## Overview

Financial institutions need to evaluate transactions using more than a single fraud score. A production risk platform must combine transaction behavior, anomaly signals, entity relationships, velocity, model outputs, and operational context while remaining explainable, observable, and safe to operate at scale.

This repository implements that workflow as a modular Python service and streaming system:

```text
Transaction
    │
    ▼
Ingestion / Data Contracts
    │
    ▼
Leakage-Safe Feature Generation
    │
    ├──────────────┬───────────────┬──────────────┐
    ▼              ▼               ▼              ▼
Fraud Model    Anomaly Model   Graph Risk    Velocity
    │              │               │              │
    └──────────────┴───────┬───────┴──────────────┘
                           ▼
                    Risk Decision Engine
                           │
                ┌──────────┴──────────┐
                ▼                     ▼
          Risk-Scored Event     Investigation Case
                                      │
                              Evidence / Reason Codes
                                      │
                                      ▼
                         Grounded Retrieval Context
                                      │
                                      ▼
                                Analyst Workflow
```

The system is designed around explicit interfaces between data processing, model inference, graph enrichment, decisioning, investigation, and operations.

## Interactive Portfolio Dashboard

The Streamlit dashboard trains an XGBoost model on historical synthetic transactions, scores
a strictly later window, combines fraud, anomaly, network, community, and velocity signals,
and exposes a prioritized decision queue with evidence-grounded investigation cases.

```bash
python -m pip install -r requirements.txt -r requirements-dashboard.txt
streamlit run dashboards/financial_risk_app.py
```

The dashboard includes executive metrics, decision distribution, daily risk, signal-level
scores, operational actions, case evidence, held-out model metrics, and optional walk-forward
backtest results. GitHub Actions executes the complete dashboard in headless mode.

## System Architecture

### Real-time processing

```text
Kafka
  │
  ├── transaction.created
  │
  ▼
Consumer Group
  │
  ├── partitioning by customer_id
  ├── retry / DLQ handling
  └── duplicate-event protection
  │
  ▼
Durable Feature State
  │
  ├── prior transaction history
  ├── velocity state
  ├── entity relationships
  └── community state
  │
  ▼
Risk Enrichment
  │
  ├── XGBoost fraud probability
  ├── anomaly signal
  ├── network risk
  └── velocity signal
  │
  ▼
Risk Decision Engine
  │
  ▼
transaction.risk_scored
```

The streaming design preserves prior-only feature semantics so the transaction being scored cannot influence its own historical features.

### Risk decisioning

The decision layer combines independently generated signals into an operational risk score and risk band. Graph intelligence is integrated into the final decision rather than exposed only as an analytical side output.

The platform supports:

- supervised fraud classification
- anomaly detection
- behavioral and velocity signals
- network and entity-reuse risk
- community-level risk
- configurable signal weighting
- risk bands and actions
- model explainability and reason codes

## Model Lifecycle

Models are treated as deployable artifacts with measurable promotion criteria.

```text
Training
   │
   ▼
Validation
   │
   ▼
Quality Gates
   │
   ▼
Champion / Challenger Evaluation
   │
   ▼
Model Registry
   │
   ▼
Promotion
   │
   ▼
Persisted Serving Artifact
   │
   ▼
Runtime Monitoring
   ├── drift
   ├── performance
   └── calibration
```

The repository includes model comparison, quality gates, registry metadata, automated promotion controls, persisted artifacts, and monitoring utilities.

### Verified model benchmark

The CI benchmark uses 20,000 deterministic synthetic transactions and a chronological
train/validation/test split. The current held-out test result is intentionally reported
against the class prevalence so the result is interpretable for an imbalanced problem.

| Metric | Current CI result |
|---|---:|
| Test fraud prevalence | 1.20% |
| XGBoost ROC-AUC | 0.6850 |
| XGBoost PR-AUC | 0.0832 |
| Top-250 precision | 4.40% |
| Top-250 recall | 28.21% |
| Top-250 lift | 3.68x |

These figures describe a synthetic portfolio benchmark, not production fraud performance.
They provide a reproducible baseline for future temporal backtests, ablation studies, and
cost-based operating-point comparisons.

Reproduce the complete model benchmark and create CSV evidence plus a Markdown portfolio
report with one command:

```bash
python -m financial_risk.models.benchmark
```

Outputs are written under `artifacts/`, including `model-benchmark-report.md`. Pull-request
CI runs the same command, enforces the quality gate, and publishes the report as the
`financial-risk-model-benchmark` workflow artifact.

### Temporal stability backtesting

A walk-forward backtest retrains XGBoost on expanding historical windows and evaluates three
strictly later two-month periods. It reports fold-level ROC-AUC, PR-AUC, precision, recall,
and investigation-capacity lift so temporal instability cannot be hidden by one aggregate
test split.

```bash
python -m financial_risk.models.backtesting
```

The command creates `artifacts/temporal-backtest.csv` and a Markdown stability report. CI
executes a 12,000-row profile and includes both files in the model benchmark artifact.

## Graph Intelligence

Financial crime frequently involves relationships that are difficult to detect from an individual transaction. The graph layer models relationships across entities such as customers, accounts, devices, IP addresses, and merchants.

Implemented capabilities include:

- heterogeneous entity relationships
- weighted network features
- shared-entity reuse detection
- deterministic connected-component grouping
- stable online component identifiers
- incremental component updates
- streaming graph enrichment
- graph risk integration into the final decision

## Investigation Workflow

High-risk decisions can be converted into persistent investigation cases.

```text
Risk Decision
     │
     ▼
Case Creation
     │
     ├── transaction evidence
     ├── model signals
     ├── graph context
     ├── reason codes
     └── retrieved references
     │
     ▼
Investigator Review
     │
     ├── OPEN
     ├── IN_REVIEW
     ├── ESCALATED
     ├── RESOLVED
     └── DISMISSED
     │
     ▼
Audit Trail
```

Case creation supports idempotency. Status transitions are validated, actor information is recorded for workflow actions, and evidence/reference provenance is retained for downstream analysis.

## Investigation Copilot

The investigation copilot provides retrieval and grounded prompt construction around case evidence.

The current implementation uses TF-IDF retrieval and produces an evidence-constrained prompt
and analyst brief. It does not call a hosted language model or autonomously adjudicate cases.

The workflow separates:

1. case evidence
2. retrieved reference material
3. analyst-facing summaries
4. model-generated interpretation
5. human investigation decisions

The copilot is constrained to supplied evidence and references. It is not an autonomous case adjudication system and is designed to surface uncertainty when evidence is insufficient.

## API

The service exposes a FastAPI interface for operational and investigation workflows.

| Endpoint | Purpose |
|---|---|
| `GET /health` | Liveness check |
| `GET /ready` | Readiness check |
| `GET /version` | Service version |
| `GET /metrics` | Prometheus metrics |
| `POST /v1/model/score` | Model scoring |
| `POST /v1/investigations/cases` | Create investigation case |
| `GET /v1/investigations/cases` | List/filter cases |
| `GET /v1/investigations/cases/{case_id}` | Retrieve case |
| `PATCH /v1/investigations/cases/{case_id}/status` | Update case state |
| `GET /v1/investigations/cases/{case_id}/audit` | Retrieve audit trail |
| `POST /v1/copilot/prompt` | Build grounded investigation prompt |
| `POST /v1/copilot/cases/{case_id}/brief` | Generate analyst brief context |

## Reliability & Operations

The runtime is designed around common distributed-system failure modes:

- Kafka consumer groups for horizontal processing
- customer-keyed partitioning for ordering
- durable feature state
- atomic idempotency claims and leases
- retry and dead-letter paths
- versioned event envelopes
- structured operational logging
- health and readiness endpoints
- Prometheus-compatible metrics
- Docker-based local deployment
- automated CI validation

## Observability

Operational telemetry covers service health and streaming/inference behavior. Prometheus-compatible metrics and Grafana dashboards provide a local operational view for development and validation.

The repository also includes model monitoring for:

- feature/data drift
- model performance
- calibration
- prediction behavior

## Scale Validation

A reproducible benchmark runner supports controlled synthetic transaction workloads.

A measured local run generated **1,000,000 records in 1.732874 seconds (577,075.9 rows/second)** using 8 partitions.

```powershell
python scripts/benchmark_scale.py --rows 1000000 --partitions 8 --output-dir artifacts/benchmarks
```

This benchmark measures the repository's bounded-memory synthetic transaction generation/output path. It is not an end-to-end throughput claim for fraud inference, graph enrichment, Kafka processing, or cloud infrastructure.

Detailed methodology and results are documented in `docs/PHASE_49_SCALE_BENCHMARK.md` and `docs/PHASE_49_BENCHMARK_RESULTS.md`.

## Cloud Deployment Model

The repository includes a Terraform AWS foundation and a production reference architecture.

```text
                         Clients
                            │
                            ▼
                      ALB / API Gateway
                            │
                  ┌─────────┴─────────┐
                  ▼                   ▼
             ECS API             ECS Worker
                  │                   │
                  └─────────┬─────────┘
                            │
        ┌───────────────────┼───────────────────┐
        ▼                   ▼                   ▼
   Managed Kafka       Redis-compatible        S3
                        durable state       artifacts
                            │
                            ▼
                     Observability
```

The Terraform foundation currently provides ECR, an ECS cluster, S3 artifact storage, and CloudWatch logging. Managed Kafka/Redis, networking, ingress, certificates, secrets, IAM policies, autoscaling, and production capacity configuration are deployment-specific.

No live production AWS environment is implied by the infrastructure code.

## Technology

| Layer | Implementation |
|---|---|
| Language | Python |
| API | FastAPI / Uvicorn |
| ML | XGBoost / scikit-learn / SHAP |
| Data | Pandas / NumPy / PyArrow |
| Streaming | Apache Kafka |
| State | Redis-compatible durable state |
| Graph | NetworkX / online communities |
| AI-ready retrieval | TF-IDF retrieval + grounded prompt construction |
| Observability | Prometheus / Grafana / structured logs |
| Containers | Docker / Docker Compose |
| MLOps | Model registry / quality gates / CI/CD |
| Cloud | AWS / Terraform |
| Testing | pytest / Ruff |

## Repository Structure

```text
financial-risk-intelligence/
├── src/financial_risk/
│   ├── api/              # HTTP service and endpoints
│   ├── models/           # Risk models, decisioning, governance
│   ├── graph/            # Network and community intelligence
│   ├── streaming/        # Kafka consumers, events, state, enrichment
│   ├── investigation/    # Cases, evidence, copilot
│   └── monitoring/       # Runtime/model monitoring
├── tests/                # Automated test suite
├── dashboards/           # Interactive portfolio dashboard
├── scripts/              # Benchmark and operational utilities
├── docs/                 # Architecture and operational documentation
├── infra/terraform/aws/  # AWS infrastructure foundation
├── .github/workflows/    # CI and validation workflows
├── docker-compose.yml    # Local service stack
├── requirements.txt
└── SECURITY.md
```

## Local Development

```bash
git clone https://github.com/mahaveerreddysamala/financial-risk-intelligence.git
cd financial-risk-intelligence
python -m venv .venv
source .venv/bin/activate       # Windows: .venv\Scripts\activate
pip install -r requirements.txt
```

Run validation (CI executes the suite on Python 3.11 and 3.12):

```bash
ruff check src tests scripts
pytest
```

Run the API:

```bash
uvicorn financial_risk.api.app:app --host 0.0.0.0 --port 8000
```

Run the local service stack:

```bash
docker compose up --build
```

## Documentation

- `docs/PHASE_49_SCALE_BENCHMARK.md` — benchmark methodology
- `docs/PHASE_49_BENCHMARK_RESULTS.md` — measured scale results
- `docs/PHASE_50_CLOUD_ARCHITECTURE.md` — deployment architecture
- `docs/PHASE_50_STATUS.md` — infrastructure status
- `docs/production-readiness.md` — operational readiness
- `infra/README.md` — Terraform usage
- `SECURITY.md` — security guidance

## Security

The repository uses synthetic transaction data and does not contain production financial data or credentials.

Production deployments should use managed secrets, least-privilege IAM, private networking, TLS, encryption at rest and in transit, dependency/container scanning, centralized audit logging, and controlled model artifact access.

See `SECURITY.md` for security guidance and reporting procedures.

## Status

The current implementation covers the core risk-decisioning, streaming, connected-component graph enrichment, investigation, evidence-grounded retrieval, monitoring, and cloud-foundation components. CI validates the application and infrastructure configuration.

The system is intended to be operated as a reference implementation of a financial risk platform, with environment-specific production infrastructure configured separately.
