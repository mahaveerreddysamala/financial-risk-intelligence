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
                         GenAI Investigation
                              Copilot + RAG
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

## Planned ML Capabilities

- Random Forest comparison
- LightGBM comparison
- Cost-sensitive expected-loss decisioning
- Model calibration and monitoring
- Graph/network intelligence expansion

## Planned GenAI Capabilities

- RAG over financial investigation policies and fraud typologies
- Evidence-grounded investigation summaries
- Transaction explanation assistant
- Controlled investigation tools / agent workflow

## Planned Engineering

- Python, SQL, PySpark
- Kafka / batch ingestion
- AWS S3 / EC2
- MLflow
- FastAPI
- Docker
- Airflow
- GitHub Actions
- Terraform

## Scale Targets

The platform will be benchmarked at progressively larger synthetic workloads, including 100K, 1M, 10M, and 50M transactions. Performance figures will be added only after they are measured and verified.

## Repository Status

**Current phase:** Phase 6 — SHAP explainability and human-readable fraud reason codes.
