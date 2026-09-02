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

See [`docs/model-benchmark.md`](docs/model-benchmark.md) for the evaluation protocol. Measured model results will be added only after the training workflow is executed and verified.

## Planned ML Capabilities

- Random Forest comparison
- LightGBM comparison
- Isolation Forest anomaly detection
- Advanced behavioral and velocity features
- Geographic and network risk features
- Cost-sensitive expected-loss decisioning
- SHAP explainability and reason codes
- Model calibration and monitoring

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

**Current phase:** Phase 4 — baseline and XGBoost fraud modeling with threshold and top-K evaluation.
