# Financial Crime & Risk Intelligence Platform

A production-oriented financial AI/ML platform for transaction fraud detection, anomaly detection, graph-based risk intelligence, explainable AI, and GenAI-assisted investigations.

## Project Vision

This project is being built as a senior-level Data Scientist / AI-ML portfolio system rather than a single fraud-classification notebook. The platform will combine behavioral machine learning, unsupervised anomaly detection, network risk signals, cost-sensitive decisioning, explainability, MLOps, and a grounded investigation copilot.

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
Feature Engineering
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

## Planned ML Capabilities

- Logistic Regression baseline
- Random Forest baseline
- XGBoost / LightGBM fraud modeling
- Imbalanced-learning evaluation
- Temporal and leakage-aware validation
- Isolation Forest anomaly detection
- Behavioral and velocity features
- Geographic and network risk features
- Cost-sensitive threshold optimization
- SHAP explainability
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

**Current phase:** Phase 1 — platform foundation and deterministic financial transaction generation.
