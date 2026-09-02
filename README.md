# Financial Crime & Risk Intelligence Platform

A production-oriented financial AI/ML platform for transaction fraud detection, anomaly detection, graph-based risk intelligence, explainable AI, and GenAI-assisted investigations.

## Project Vision

This project is being built as a senior-level Data Scientist / AI-ML portfolio system rather than a single fraud-classification notebook. The platform combines behavioral machine learning, unsupervised anomaly detection, network risk signals, cost-sensitive decisioning, explainability, MLOps, and a grounded investigation copilot.

## Architecture

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
- Explicitly avoids claims of a live broker, exactly-once semantics, or production model serving until those infrastructure components are deployed and verified

See [`docs/real-time-risk-scoring.md`](docs/real-time-risk-scoring.md) and [`docs/streaming-worker.md`](docs/streaming-worker.md) for the streaming inference and worker contracts.

## Verified 20K Fraud Benchmark

| Model | ROC-AUC | PR-AUC | Precision | Recall | F1 |
|---|---:|---:|---:|---:|---:|
| Logistic Regression | 0.6131 | 0.0573 | 2.45% | 51.28% | 0.0468 |
| XGBoost | **0.6667** | **0.0802** | **4.44%** | 10.26% | **0.0620** |

XGBoost achieved a **12.87x lift** at the validation-selected `0.85` operating threshold on the synthetic test workload. These values are reproducible portfolio benchmarks, not production fraud-performance claims.

## Planned ML / Engineering Extensions

- Random Forest comparison
- LightGBM comparison
- Advanced graph/community detection
- Embedding-based RAG over financial investigation policies and fraud typologies
- Controlled investigation tools / agent workflow
- Managed Kafka, object storage, managed MLflow, and cloud deployment integrations

## Scale Targets

The platform will be benchmarked at progressively larger synthetic workloads, including 100K, 1M, 10M, and 50M transactions. Performance figures will be added only after they are measured and verified.

## Repository Status

**Current phase:** Phase 22 — real-time risk scoring consumer.
