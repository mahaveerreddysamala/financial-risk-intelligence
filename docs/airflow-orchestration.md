# Airflow Orchestration

## Purpose

Phase 20 adds a thin Apache Airflow 3 DAG around the existing, tested financial-risk components. Airflow owns scheduling, retries, and task dependencies; the application package remains the source of truth for generation, validation, benchmarking, and quality-gate logic.

## DAG

`dags/financial_risk_pipeline.py`

DAG ID: `financial_risk_training_pipeline`

Schedule: weekly (`@weekly`)

Task flow:

```text
Generate synthetic transactions
        |
        v
Validate transaction contract
        |
        v
Run deterministic 20K model benchmark
        |
        v
Enforce model quality gate
```

The benchmark uses the same deterministic 20K workload and seed already used by CI. The final task fails the DAG run when the configured promotion-quality thresholds are not met.

## Design boundary

The DAG intentionally does not duplicate model or validation logic. It imports:

- `financial_risk.data_generation.generator`
- `financial_risk.validation.contracts`
- `financial_risk.models.benchmark`
- `financial_risk.mlops.ci_quality`

This keeps orchestration separate from business logic and makes each component independently testable.

## Optional Airflow environment

The base project and CI workflow remain lightweight. Airflow is isolated in `requirements-airflow.txt` and is not required for the normal unit-test suite.

This project currently pins Apache Airflow 3.3.1 for the optional orchestration environment. Airflow 3 uses the stable `airflow.sdk` authoring namespace for DAG constructs. The current Airflow release line and support status should be reviewed before a production deployment.

For local development, a Linux container or other supported Airflow environment is recommended rather than treating the Windows Conda environment as the Airflow runtime.

## Production extension

A production deployment can replace the synthetic generator task with object-storage or streaming ingestion, add dataset partitioning, route model artifacts to managed storage, and connect the final quality gate to an MLflow registry promotion step. Those integrations are intentionally not claimed by this phase until implemented and verified.
