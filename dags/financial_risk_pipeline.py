"""Airflow 3 DAG for the financial-risk training quality pipeline.

This DAG is intentionally thin: business logic remains in the tested Python
package and CLI scripts. Airflow is responsible for scheduling, retries, and
stage ordering only.
"""

from __future__ import annotations

from airflow.sdk import dag, task


@dag(
    dag_id="financial_risk_training_pipeline",
    schedule="@weekly",
    start_date=None,
    catchup=False,
    tags=["financial-risk", "ml", "quality-gate"],
    description="Generate, validate, benchmark, and quality-gate the fraud model.",
)
def financial_risk_training_pipeline():
    """Define the end-to-end scheduled fraud-model validation workflow."""

    @task
    def generate_data() -> str:
        from financial_risk.data_generation.generator import generate_transactions, write_dataset

        output = "data/raw/airflow_transactions.parquet"
        df = generate_transactions(rows=20_000, seed=42)
        write_dataset(df, output)
        return output

    @task
    def validate_data(input_path: str) -> str:
        import pandas as pd

        from financial_risk.validation.contracts import assert_valid_transactions

        df = pd.read_parquet(input_path)
        assert_valid_transactions(df)
        return input_path

    @task
    def benchmark_model() -> str:
        from financial_risk.models.benchmark import run_benchmark

        benchmark, _, _ = run_benchmark(rows=20_000, seed=42)
        output = "artifacts/airflow/model-benchmark.csv"
        benchmark.to_csv(output, index=False)
        return output

    @task
    def enforce_quality_gate(benchmark_path: str) -> str:
        from financial_risk.mlops.ci_quality import evaluate_quality_gates

        import pandas as pd

        benchmark = pd.read_csv(benchmark_path)
        xgb = benchmark.loc[benchmark["model"].eq("XGBoost")].iloc[0]
        result = evaluate_quality_gates(
            {
                "test_pr_auc": float(xgb["pr_auc"]),
                "test_recall": float(xgb["recall"]),
                "test_precision": float(xgb["precision"]),
            }
        )
        if not result.passed:
            raise ValueError("Model quality gate failed: " + "; ".join(result.failures))
        return "QUALITY GATE: PASS"

    raw = generate_data()
    validated = validate_data(raw)
    benchmark = benchmark_model()
    validated >> benchmark
    enforce_quality_gate(benchmark)


financial_risk_training_pipeline()
