from __future__ import annotations

import ast
from pathlib import Path


DAG_PATH = Path(__file__).parents[1] / "dags" / "financial_risk_pipeline.py"


def _source() -> str:
    return DAG_PATH.read_text(encoding="utf-8")


def test_airflow_dag_is_valid_python() -> None:
    ast.parse(_source(), filename=str(DAG_PATH))


def test_airflow_dag_declares_expected_pipeline_stages() -> None:
    source = _source()
    for task_name in ("generate_data", "validate_data", "benchmark_model", "enforce_quality_gate"):
        assert f"def {task_name}(" in source
    assert 'dag_id="financial_risk_training_pipeline"' in source
    assert 'schedule="@weekly"' in source
    assert "benchmark_model(validated)" in source
    assert "enforce_quality_gate(benchmark)" in source


def test_dag_keeps_business_logic_in_package() -> None:
    source = _source()
    assert "generate_transactions" in source
    assert "assert_valid_transactions" in source
    assert "run_benchmark" in source
    assert "evaluate_quality_gates" in source
