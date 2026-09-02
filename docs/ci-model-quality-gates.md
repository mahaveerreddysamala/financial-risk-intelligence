# CI Model Quality Gates

Phase 19 makes model-quality validation executable in GitHub Actions.

## Workflow

The CI workflow first installs the base project dependencies, runs Ruff and pytest, then executes the deterministic 20K synthetic fraud benchmark. The XGBoost row from that benchmark is evaluated against portfolio operating thresholds.

```text
Ruff + pytest
      |
      v
20K deterministic benchmark
      |
      v
XGBoost quality metrics
      |
      +--> PR-AUC >= 0.05
      +--> Recall >= 0.05
      +--> Precision >= 0.02
      |
      v
PASS / FAIL
```

A failed gate exits with a non-zero status and fails CI. A passing gate means the benchmark satisfies these portfolio-defined minimums; it does not represent a regulatory or production approval decision.

## Why synthetic benchmarks are used

The repository uses a deterministic synthetic financial dataset so CI can exercise the complete quality-gate path without external data, credentials, or services. These checks validate engineering behavior and promotion logic. They are not evidence of real-world fraud-detection performance.

## Local execution

```text
python -m financial_risk.models.benchmark --rows 20000 --seed 42
python scripts/check_model_quality.py --benchmark artifacts/model-benchmark.csv
```
