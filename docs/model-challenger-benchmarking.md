# Phase 40: Model Challenger Benchmarking

Phase 40 adds Random Forest and LightGBM as challenger models to the established XGBoost fraud benchmark.

## Benchmark contract

All three tree-based models use the same leakage-aware feature table, the same chronological train/validation/test split, and the same imbalance ratio derived from the training partition. Precision, recall, F1, ROC-AUC, and PR-AUC are evaluated on the held-out test partition.

The benchmark is intended to answer a model-selection question rather than maximize a single metric. XGBoost remains the current production-serving model; Random Forest and LightGBM are evaluated as challengers before any model promotion decision.

## Run

```powershell
conda activate portfolio311
python scripts/run_challenger_benchmark.py --rows 20000 --seed 42
```

Results are written to `artifacts/challenger-benchmark.csv`.

## Interpretation

PR-AUC is especially useful for comparing classifiers on the project's imbalanced fraud workload. Threshold-based precision, recall, and F1 remain operating-point metrics and should be interpreted separately from ranking metrics such as ROC-AUC and PR-AUC.

A challenger should only replace the current model after reproducible benchmark evidence, threshold analysis, operational validation, artifact compatibility, and downstream streaming verification.

## Production boundary

Random Forest and LightGBM are benchmark challengers in Phase 40. The persisted streaming artifact remains the existing XGBoost model until a separate promotion workflow validates a challenger end to end.
