# Phase 41: Challenger Operating-Point Analysis

Phase 40 established Random Forest and LightGBM challengers against the existing XGBoost model. Phase 41 evaluates all three models under the same decision framework rather than comparing only their default 0.5 threshold metrics.

## Method

1. Generate the deterministic synthetic transaction workload and build the existing leakage-aware feature table.
2. Preserve the existing chronological train / validation / test split.
3. Fit XGBoost, Random Forest, and LightGBM using the same feature contract and imbalance strategy.
4. Select the F1-maximizing threshold using **validation data only**.
5. Evaluate the selected threshold on the untouched test set.
6. Measure investigator-oriented Top-K performance at K=500, including precision, recall, and lift.
7. Measure probability quality with the existing Brier-score calibration diagnostic.
8. Apply the existing cost policy (`FP=5`, `FN=100`, `review=3`) to compute realized decision cost.

## Interpretation boundary

The output is a portfolio benchmark for comparing modeling and decisioning strategies. Thresholds, costs, and resulting operating points are illustrative business-policy values and are not regulatory requirements or production fraud-performance claims.

## Reproducibility

```powershell
python scripts/run_operating_point_analysis.py --rows 20000 --seed 42
```

Results are written to `artifacts/challenger-operating-points.csv`.
