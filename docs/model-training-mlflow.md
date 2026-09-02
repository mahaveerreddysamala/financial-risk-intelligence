# Model Training & MLflow Lifecycle

Phase 17 connects the existing fraud-model training pipeline to MLflow so a fitted XGBoost pipeline can be tracked as a reproducible experiment and, when configured, registered for downstream promotion.

## Lifecycle

```text
Temporal train / validation / test data
              |
              v
     Fit XGBoost pipeline
              |
              +--> validation metrics
              |
              +--> test metrics
              |
              v
      MLflow experiment run
              |
        +-----+-----+
        |           |
     params       metrics
        |           |
        +-----+-----+
              v
       logged model artifact
              |
              v
       registered model
              |
              v
      optional alias: champion
```

## Training contract

`financial_risk.mlops.training.train_and_log_xgboost()` accepts already-created temporal train, validation, and test frames. It uses the project XGBoost feature contract, derives `scale_pos_weight` from the training labels only, fits the pipeline, evaluates validation and test performance at the supplied threshold, and sends parameters, metrics, tags, and the fitted scikit-learn-compatible pipeline to the MLflow adapter.

The default operating threshold remains `0.85`, matching the validated project benchmark. It is an operating-policy input, not a claim of a universally optimal fraud threshold.

## Reproducibility metadata

Each tracked run records model configuration, class-imbalance weighting, data-split row counts, feature count, threshold, validation metrics, test metrics, and synthetic-data context tags. This makes the run auditable without treating synthetic benchmark metrics as production outcomes.

## MLflow configuration

MLflow is intentionally an optional dependency. Install it with:

```text
pip install -r requirements-mlops.txt
```

A local file-backed tracking store can be supplied through the `tracking_uri` argument, for example:

```text
file:./mlruns
```

For a managed deployment, point `tracking_uri` at the organization-approved MLflow tracking backend. The same adapter supports registering the logged model under a registered-model name and assigning an alias such as `champion`.

MLflow's current Model Registry supports model versions and mutable aliases; aliases can be referenced by model URIs such as `models:/<model-name>@champion`. citeturn791407search0turn791407search2

## Production boundary

This phase does **not** claim a hosted MLflow service, cloud artifact store, or production model promotion. Those require an actual configured backend and verified deployment workflow. The repository only claims the executable integration and local/testable contract.

## Next extension

A later deployment phase can add signed model metadata, approval gates, champion/challenger comparison, automated promotion checks, and managed MLflow infrastructure through AWS or another approved platform.
