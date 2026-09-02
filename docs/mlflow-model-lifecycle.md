# MLflow Model Lifecycle

Phase 16 adds an optional MLflow integration on top of the project's existing lightweight JSON model-run registry.

## Lifecycle

```text
Model Training
      |
      v
Experiment Run
  - parameters
  - metrics
  - tags
      |
      v
MLflow Model Artifact
      |
      v
Registered Model
      |
      v
Version + Alias
  - candidate
  - champion
```

## Why the adapter is optional

The core repository remains usable without a running MLflow server. The MLflow dependency is isolated in `requirements-mlops.txt`, while `financial_risk.mlops` imports MLflow lazily. This keeps the base CI workflow lightweight and allows the portfolio project to demonstrate a real tracking integration without claiming a managed MLflow deployment.

## Local setup

From the repository root:

```powershell
pip install -r requirements-mlops.txt
```

A local file-based tracking URI can then be used for experiments:

```python
from financial_risk.mlops import log_sklearn_run

result = log_sklearn_run(
    model,
    model_name="fraud-xgboost",
    experiment_name="financial-fraud",
    parameters={"max_depth": 6, "learning_rate": 0.05},
    metrics={"pr_auc": 0.0802, "roc_auc": 0.6667},
    tracking_uri="file:./mlruns",
    registered_model_name="financial-fraud-xgboost",
)
```

The adapter returns the MLflow run ID, experiment ID, model URI, and registered version when the registry backend returns a matching model version.

## Registration and aliases

An existing MLflow model URI can be registered and assigned an alias:

```python
from financial_risk.mlops import register_model_version

version = register_model_version(
    "runs:/<run_id>/model",
    registered_model_name="financial-fraud-xgboost",
    alias="champion",
)
```

Aliases are preferred to hard-coding a stage name in application code because the serving layer can resolve the model version by a stable alias such as `champion`.

## Production migration path

The same adapter can point at a managed MLflow tracking backend through configuration. Production deployment should additionally provide an authenticated artifact store, access controls, retention policies, model approval controls, and monitoring. None of those managed-service integrations are claimed by this phase.

## Testing boundary

Unit tests use a small fake MLflow client so CI does not need a live tracking server. Real MLflow integration should be exercised separately against a local or managed backend before production use.
