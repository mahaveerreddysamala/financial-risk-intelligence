"""Provider-neutral MLflow tracking and model-registration helpers."""
from __future__ import annotations

from dataclasses import dataclass
from typing import Any


@dataclass(frozen=True)
class MLflowRunResult:
    """Identifiers returned from a tracked model run."""

    run_id: str
    experiment_id: str
    model_uri: str
    registered_model_name: str | None = None
    registered_model_version: str | None = None


def _require_mlflow() -> Any:
    """Import MLflow lazily so core package imports do not require the optional service."""
    try:
        import mlflow
    except ImportError as exc:  # pragma: no cover - exercised through the public error path
        raise RuntimeError(
            "MLflow is not installed. Install the project requirements before using MLOps tracking."
        ) from exc
    return mlflow


def log_sklearn_run(
    model: Any,
    *,
    model_name: str,
    experiment_name: str,
    parameters: dict[str, Any],
    metrics: dict[str, float],
    tags: dict[str, str] | None = None,
    tracking_uri: str | None = None,
    registered_model_name: str | None = None,
    artifact_name: str = "model",
) -> MLflowRunResult:
    """Log a scikit-learn-compatible model, metadata, and metrics to MLflow.

    The helper does not assume a hosted MLflow server. Pass a tracking URI for a
    local file store or managed tracking backend through configuration.
    """
    if not model_name.strip():
        raise ValueError("model_name must not be empty")
    if not experiment_name.strip():
        raise ValueError("experiment_name must not be empty")
    if not artifact_name.strip():
        raise ValueError("artifact_name must not be empty")

    mlflow = _require_mlflow()
    if tracking_uri:
        mlflow.set_tracking_uri(tracking_uri)
    experiment = mlflow.set_experiment(experiment_name)

    with mlflow.start_run() as run:
        mlflow.set_tag("model_name", model_name)
        if tags:
            mlflow.set_tags(tags)
        if parameters:
            mlflow.log_params(parameters)
        if metrics:
            mlflow.log_metrics({key: float(value) for key, value in metrics.items()})

        model_info = mlflow.sklearn.log_model(
            sk_model=model,
            name=artifact_name,
            registered_model_name=registered_model_name,
        )
        model_uri = getattr(model_info, "model_uri", f"runs:/{run.info.run_id}/{artifact_name}")

        registered_version = None
        if registered_model_name:
            client = mlflow.MlflowClient()
            latest = client.search_model_versions(
                filter_string=f"name='{registered_model_name}'",
                max_results=100,
                order_by=["version_number DESC"],
            )
            run_versions = [
                item for item in latest if getattr(item, "run_id", None) == run.info.run_id
            ]
            if run_versions:
                registered_version = str(run_versions[0].version)

        return MLflowRunResult(
            run_id=run.info.run_id,
            experiment_id=str(experiment.experiment_id),
            model_uri=model_uri,
            registered_model_name=registered_model_name,
            registered_model_version=registered_version,
        )


def register_model_version(
    model_uri: str,
    *,
    registered_model_name: str,
    alias: str | None = None,
    tracking_uri: str | None = None,
) -> str:
    """Register an existing MLflow model URI and optionally assign a registry alias."""
    if not model_uri.strip():
        raise ValueError("model_uri must not be empty")
    if not registered_model_name.strip():
        raise ValueError("registered_model_name must not be empty")

    mlflow = _require_mlflow()
    if tracking_uri:
        mlflow.set_tracking_uri(tracking_uri)

    version = mlflow.register_model(model_uri, registered_model_name)
    version_number = str(version.version)
    if alias:
        client = mlflow.MlflowClient()
        client.set_registered_model_alias(registered_model_name, alias, version_number)
    return version_number
