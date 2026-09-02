"""Lightweight model artifact and experiment tracking for the portfolio project."""
from __future__ import annotations

import hashlib
import json
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any


@dataclass(frozen=True)
class ModelRun:
    run_id: str
    model_name: str
    parameters: dict[str, Any]
    metrics: dict[str, float]
    feature_count: int
    artifact_path: str


def build_run_id(model_name: str, parameters: dict[str, Any], metrics: dict[str, float]) -> str:
    """Create a deterministic run identifier from experiment metadata."""
    payload = json.dumps(
        {"model_name": model_name, "parameters": parameters, "metrics": metrics},
        sort_keys=True,
        default=str,
    ).encode("utf-8")
    return hashlib.sha256(payload).hexdigest()[:12]


def save_model_run(
    output_dir: str | Path,
    model_name: str,
    parameters: dict[str, Any],
    metrics: dict[str, float],
    feature_count: int,
    artifact_path: str,
) -> Path:
    """Persist experiment metadata as a versioned JSON model-run record."""
    if feature_count < 1:
        raise ValueError("feature_count must be greater than zero")
    if not model_name.strip():
        raise ValueError("model_name must not be empty")
    if not artifact_path.strip():
        raise ValueError("artifact_path must not be empty")

    run_id = build_run_id(model_name, parameters, metrics)
    run = ModelRun(
        run_id=run_id,
        model_name=model_name,
        parameters=parameters,
        metrics={key: float(value) for key, value in metrics.items()},
        feature_count=feature_count,
        artifact_path=artifact_path,
    )
    destination = Path(output_dir)
    destination.mkdir(parents=True, exist_ok=True)
    path = destination / f"{run_id}.json"
    path.write_text(json.dumps(asdict(run), indent=2, sort_keys=True), encoding="utf-8")
    return path


def load_model_run(path: str | Path) -> ModelRun:
    """Load and validate one persisted model-run record."""
    payload = json.loads(Path(path).read_text(encoding="utf-8"))
    required = {"run_id", "model_name", "parameters", "metrics", "feature_count", "artifact_path"}
    missing = required.difference(payload)
    if missing:
        raise ValueError(f"Missing model-run fields: {sorted(missing)}")
    return ModelRun(
        run_id=str(payload["run_id"]),
        model_name=str(payload["model_name"]),
        parameters=dict(payload["parameters"]),
        metrics={key: float(value) for key, value in payload["metrics"].items()},
        feature_count=int(payload["feature_count"]),
        artifact_path=str(payload["artifact_path"]),
    )
