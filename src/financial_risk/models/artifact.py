"""Persisted-model loading and inference contract."""
from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any

import joblib
import pandas as pd

from financial_risk.models.baseline import CATEGORICAL_FEATURES, NUMERIC_FEATURES

FEATURE_COLUMNS = NUMERIC_FEATURES + CATEGORICAL_FEATURES


@dataclass(frozen=True)
class PersistedModelMetadata:
    """Metadata stored alongside a trained serving artifact."""

    model_name: str
    model_version: str
    feature_contract_version: str


@dataclass(frozen=True)
class ModelPrediction:
    """Prediction returned by the persisted model boundary."""

    fraud_probability: float
    model_name: str
    model_version: str
    feature_contract_version: str


class PersistedModelService:
    """Load a serialized sklearn-compatible model and expose safe inference."""

    def __init__(self, artifact_path: str | Path) -> None:
        self.artifact_path = Path(artifact_path)
        self._model: Any | None = None
        self._metadata: PersistedModelMetadata | None = None

    def _load(self) -> None:
        if self._model is not None and self._metadata is not None:
            return
        if not self.artifact_path.is_file():
            raise FileNotFoundError(f"Model artifact not found: {self.artifact_path}")
        artifact = joblib.load(self.artifact_path)
        if not isinstance(artifact, dict) or "model" not in artifact or "metadata" not in artifact:
            raise ValueError("model artifact must contain model and metadata entries")
        metadata = artifact["metadata"]
        if not isinstance(metadata, dict):
            raise ValueError("model artifact metadata must be a dictionary")
        required_metadata = {"model_name", "model_version", "feature_contract_version"}
        if not required_metadata.issubset(metadata):
            raise ValueError("model artifact metadata is missing required fields")
        self._model = artifact["model"]
        self._metadata = PersistedModelMetadata(
            model_name=str(metadata["model_name"]),
            model_version=str(metadata["model_version"]),
            feature_contract_version=str(metadata["feature_contract_version"]),
        )

    def predict(self, features: dict[str, Any]) -> ModelPrediction:
        """Score one feature row with the persisted model artifact."""
        self._load()
        missing = [column for column in FEATURE_COLUMNS if column not in features]
        if missing:
            raise ValueError(f"Model input is missing required features: {sorted(missing)}")
        frame = pd.DataFrame([{column: features[column] for column in FEATURE_COLUMNS}])
        probabilities = self._model.predict_proba(frame)[:, 1]
        metadata = self._metadata
        if metadata is None:  # pragma: no cover - guarded by _load
            raise RuntimeError("Model metadata was not loaded")
        return ModelPrediction(
            fraud_probability=float(probabilities[0]),
            model_name=metadata.model_name,
            model_version=metadata.model_version,
            feature_contract_version=metadata.feature_contract_version,
        )
