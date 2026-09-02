"""Environment-driven API configuration."""
from __future__ import annotations

import os
from dataclasses import dataclass


@dataclass(frozen=True)
class Settings:
    """Runtime settings resolved from environment variables."""

    app_env: str = "development"
    log_level: str = "INFO"
    model_artifact_path: str = "artifacts"
    app_version: str = "0.1.0"

    @classmethod
    def from_env(cls) -> "Settings":
        """Build settings from environment variables with safe defaults."""
        return cls(
            app_env=os.getenv("APP_ENV", "development"),
            log_level=os.getenv("LOG_LEVEL", "INFO").upper(),
            model_artifact_path=os.getenv("MODEL_ARTIFACT_PATH", "artifacts"),
            app_version=os.getenv("APP_VERSION", "0.1.0"),
        )


settings = Settings.from_env()
