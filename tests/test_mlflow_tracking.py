from __future__ import annotations

from types import SimpleNamespace

import pytest

from financial_risk.mlops.tracking import (
    MLflowRunResult,
    _require_mlflow,
    log_sklearn_run,
    register_model_version,
)


class _RunContext:
    def __init__(self, run_id: str) -> None:
        self.info = SimpleNamespace(run_id=run_id)

    def __enter__(self) -> _RunContext:
        return self

    def __exit__(self, *args: object) -> None:
        return None


class _FakeClient:
    def __init__(self) -> None:
        self.aliases: list[tuple[str, str, str]] = []

    def search_model_versions(
        self, *, filter_string: str, max_results: int, order_by: list[str]
    ) -> list[SimpleNamespace]:
        assert filter_string == "name='fraud-model'"
        assert max_results == 100
        assert order_by == ["version_number DESC"]
        return [SimpleNamespace(run_id="run-123", version="7")]

    def set_registered_model_alias(self, name: str, alias: str, version: str) -> None:
        self.aliases.append((name, alias, version))


class _FakeMLflow:
    def __init__(self) -> None:
        self.tracking_uri: str | None = None
        self.tags: dict[str, str] = {}
        self.params: dict[str, object] = {}
        self.metrics: dict[str, float] = {}
        self.client = _FakeClient()
        self.sklearn = SimpleNamespace(
            log_model=lambda **kwargs: SimpleNamespace(model_uri="runs:/run-123/model")
        )
        self.registered_calls: list[tuple[str, str]] = []

    def set_tracking_uri(self, uri: str) -> None:
        self.tracking_uri = uri

    def set_experiment(self, name: str) -> SimpleNamespace:
        return SimpleNamespace(experiment_id="42", name=name)

    def start_run(self) -> _RunContext:
        return _RunContext("run-123")

    def set_tag(self, key: str, value: str) -> None:
        self.tags[key] = value

    def set_tags(self, tags: dict[str, str]) -> None:
        self.tags.update(tags)

    def log_params(self, params: dict[str, object]) -> None:
        self.params.update(params)

    def log_metrics(self, metrics: dict[str, float]) -> None:
        self.metrics.update(metrics)

    def MlflowClient(self) -> _FakeClient:
        return self.client

    def register_model(self, model_uri: str, name: str) -> SimpleNamespace:
        self.registered_calls.append((model_uri, name))
        return SimpleNamespace(version=8)


def test_require_mlflow_reports_optional_dependency(monkeypatch: pytest.MonkeyPatch) -> None:
    real_import = __import__

    def missing_mlflow(name: str, *args: object, **kwargs: object):
        if name == "mlflow":
            raise ImportError("missing")
        return real_import(name, *args, **kwargs)

    monkeypatch.setattr("builtins.__import__", missing_mlflow)
    with pytest.raises(RuntimeError, match="MLflow is not installed"):
        _require_mlflow()


def test_log_sklearn_run(monkeypatch: pytest.MonkeyPatch) -> None:
    fake = _FakeMLflow()
    monkeypatch.setattr("financial_risk.mlops.tracking._require_mlflow", lambda: fake)

    result = log_sklearn_run(
        object(),
        model_name="fraud-model",
        experiment_name="fraud-detection",
        parameters={"depth": 6},
        metrics={"pr_auc": 0.08},
        tags={"stage": "validation"},
        tracking_uri="file:./mlruns",
        registered_model_name="fraud-model",
    )

    assert result == MLflowRunResult(
        run_id="run-123",
        experiment_id="42",
        model_uri="runs:/run-123/model",
        registered_model_name="fraud-model",
        registered_model_version="7",
    )
    assert fake.tracking_uri == "file:./mlruns"
    assert fake.params == {"depth": 6}
    assert fake.metrics == {"pr_auc": 0.08}
    assert fake.tags == {"model_name": "fraud-model", "stage": "validation"}


def test_register_model_version_sets_alias(monkeypatch: pytest.MonkeyPatch) -> None:
    fake = _FakeMLflow()
    monkeypatch.setattr("financial_risk.mlops.tracking._require_mlflow", lambda: fake)

    version = register_model_version(
        "runs:/run-123/model",
        registered_model_name="fraud-model",
        alias="champion",
        tracking_uri="file:./mlruns",
    )

    assert version == "8"
    assert fake.registered_calls == [("runs:/run-123/model", "fraud-model")]
    assert fake.client.aliases == [("fraud-model", "champion", "8")]


def test_tracking_validates_required_names(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr("financial_risk.mlops.tracking._require_mlflow", lambda: _FakeMLflow())

    with pytest.raises(ValueError, match="model_name"):
        log_sklearn_run(
            object(),
            model_name=" ",
            experiment_name="exp",
            parameters={},
            metrics={},
        )

    with pytest.raises(ValueError, match="experiment_name"):
        log_sklearn_run(
            object(),
            model_name="model",
            experiment_name=" ",
            parameters={},
            metrics={},
        )

    with pytest.raises(ValueError, match="registered_model_name"):
        register_model_version(
            "runs:/run-123/model",
            registered_model_name=" ",
        )
