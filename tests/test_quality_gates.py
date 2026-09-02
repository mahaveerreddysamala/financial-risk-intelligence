from __future__ import annotations

import pytest

from financial_risk.mlops.quality_gates import promote_if_approved, run_quality_gates


def _metrics() -> dict[str, float]:
    return {
        "test_pr_auc": 0.08,
        "test_recall": 0.10,
        "test_precision": 0.04,
    }


def test_quality_gates_pass_with_metrics_and_drift() -> None:
    report = run_quality_gates(
        _metrics(),
        min_pr_auc=0.05,
        min_recall=0.05,
        min_precision=0.02,
        drift_report={"feature_psi": {"amount": 0.08, "velocity": 0.12}},
        max_psi=0.20,
    )

    assert report.passed is True
    assert report.failed_gates == ()
    assert len(report.gates) == 4


def test_quality_gates_fail_missing_metric() -> None:
    report = run_quality_gates({"test_pr_auc": 0.08})

    assert report.passed is False
    assert {gate.name for gate in report.failed_gates} == {"test_recall", "test_precision"}


def test_quality_gates_fail_excessive_drift() -> None:
    report = run_quality_gates(
        _metrics(),
        drift_report={"feature_psi": {"amount": 0.35}},
        max_psi=0.20,
    )

    assert report.passed is False
    assert report.failed_gates[-1].name == "max_feature_psi"
    assert report.failed_gates[-1].observed == pytest.approx(0.35)


def test_promotion_requires_all_gates(monkeypatch: pytest.MonkeyPatch) -> None:
    report = run_quality_gates(_metrics())
    captured: dict[str, object] = {}

    def fake_register(
        model_uri: str,
        *,
        registered_model_name: str,
        alias: str | None,
        tracking_uri: str | None,
    ) -> str:
        captured.update(
            {
                "model_uri": model_uri,
                "registered_model_name": registered_model_name,
                "alias": alias,
                "tracking_uri": tracking_uri,
            }
        )
        return "9"

    monkeypatch.setattr("financial_risk.mlops.quality_gates.register_model_version", fake_register)

    version = promote_if_approved(
        report,
        "runs:/run-123/model",
        registered_model_name="financial-fraud-xgboost",
        alias="champion",
        tracking_uri="file:./mlruns",
    )

    assert version == "9"
    assert captured == {
        "model_uri": "runs:/run-123/model",
        "registered_model_name": "financial-fraud-xgboost",
        "alias": "champion",
        "tracking_uri": "file:./mlruns",
    }


def test_promotion_rejects_failed_model() -> None:
    report = run_quality_gates({"test_pr_auc": 0.01})

    with pytest.raises(ValueError, match="failed quality gates"):
        promote_if_approved(
            report,
            "runs:/run-123/model",
            registered_model_name="financial-fraud-xgboost",
        )


def test_invalid_thresholds_are_rejected() -> None:
    with pytest.raises(ValueError, match="between zero and one"):
        run_quality_gates(_metrics(), min_pr_auc=1.1)

    with pytest.raises(ValueError, match="non-negative"):
        run_quality_gates(_metrics(), max_psi=-0.1)
