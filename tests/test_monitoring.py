import numpy as np
import pandas as pd
import pytest

from financial_risk.monitoring.drift import (
    drift_report,
    population_stability_index,
    prediction_rate_shift,
)


def test_psi_detects_distribution_shift():
    reference = pd.Series(np.linspace(0, 10, 500))
    current = pd.Series(np.linspace(20, 30, 500))
    result = population_stability_index(reference, current, bins=5, threshold=0.20)

    assert result.metric == "psi"
    assert result.statistic >= 0
    assert result.drift_detected is True
    assert result.reference_size == 500
    assert result.current_size == 500


def test_psi_does_not_flag_same_distribution():
    rng = np.random.default_rng(42)
    reference = pd.Series(rng.normal(0, 1, 1000))
    current = pd.Series(rng.normal(0, 1, 1000))
    result = population_stability_index(reference, current, bins=10, threshold=0.20)

    assert result.statistic < 0.20
    assert result.drift_detected is False


def test_prediction_rate_shift():
    result = prediction_rate_shift(0.01, 0.05, threshold=0.02)
    assert result.statistic == pytest.approx(0.04)
    assert result.drift_detected is True


def test_drift_report():
    reference = pd.DataFrame(
        {
            "amount": np.linspace(10, 100, 200),
            "txn_count_1h": np.ones(200),
        }
    )
    current = pd.DataFrame(
        {
            "amount": np.linspace(10, 100, 200),
            "txn_count_1h": np.full(200, 3.0),
        }
    )
    report = drift_report(reference, current, ["amount", "txn_count_1h"], psi_threshold=0.20)

    assert list(report.columns) == [
        "feature",
        "metric",
        "statistic",
        "threshold",
        "drift_detected",
        "reference_size",
        "current_size",
    ]
    assert len(report) == 2
    assert report.loc[report["feature"] == "txn_count_1h", "drift_detected"].item() is True
    assert report.loc[report["feature"] == "amount", "drift_detected"].item() is False


def test_monitoring_validates_inputs():
    with pytest.raises(ValueError, match="at least 2"):
        population_stability_index(pd.Series([1]), pd.Series([1, 2]))
    with pytest.raises(ValueError, match="fraud rates"):
        prediction_rate_shift(-0.1, 0.1)
    with pytest.raises(ValueError, match="Missing features"):
        drift_report(pd.DataFrame({"amount": [1, 2]}), pd.DataFrame(), ["amount"])
