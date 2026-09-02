"""Feature and prediction drift monitoring utilities."""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd


@dataclass(frozen=True)
class DriftResult:
    metric: str
    statistic: float
    threshold: float
    drift_detected: bool
    reference_size: int
    current_size: int


def _validate_numeric(reference: pd.Series, current: pd.Series) -> tuple[np.ndarray, np.ndarray]:
    ref = pd.to_numeric(reference, errors="coerce").dropna().to_numpy(dtype=float)
    cur = pd.to_numeric(current, errors="coerce").dropna().to_numpy(dtype=float)
    if len(ref) < 2 or len(cur) < 2:
        raise ValueError("reference and current samples must each contain at least 2 numeric values")
    return ref, cur


def population_stability_index(
    reference: pd.Series,
    current: pd.Series,
    bins: int = 10,
    threshold: float = 0.20,
) -> DriftResult:
    """Compute PSI using robust reference bins and flag material distribution shift."""
    if bins < 2:
        raise ValueError("bins must be at least 2")
    if threshold < 0:
        raise ValueError("threshold must be non-negative")

    ref, cur = _validate_numeric(reference, current)
    unique_ref = np.unique(ref)
    if len(unique_ref) == 1:
        # A constant reference feature still has a meaningful PSI when the
        # current population moves away from that value. Use a small window
        # around the reference value so the comparison remains well-defined.
        value = float(unique_ref[0])
        span = max(abs(value) * 1e-6, 1e-6)
        edges = np.array([value - span, value + span], dtype=float)
    else:
        quantiles = np.quantile(ref, np.linspace(0.0, 1.0, bins + 1))
        edges = np.unique(quantiles)
        if len(edges) < 2:
            raise ValueError("reference values must contain enough variation for drift bins")

        # Expand the outer boundaries so values outside the reference range
        # are represented in the current distribution rather than dropped.
        lower = np.nextafter(edges[0], -np.inf)
        upper = np.nextafter(edges[-1], np.inf)
        edges = np.concatenate(([lower], edges[1:-1], [upper]))

    ref_counts, _ = np.histogram(ref, bins=edges)
    cur_counts, _ = np.histogram(cur, bins=edges)
    ref_pct = np.clip(ref_counts / len(ref), 1e-6, None)
    cur_pct = np.clip(cur_counts / len(cur), 1e-6, None)
    psi = float(np.sum((cur_pct - ref_pct) * np.log(cur_pct / ref_pct)))
    return DriftResult("psi", psi, threshold, psi >= threshold, len(ref), len(cur))


def prediction_rate_shift(
    reference_fraud_rate: float,
    current_fraud_rate: float,
    threshold: float = 0.02,
) -> DriftResult:
    """Compare observed fraud rates between reference and current periods."""
    if not 0 <= reference_fraud_rate <= 1 or not 0 <= current_fraud_rate <= 1:
        raise ValueError("fraud rates must be in [0, 1]")
    if threshold < 0:
        raise ValueError("threshold must be non-negative")
    shift = float(abs(current_fraud_rate - reference_fraud_rate))
    return DriftResult("fraud_rate_shift", shift, threshold, shift >= threshold, 0, 0)


def drift_report(
    reference: pd.DataFrame,
    current: pd.DataFrame,
    numeric_features: list[str],
    psi_threshold: float = 0.20,
) -> pd.DataFrame:
    """Build a feature-level drift report using PSI."""
    missing_reference = [c for c in numeric_features if c not in reference.columns]
    missing_current = [c for c in numeric_features if c not in current.columns]
    if missing_reference or missing_current:
        raise ValueError(
            f"Missing features: reference={missing_reference}, current={missing_current}"
        )

    rows = []
    for feature in numeric_features:
        result = population_stability_index(reference[feature], current[feature], threshold=psi_threshold)
        rows.append(
            {
                "feature": feature,
                "metric": result.metric,
                "statistic": result.statistic,
                "threshold": result.threshold,
                "drift_detected": result.drift_detected,
                "reference_size": result.reference_size,
                "current_size": result.current_size,
            }
        )
    return pd.DataFrame(rows)
