"""Unsupervised anomaly detection for transaction-risk signals."""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd
from sklearn.ensemble import IsolationForest
from sklearn.preprocessing import StandardScaler

ANOMALY_FEATURES = [
    "amount",
    "is_international",
    "is_night",
    "shared_device_account_count",
    "customer_txn_count_7d",
    "customer_avg_amount_30d",
    "customer_std_amount_30d",
    "customer_unique_merchants_7d",
    "customer_unique_devices_30d",
    "customer_international_rate_30d",
    "customer_night_txn_rate_30d",
    "amount_vs_customer_avg",
    "amount_zscore",
    "txn_count_5m",
    "txn_count_1h",
    "txn_count_24h",
]


@dataclass(frozen=True)
class AnomalyResult:
    scores: np.ndarray
    flags: np.ndarray


def build_anomaly_detector(
    contamination: float = 0.01,
    random_state: int = 42,
) -> IsolationForest:
    """Create a reproducible Isolation Forest anomaly detector."""
    if not 0.0 < contamination < 0.5:
        raise ValueError("contamination must be between 0 and 0.5")
    return IsolationForest(
        n_estimators=250,
        contamination=contamination,
        random_state=random_state,
        n_jobs=4,
    )


def fit_anomaly_detector(train: pd.DataFrame, contamination: float = 0.01) -> tuple[StandardScaler, IsolationForest]:
    """Fit scaler and Isolation Forest on training observations only."""
    missing = set(ANOMALY_FEATURES).difference(train.columns)
    if missing:
        raise ValueError(f"Missing anomaly features: {sorted(missing)}")

    scaler = StandardScaler()
    matrix = scaler.fit_transform(train[ANOMALY_FEATURES].fillna(0.0))
    detector = build_anomaly_detector(contamination=contamination)
    detector.fit(matrix)
    return scaler, detector


def score_anomalies(
    frame: pd.DataFrame,
    scaler: StandardScaler,
    detector: IsolationForest,
) -> AnomalyResult:
    """Return normalized anomaly scores where higher means more anomalous."""
    missing = set(ANOMALY_FEATURES).difference(frame.columns)
    if missing:
        raise ValueError(f"Missing anomaly features: {sorted(missing)}")

    matrix = scaler.transform(frame[ANOMALY_FEATURES].fillna(0.0))
    # IsolationForest's decision boundary is zero: negative values are
    # anomalies and positive values are inliers. Map that stable boundary to
    # 0.5 instead of min-max normalizing each scoring batch. Batch-local
    # normalization makes the same observation receive a different risk score
    # depending on which other observations happen to be scored with it, and a
    # single-row request always collapses to zero.
    decision = detector.decision_function(matrix)
    scores = np.clip(0.5 - decision, 0.0, 1.0)
    flags = detector.predict(matrix) == -1
    return AnomalyResult(scores=scores.astype(float), flags=flags.astype(bool))
