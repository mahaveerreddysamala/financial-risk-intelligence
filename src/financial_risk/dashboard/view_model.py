"""Deterministic data and model outputs for the Financial Risk dashboard."""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd
from sklearn.metrics import (
    average_precision_score,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
)

from financial_risk.data_generation.generator import generate_transactions
from financial_risk.features.pipeline import build_feature_table
from financial_risk.investigation.case_builder import (
    build_investigation_case,
    case_to_dict,
)
from financial_risk.models.anomaly import fit_anomaly_detector, score_anomalies
from financial_risk.models.baseline import CATEGORICAL_FEATURES, NUMERIC_FEATURES
from financial_risk.models.graph_risk import integrate_graph_risk
from financial_risk.models.split import temporal_split
from financial_risk.models.xgboost_model import build_xgboost_model

FEATURE_COLUMNS = NUMERIC_FEATURES + CATEGORICAL_FEATURES
RISK_ORDER = ["LOW", "MEDIUM", "HIGH", "CRITICAL"]


@dataclass(frozen=True)
class DashboardSnapshot:
    """All tables needed to render the dashboard without UI coupling."""

    transactions: pd.DataFrame
    daily_risk: pd.DataFrame
    band_counts: pd.DataFrame
    model_metrics: dict[str, float]
    train_rows: int
    test_rows: int


def _dominant_reason(
    fraud_probability: float,
    anomaly_score: float,
    network_score: float,
    velocity_score: float,
) -> str:
    contributions = {
        "Fraud model": fraud_probability * 0.50,
        "Anomaly signal": anomaly_score * 0.30,
        "Network reuse": network_score * 0.10,
        "Transaction velocity": velocity_score * 0.10,
    }
    return max(contributions, key=contributions.get)


def build_dashboard_snapshot(rows: int = 5_000, seed: int = 42) -> DashboardSnapshot:
    """Train on past synthetic transactions and score a future dashboard window."""
    if rows < 2_000:
        raise ValueError("rows must be at least 2,000 for a stable temporal demo")

    features = build_feature_table(generate_transactions(rows, seed=seed))
    train, _, test = temporal_split(features)
    y_train = train["is_fraud"].astype(int)
    if y_train.nunique() < 2 or test["is_fraud"].nunique() < 2:
        raise ValueError("Dashboard temporal split must contain both target classes")

    positive = int(y_train.sum())
    model = build_xgboost_model(scale_pos_weight=(len(y_train) - positive) / positive)
    model.fit(train[FEATURE_COLUMNS], y_train)
    probabilities = np.asarray(model.predict_proba(test[FEATURE_COLUMNS])[:, 1], dtype=float)

    scaler, detector = fit_anomaly_detector(train)
    anomaly = score_anomalies(test, scaler, detector)
    network_scores = np.clip(
        (test["shared_device_account_count"].to_numpy(dtype=float) - 1.0) / 8.0,
        0.0,
        1.0,
    )
    velocity_scores = np.clip(
        test["txn_count_1h"].to_numpy(dtype=float) / 5.0,
        0.0,
        1.0,
    )
    community_scores = np.clip(network_scores * 0.80, 0.0, 1.0)

    decisions = [
        integrate_graph_risk(
            float(fraud_probability),
            float(anomaly_score),
            float(network_score),
            float(velocity_score),
            float(community_score),
        ).decision
        for fraud_probability, anomaly_score, network_score, velocity_score, community_score in zip(
            probabilities,
            anomaly.scores,
            network_scores,
            velocity_scores,
            community_scores,
            strict=True,
        )
    ]

    scored = test[
        [
            "transaction_id",
            "timestamp",
            "customer_id",
            "amount",
            "country",
            "channel",
            "payment_method",
            "merchant_id",
            "device_id",
            "ip_id",
            "shared_device_account_count",
            "is_fraud",
        ]
    ].copy()
    scored["fraud_probability"] = probabilities
    scored["anomaly_score"] = anomaly.scores
    scored["network_score"] = network_scores
    scored["velocity_score"] = velocity_scores
    scored["community_score"] = community_scores
    scored["risk_score"] = [decision.score for decision in decisions]
    scored["risk_band"] = [decision.level for decision in decisions]
    scored["action"] = [decision.action for decision in decisions]
    scored["primary_reason"] = [
        _dominant_reason(*signals)
        for signals in zip(
            probabilities,
            anomaly.scores,
            network_scores,
            velocity_scores,
            strict=True,
        )
    ]
    scored = scored.sort_values("risk_score", ascending=False).reset_index(drop=True)

    daily = scored.assign(day=pd.to_datetime(scored["timestamp"]).dt.floor("D"))
    daily_risk = (
        daily.groupby("day", as_index=False)
        .agg(transactions=("transaction_id", "size"), mean_risk=("risk_score", "mean"))
        .sort_values("day")
    )
    counts = scored["risk_band"].value_counts().reindex(RISK_ORDER, fill_value=0)
    band_counts = counts.rename_axis("risk_band").reset_index(name="transactions")

    y_test = scored["is_fraud"].astype(int).to_numpy()
    ordered_probabilities = scored["fraud_probability"].to_numpy(dtype=float)
    predictions = (ordered_probabilities >= 0.5).astype(int)
    metrics = {
        "roc_auc": float(roc_auc_score(y_test, ordered_probabilities)),
        "pr_auc": float(average_precision_score(y_test, ordered_probabilities)),
        "precision": float(precision_score(y_test, predictions, zero_division=0)),
        "recall": float(recall_score(y_test, predictions, zero_division=0)),
        "f1": float(f1_score(y_test, predictions, zero_division=0)),
    }
    return DashboardSnapshot(scored, daily_risk, band_counts, metrics, len(train), len(test))


def build_investigation_payload(row: pd.Series) -> dict[str, object]:
    """Build the evidence payload displayed for a selected scored transaction."""
    case = build_investigation_case(
        row,
        fraud_probability=float(row["fraud_probability"]),
        anomaly_score=float(row["anomaly_score"]),
        network_risk=float(row["network_score"]),
        velocity_risk=float(row["velocity_score"]),
        risk_score=float(row["risk_score"]),
        risk_band=str(row["risk_band"]),
        action=str(row["action"]),
    )
    return case_to_dict(case)
