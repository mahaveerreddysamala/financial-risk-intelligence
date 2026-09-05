from __future__ import annotations

import pandas as pd
import pytest

from financial_risk.dashboard import (
    build_dashboard_snapshot,
    build_investigation_payload,
    build_operating_point_curve,
    operating_point_summary,
)


def test_dashboard_scores_future_transactions_and_builds_cases() -> None:
    snapshot = build_dashboard_snapshot(rows=3_000, seed=42)

    assert snapshot.train_rows > snapshot.test_rows > 0
    assert snapshot.transactions["risk_score"].between(0, 1).all()
    assert set(snapshot.transactions["risk_band"]).issubset(
        {"LOW", "MEDIUM", "HIGH", "CRITICAL"}
    )
    assert snapshot.transactions["primary_reason"].notna().all()
    assert snapshot.band_counts["transactions"].sum() == snapshot.test_rows
    assert 0.0 <= snapshot.model_metrics["pr_auc"] <= 1.0

    payload = build_investigation_payload(snapshot.transactions.iloc[0])
    assert payload["transaction_id"] == snapshot.transactions.iloc[0]["transaction_id"]
    assert len(payload["evidence"]) >= 4


def test_operating_point_reports_capacity_precision_recall_and_lift() -> None:
    transactions = pd.DataFrame(
        {
            "risk_score": [0.99, 0.90, 0.80, 0.20, 0.10],
            "is_fraud": [1, 1, 0, 0, 0],
        }
    )

    point = operating_point_summary(transactions, review_rate=0.40)

    assert point == {
        "review_rate": 0.4,
        "review_count": 2,
        "captured_fraud": 2,
        "precision": 1.0,
        "recall": 1.0,
        "lift": 2.5,
    }

    curve = build_operating_point_curve(transactions, (0.20, 0.40, 1.00))
    assert curve["review_count"].tolist() == [1, 2, 5]
    assert curve["recall"].is_monotonic_increasing


def test_operating_point_rejects_invalid_inputs() -> None:
    with pytest.raises(ValueError, match="review_rate"):
        operating_point_summary(pd.DataFrame({"risk_score": [0.5], "is_fraud": [0]}), 0.0)
    with pytest.raises(ValueError, match="missing columns"):
        operating_point_summary(pd.DataFrame({"risk_score": [0.5]}), 0.5)
