from __future__ import annotations

from financial_risk.dashboard import build_dashboard_snapshot, build_investigation_payload


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
