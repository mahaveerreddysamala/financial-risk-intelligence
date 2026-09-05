"""Interactive portfolio dashboard for financial risk decisioning."""
from __future__ import annotations

import os
from pathlib import Path
import sys

import pandas as pd
import streamlit as st

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from financial_risk.dashboard import (
    build_dashboard_snapshot,
    build_investigation_payload,
)

st.set_page_config(page_title="Financial Risk Intelligence", page_icon="🛡️", layout="wide")


@st.cache_data(show_spinner="Training the temporal model and scoring future transactions…")
def load_snapshot(rows: int):
    return build_dashboard_snapshot(rows=rows, seed=42)


default_rows = int(os.getenv("FINANCIAL_DASHBOARD_ROWS", "5000"))
st.title("Financial Risk Intelligence")
st.caption(
    "Leakage-safe synthetic demonstration: train on historical transactions, score a future "
    "window, combine model/anomaly/network/velocity signals, and assemble investigation evidence."
)

with st.sidebar:
    st.header("Demo controls")
    rows = st.slider("Synthetic transactions", 2_000, 20_000, default_rows, 1_000)
    minimum_risk = st.slider("Minimum queue score", 0.0, 1.0, 0.30, 0.05)
    selected_bands = st.multiselect(
        "Risk bands",
        ["CRITICAL", "HIGH", "MEDIUM", "LOW"],
        default=["CRITICAL", "HIGH", "MEDIUM"],
    )
    st.info("Synthetic data only. Labels are shown for evaluation, never as scoring inputs.")

snapshot = load_snapshot(rows)
transactions = snapshot.transactions
flagged = int(transactions["risk_band"].ne("LOW").sum())
critical = int(transactions["risk_band"].eq("CRITICAL").sum())

metric_columns = st.columns(5)
metric_columns[0].metric("Future transactions", f"{snapshot.test_rows:,}")
metric_columns[1].metric("Flagged for monitoring", f"{flagged:,}")
metric_columns[2].metric("Critical cases", f"{critical:,}")
metric_columns[3].metric("PR-AUC", f'{snapshot.model_metrics["pr_auc"]:.4f}')
metric_columns[4].metric("ROC-AUC", f'{snapshot.model_metrics["roc_auc"]:.4f}')

overview_tab, queue_tab, evidence_tab, model_tab = st.tabs(
    ["Executive overview", "Decision queue", "Case evidence", "Model evidence"]
)

with overview_tab:
    left, right = st.columns(2)
    with left:
        st.subheader("Decision distribution")
        st.bar_chart(snapshot.band_counts.set_index("risk_band"), color="#d97706")
    with right:
        st.subheader("Daily mean risk")
        st.line_chart(snapshot.daily_risk.set_index("day")[["mean_risk"]], color="#2563eb")
    st.markdown(
        "**Decision path:** transaction → leakage-safe features → XGBoost + anomaly + network + "
        "velocity → bounded ensemble score → operational action → investigation evidence"
    )

with queue_tab:
    st.subheader("Prioritized transaction queue")
    queue = transactions[
        transactions["risk_band"].isin(selected_bands)
        & transactions["risk_score"].ge(minimum_risk)
    ].copy()
    queue_columns = [
        "transaction_id",
        "timestamp",
        "amount",
        "country",
        "fraud_probability",
        "anomaly_score",
        "network_score",
        "velocity_score",
        "risk_score",
        "risk_band",
        "action",
        "primary_reason",
    ]
    st.dataframe(
        queue[queue_columns].head(100),
        hide_index=True,
        width="stretch",
        column_config={
            "amount": st.column_config.NumberColumn(format="$%.2f"),
            "fraud_probability": st.column_config.ProgressColumn(min_value=0.0, max_value=1.0),
            "anomaly_score": st.column_config.ProgressColumn(min_value=0.0, max_value=1.0),
            "risk_score": st.column_config.ProgressColumn(min_value=0.0, max_value=1.0),
        },
    )
    st.caption(f"Showing {min(len(queue), 100):,} of {len(queue):,} matching transactions.")

with evidence_tab:
    st.subheader("Evidence-grounded investigation case")
    case_candidates = transactions.head(50)
    selected_id = st.selectbox("Select a high-priority transaction", case_candidates["transaction_id"])
    selected_row = case_candidates.loc[case_candidates["transaction_id"].eq(selected_id)].iloc[0]
    detail_columns = st.columns(4)
    detail_columns[0].metric("Risk score", f'{selected_row["risk_score"]:.3f}')
    detail_columns[1].metric("Risk band", selected_row["risk_band"])
    detail_columns[2].metric("Amount", f'${selected_row["amount"]:,.2f}')
    detail_columns[3].metric("Action", selected_row["action"])
    st.json(build_investigation_payload(selected_row), expanded=False)

with model_tab:
    st.subheader("Held-out temporal evaluation")
    st.caption(
        f"Model trained on {snapshot.train_rows:,} earlier rows and evaluated on "
        f"{snapshot.test_rows:,} strictly later rows."
    )
    metrics = pd.DataFrame(
        {"metric": list(snapshot.model_metrics), "value": list(snapshot.model_metrics.values())}
    )
    st.dataframe(metrics, hide_index=True, width="stretch")
    backtest_path = ROOT / "artifacts" / "temporal-backtest.csv"
    if backtest_path.exists():
        st.subheader("Walk-forward stability")
        st.dataframe(pd.read_csv(backtest_path), hide_index=True, width="stretch")
    else:
        st.info(
            "Run `python -m financial_risk.models.backtesting` to add the three-fold "
            "walk-forward table."
        )
    st.warning(
        "All results use deterministic synthetic data and demonstrate system behavior, not "
        "production fraud performance."
    )
