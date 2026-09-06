"""Interactive portfolio dashboard for financial risk decisioning."""
from __future__ import annotations

import os
from pathlib import Path
import sys

import pandas as pd
import streamlit as st

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from financial_risk.investigation.rag import answer_question, load_chunks
from financial_risk.investigation.evidence_scope import REFERENCE_SCOPE

from financial_risk.dashboard import (
    build_dashboard_snapshot,
    build_investigation_payload,
)
from financial_risk.dashboard.operating_points import (
    build_operating_point_curve,
    operating_point_summary,
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
    review_capacity = st.slider("Review capacity (% of future transactions)", 1, 25, 5)
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

overview_tab, queue_tab, evidence_tab, model_tab, copilot_tab = st.tabs(
    ["Executive overview", "Decision queue", "Case evidence", "Model evidence", "Ask the copilot"]
)

with copilot_tab:
    st.subheader("Investigation reference copilot")
    st.caption("Offline reference retrieval. No API calls, credentials, or transaction actions.")
    with st.form("reference_question"):
        case_id = st.selectbox("Case context", ["Reference only"] +
                               transactions.head(50)["transaction_id"].tolist())
        question = st.text_input("Question", "How do shared devices affect an investigation?",
                                 max_chars=2000)
        submitted = st.form_submit_button("Retrieve evidence")
    if submitted:
        if not question.strip():
            st.warning("Enter a question first.")
        else:
            case = None
            if case_id != "Reference only":
                row = transactions.loc[transactions["transaction_id"].eq(case_id)].iloc[0]
                case = build_investigation_payload(row)
            answer = answer_question(question, load_chunks(ROOT), case=case,
                                     evidence_scope=REFERENCE_SCOPE)
            st.text(answer.text)
            for source in answer.sources:
                with st.expander(f'{source["id"]}: {source["source"]}'):
                    st.text(source["text"])
                    if "score" in source:
                        st.caption(f'Retrieval similarity: {source["score"]:.3f} (not confidence)')
    st.info("Selected cases contribute only allowlisted numeric signals; identifiers and free text "
            "are excluded. Excerpts and observations are not LLM conclusions. "
            "The optional hosted adapter is available to developers but disabled in this app.")

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

    st.subheader("Capacity-aware operating point")
    operating_point = operating_point_summary(transactions, review_capacity / 100.0)
    operating_columns = st.columns(5)
    operating_columns[0].metric("Transactions reviewed", f'{operating_point["review_count"]:,}')
    operating_columns[1].metric("Fraud captured", f'{operating_point["captured_fraud"]:,}')
    operating_columns[2].metric("Queue precision", f'{operating_point["precision"]:.1%}')
    operating_columns[3].metric("Fraud recall", f'{operating_point["recall"]:.1%}')
    operating_columns[4].metric("Lift vs. random", f'{operating_point["lift"]:.2f}x')
    capacity_curve = build_operating_point_curve(transactions).set_index("review_rate_pct")
    st.line_chart(capacity_curve[["precision", "recall"]], color=["#7c3aed", "#ea580c"])
    st.caption(
        "The simulator ranks the held-out future window by the final ensemble risk score and "
        "measures the operational trade-off at a fixed investigation capacity."
    )

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
