"""Recruiter-facing dashboard data preparation."""

from financial_risk.dashboard.view_model import (
    DashboardSnapshot,
    build_dashboard_snapshot,
    build_investigation_payload,
)

__all__ = [
    "DashboardSnapshot",
    "build_dashboard_snapshot",
    "build_investigation_payload",
]
