"""Minimal numeric context for case-aware retrieval; not a general PII detector."""
from __future__ import annotations

import math

FIELDS = {
    "fraud_probability": "Fraud model probability",
    "anomaly_score": "Anomaly score",
    "network_risk": "Network reuse risk",
    "velocity_risk": "Transaction velocity risk",
    "shared_device_account_count": "Shared devices account count",
}


def safe_case_sources(case: dict) -> tuple[dict, ...]:
    """Allow only finite validated numeric signals; drop identifiers and free text."""
    rows = []
    seen = set()
    for item in case.get("evidence", ()):
        field, value = item.get("field"), item.get("value")
        if field not in FIELDS or field in seen:
            continue
        if isinstance(value, bool) or not isinstance(value, (float, int)):
            continue
        if not math.isfinite(value) or value < 0:
            continue
        if field == "shared_device_account_count":
            if value != int(value):
                continue
        elif value > 1:
            continue
        seen.add(field)
        rows.append({"id": f"E{len(rows) + 1}", "source": f"selected_case.{field}",
                     "text": f"{FIELDS[field]}: {value}", "field": field,
                     "value": value})
    return tuple(rows)
