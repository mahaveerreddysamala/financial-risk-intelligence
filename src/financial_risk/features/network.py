"""Prior-only network reuse features for fraud-ring signals."""

from __future__ import annotations

import pandas as pd


def add_network_features(df: pd.DataFrame) -> pd.DataFrame:
    """Add historical reuse counts for devices, IPs, and merchants."""
    required = {"transaction_id", "customer_id", "timestamp", "device_id", "ip_id", "merchant_id"}
    missing = required.difference(df.columns)
    if missing:
        raise ValueError(f"Missing required columns: {sorted(missing)}")

    result = df.copy()
    result["timestamp"] = pd.to_datetime(result["timestamp"], errors="raise")
    result["_row_id"] = range(len(result))
    result = result.sort_values(["timestamp", "transaction_id"], kind="mergesort")

    prior_device = result.groupby("device_id", sort=False).cumcount()
    prior_ip = result.groupby("ip_id", sort=False).cumcount()
    prior_merchant = result.groupby("merchant_id", sort=False).cumcount()
    prior_customer_device = result.groupby(["customer_id", "device_id"], sort=False).cumcount()

    result["device_prior_transaction_count"] = prior_device.astype("int64")
    result["ip_prior_transaction_count"] = prior_ip.astype("int64")
    result["merchant_prior_transaction_count"] = prior_merchant.astype("int64")
    result["customer_device_prior_transaction_count"] = prior_customer_device.astype("int64")

    if "shared_device_account_count" in result.columns:
        result["shared_device_risk_signal"] = result["shared_device_account_count"].clip(lower=1).astype(float).rpow(1.0)
    else:
        result["shared_device_risk_signal"] = 1.0

    result["network_reuse_score"] = (
        result["device_prior_transaction_count"]
        + result["ip_prior_transaction_count"]
        + result["merchant_prior_transaction_count"]
    ).astype(float)

    return result.sort_values("_row_id", kind="mergesort").drop(columns="_row_id")
