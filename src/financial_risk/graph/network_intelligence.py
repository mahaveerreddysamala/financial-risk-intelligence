"""Graph-style financial crime network intelligence without a graph database."""
from __future__ import annotations

import numpy as np
import pandas as pd

ENTITY_COLUMNS = ["customer_id", "account_id", "device_id", "ip_id", "merchant_id"]


def build_entity_graph_features(df: pd.DataFrame) -> pd.DataFrame:
    """Build entity-degree and shared-entity features from transaction relationships."""
    required = {"transaction_id", *ENTITY_COLUMNS}
    missing = required.difference(df.columns)
    if missing:
        raise ValueError(f"Missing required columns: {sorted(missing)}")

    result = df.copy()

    device_accounts = result.groupby("device_id")["account_id"].transform("nunique")
    ip_accounts = result.groupby("ip_id")["account_id"].transform("nunique")
    customer_devices = result.groupby("customer_id")["device_id"].transform("nunique")
    customer_ips = result.groupby("customer_id")["ip_id"].transform("nunique")
    merchant_customers = result.groupby("merchant_id")["customer_id"].transform("nunique")

    result["shared_device_accounts"] = device_accounts.astype("int64")
    result["shared_ip_accounts"] = ip_accounts.astype("int64")
    result["customer_device_degree"] = customer_devices.astype("int64")
    result["customer_ip_degree"] = customer_ips.astype("int64")
    result["merchant_customer_degree"] = merchant_customers.astype("int64")

    result["network_entity_degree"] = (
        result["shared_device_accounts"]
        + result["shared_ip_accounts"]
        + result["customer_device_degree"]
        + result["customer_ip_degree"]
        + result["merchant_customer_degree"]
    ).astype("int64")

    result["network_risk_score"] = (
        np.log1p(result["shared_device_accounts"])
        + np.log1p(result["shared_ip_accounts"])
        + np.log1p(result["merchant_customer_degree"])
    ) / 3.0
    result["network_risk_score"] = result["network_risk_score"].clip(0.0, None).astype(float)

    return result
