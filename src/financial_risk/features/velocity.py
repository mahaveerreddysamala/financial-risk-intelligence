"""Short-horizon transaction velocity features with no future-data leakage."""

from __future__ import annotations

import pandas as pd


def add_velocity_features(df: pd.DataFrame) -> pd.DataFrame:
    """Add prior 5-minute, 1-hour, and 24-hour customer velocity features."""
    required = {"transaction_id", "customer_id", "timestamp", "amount"}
    missing = required.difference(df.columns)
    if missing:
        raise ValueError(f"Missing required columns: {sorted(missing)}")

    result = df.copy()
    result["timestamp"] = pd.to_datetime(result["timestamp"], errors="raise")
    result["_row_id"] = range(len(result))
    result = result.sort_values(["customer_id", "timestamp", "transaction_id"], kind="mergesort")

    def _features(group: pd.DataFrame) -> pd.DataFrame:
        work = group.sort_values("timestamp", kind="mergesort").set_index("timestamp")
        prior_amount = work["amount"].shift(1)
        out = pd.DataFrame({"_row_id": group.sort_values("timestamp", kind="mergesort")["_row_id"].to_numpy()})
        out["txn_count_5m"] = prior_amount.rolling("5min", closed="both").count().to_numpy()
        out["txn_count_1h"] = prior_amount.rolling("1h", closed="both").count().to_numpy()
        out["txn_count_24h"] = prior_amount.rolling("24h", closed="both").count().to_numpy()
        out["amount_sum_1h"] = prior_amount.rolling("1h", closed="both").sum().fillna(0).to_numpy()
        out["amount_sum_24h"] = prior_amount.rolling("24h", closed="both").sum().fillna(0).to_numpy()
        return out

    parts = [
        _features(group)
        for _, group in result.groupby("customer_id", sort=False)
    ]
    features = pd.concat(parts, ignore_index=True).sort_values("_row_id", kind="mergesort")
    feature_cols = [
        "txn_count_5m",
        "txn_count_1h",
        "txn_count_24h",
        "amount_sum_1h",
        "amount_sum_24h",
    ]
    feature_map = features.set_index("_row_id")[feature_cols]
    result[feature_cols] = feature_map.reindex(result["_row_id"]).to_numpy()
    return result.sort_values("_row_id", kind="mergesort").drop(columns="_row_id")
