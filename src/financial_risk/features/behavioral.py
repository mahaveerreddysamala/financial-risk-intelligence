"""Customer behavioral features computed from transaction history only."""

from __future__ import annotations

import pandas as pd


def add_behavioral_features(df: pd.DataFrame) -> pd.DataFrame:
    """Add customer behavior features using observations strictly before each transaction."""
    required = {"transaction_id", "customer_id", "timestamp", "amount", "merchant_id", "device_id", "country", "is_night"}
    missing = required.difference(df.columns)
    if missing:
        raise ValueError(f"Missing required columns: {sorted(missing)}")

    result = df.copy()
    result["timestamp"] = pd.to_datetime(result["timestamp"], errors="raise")
    result["_row_order"] = range(len(result))
    result = result.sort_values(["customer_id", "timestamp", "transaction_id"], kind="mergesort")

    def _features(group: pd.DataFrame) -> pd.DataFrame:
        indexed = group.set_index("timestamp")
        prior_amount = indexed["amount"].shift(1)
        rolling_7d = prior_amount.rolling("7D", closed="both")
        rolling_30d = prior_amount.rolling("30D", closed="both")

        out = pd.DataFrame(index=group.index)
        out["customer_txn_count_7d"] = rolling_7d.count().to_numpy()
        out["customer_avg_amount_30d"] = rolling_30d.mean().to_numpy()
        out["customer_std_amount_30d"] = rolling_30d.std(ddof=0).fillna(0).to_numpy()
        out["customer_unique_merchants_7d"] = (
            indexed["merchant_id"].shift(1).rolling("7D", closed="both").apply(
                lambda values: len(set(values)), raw=False
            ).fillna(0).to_numpy()
        )
        out["customer_unique_devices_30d"] = (
            indexed["device_id"].shift(1).rolling("30D", closed="both").apply(
                lambda values: len(set(values)), raw=False
            ).fillna(0).to_numpy()
        )
        out["customer_international_rate_30d"] = (
            indexed["country"].shift(1).ne("US").astype(float).rolling("30D", closed="both").mean().fillna(0).to_numpy()
        )
        out["customer_night_txn_rate_30d"] = (
            indexed["is_night"].shift(1).astype(float).rolling("30D", closed="both").mean().fillna(0).to_numpy()
        )
        return out

    features = result.groupby("customer_id", group_keys=False, sort=False).apply(_features, include_groups=False)
    feature_cols = [
        "customer_txn_count_7d",
        "customer_avg_amount_30d",
        "customer_std_amount_30d",
        "customer_unique_merchants_7d",
        "customer_unique_devices_30d",
        "customer_international_rate_30d",
        "customer_night_txn_rate_30d",
    ]
    result[feature_cols] = features[feature_cols]
    result["amount_vs_customer_avg"] = (
        result["amount"] / result["customer_avg_amount_30d"].replace(0, pd.NA)
    ).fillna(1.0)
    result["amount_zscore"] = (
        (result["amount"] - result["customer_avg_amount_30d"])
        / result["customer_std_amount_30d"].replace(0, pd.NA)
    ).fillna(0.0)

    return result.sort_values("_row_order", kind="mergesort").drop(columns="_row_order")
