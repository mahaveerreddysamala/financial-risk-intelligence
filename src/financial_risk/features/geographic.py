"""Geographic and temporal transaction behavior features."""

from __future__ import annotations

import pandas as pd


def add_geographic_features(df: pd.DataFrame) -> pd.DataFrame:
    """Add prior-location and temporal deviation features per customer."""
    required = {"transaction_id", "customer_id", "timestamp", "country", "is_international", "is_night"}
    missing = required.difference(df.columns)
    if missing:
        raise ValueError(f"Missing required columns: {sorted(missing)}")

    result = df.copy()
    result["timestamp"] = pd.to_datetime(result["timestamp"], errors="raise")
    result["_row_id"] = range(len(result))
    result = result.sort_values(["customer_id", "timestamp", "transaction_id"], kind="mergesort")

    def _features(group: pd.DataFrame) -> pd.DataFrame:
        work = group.sort_values("timestamp", kind="mergesort").reset_index(drop=True)
        previous_country = work["country"].shift(1)
        previous_timestamp = work["timestamp"].shift(1)
        prior_country_count = previous_country.expanding().count()
        prior_international = work["is_international"].shift(1).expanding().mean()
        hour = work["timestamp"].dt.hour
        previous_night_rate = work["is_night"].shift(1).expanding().mean()
        out = pd.DataFrame({"_row_id": work["_row_id"].to_numpy()})
        out["location_changed"] = (previous_country.notna() & previous_country.ne(work["country"])).astype("int8")
        out["minutes_since_customer_txn"] = (
            (work["timestamp"] - previous_timestamp).dt.total_seconds().div(60).fillna(-1.0)
        )
        out["customer_prior_international_rate"] = prior_international.fillna(0.0).to_numpy()
        out["customer_prior_night_rate"] = previous_night_rate.fillna(0.0).to_numpy()
        out["customer_prior_transaction_count"] = prior_country_count.fillna(0).to_numpy()
        out["transaction_hour"] = hour.to_numpy()
        return out

    parts = [_features(group) for _, group in result.groupby("customer_id", sort=False)]
    features = pd.concat(parts, ignore_index=True).sort_values("_row_id", kind="mergesort")
    feature_cols = [
        "location_changed",
        "minutes_since_customer_txn",
        "customer_prior_international_rate",
        "customer_prior_night_rate",
        "customer_prior_transaction_count",
        "transaction_hour",
    ]
    feature_map = features.set_index("_row_id")[feature_cols]
    result[feature_cols] = feature_map.reindex(result["_row_id"]).to_numpy()
    return result.sort_values("_row_id", kind="mergesort").drop(columns="_row_id")
