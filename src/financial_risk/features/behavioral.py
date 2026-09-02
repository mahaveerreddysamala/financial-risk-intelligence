"""Customer behavioral features computed from transaction history only."""

from __future__ import annotations

from collections import Counter, deque

import numpy as np
import pandas as pd


def add_behavioral_features(df: pd.DataFrame) -> pd.DataFrame:
    """Add customer behavior features using only observations before each transaction."""
    required = {"transaction_id", "customer_id", "timestamp", "amount", "merchant_id", "device_id", "country", "is_night"}
    missing = required.difference(df.columns)
    if missing:
        raise ValueError(f"Missing required columns: {sorted(missing)}")

    result = df.copy()
    result["timestamp"] = pd.to_datetime(result["timestamp"], errors="raise")
    # Avoid a leading-underscore column name because DataFrame.itertuples()
    # sanitizes such names (for example, _row_id -> _1) on some pandas versions.
    result["row_id"] = np.arange(len(result))
    result = result.sort_values(["customer_id", "timestamp", "transaction_id"], kind="mergesort")

    feature_cols = [
        "customer_txn_count_7d",
        "customer_avg_amount_30d",
        "customer_std_amount_30d",
        "customer_unique_merchants_7d",
        "customer_unique_devices_30d",
        "customer_international_rate_30d",
        "customer_night_txn_rate_30d",
        "amount_vs_customer_avg",
        "amount_zscore",
    ]

    def _features(group: pd.DataFrame) -> pd.DataFrame:
        work = group.sort_values("timestamp", kind="mergesort")
        short: deque[tuple[pd.Timestamp, float, str, str]] = deque()
        long: deque[tuple[pd.Timestamp, float, str, str, bool, int]] = deque()
        merchant_counts: Counter[str] = Counter()
        device_counts: Counter[str] = Counter()
        long_country = 0
        long_night = 0
        amount_sum = 0.0
        amount_sq_sum = 0.0
        output: list[dict[str, object]] = []

        for row in work.itertuples(index=False):
            timestamp = row.timestamp
            amount = float(row.amount)

            short_cutoff = timestamp - pd.Timedelta(days=7)
            while short and short[0][0] <= short_cutoff:
                _, _, merchant, device = short.popleft()
                merchant_counts[merchant] -= 1
                device_counts[device] -= 1
                if merchant_counts[merchant] <= 0:
                    del merchant_counts[merchant]
                if device_counts[device] <= 0:
                    del device_counts[device]

            long_cutoff = timestamp - pd.Timedelta(days=30)
            while long and long[0][0] <= long_cutoff:
                _, old_amount, _, _, international, night = long.popleft()
                amount_sum -= old_amount
                amount_sq_sum -= old_amount * old_amount
                long_country -= int(international)
                long_night -= int(night)

            count_7d = len(short)
            count_30d = len(long)
            avg_30d = amount_sum / count_30d if count_30d else 0.0
            variance = amount_sq_sum / count_30d - avg_30d**2 if count_30d else 0.0
            std_30d = float(np.sqrt(max(variance, 0.0)))
            amount_vs_avg = amount / avg_30d if avg_30d > 0 else 1.0
            amount_zscore = (amount - avg_30d) / std_30d if std_30d > 0 else 0.0

            output.append(
                {
                    "row_id": row.row_id,
                    "customer_txn_count_7d": count_7d,
                    "customer_avg_amount_30d": avg_30d,
                    "customer_std_amount_30d": std_30d,
                    "customer_unique_merchants_7d": len(merchant_counts),
                    "customer_unique_devices_30d": len(device_counts),
                    "customer_international_rate_30d": long_country / count_30d if count_30d else 0.0,
                    "customer_night_txn_rate_30d": long_night / count_30d if count_30d else 0.0,
                    "amount_vs_customer_avg": amount_vs_avg,
                    "amount_zscore": amount_zscore,
                }
            )

            short.append((timestamp, amount, row.merchant_id, row.device_id))
            merchant_counts[row.merchant_id] += 1
            device_counts[row.device_id] += 1
            is_international = row.country != "US"
            is_night = bool(row.is_night)
            long.append((timestamp, amount, row.merchant_id, row.device_id, is_international, is_night))
            amount_sum += amount
            amount_sq_sum += amount * amount
            long_country += int(is_international)
            long_night += int(is_night)

        return pd.DataFrame(output)

    parts = [_features(group) for _, group in result.groupby("customer_id", sort=False)]
    features = pd.concat(parts, ignore_index=True).set_index("row_id")
    result[feature_cols] = features.reindex(result["row_id"])[feature_cols].to_numpy()
    return result.sort_values("row_id", kind="mergesort").drop(columns="row_id")
