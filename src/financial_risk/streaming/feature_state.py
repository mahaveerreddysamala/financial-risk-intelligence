"""Stateful prior-only feature generation for real-time transaction events."""
from __future__ import annotations

from collections import defaultdict
from datetime import UTC
from typing import Any

import pandas as pd

from financial_risk.features.behavioral import add_behavioral_features
from financial_risk.features.velocity import add_velocity_features


MODEL_FEATURE_COLUMNS = [
    "amount",
    "is_international",
    "is_night",
    "shared_device_account_count",
    "customer_txn_count_7d",
    "customer_avg_amount_30d",
    "customer_std_amount_30d",
    "customer_unique_merchants_7d",
    "customer_unique_devices_30d",
    "customer_international_rate_30d",
    "customer_night_txn_rate_30d",
    "amount_vs_customer_avg",
    "amount_zscore",
    "txn_count_5m",
    "txn_count_1h",
    "txn_count_24h",
    "merchant_category",
    "payment_method",
    "channel",
    "country",
]


class StreamingFeatureService:
    """Generate model features from prior transaction history in memory.

    The current transaction is scored before it is committed to history, so every
    feature is based only on observations available before that transaction.
    """

    def __init__(self, history_days: int = 30) -> None:
        if history_days <= 0:
            raise ValueError("history_days must be greater than zero")
        self.history_days = history_days
        self._history: dict[str, list[dict[str, Any]]] = defaultdict(list)

    @staticmethod
    def _timestamp(value: Any) -> pd.Timestamp:
        timestamp = pd.Timestamp(value)
        if timestamp.tzinfo is None:
            timestamp = timestamp.tz_localize(UTC)
        else:
            timestamp = timestamp.tz_convert(UTC)
        return timestamp

    def _base_row(self, event_payload: dict[str, Any], occurred_at: str) -> dict[str, Any]:
        required = {
            "transaction_id",
            "customer_id",
            "amount",
            "merchant_id",
            "device_id",
            "country",
            "shared_device_account_count",
            "merchant_category",
            "payment_method",
            "channel",
        }
        missing = sorted(required.difference(event_payload))
        if missing:
            raise ValueError(f"real-time feature generation missing fields: {missing}")

        timestamp = self._timestamp(occurred_at)
        hour = timestamp.hour
        is_night = int(event_payload.get("is_night", hour < 5 or hour >= 23))
        country = str(event_payload["country"])
        return {
            "transaction_id": str(event_payload["transaction_id"]),
            "customer_id": str(event_payload["customer_id"]),
            "timestamp": timestamp,
            "amount": float(event_payload["amount"]),
            "merchant_id": str(event_payload["merchant_id"]),
            "device_id": str(event_payload["device_id"]),
            "country": country,
            "is_night": is_night,
            "is_international": int(event_payload.get("is_international", country != "US")),
            "shared_device_account_count": int(event_payload["shared_device_account_count"]),
            "merchant_category": str(event_payload["merchant_category"]),
            "payment_method": str(event_payload["payment_method"]),
            "channel": str(event_payload["channel"]),
        }

    def prepare(self, event_payload: dict[str, Any], occurred_at: str) -> dict[str, Any]:
        """Build model features without mutating history."""
        current = self._base_row(event_payload, occurred_at)
        customer_id = current["customer_id"]
        current_time = current["timestamp"]
        cutoff = current_time - pd.Timedelta(days=self.history_days)
        prior = [row for row in self._history[customer_id] if row["timestamp"] >= cutoff]
        frame = pd.DataFrame([*prior, current])
        frame = add_behavioral_features(frame)
        frame = add_velocity_features(frame)
        result = frame.iloc[-1]
        return {column: result[column].item() if hasattr(result[column], "item") else result[column] for column in MODEL_FEATURE_COLUMNS}

    def commit(self, event_payload: dict[str, Any], occurred_at: str) -> None:
        """Commit one successfully processed transaction to feature history."""
        current = self._base_row(event_payload, occurred_at)
        customer_id = current["customer_id"]
        cutoff = current["timestamp"] - pd.Timedelta(days=self.history_days)
        history = self._history[customer_id]
        history.append(current)
        self._history[customer_id] = [row for row in history if row["timestamp"] >= cutoff]
