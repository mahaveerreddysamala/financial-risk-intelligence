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
    """Generate model features from prior transaction history.

    The current transaction is scored before it is committed to history, so every
    feature is based only on observations available before that transaction.
    """

    def __init__(self, history_days: int = 30, state_store: Any | None = None) -> None:
        if history_days <= 0:
            raise ValueError("history_days must be greater than zero")
        self.history_days = history_days
        self._history: dict[str, list[dict[str, Any]]] = defaultdict(list)
        self._state_store = state_store

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

    def _get_history(self, customer_id: str) -> list[dict[str, Any]]:
        if self._state_store is not None:
            return self._state_store.get_history(customer_id)
        return self._history[customer_id]

    def prepare(self, event_payload: dict[str, Any], occurred_at: str) -> dict[str, Any]:
        """Build model features without mutating history."""
        current = self._base_row(event_payload, occurred_at)
        customer_id = current["customer_id"]
        current_time = current["timestamp"]
        cutoff = current_time - pd.Timedelta(days=self.history_days)
        prior = [row for row in self._get_history(customer_id) if self._timestamp(row["timestamp"]) >= cutoff]
        frame = pd.DataFrame([*prior, current])
        frame["timestamp"] = pd.to_datetime(frame["timestamp"], utc=True)
        frame = add_behavioral_features(frame)
        frame = add_velocity_features(frame)
        result = frame.iloc[-1]
        return {
            column: result[column].item() if hasattr(result[column], "item") else result[column]
            for column in MODEL_FEATURE_COLUMNS
        }

    def commit(self, event_payload: dict[str, Any], occurred_at: str) -> None:
        """Commit one successfully processed transaction to feature history."""
        current = self._base_row(event_payload, occurred_at)
        customer_id = current["customer_id"]
        cutoff = current["timestamp"] - pd.Timedelta(days=self.history_days)
        if self._state_store is not None and hasattr(self._state_store, "append_history"):
            self._state_store.append_history(
                customer_id,
                current,
                cutoff_timestamp=cutoff.isoformat(),
            )
            return
        history = [row for row in self._get_history(customer_id) if self._timestamp(row["timestamp"]) >= cutoff]
        history.append(current)
        history = [row for row in history if self._timestamp(row["timestamp"]) >= cutoff]
        if self._state_store is not None:
            self._state_store.set_history(customer_id, history)
        else:
            self._history[customer_id] = history
