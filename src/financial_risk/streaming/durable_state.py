"""Redis-backed durable state adapters for streaming inference."""
from __future__ import annotations

import json
from typing import Any


class RedisStateStore:
    """Small Redis adapter for customer feature history and idempotency keys."""

    def __init__(self, url: str, *, prefix: str = "financial-risk") -> None:
        if not url.strip():
            raise ValueError("url must not be empty")
        if not prefix.strip():
            raise ValueError("prefix must not be empty")
        try:
            import redis
        except ImportError as exc:  # pragma: no cover - dependency boundary
            raise RuntimeError("redis package is required for Redis-backed state") from exc
        self._client = redis.Redis.from_url(url, decode_responses=True)
        self._prefix = prefix

    def ping(self) -> bool:
        """Verify connectivity to Redis."""
        return bool(self._client.ping())

    def contains(self, event_id: str) -> bool:
        """Return whether an event has already been completed."""
        return bool(self._client.exists(self._key("idempotency", event_id)))

    def mark(self, event_id: str, *, ttl_seconds: int = 86_400) -> None:
        """Mark an event complete with a bounded retention period."""
        if ttl_seconds <= 0:
            raise ValueError("ttl_seconds must be greater than zero")
        self._client.set(self._key("idempotency", event_id), "1", ex=ttl_seconds)

    def get_history(self, customer_id: str) -> list[dict[str, Any]]:
        """Read one customer's serialized transaction history."""
        value = self._client.get(self._key("customer", customer_id))
        if value is None:
            return []
        payload = json.loads(value)
        if not isinstance(payload, list):
            raise ValueError("stored customer history must be a list")
        return [dict(item) for item in payload if isinstance(item, dict)]

    def set_history(self, customer_id: str, history: list[dict[str, Any]]) -> None:
        """Persist one customer's transaction history."""
        self._client.set(
            self._key("customer", customer_id),
            json.dumps(history, separators=(",", ":"), default=str),
        )

    def _key(self, namespace: str, value: str) -> str:
        return f"{self._prefix}:{namespace}:{value}"


class RedisIdempotencyStore:
    """Idempotency-only Redis adapter compatible with the runtime contract."""

    def __init__(self, state: RedisStateStore, *, ttl_seconds: int = 86_400) -> None:
        if ttl_seconds <= 0:
            raise ValueError("ttl_seconds must be greater than zero")
        self._state = state
        self._ttl_seconds = ttl_seconds

    def contains(self, event_id: str) -> bool:
        """Return whether the event is already marked complete."""
        return self._state.contains(event_id)

    def mark(self, event_id: str) -> None:
        """Mark the event complete."""
        self._state.mark(event_id, ttl_seconds=self._ttl_seconds)
