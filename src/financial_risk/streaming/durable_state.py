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

    def claim(self, event_id: str, *, ttl_seconds: int = 120) -> bool:
        """Atomically claim an event with a lease that expires after a crash."""
        if ttl_seconds <= 0:
            raise ValueError("ttl_seconds must be greater than zero")
        if self.contains(event_id):
            return False
        return bool(
            self._client.set(
                self._key("claim", event_id),
                "1",
                ex=ttl_seconds,
                nx=True,
            )
        )

    def mark(self, event_id: str, *, ttl_seconds: int = 86_400) -> None:
        """Mark an event complete with a bounded retention period."""
        if ttl_seconds <= 0:
            raise ValueError("ttl_seconds must be greater than zero")
        pipeline = self._client.pipeline(transaction=True)
        pipeline.set(self._key("idempotency", event_id), "1", ex=ttl_seconds)
        pipeline.delete(self._key("claim", event_id))
        pipeline.execute()

    def release(self, event_id: str) -> None:
        """Release an in-progress claim after a failed processing attempt."""
        self._client.delete(self._key("claim", event_id))

    def get_history(self, customer_id: str) -> list[dict[str, Any]]:
        """Read one customer's serialized transaction history."""
        value = self._client.get(self._key("customer", customer_id))
        if value is None:
            return []
        payload = json.loads(value)
        if not isinstance(payload, list):
            raise TypeError("stored customer history must be a list")
        return [dict(item) for item in payload if isinstance(item, dict)]

    def set_history(self, customer_id: str, history: list[dict[str, Any]]) -> None:
        """Persist one customer's transaction history."""
        self._client.set(
            self._key("customer", customer_id),
            json.dumps(history, separators=(",", ":"), default=str),
        )

    def append_history(
        self,
        customer_id: str,
        transaction: dict[str, Any],
        *,
        cutoff_timestamp: str,
    ) -> list[dict[str, Any]]:
        """Atomically append one transaction to customer history with optimistic locking."""
        import redis

        key = self._key("customer", customer_id)
        for _ in range(5):
            with self._client.pipeline() as pipeline:
                try:
                    pipeline.watch(key)
                    value = pipeline.get(key)
                    history = [] if value is None else json.loads(value)
                    if not isinstance(history, list):
                        raise TypeError("stored customer history must be a list")
                    history = [
                        dict(item)
                        for item in history
                        if isinstance(item, dict)
                        and str(item.get("timestamp", "")) >= cutoff_timestamp
                    ]
                    history.append(dict(transaction))
                    pipeline.multi()
                    pipeline.set(
                        key,
                        json.dumps(history, separators=(",", ":"), default=str),
                    )
                    pipeline.execute()
                    return history
                except redis.WatchError:
                    continue
        raise RuntimeError(f"concurrent history update failed for customer {customer_id}")

    def _key(self, namespace: str, value: str) -> str:
        return f"{self._prefix}:{namespace}:{value}"


class RedisIdempotencyStore:
    """Idempotency adapter compatible with the streaming runtime contract."""

    def __init__(self, state: RedisStateStore, *, ttl_seconds: int = 86_400) -> None:
        if ttl_seconds <= 0:
            raise ValueError("ttl_seconds must be greater than zero")
        self._state = state
        self._ttl_seconds = ttl_seconds
        self._claim_ttl_seconds = 120

    def contains(self, event_id: str) -> bool:
        """Return whether the event is already marked complete."""
        return self._state.contains(event_id)

    def claim(self, event_id: str) -> bool:
        """Atomically reserve an event for one worker."""
        return self._state.claim(event_id, ttl_seconds=self._claim_ttl_seconds)

    def mark(self, event_id: str) -> None:
        """Mark the event complete and clear its processing lease."""
        self._state.mark(event_id, ttl_seconds=self._ttl_seconds)

    def release(self, event_id: str) -> None:
        """Release the event processing lease."""
        self._state.release(event_id)
