"""Versioned event envelopes for financial risk streaming."""
from __future__ import annotations

import json
from dataclasses import asdict, dataclass
from datetime import UTC, datetime
from typing import Any


@dataclass(frozen=True)
class EventEnvelope:
    """Transport-neutral envelope for a financial risk event."""

    event_id: str
    event_type: str
    schema_version: int
    occurred_at: str
    payload: dict[str, Any]

    def to_dict(self) -> dict[str, Any]:
        """Return a JSON-serializable envelope."""
        return asdict(self)

    def to_json(self) -> str:
        """Serialize the envelope to compact JSON."""
        return json.dumps(self.to_dict(), separators=(",", ":"), sort_keys=True, default=str)

    @classmethod
    def from_json(cls, value: str | bytes) -> EventEnvelope:
        """Deserialize and validate one event envelope."""
        try:
            payload = json.loads(value)
        except (TypeError, json.JSONDecodeError) as exc:
            raise ValueError("event envelope must contain valid JSON") from exc
        if not isinstance(payload, dict):
            raise ValueError("event envelope must be a JSON object")
        required = {"event_id", "event_type", "schema_version", "occurred_at", "payload"}
        missing = required.difference(payload)
        if missing:
            raise ValueError(f"event envelope missing fields: {sorted(missing)}")
        if not isinstance(payload["payload"], dict):
            raise ValueError("event envelope payload must be an object")
        schema_version = int(payload["schema_version"])
        if schema_version < 1:
            raise ValueError("schema_version must be >= 1")
        return cls(
            event_id=str(payload["event_id"]),
            event_type=str(payload["event_type"]),
            schema_version=schema_version,
            occurred_at=str(payload["occurred_at"]),
            payload=dict(payload["payload"]),
        )


def serialize_event(
    event_id: str,
    event_type: str,
    payload: dict[str, Any],
    *,
    schema_version: int = 1,
    occurred_at: datetime | None = None,
) -> bytes:
    """Build a versioned event and serialize it for Kafka transport."""
    if not event_id.strip():
        raise ValueError("event_id must not be empty")
    if not event_type.strip():
        raise ValueError("event_type must not be empty")
    if schema_version < 1:
        raise ValueError("schema_version must be >= 1")
    timestamp = occurred_at or datetime.now(UTC)
    if timestamp.tzinfo is None:
        raise ValueError("occurred_at must be timezone-aware")
    envelope = EventEnvelope(
        event_id=event_id,
        event_type=event_type,
        schema_version=schema_version,
        occurred_at=timestamp.astimezone(UTC).isoformat(),
        payload=dict(payload),
    )
    return envelope.to_json().encode("utf-8")
