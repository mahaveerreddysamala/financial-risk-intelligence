"""Reliable, transport-agnostic runtime primitives for streaming inference."""
from __future__ import annotations

import logging
import time
from collections.abc import Callable
from dataclasses import asdict, dataclass, field
from datetime import UTC, datetime
from typing import Protocol

from financial_risk.streaming.events import EventEnvelope
from financial_risk.streaming.risk_consumer import RiskScoringResult


class EventConsumer(Protocol):
    """Minimal consumer contract required by the streaming runtime."""

    def poll(self, timeout: float = 1.0) -> EventEnvelope | None:
        """Return one event or None when no event is available."""

    def close(self) -> None:
        """Release consumer resources."""


class IdempotencyStore(Protocol):
    """Minimal contract for duplicate-event suppression."""

    def contains(self, event_id: str) -> bool:
        """Return whether an event has already been completed."""

    def mark(self, event_id: str) -> None:
        """Record an event as successfully completed."""


@dataclass
class InMemoryIdempotencyStore:
    """Simple process-local idempotency store for tests and local runs."""

    _event_ids: set[str] = field(default_factory=set)

    def contains(self, event_id: str) -> bool:
        """Return whether the event has already been completed."""
        return event_id in self._event_ids

    def mark(self, event_id: str) -> None:
        """Mark an event as successfully completed."""
        self._event_ids.add(event_id)


@dataclass(frozen=True)
class RetryPolicy:
    """Bounded retry configuration for transient stream-processing failures."""

    max_attempts: int = 3
    backoff_seconds: float = 1.0

    def __post_init__(self) -> None:
        if self.max_attempts < 1:
            raise ValueError("max_attempts must be at least 1")
        if self.backoff_seconds < 0:
            raise ValueError("backoff_seconds must be non-negative")


@dataclass(frozen=True)
class DeadLetterRecord:
    """Structured failure record suitable for publishing to a DLQ topic."""

    event_id: str
    event_type: str
    attempts: int
    error_type: str
    error_message: str
    failed_at: str
    event: EventEnvelope

    def to_dict(self) -> dict[str, object]:
        """Return a JSON-serializable dead-letter record."""
        payload = asdict(self.event)
        return {
            "event_id": self.event_id,
            "event_type": self.event_type,
            "attempts": self.attempts,
            "error_type": self.error_type,
            "error_message": self.error_message,
            "failed_at": self.failed_at,
            "event": payload,
        }


@dataclass
class StreamingStats:
    """Operational counters for one long-running consumer process."""

    received: int = 0
    succeeded: int = 0
    retried: int = 0
    duplicates: int = 0
    dead_lettered: int = 0


class StreamingRuntime:
    """Run reliable event processing independently of Kafka transport details."""

    def __init__(
        self,
        *,
        consumer: EventConsumer,
        processor: Callable[[EventEnvelope], RiskScoringResult],
        publisher: Callable[[RiskScoringResult], None] | None = None,
        dead_letter: Callable[[DeadLetterRecord], None] | None = None,
        idempotency: IdempotencyStore | None = None,
        retry_policy: RetryPolicy | None = None,
        sleep: Callable[[float], None] = time.sleep,
        logger: logging.Logger | None = None,
    ) -> None:
        self.consumer = consumer
        self.processor = processor
        self.publisher = publisher
        self.dead_letter = dead_letter
        self.idempotency = idempotency or InMemoryIdempotencyStore()
        self.retry_policy = retry_policy or RetryPolicy()
        self.sleep = sleep
        self.logger = logger or logging.getLogger(__name__)
        self.stats = StreamingStats()

    def process_event(self, event: EventEnvelope) -> bool:
        """Process one event, retry failures, and route exhausted failures to DLQ."""
        self.stats.received += 1

        if self.idempotency.contains(event.event_id):
            self.stats.duplicates += 1
            self.logger.info("stream_event_duplicate event_id=%s", event.event_id)
            return False

        last_error: Exception | None = None
        for attempt in range(1, self.retry_policy.max_attempts + 1):
            try:
                result = self.processor(event)
                if self.publisher is not None:
                    self.publisher(result)
                self.idempotency.mark(event.event_id)
                self.stats.succeeded += 1
                self.logger.info(
                    "stream_event_succeeded event_id=%s attempt=%s",
                    event.event_id,
                    attempt,
                )
                return True
            except (OSError, RuntimeError, TypeError, ValueError) as exc:
                last_error = exc
                if attempt < self.retry_policy.max_attempts:
                    self.stats.retried += 1
                    self.logger.warning(
                        "stream_event_retry event_id=%s attempt=%s error=%s",
                        event.event_id,
                        attempt,
                        exc,
                    )
                    self.sleep(self.retry_policy.backoff_seconds)
                    continue
                break

        assert last_error is not None
        record = DeadLetterRecord(
            event_id=event.event_id,
            event_type=event.event_type,
            attempts=self.retry_policy.max_attempts,
            error_type=type(last_error).__name__,
            error_message=str(last_error),
            failed_at=datetime.now(UTC).isoformat(),
            event=event,
        )
        if self.dead_letter is not None:
            self.dead_letter(record)
        self.stats.dead_lettered += 1
        self.logger.error(
            "stream_event_dead_lettered event_id=%s attempts=%s error=%s",
            event.event_id,
            record.attempts,
            record.error_message,
        )
        return False

    def run_once(self, *, timeout: float = 1.0) -> bool:
        """Poll one event and process it; return False when the poll times out."""
        event = self.consumer.poll(timeout=timeout)
        if event is None:
            return False
        self.process_event(event)
        return True

    def run_forever(self, *, timeout: float = 1.0) -> None:
        """Continuously poll and process events until the caller stops the runtime."""
        while True:
            self.run_once(timeout=timeout)
