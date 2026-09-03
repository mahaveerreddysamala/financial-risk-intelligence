"""Operational metrics for long-running streaming inference."""
from __future__ import annotations

import time
from collections import Counter
from dataclasses import dataclass, field
from threading import Lock
from typing import Any


@dataclass
class StreamingMetrics:
    """Thread-safe in-process metrics for streaming operations."""

    _counts: Counter[str] = field(default_factory=Counter)
    _latencies_ms: list[float] = field(default_factory=list)
    _lock: Lock = field(default_factory=Lock)

    def increment(self, name: str, value: int = 1) -> None:
        """Increment a named counter."""
        if not name.strip():
            raise ValueError("metric name must not be empty")
        if value < 0:
            raise ValueError("counter increment must be non-negative")
        with self._lock:
            self._counts[name] += value

    def observe_latency_ms(self, value: float) -> None:
        """Record one non-negative processing-latency observation."""
        if value < 0:
            raise ValueError("latency must be non-negative")
        with self._lock:
            self._latencies_ms.append(float(value))

    def time(self) -> "LatencyTimer":
        """Return a context manager that records elapsed latency."""
        return LatencyTimer(self)

    def snapshot(self) -> dict[str, Any]:
        """Return a point-in-time metrics snapshot."""
        with self._lock:
            latencies = sorted(self._latencies_ms)
            counts = dict(self._counts)
        return {
            "counters": counts,
            "latency_ms": {
                "count": len(latencies),
                "min": latencies[0] if latencies else 0.0,
                "max": latencies[-1] if latencies else 0.0,
                "avg": sum(latencies) / len(latencies) if latencies else 0.0,
                "p95": _percentile(latencies, 0.95),
            },
        }

    def prometheus(self, prefix: str = "financial_risk_stream") -> str:
        """Render counters and latency summary as Prometheus text exposition."""
        if not prefix.strip():
            raise ValueError("prefix must not be empty")
        snapshot = self.snapshot()
        lines: list[str] = []
        for name, value in sorted(snapshot["counters"].items()):
            metric_name = _metric_name(prefix, name)
            lines.append(f"{metric_name} {value}")
        latency = snapshot["latency_ms"]
        lines.extend(
            [
                f"{prefix}_processing_latency_ms_count {latency['count']}",
                f"{prefix}_processing_latency_ms_sum "
                f"{latency['avg'] * latency['count']}",
                f"{prefix}_processing_latency_ms_max {latency['max']}",
                f"{prefix}_processing_latency_ms_p95 {latency['p95']}",
            ]
        )
        return "\n".join(lines) + "\n"


class LatencyTimer:
    """Context manager used to measure elapsed processing time."""

    def __init__(self, metrics: StreamingMetrics) -> None:
        self._metrics = metrics
        self._started = 0.0

    def __enter__(self) -> "LatencyTimer":
        self._started = time.perf_counter()
        return self

    def __exit__(self, exc_type: object, exc: object, traceback: object) -> None:
        elapsed_ms = (time.perf_counter() - self._started) * 1000.0
        self._metrics.observe_latency_ms(elapsed_ms)


def _percentile(values: list[float], quantile: float) -> float:
    """Return the nearest-rank percentile for a sorted list."""
    if not values:
        return 0.0
    index = max(0, min(len(values) - 1, int(round((len(values) - 1) * quantile))))
    return values[index]


def _metric_name(prefix: str, name: str) -> str:
    """Normalize an application counter name for Prometheus."""
    normalized = "".join(char if char.isalnum() or char == "_" else "_" for char in name)
    return f"{prefix}_{normalized}"
