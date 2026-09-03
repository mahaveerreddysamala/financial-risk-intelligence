# Phase 25 — Streaming Observability

Phase 25 adds an explicit observability boundary around the long-running streaming runtime.

## Metrics model

`StreamingMetrics` keeps thread-safe in-process counters and processing-latency observations.

Tracked runtime counters include:

- `events_received`
- `events_succeeded`
- `events_failed`
- `events_retried`
- `events_duplicates`
- `events_dead_lettered`
- `poll_timeouts`
- `risk_band_low`
- `risk_band_medium`
- `risk_band_high`
- `risk_band_critical`

Latency observations expose count, minimum, maximum, average, and p95 values through the snapshot API.

## Prometheus boundary

`StreamingMetrics.prometheus()` emits a simple Prometheus text-exposition payload. This keeps the metrics representation independent of a specific metrics server or dashboard product.

Example:

```text
financial_risk_stream_events_received 100
financial_risk_stream_events_succeeded 98
financial_risk_stream_events_dead_lettered 2
financial_risk_stream_processing_latency_ms_count 98
financial_risk_stream_processing_latency_ms_p95 42.5
```

Values above are illustrative only; the repository does not claim these figures as measured production performance.

## Runtime integration

The Phase 24 `StreamingRuntime` records metrics while retaining its existing retry, idempotency, and dead-letter behavior. Processing latency is measured around the processor call, while outcome and risk-band counters are recorded after successful processing.

The metrics object is injectable, so tests or a future deployment can provide a compatible boundary without changing the event or Kafka contracts.

## Production boundary

The current implementation is deliberately lightweight:

- Metrics live in the process memory.
- There is no external Prometheus server in the repository.
- There is no Grafana dashboard yet.
- Metrics are not durable across process restarts.

A production deployment can expose the rendered metrics through an HTTP endpoint or replace this in-process representation with a metrics client while preserving the runtime contract.

## Validation

Phase 25 should be considered complete only after the full repository lint and test suite passes with the new observability tests included.
