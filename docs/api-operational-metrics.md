# API Operational Metrics

Phase 26 exposes a Prometheus-compatible `/metrics` endpoint for the FastAPI service.

## Signals

The endpoint reports in-process counters for total API requests, failed requests, and HTTP status codes, plus processing-latency summary metrics including count, sum, maximum, and p95.

The metrics layer is intentionally dependency-light and reuses the thread-safe `StreamingMetrics` implementation already used by the streaming runtime.

## Endpoint

```text
GET /metrics
```

The response uses Prometheus text exposition media type:

```text
text/plain; version=0.0.4; charset=utf-8
```

Example metric names include:

```text
financial_risk_api_api_requests_total
financial_risk_api_api_requests_failed
financial_risk_api_api_status_200
financial_risk_api_processing_latency_ms_count
financial_risk_api_processing_latency_ms_p95
```

The `/metrics` endpoint does not increment request counters for itself, preventing self-scrape traffic from distorting the application request count.

## Scope

These metrics are local process metrics. They provide a production-style integration point for a Prometheus scraper but are not a distributed metrics store. Multi-instance aggregation should be handled by the monitoring platform.
