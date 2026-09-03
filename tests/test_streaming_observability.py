from __future__ import annotations

from financial_risk.streaming.observability import StreamingMetrics


def test_snapshot_tracks_counters_and_latency() -> None:
    metrics = StreamingMetrics()
    metrics.increment("events_received", 3)
    metrics.increment("events_succeeded")
    metrics.observe_latency_ms(10.0)
    metrics.observe_latency_ms(20.0)

    snapshot = metrics.snapshot()

    assert snapshot["counters"] == {"events_received": 3, "events_succeeded": 1}
    assert snapshot["latency_ms"]["count"] == 2
    assert snapshot["latency_ms"]["min"] == 10.0
    assert snapshot["latency_ms"]["max"] == 20.0
    assert snapshot["latency_ms"]["avg"] == 15.0
    assert snapshot["latency_ms"]["p95"] == 20.0


def test_latency_timer_records_observation() -> None:
    metrics = StreamingMetrics()
    with metrics.time():
        pass

    assert metrics.snapshot()["latency_ms"]["count"] == 1


def test_prometheus_output_contains_runtime_metrics() -> None:
    metrics = StreamingMetrics()
    metrics.increment("events_received", 2)
    metrics.increment("risk_band_critical")
    metrics.observe_latency_ms(5.0)

    output = metrics.prometheus()

    assert "financial_risk_stream_events_received 2" in output
    assert "financial_risk_stream_risk_band_critical 1" in output
    assert "financial_risk_stream_processing_latency_ms_count 1" in output
    assert "financial_risk_stream_processing_latency_ms_p95 5.0" in output


def test_invalid_metric_inputs_are_rejected() -> None:
    metrics = StreamingMetrics()

    for call in (
        lambda: metrics.increment("", 1),
        lambda: metrics.increment("events", -1),
        lambda: metrics.observe_latency_ms(-1),
        lambda: metrics.prometheus(""),
    ):
        try:
            call()
        except ValueError:
            continue
        raise AssertionError("expected ValueError")
