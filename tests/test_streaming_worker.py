from __future__ import annotations

from dataclasses import dataclass

import pytest

from financial_risk.streaming.events import EventEnvelope
from financial_risk.streaming.risk_consumer import RiskScoringResult
from financial_risk.streaming.worker import process_event


def _event() -> EventEnvelope:
    return EventEnvelope(
        event_id="evt-1",
        event_type="transaction.created",
        schema_version=1,
        occurred_at="2026-09-02T12:00:00+00:00",
        payload={
            "transaction_id": "TXN-1",
            "fraud_probability": 0.90,
            "anomaly_score": 0.80,
            "network_risk": 0.70,
            "velocity_risk": 0.60,
        },
    )


def test_process_event_scores_and_publishes() -> None:
    captured: list[RiskScoringResult] = []
    result = process_event(_event(), publish=captured.append)

    assert result.risk_band == "CRITICAL"
    assert result.action == "hold_and_investigate"
    assert captured == [result]


def test_process_event_returns_result_without_publisher() -> None:
    result = process_event(_event(), publish=None)
    assert result.transaction_id == "TXN-1"
    assert result.risk_score == pytest.approx(0.80)


def test_process_event_requires_publish_callable_when_provided() -> None:
    with pytest.raises(TypeError, match="publish"):
        process_event(_event(), publish=object())
