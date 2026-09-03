"""FastAPI application exposing portfolio risk and investigation workflows."""
from __future__ import annotations

import time
from contextlib import nullcontext
from typing import Any

from fastapi import FastAPI, HTTPException, Request
from fastapi.responses import PlainTextResponse
from pydantic import BaseModel, Field

from financial_risk.api.config import settings
from financial_risk.api.observability import configure_logging
from financial_risk.investigation.case_builder import build_investigation_case, case_to_dict
from financial_risk.investigation.copilot import (
    RetrievalResult,
    build_copilot_context,
    build_grounded_prompt,
)
from financial_risk.models.risk_score import combine_risk_signals, decision_from_score
from financial_risk.streaming.observability import StreamingMetrics

logger = configure_logging(settings.log_level)
api_metrics = StreamingMetrics()

app = FastAPI(
    title="Financial Crime & Risk Intelligence API",
    version=settings.app_version,
    description="REST interface for risk scoring, investigation cases, and grounded copilot context.",
)


class RiskRequest(BaseModel):
    fraud_probability: float = Field(ge=0, le=1)
    anomaly_score: float = Field(ge=0, le=1)
    network_score: float = Field(ge=0, le=1)
    velocity_score: float = Field(ge=0, le=1)


class InvestigationRequest(RiskRequest):
    transaction: dict[str, Any]


class CopilotRequest(BaseModel):
    case_id: str = Field(min_length=1)
    evidence: list[dict[str, Any]] = Field(default_factory=list)
    references: list[dict[str, Any]] = Field(default_factory=list)


def _risk_payload(request: RiskRequest) -> dict[str, Any]:
    score = combine_risk_signals(
        fraud_probability=request.fraud_probability,
        anomaly_score=request.anomaly_score,
        network_score=request.network_score,
        velocity_score=request.velocity_score,
    )
    decision = decision_from_score(score)
    return {
        "risk_score": score,
        "risk_band": decision.level,
        "action": decision.action,
    }


@app.middleware("http")
async def request_logging(request: Request, call_next: Any) -> Any:
    """Log request metadata and collect operational metrics without bodies."""
    started = time.perf_counter()
    collect_metrics = request.url.path != "/metrics"
    if collect_metrics:
        api_metrics.increment("api_requests_total")

    try:
        with api_metrics.time() if collect_metrics else nullcontext():
            response = await call_next(request)
    except Exception:
        elapsed_ms = round((time.perf_counter() - started) * 1000, 2)
        if collect_metrics:
            api_metrics.increment("api_requests_failed")
        logger.exception(
            "request_failed",
            extra={"method": request.method, "path": request.url.path, "duration_ms": elapsed_ms},
        )
        raise
    elapsed_ms = round((time.perf_counter() - started) * 1000, 2)
    if collect_metrics:
        api_metrics.increment(f"api_status_{response.status_code}")
    logger.info(
        "request_completed",
        extra={
            "method": request.method,
            "path": request.url.path,
            "status_code": response.status_code,
            "duration_ms": elapsed_ms,
        },
    )
    return response


@app.get("/health")
def health() -> dict[str, str]:
    return {"status": "ok"}


@app.get("/ready")
def ready() -> dict[str, str]:
    """Report application readiness for container orchestration checks."""
    return {"status": "ready"}


@app.get("/version")
def version() -> dict[str, str]:
    return {"version": app.version, "environment": settings.app_env}


@app.get("/metrics", response_class=PlainTextResponse)
def metrics() -> PlainTextResponse:
    """Expose application metrics in Prometheus text exposition format."""
    return PlainTextResponse(
        api_metrics.prometheus(prefix="financial_risk_api"),
        media_type="text/plain; version=0.0.4; charset=utf-8",
    )


@app.post("/v1/risk/score")
def score_risk(request: RiskRequest) -> dict[str, Any]:
    return _risk_payload(request)


@app.post("/v1/investigations/cases")
def create_investigation_case(request: InvestigationRequest) -> dict[str, Any]:
    risk = _risk_payload(request)
    case = build_investigation_case(
        request.transaction,
        fraud_probability=request.fraud_probability,
        anomaly_score=request.anomaly_score,
        network_risk=request.network_score,
        velocity_risk=request.velocity_score,
        risk_score=risk["risk_score"],
        risk_band=risk["risk_band"],
        action=risk["action"],
    )
    return case_to_dict(case)


@app.post("/v1/copilot/prompt")
def build_copilot_prompt(request: CopilotRequest) -> dict[str, Any]:
    references = []
    for item in request.references:
        if not {"document_id", "score", "text"}.issubset(item):
            raise HTTPException(status_code=422, detail="references require document_id, score, and text")
        references.append(
            RetrievalResult(
                document_id=str(item["document_id"]),
                score=float(item["score"]),
                text=str(item["text"]),
            )
        )
    context = build_copilot_context(request.case_id, request.evidence, references)
    return {"case_id": request.case_id, "grounded_prompt": build_grounded_prompt(context)}
