"""FastAPI application exposing portfolio risk and investigation workflows."""
from __future__ import annotations

from typing import Any

from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field

from financial_risk.investigation.case_builder import build_investigation_case, case_to_dict
from financial_risk.investigation.copilot import (
    build_copilot_context,
    build_grounded_prompt,
)
from financial_risk.models.risk_score import combine_risk_signals, decision_from_score

app = FastAPI(
    title="Financial Crime & Risk Intelligence API",
    version="0.1.0",
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


@app.get("/health")
def health() -> dict[str, str]:
    return {"status": "ok"}


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
        from financial_risk.investigation.copilot import RetrievalResult

        references.append(
            RetrievalResult(
                document_id=str(item["document_id"]),
                score=float(item["score"]),
                text=str(item["text"]),
            )
        )
    context = build_copilot_context(request.case_id, request.evidence, references)
    return {"case_id": request.case_id, "grounded_prompt": build_grounded_prompt(context)}
