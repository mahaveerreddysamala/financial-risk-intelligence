"""In-memory investigation case lifecycle store for the portfolio API."""
from __future__ import annotations

from dataclasses import dataclass, field
from datetime import UTC, datetime
from threading import RLock
from typing import Any
from uuid import uuid4


VALID_STATUSES = {"OPEN", "IN_REVIEW", "ESCALATED", "RESOLVED", "DISMISSED"}
VALID_TRANSITIONS = {
    "OPEN": {"IN_REVIEW", "ESCALATED", "DISMISSED"},
    "IN_REVIEW": {"ESCALATED", "RESOLVED", "DISMISSED"},
    "ESCALATED": {"IN_REVIEW", "RESOLVED", "DISMISSED"},
    "RESOLVED": set(),
    "DISMISSED": set(),
}


@dataclass
class AuditEvent:
    event_id: str
    event_type: str
    timestamp: str
    actor: str
    details: dict[str, Any] = field(default_factory=dict)


@dataclass
class StoredCase:
    case_id: str
    transaction_id: str
    risk_score: float
    risk_band: str
    action: str
    evidence: list[dict[str, Any]]
    status: str
    created_at: str
    updated_at: str
    audit_log: list[AuditEvent] = field(default_factory=list)

    def to_dict(self, *, include_audit: bool = False) -> dict[str, Any]:
        payload = {
            "case_id": self.case_id,
            "transaction_id": self.transaction_id,
            "risk_score": self.risk_score,
            "risk_band": self.risk_band,
            "action": self.action,
            "evidence": self.evidence,
            "status": self.status,
            "created_at": self.created_at,
            "updated_at": self.updated_at,
        }
        if include_audit:
            payload["audit_log"] = [event.__dict__ for event in self.audit_log]
        return payload


class InvestigationCaseStore:
    """Thread-safe store with idempotent creation and explicit state transitions."""

    def __init__(self) -> None:
        self._cases: dict[str, StoredCase] = {}
        self._idempotency: dict[str, str] = {}
        self._lock = RLock()

    @staticmethod
    def _now() -> str:
        return datetime.now(UTC).isoformat()

    def create(self, payload: dict[str, Any], *, idempotency_key: str | None = None, actor: str = "api") -> tuple[StoredCase, bool]:
        with self._lock:
            if idempotency_key:
                existing_id = self._idempotency.get(idempotency_key)
                if existing_id:
                    return self._cases[existing_id], True

            now = self._now()
            case = StoredCase(
                case_id=f"CASE-{uuid4().hex[:12].upper()}",
                transaction_id=str(payload["transaction_id"]),
                risk_score=float(payload["risk_score"]),
                risk_band=str(payload["risk_band"]),
                action=str(payload["action"]),
                evidence=list(payload["evidence"]),
                status="OPEN",
                created_at=now,
                updated_at=now,
            )
            case.audit_log.append(
                AuditEvent(uuid4().hex, "CASE_CREATED", now, actor, {"risk_band": case.risk_band})
            )
            self._cases[case.case_id] = case
            if idempotency_key:
                self._idempotency[idempotency_key] = case.case_id
            return case, False

    def get(self, case_id: str) -> StoredCase | None:
        with self._lock:
            return self._cases.get(case_id)

    def list(
        self,
        *,
        status: str | None = None,
        risk_band: str | None = None,
        transaction_id: str | None = None,
        offset: int = 0,
        limit: int = 50,
    ) -> tuple[list[StoredCase], int]:
        with self._lock:
            cases = list(self._cases.values())
            if status:
                cases = [case for case in cases if case.status == status]
            if risk_band:
                cases = [case for case in cases if case.risk_band == risk_band]
            if transaction_id:
                cases = [case for case in cases if case.transaction_id == transaction_id]
            cases.sort(key=lambda case: case.created_at, reverse=True)
            total = len(cases)
            return cases[offset : offset + limit], total

    def transition(self, case_id: str, new_status: str, *, actor: str = "api", note: str | None = None) -> StoredCase | None:
        with self._lock:
            case = self._cases.get(case_id)
            if case is None:
                return None
            if new_status not in VALID_STATUSES:
                raise ValueError(f"unsupported status: {new_status}")
            if new_status not in VALID_TRANSITIONS[case.status]:
                raise ValueError(f"invalid transition: {case.status} -> {new_status}")
            previous = case.status
            case.status = new_status
            case.updated_at = self._now()
            details: dict[str, Any] = {"from": previous, "to": new_status}
            if note:
                details["note"] = note
            case.audit_log.append(AuditEvent(uuid4().hex, "STATUS_CHANGED", case.updated_at, actor, details))
            return case

    def audit(self, case_id: str) -> list[AuditEvent] | None:
        with self._lock:
            case = self._cases.get(case_id)
            return None if case is None else list(case.audit_log)


case_store = InvestigationCaseStore()
