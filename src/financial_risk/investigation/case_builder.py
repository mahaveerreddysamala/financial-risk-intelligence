"""Evidence-grounded investigation case assembly."""
from __future__ import annotations

from dataclasses import asdict, dataclass
from typing import Any

import pandas as pd


@dataclass(frozen=True)
class EvidenceItem:
    source: str
    field: str
    value: Any
    signal: str
    severity: str


@dataclass(frozen=True)
class InvestigationCase:
    transaction_id: str
    risk_score: float
    risk_band: str
    action: str
    evidence: tuple[EvidenceItem, ...]


def _severity_from_signal(signal: str) -> str:
    if signal in {"fraud_probability", "network_risk", "velocity_risk", "anomaly_score"}:
        return "high"
    return "medium"


def build_investigation_case(
    transaction: pd.Series | dict[str, Any],
    *,
    fraud_probability: float,
    anomaly_score: float,
    network_risk: float,
    velocity_risk: float,
    risk_score: float,
    risk_band: str,
    action: str,
) -> InvestigationCase:
    """Assemble a traceable case from observed transaction fields and model signals."""
    row = transaction.to_dict() if isinstance(transaction, pd.Series) else dict(transaction)
    transaction_id = str(row.get("transaction_id", "unknown"))
    signals = {
        "fraud_probability": fraud_probability,
        "anomaly_score": anomaly_score,
        "network_risk": network_risk,
        "velocity_risk": velocity_risk,
    }

    evidence: list[EvidenceItem] = []
    for field, value in signals.items():
        evidence.append(
            EvidenceItem(
                source="risk_engine",
                field=field,
                value=float(value),
                signal=field,
                severity=_severity_from_signal(field),
            )
        )

    observed_fields = (
        "amount",
        "country",
        "channel",
        "payment_method",
        "device_id",
        "ip_id",
        "merchant_id",
        "shared_device_account_count",
    )
    for field in observed_fields:
        if field in row and pd.notna(row[field]):
            evidence.append(
                EvidenceItem(
                    source="transaction",
                    field=field,
                    value=row[field],
                    signal="observed_attribute",
                    severity="context",
                )
            )

    return InvestigationCase(
        transaction_id=transaction_id,
        risk_score=float(risk_score),
        risk_band=str(risk_band),
        action=str(action),
        evidence=tuple(evidence),
    )


def case_to_dict(case: InvestigationCase) -> dict[str, Any]:
    """Serialize an investigation case without adding generated or inferred facts."""
    payload = asdict(case)
    payload["evidence"] = [asdict(item) for item in case.evidence]
    return payload
