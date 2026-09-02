import pandas as pd

from financial_risk.investigation.case_builder import (
    EvidenceItem,
    InvestigationCase,
    build_investigation_case,
    case_to_dict,
)


def test_build_investigation_case_preserves_evidence():
    transaction = pd.Series(
        {
            "transaction_id": "TXN000000001",
            "amount": 250.0,
            "country": "GB",
            "channel": "ecommerce",
            "payment_method": "credit",
            "device_id": "D1",
            "ip_id": "IP1",
            "merchant_id": "M1",
            "shared_device_account_count": 6,
        }
    )
    case = build_investigation_case(
        transaction,
        fraud_probability=0.91,
        anomaly_score=0.82,
        network_risk=0.74,
        velocity_risk=0.66,
        risk_score=0.83,
        risk_band="CRITICAL",
        action="hold_and_investigate",
    )

    assert isinstance(case, InvestigationCase)
    assert case.transaction_id == "TXN000000001"
    assert case.risk_band == "CRITICAL"
    assert len(case.evidence) == 12
    assert all(isinstance(item, EvidenceItem) for item in case.evidence)
    assert {item.source for item in case.evidence} == {"risk_engine", "transaction"}


def test_case_serialization_is_structured():
    case = build_investigation_case(
        {"transaction_id": "TXN1", "amount": 10.0},
        fraud_probability=0.2,
        anomaly_score=0.1,
        network_risk=0.3,
        velocity_risk=0.0,
        risk_score=0.19,
        risk_band="LOW",
        action="approve",
    )

    payload = case_to_dict(case)
    assert payload["transaction_id"] == "TXN1"
    assert isinstance(payload["evidence"], list)
    assert all("source" in item and "field" in item and "value" in item for item in payload["evidence"])


def test_case_uses_unknown_id_when_missing():
    case = build_investigation_case(
        {"amount": 10.0},
        fraud_probability=0.1,
        anomaly_score=0.1,
        network_risk=0.1,
        velocity_risk=0.1,
        risk_score=0.1,
        risk_band="LOW",
        action="approve",
    )
    assert case.transaction_id == "unknown"
