from fastapi.testclient import TestClient

from financial_risk.api.app import app
from financial_risk.investigation.case_store import InvestigationCaseStore


client = TestClient(app)


def _payload(transaction_id: str = "TXN-LIFE-1") -> dict:
    return {
        "fraud_probability": 0.9,
        "anomaly_score": 0.8,
        "network_score": 0.7,
        "velocity_score": 0.6,
        "transaction": {"transaction_id": transaction_id, "amount": 250.0, "country": "US"},
    }


def test_case_store_transition_rules_and_audit():
    store = InvestigationCaseStore()
    case, replayed = store.create(
        {
            "transaction_id": "TXN-1",
            "risk_score": 0.8,
            "risk_band": "HIGH",
            "action": "investigate",
            "evidence": [],
        },
        idempotency_key="request-1",
        actor="analyst-1",
    )
    assert replayed is False
    same_case, replayed = store.create(
        {
            "transaction_id": "different",
            "risk_score": 0.1,
            "risk_band": "LOW",
            "action": "allow",
            "evidence": [],
        },
        idempotency_key="request-1",
    )
    assert replayed is True
    assert same_case.case_id == case.case_id

    updated = store.transition(case.case_id, "IN_REVIEW", actor="analyst-2", note="Started review")
    assert updated is not None
    assert updated.status == "IN_REVIEW"
    assert len(updated.audit_log) == 2
    assert updated.audit_log[-1].details["note"] == "Started review"

    try:
        store.transition(case.case_id, "OPEN")
        raise AssertionError("expected invalid transition")
    except ValueError as exc:
        assert "invalid transition" in str(exc)


def test_case_lifecycle_api():
    response = client.post(
        "/v1/investigations/cases",
        headers={"Idempotency-Key": "api-lifecycle-1"},
        json=_payload("TXN-LIFE-API"),
    )
    assert response.status_code == 200
    created = response.json()
    assert created["status"] == "OPEN"
    assert created["case_id"].startswith("CASE-")
    assert created["idempotent_replay"] is False

    replay = client.post(
        "/v1/investigations/cases",
        headers={"Idempotency-Key": "api-lifecycle-1"},
        json=_payload("TXN-DIFFERENT"),
    )
    assert replay.status_code == 200
    assert replay.json()["case_id"] == created["case_id"]
    assert replay.json()["idempotent_replay"] is True

    case_id = created["case_id"]
    fetched = client.get(f"/v1/investigations/cases/{case_id}")
    assert fetched.status_code == 200
    assert fetched.json()["transaction_id"] == "TXN-LIFE-API"

    transitioned = client.patch(
        f"/v1/investigations/cases/{case_id}/status",
        headers={"X-Actor": "investigator-7"},
        json={"status": "IN_REVIEW", "note": "Manual review started"},
    )
    assert transitioned.status_code == 200
    assert transitioned.json()["status"] == "IN_REVIEW"

    invalid_transition = client.patch(
        f"/v1/investigations/cases/{case_id}/status",
        json={"status": "OPEN"},
    )
    assert invalid_transition.status_code == 409

    audit = client.get(f"/v1/investigations/cases/{case_id}/audit")
    assert audit.status_code == 200
    assert audit.json()["events"][-1]["event_type"] == "STATUS_CHANGED"


def test_case_listing_filters_and_pagination():
    first = client.post("/v1/investigations/cases", json=_payload("TXN-LIST-1")).json()
    client.post("/v1/investigations/cases", json=_payload("TXN-LIST-2"))

    listing = client.get("/v1/investigations/cases", params={"status": "open", "limit": 1})
    assert listing.status_code == 200
    body = listing.json()
    assert body["total"] >= 2
    assert len(body["items"]) == 1
    assert body["items"][0]["status"] == "OPEN"

    missing = client.get("/v1/investigations/cases/CASE-NOT-FOUND")
    assert missing.status_code == 404
    assert first["case_id"] != "CASE-NOT-FOUND"
