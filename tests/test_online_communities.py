import pytest

from financial_risk.graph.online_communities import OnlineCommunityTracker


def _transaction(customer: str, account: str, device: str, ip: str, merchant: str) -> dict[str, object]:
    return {
        "transaction_id": f"TXN-{customer}",
        "customer_id": customer,
        "account_id": account,
        "device_id": device,
        "ip_id": ip,
        "merchant_id": merchant,
    }


def test_incremental_transactions_join_shared_entity_communities() -> None:
    tracker = OnlineCommunityTracker()

    first_id = tracker.add_transaction(_transaction("C-1", "A-1", "D-1", "IP-1", "M-1"))
    second_id = tracker.add_transaction(_transaction("C-2", "A-2", "D-1", "IP-1", "M-2"))

    assert first_id == second_id
    members = tracker.community_members("C-2")
    assert "customer:C-1" in members
    assert "customer:C-2" in members
    assert "device:D-1" in members


def test_disconnected_customers_remain_in_separate_communities() -> None:
    tracker = OnlineCommunityTracker()

    first_id = tracker.add_transaction(_transaction("C-1", "A-1", "D-1", "IP-1", "M-1"))
    second_id = tracker.add_transaction(_transaction("C-2", "A-2", "D-2", "IP-2", "M-2"))

    assert first_id != second_id
    assert "customer:C-2" not in tracker.community_members("C-1")


def test_late_shared_entity_merges_existing_communities() -> None:
    tracker = OnlineCommunityTracker()

    first_id = tracker.add_transaction(_transaction("C-1", "A-1", "D-1", "IP-1", "M-1"))
    second_id = tracker.add_transaction(_transaction("C-2", "A-2", "D-2", "IP-2", "M-2"))
    assert first_id != second_id

    merged_id = tracker.add_transaction(_transaction("C-3", "A-3", "D-1", "IP-2", "M-3"))

    assert merged_id == tracker.community_id("C-1")
    assert merged_id == tracker.community_id("C-2")
    assert {"customer:C-1", "customer:C-2", "customer:C-3"}.issubset(
        tracker.community_members("C-3")
    )


def test_missing_entity_column_raises_clear_error() -> None:
    tracker = OnlineCommunityTracker()
    with pytest.raises(ValueError, match="Missing required columns"):
        tracker.add_transaction({"customer_id": "C-1"})
