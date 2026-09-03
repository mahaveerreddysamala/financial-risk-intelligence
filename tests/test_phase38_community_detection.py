import pandas as pd
import pytest

from financial_risk.graph.community_detection import (
    add_community_features,
    build_entity_graph,
    detect_communities,
)


def _transactions() -> pd.DataFrame:
    return pd.DataFrame(
        [
            {
                "transaction_id": "TXN-1",
                "customer_id": "C-1",
                "account_id": "A-1",
                "device_id": "D-1",
                "ip_id": "IP-1",
                "merchant_id": "M-1",
            },
            {
                "transaction_id": "TXN-2",
                "customer_id": "C-2",
                "account_id": "A-2",
                "device_id": "D-1",
                "ip_id": "IP-1",
                "merchant_id": "M-1",
            },
            {
                "transaction_id": "TXN-3",
                "customer_id": "C-3",
                "account_id": "A-3",
                "device_id": "D-3",
                "ip_id": "IP-3",
                "merchant_id": "M-3",
            },
            {
                "transaction_id": "TXN-4",
                "customer_id": "C-4",
                "account_id": "A-4",
                "device_id": "D-3",
                "ip_id": "IP-3",
                "merchant_id": "M-3",
            },
        ]
    )


def test_build_entity_graph_contains_typed_nodes_and_weighted_edges() -> None:
    graph = build_entity_graph(_transactions())

    assert "customer:C-1" in graph
    assert "device:D-1" in graph
    assert graph["customer:C-1"]["account:A-1"]["weight"] == 1
    assert graph["device:D-1"]["ip:IP-1"]["weight"] == 2


def test_detect_communities_groups_shared_entities_deterministically() -> None:
    graph = build_entity_graph(_transactions())
    assignments = detect_communities(graph)

    assert assignments["customer:C-1"] == assignments["customer:C-2"]
    assert assignments["customer:C-3"] == assignments["customer:C-4"]
    assert assignments["customer:C-1"] != assignments["customer:C-3"]


def test_add_community_features_returns_customer_level_signals() -> None:
    result = add_community_features(_transactions())

    assert {
        "community_id",
        "community_customer_count",
        "customer_weighted_network_degree",
        "community_risk_signal",
    }.issubset(result.columns)
    shared = result[result["customer_id"].isin(["C-1", "C-2"])]
    second_shared = result[result["customer_id"].isin(["C-3", "C-4"])]

    assert shared["community_customer_count"].nunique() == 1
    assert shared["community_customer_count"].iloc[0] == 2
    assert second_shared["community_customer_count"].nunique() == 1
    assert second_shared["community_customer_count"].iloc[0] == 2


def test_missing_columns_raise_clear_error() -> None:
    with pytest.raises(ValueError, match="Missing required columns"):
        build_entity_graph(pd.DataFrame({"transaction_id": ["TXN-1"]}))
