import pandas as pd
import pytest

from financial_risk.graph.network_intelligence import build_entity_graph_features


def test_build_entity_graph_features():
    frame = pd.DataFrame(
        {
            "transaction_id": ["T1", "T2", "T3", "T4"],
            "customer_id": ["C1", "C2", "C1", "C3"],
            "account_id": ["A1", "A2", "A1", "A3"],
            "device_id": ["D1", "D1", "D2", "D3"],
            "ip_id": ["IP1", "IP1", "IP2", "IP3"],
            "merchant_id": ["M1", "M1", "M1", "M2"],
        }
    )

    result = build_entity_graph_features(frame)

    assert result["shared_device_accounts"].tolist() == [2, 2, 1, 1]
    assert result["shared_ip_accounts"].tolist() == [2, 2, 1, 1]
    assert result.loc[0, "merchant_customer_degree"] == 2
    assert result.loc[2, "customer_device_degree"] == 2
    assert result["network_entity_degree"].ge(0).all()
    assert result["network_risk_score"].ge(0).all()


def test_graph_feature_validation():
    with pytest.raises(ValueError, match="Missing required columns"):
        build_entity_graph_features(pd.DataFrame({"transaction_id": ["T1"]}))
