import pandas as pd
import pytest

from financial_risk.data_generation.generator import generate_transactions
from financial_risk.features.pipeline import build_feature_table


FEATURE_COLUMNS = {
    "customer_txn_count_7d",
    "customer_avg_amount_30d",
    "customer_std_amount_30d",
    "customer_unique_merchants_7d",
    "customer_unique_devices_30d",
    "customer_international_rate_30d",
    "customer_night_txn_rate_30d",
    "amount_vs_customer_avg",
    "amount_zscore",
    "txn_count_5m",
    "txn_count_1h",
    "txn_count_24h",
    "amount_sum_1h",
    "amount_sum_24h",
    "location_changed",
    "minutes_since_customer_txn",
    "customer_prior_international_rate",
    "customer_prior_night_rate",
    "customer_prior_transaction_count",
    "transaction_hour",
    "device_prior_transaction_count",
    "ip_prior_transaction_count",
    "merchant_prior_transaction_count",
    "customer_device_prior_transaction_count",
    "shared_device_risk_signal",
    "network_reuse_score",
}


def test_feature_pipeline_adds_expected_features_without_changing_rows() -> None:
    df = generate_transactions(1_000, seed=42)
    features = build_feature_table(df)

    assert len(features) == len(df)
    assert features["transaction_id"].tolist() == df["transaction_id"].tolist()
    assert FEATURE_COLUMNS <= set(features.columns)
    assert features[list(FEATURE_COLUMNS)].notna().all().all()


def test_first_customer_transaction_has_no_prior_history() -> None:
    df = pd.DataFrame(
        [
            {
                "transaction_id": "T1",
                "customer_id": "C1",
                "account_id": "A1",
                "merchant_id": "M1",
                "timestamp": "2025-01-01 10:00:00",
                "amount": 100.0,
                "currency": "USD",
                "merchant_category": "grocery",
                "payment_method": "debit",
                "channel": "card_present",
                "device_id": "D1",
                "ip_id": "IP1",
                "country": "US",
                "is_international": 0,
                "is_night": 0,
                "shared_device_account_count": 1,
                "is_fraud": 0,
                "fraud_type": "legitimate",
            },
            {
                "transaction_id": "T2",
                "customer_id": "C1",
                "account_id": "A1",
                "merchant_id": "M2",
                "timestamp": "2025-01-01 10:03:00",
                "amount": 150.0,
                "currency": "USD",
                "merchant_category": "fuel",
                "payment_method": "credit",
                "channel": "ecommerce",
                "device_id": "D1",
                "ip_id": "IP1",
                "country": "US",
                "is_international": 0,
                "is_night": 0,
                "shared_device_account_count": 1,
                "is_fraud": 0,
                "fraud_type": "legitimate",
            },
        ]
    )

    features = build_feature_table(df)
    first = features.iloc[0]
    second = features.iloc[1]

    assert first["customer_txn_count_7d"] == 0
    assert first["customer_avg_amount_30d"] == 0
    assert first["txn_count_5m"] == 0
    assert first["txn_count_1h"] == 0
    assert first["network_reuse_score"] == 0
    assert second["customer_txn_count_7d"] == 1
    assert second["txn_count_5m"] == 1
    assert second["txn_count_1h"] == 1
    assert second["amount_sum_1h"] == 100.0


def test_feature_pipeline_rejects_missing_columns() -> None:
    with pytest.raises(ValueError, match="Missing required columns"):
        build_feature_table(pd.DataFrame({"transaction_id": ["T1"]}))
