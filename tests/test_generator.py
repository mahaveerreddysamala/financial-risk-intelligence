import pandas as pd

from financial_risk.data_generation.generator import generate_transactions


EXPECTED_COLUMNS = {
    "transaction_id",
    "customer_id",
    "account_id",
    "merchant_id",
    "timestamp",
    "amount",
    "currency",
    "merchant_category",
    "payment_method",
    "channel",
    "device_id",
    "ip_id",
    "country",
    "is_international",
    "is_night",
    "shared_device_account_count",
    "is_fraud",
    "fraud_type",
}


def test_generator_is_reproducible_and_schema_stable() -> None:
    first = generate_transactions(2_000, seed=42)
    second = generate_transactions(2_000, seed=42)

    assert list(first.columns) == list(second.columns)
    assert set(first.columns) == EXPECTED_COLUMNS
    pd.testing.assert_frame_equal(first, second)


def test_generator_produces_valid_business_signals() -> None:
    df = generate_transactions(5_000, seed=7)

    assert len(df) == 5_000
    assert df["transaction_id"].is_unique
    assert (df["amount"] > 0).all()
    assert set(df["is_fraud"].unique()) <= {0, 1}
    assert set(df["fraud_type"].unique()) <= {
        "legitimate",
        "account_takeover",
        "card_testing",
        "velocity_fraud",
        "geographic_anomaly",
        "mule_activity",
    }
    assert df["timestamp"].dtype == "datetime64[ns]"
