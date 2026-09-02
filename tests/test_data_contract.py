import pandas as pd
import pytest

from financial_risk.data_generation.generator import generate_transactions
from financial_risk.validation.contracts import assert_valid_transactions, validate_transactions


def test_generated_transactions_satisfy_contract() -> None:
    df = generate_transactions(5_000, seed=42)
    result = validate_transactions(df)

    assert result.valid, result.errors


def test_contract_rejects_duplicate_transaction_ids() -> None:
    df = generate_transactions(100, seed=42)
    df.loc[1, "transaction_id"] = df.loc[0, "transaction_id"]

    result = validate_transactions(df)

    assert not result.valid
    assert "transaction_id must be unique" in result.errors


def test_contract_rejects_label_mismatch() -> None:
    df = generate_transactions(100, seed=42)
    df.loc[0, "fraud_type"] = "account_takeover"
    df.loc[0, "is_fraud"] = 0

    result = validate_transactions(df)

    assert not result.valid
    assert "is_fraud must agree with fraud_type" in result.errors


def test_assert_valid_transactions_raises_with_actionable_message() -> None:
    df = generate_transactions(50, seed=7)
    df.loc[0, "amount"] = 0.0

    with pytest.raises(ValueError, match="amount must be non-null and greater than zero"):
        assert_valid_transactions(df)


def test_contract_requires_datetime_timestamp() -> None:
    df = generate_transactions(50, seed=7)
    df["timestamp"] = pd.Series(df["timestamp"].dt.strftime("%Y-%m-%d"))

    result = validate_transactions(df)

    assert not result.valid
    assert "timestamp must be datetime64" in result.errors
