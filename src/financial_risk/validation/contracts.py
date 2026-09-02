"""Schema and business-rule validation for transaction events."""

from __future__ import annotations

from dataclasses import dataclass

import pandas as pd

EXPECTED_COLUMNS = (
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
)

ALLOWED_FRAUD_TYPES = frozenset(
    {
        "legitimate",
        "account_takeover",
        "card_testing",
        "velocity_fraud",
        "geographic_anomaly",
        "mule_activity",
    }
)

@dataclass(frozen=True)
class ValidationResult:
    """Outcome of a transaction data-contract validation."""

    valid: bool
    errors: tuple[str, ...]


def validate_transactions(df: pd.DataFrame) -> ValidationResult:
    """Validate schema, types, identifiers, domains, and label consistency."""
    errors: list[str] = []

    missing = [column for column in EXPECTED_COLUMNS if column not in df.columns]
    if missing:
        errors.append(f"missing columns: {missing}")
        return ValidationResult(False, tuple(errors))

    if df["transaction_id"].isna().any():
        errors.append("transaction_id contains nulls")
    if not df["transaction_id"].is_unique:
        errors.append("transaction_id must be unique")

    if df["timestamp"].isna().any():
        errors.append("timestamp contains nulls")
    elif not pd.api.types.is_datetime64_any_dtype(df["timestamp"]):
        errors.append("timestamp must be datetime64")

    if df["amount"].isna().any() or (df["amount"] <= 0).any():
        errors.append("amount must be non-null and greater than zero")

    if not set(df["is_fraud"].dropna().unique()).issubset({0, 1}):
        errors.append("is_fraud must contain only 0/1")

    fraud_types = set(df["fraud_type"].dropna().unique())
    if not fraud_types.issubset(ALLOWED_FRAUD_TYPES):
        errors.append(f"unknown fraud_type values: {sorted(fraud_types - ALLOWED_FRAUD_TYPES)}")

    expected_fraud = df["fraud_type"].ne("legitimate").astype("int8")
    if not df["is_fraud"].astype("int8").equals(expected_fraud):
        errors.append("is_fraud must agree with fraud_type")

    for column in ("is_international", "is_night"):
        if not set(df[column].dropna().unique()).issubset({0, 1}):
            errors.append(f"{column} must contain only 0/1")

    if df["shared_device_account_count"].isna().any() or (df["shared_device_account_count"] < 1).any():
        errors.append("shared_device_account_count must be non-null and >= 1")

    return ValidationResult(not errors, tuple(errors))


def assert_valid_transactions(df: pd.DataFrame) -> None:
    """Raise ValueError with actionable contract failures."""
    result = validate_transactions(df)
    if not result.valid:
        raise ValueError("Transaction data contract failed: " + "; ".join(result.errors))
