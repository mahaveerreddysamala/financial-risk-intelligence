"""Deterministic synthetic financial transaction generator.

The generator creates a compact relational-style transaction dataset with
multiple fraud typologies so downstream feature engineering and model
experiments can be reproduced without external data dependencies.
"""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd

MERCHANT_CATEGORIES = np.array(
    [
        "grocery",
        "fuel",
        "travel",
        "electronics",
        "fashion",
        "digital_goods",
        "restaurant",
        "healthcare",
        "utilities",
        "marketplace",
    ]
)
COUNTRIES = np.array(["US", "CA", "GB", "DE", "FR", "SG", "AU", "MX"])
CHANNELS = np.array(["card_present", "ecommerce", "mobile_wallet", "ach"])
PAYMENT_METHODS = np.array(["debit", "credit", "wallet", "bank_transfer"])


def _entity_ids(prefix: str, size: int, rng: np.random.Generator) -> np.ndarray:
    """Generate stable-looking entity identifiers with repeated references."""
    values = rng.integers(1, max(size // 4, 2), size=size)
    return np.char.add(prefix, values.astype(str))


def generate_transactions(rows: int, seed: int = 42) -> pd.DataFrame:
    """Generate deterministic synthetic transactions with labeled fraud patterns."""
    if rows <= 0:
        raise ValueError("rows must be greater than zero")

    rng = np.random.default_rng(seed)
    timestamps = pd.Timestamp("2025-01-01") + pd.to_timedelta(
        rng.integers(0, 365 * 24 * 60, rows), unit="m"
    )
    customer_id = _entity_ids("C", rows, rng)
    account_id = np.char.add("A", rng.integers(1, max(rows // 3, 2), rows).astype(str))
    merchant_id = np.char.add("M", rng.integers(1, max(rows // 20, 2), rows).astype(str))
    device_id = np.char.add("D", rng.integers(1, max(rows // 6, 2), rows).astype(str))
    ip_id = np.char.add("IP", rng.integers(1, max(rows // 8, 2), rows).astype(str))

    amount = np.clip(rng.lognormal(mean=3.8, sigma=1.0, size=rows), 1.0, 2500.0)
    country = rng.choice(COUNTRIES, size=rows, p=[0.78, 0.04, 0.04, 0.03, 0.02, 0.02, 0.04, 0.03])
    channel = rng.choice(CHANNELS, size=rows, p=[0.46, 0.32, 0.14, 0.08])
    payment_method = rng.choice(PAYMENT_METHODS, size=rows, p=[0.38, 0.34, 0.16, 0.12])
    merchant_category = rng.choice(MERCHANT_CATEGORIES, size=rows)

    is_fraud = np.zeros(rows, dtype=np.int8)
    fraud_type = np.full(rows, "legitimate", dtype=object)

    # Deterministic typology masks. Later patterns overwrite earlier labels
    # intentionally, making severe patterns dominant in the training labels.
    takeover = rng.random(rows) < 0.0025
    card_testing = rng.random(rows) < 0.0020
    velocity = rng.random(rows) < 0.0020
    geographic = rng.random(rows) < 0.0015
    mule = rng.random(rows) < 0.0010

    is_fraud[takeover | card_testing | velocity | geographic | mule] = 1
    fraud_type[takeover] = "account_takeover"
    fraud_type[card_testing] = "card_testing"
    fraud_type[velocity] = "velocity_fraud"
    fraud_type[geographic] = "geographic_anomaly"
    fraud_type[mule] = "mule_activity"

    # Manipulate observable signals so the generated labels have a learnable
    # relationship with behavioral features.
    amount[takeover] *= rng.uniform(4.0, 9.0, takeover.sum())
    amount[card_testing] = rng.uniform(1.0, 15.0, card_testing.sum())
    amount[velocity] *= rng.uniform(2.0, 6.0, velocity.sum())
    country[geographic] = rng.choice(np.array(["GB", "DE", "SG", "AU"]), geographic.sum())

    hour = timestamps.hour.to_numpy()
    is_night = ((hour < 5) | (hour >= 23)).astype(np.int8)
    is_international = (country != "US").astype(np.int8)

    # Reused device/IP relationships create graph signal without requiring a
    # graph database in the initial phase.
    shared_device_account_count = rng.integers(1, 4, rows)
    shared_device_account_count[mule] = rng.integers(5, 10, mule.sum())

    return pd.DataFrame(
        {
            "transaction_id": [f"TXN{i:09d}" for i in range(1, rows + 1)],
            "customer_id": customer_id,
            "account_id": account_id,
            "merchant_id": merchant_id,
            "timestamp": timestamps,
            "amount": amount.round(2),
            "currency": "USD",
            "merchant_category": merchant_category,
            "payment_method": payment_method,
            "channel": channel,
            "device_id": device_id,
            "ip_id": ip_id,
            "country": country,
            "is_international": is_international,
            "is_night": is_night,
            "shared_device_account_count": shared_device_account_count,
            "is_fraud": is_fraud,
            "fraud_type": fraud_type,
        }
    )


def write_dataset(df: pd.DataFrame, output: str) -> Path:
    """Write transactions as a Parquet dataset and return the output path."""
    destination = Path(output)
    destination.parent.mkdir(parents=True, exist_ok=True)
    df.to_parquet(destination, index=False)
    return destination


def main() -> None:
    parser = argparse.ArgumentParser(description="Generate synthetic financial transactions.")
    parser.add_argument("--rows", type=int, default=100_000)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--output", default="data/raw/transactions.parquet")
    args = parser.parse_args()

    df = generate_transactions(args.rows, args.seed)
    output = write_dataset(df, args.output)
    fraud_rate = float(df["is_fraud"].mean())
    print(f"Generated {len(df):,} transactions")
    print(f"Fraud rate: {fraud_rate:.4%}")
    print(f"Output: {output}")


if __name__ == "__main__":
    main()
