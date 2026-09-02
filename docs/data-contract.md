# Financial Transaction Data Contract

The transaction contract is the boundary between synthetic/event ingestion and downstream feature engineering. The pipeline should fail fast when required identifiers, timestamps, numeric measures, categorical domains, or fraud labels are invalid.

## Required fields

| Field | Role | Contract |
|---|---|---|
| `transaction_id` | event key | non-null, unique |
| `customer_id` | customer key | non-null |
| `account_id` | account key | non-null |
| `merchant_id` | merchant key | non-null |
| `timestamp` | event time | datetime, non-null |
| `amount` | monetary measure | non-null, > 0 |
| `currency` | currency code | populated |
| `merchant_category` | merchant dimension | populated |
| `payment_method` | payment dimension | populated |
| `channel` | transaction channel | populated |
| `device_id` | device relationship | populated |
| `ip_id` | network relationship | populated |
| `country` | location dimension | populated |
| `is_international` | derived indicator | 0/1 |
| `is_night` | derived indicator | 0/1 |
| `shared_device_account_count` | network signal | integer >= 1 |
| `is_fraud` | supervised target | 0/1 |
| `fraud_type` | fraud typology | approved domain |

## Label consistency

`is_fraud` is derived from `fraud_type` for the generated dataset:

- `legitimate` must map to `is_fraud = 0`.
- Every non-legitimate fraud typology must map to `is_fraud = 1`.

This prevents downstream model training from receiving contradictory labels.

## Validation behavior

`validate_transactions()` returns a structured result containing a boolean status and all detected errors. `assert_valid_transactions()` raises `ValueError` with actionable contract failures so batch jobs and tests can fail fast.

The contract deliberately validates **data integrity before feature engineering**. Future ingestion paths should call the same validator so batch and streaming sources share one canonical transaction schema.
