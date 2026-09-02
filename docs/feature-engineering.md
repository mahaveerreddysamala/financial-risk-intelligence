# Feature Engineering

Phase 3 converts transaction-level events into a modeling feature table while enforcing a strict **prior-information-only** rule: a transaction can only use information that was available before its timestamp.

## Feature families

### Customer behavior

- 7-day transaction count
- 30-day average and standard deviation of transaction amount
- 7-day unique merchant count
- 30-day unique device count
- 30-day international and night-transaction rates
- Amount-to-baseline ratio and z-score

### Velocity

- Prior transaction count in 5 minutes, 1 hour, and 24 hours
- Prior amount totals in 1 hour and 24 hours

### Geographic / temporal behavior

- Country change from the customer's previous transaction
- Minutes since previous customer transaction
- Prior international and night-transaction rates
- Transaction hour

### Network reuse

- Prior device, IP, and merchant transaction counts
- Prior customer/device transaction count
- Shared-device risk signal
- Network reuse score

## Leakage controls

Rolling windows are computed before the current transaction is appended to the customer's history. The implementation also sorts by customer, timestamp, and transaction identifier with a stable ordering so tied timestamps are deterministic.

The feature tests explicitly verify that the first transaction for a customer has zero prior activity and that a later transaction can only see earlier observations.

## Scaling path

The reference implementation is intentionally dependency-light and uses Pandas for local experimentation. The same feature contract is designed to be translated to PySpark window functions for the 10M/50M transaction benchmarks planned later in the project. The feature names and semantics should remain stable across implementations so model training is reproducible across environments.
