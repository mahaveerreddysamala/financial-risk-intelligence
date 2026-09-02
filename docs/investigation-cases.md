# Investigation Cases & Evidence Aggregation

Phase 11 adds a traceable investigation-case layer between model/risk outputs and future analyst APIs or a GenAI investigation copilot.

## Design

Each case contains:

- transaction identifier
- aggregate risk score and operational risk band
- selected action
- model-derived risk signals
- observed transaction attributes used as contextual evidence

The case builder does not infer facts that are not present in the transaction or supplied risk outputs. This keeps downstream narratives grounded in explicit evidence.

## Evidence sources

`risk_engine` records the supervised fraud probability, anomaly score, network risk, and velocity risk supplied by upstream components.

`transaction` records observed values such as amount, country, channel, payment method, device, IP, merchant, and shared-device count when available.

## GenAI integration path

A later investigation copilot can consume the serialized case as structured context and produce an analyst summary. The copilot should cite or preserve the underlying evidence items and should not manufacture transaction history, customer intent, or external facts.
