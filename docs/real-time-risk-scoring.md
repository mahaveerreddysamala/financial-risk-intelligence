# Real-Time Risk Scoring

Phase 22 connects the Kafka event layer to the existing ensemble risk engine and investigation case builder.

## Flow

```text
transaction.created
        |
        v
EventEnvelope validation
        |
        v
Normalized risk signals
(fraud / anomaly / network / velocity)
        |
        v
combine_risk_signals()
        |
        v
decision_from_score()
        |
        +---- LOW / MEDIUM / HIGH
        |
        +---- CRITICAL --> investigation case
        |
        v
transaction.risk_scored
```

## Contract

The streaming inference boundary expects four normalized values in the event payload:

- `fraud_probability`
- `anomaly_score`
- `network_risk`
- `velocity_risk`

Each signal must be numeric and between 0 and 1. The component then reuses the established ensemble weights and operational risk bands rather than creating a second decisioning implementation.

## Traceability

The result preserves the original `event_id`, `transaction_id`, and `occurred_at`. A downstream `transaction.risk_scored` envelope carries the complete structured result.

For CRITICAL outcomes, the existing investigation-case builder is invoked so the event path can produce traceable evidence without inventing facts.

## Reliability boundary

Phase 22 does not claim a live Kafka cluster, production model-serving endpoint, or exactly-once processing. Broker delivery, partitions, retries, dead-letter handling, stateful feature computation, and deployment topology remain infrastructure concerns for later production integration.
