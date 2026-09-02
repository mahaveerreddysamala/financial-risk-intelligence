# Streaming Worker

The Phase 22 worker adds a transport-neutral processing boundary around real-time risk scoring.

## Flow

```text
EventEnvelope
    ↓
process_event()
    ↓
score_transaction_event()
    ↓
RiskScoringResult
    ↓
optional publisher callback
```

The worker can be invoked by Kafka consumers, replay jobs, or test harnesses without embedding broker-specific logic in the scoring component.

In a production consumer, the expected sequence is:

1. Poll and deserialize a Kafka event.
2. Call `process_event()`.
3. Publish the `transaction.risk_scored` result downstream.
4. Commit the source offset only after successful processing and publication.
5. Route malformed or repeatedly failing events to a dead-letter path.

This phase implements the application processing boundary only. It does not claim a live Kafka deployment, exactly-once semantics, or production-grade offset infrastructure.
