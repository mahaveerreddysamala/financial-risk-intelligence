# Kafka Streaming Architecture

Phase 21 adds a Kafka-compatible event transport boundary for near-real-time financial risk workflows.

## Design

```text
Financial Transaction Event
          |
          v
Versioned Event Envelope
          |
          v
Kafka Topic: transactions
          |
    +-----+-----+
    |           |
    v           v
Risk Consumer  Future Stream Processors
    |
    v
Validate -> Feature/Risk Service -> Decision
```

The event envelope contains:

- `event_id` — stable transaction/event identifier used as the Kafka key
- `event_type` — logical event name such as `transaction.created`
- `schema_version` — explicit integer schema version
- `occurred_at` — timezone-aware UTC timestamp
- `payload` — event-specific JSON object

## Implementation boundary

The core package does not require Kafka to be installed. The `confluent-kafka` client is optional and loaded lazily by the producer/consumer adapters.

Install the optional streaming dependencies with:

```text
pip install -r requirements-streaming.txt
```

The current adapter uses the Confluent Python client with standard `Producer` and `Consumer` APIs. The client is compatible with Apache Kafka brokers, Confluent Platform, and Confluent Cloud.

## Reliability contract

The producer uses the event ID as the Kafka key, making event identity available for partitioning and downstream idempotency strategies. The consumer disables automatic offset commits so a future processing layer can commit offsets only after successful validation and risk processing.

This phase intentionally does not claim a running Kafka cluster, Schema Registry, exactly-once processing, or production deployment. Those require an actual broker and infrastructure integration.

## Production path

A production implementation can extend this boundary with:

1. Kafka topic creation and ACLs through infrastructure as code.
2. Avro or JSON Schema with Schema Registry for stronger contract evolution.
3. Consumer-side idempotency and explicit offset commit after successful processing.
4. Dead-letter topics for malformed or permanently failed events.
5. Partition-aware scaling and consumer-group monitoring.
6. Integration with the existing FastAPI/risk-scoring and monitoring layers.
