# Streaming Runtime Reliability

Phase 24 adds a transport-agnostic runtime around the existing Kafka event and risk-scoring layers.

## Reliability controls

### Bounded retries

Processing failures are retried up to a configurable `max_attempts` value with configurable backoff. The default is three attempts with one second between attempts.

### Dead-letter handling

An event that still fails after the retry budget becomes a structured `DeadLetterRecord`. A caller can publish that record to a dead-letter topic for later investigation or replay.

### Idempotency boundary

The runtime checks an idempotency store before processing and marks an event only after processing and downstream publishing succeed. The default `InMemoryIdempotencyStore` is intentionally limited to local runs and tests.

A production deployment should replace it with a durable store appropriate to the processing boundary, such as a transactional database or another strongly consistent state store.

### Structured operational counters

`StreamingStats` tracks received, succeeded, retried, duplicate, and dead-lettered events for a worker process. The runtime also emits log records for successful processing, retries, duplicate suppression, and dead-letter routing.

## Runtime flow

```text
Kafka input topic
      |
      v
poll event
      |
      v
idempotency check ---- duplicate ----> skip
      |
      v
risk processor
      |
   failure?
   /     \
 yes       no
  |         |
 retry      v
  |      publish result
  |         |
 exhausted  v
  |      mark complete
  v
DLQ
```

## Continuous worker

`scripts/run_streaming_worker.py` wires the runtime to the Kafka adapters:

- `financial-risk-events` → input topic
- `financial-risk-scored` → successful risk decisions
- `financial-risk-dlq` → exhausted failures
- `financial-risk-scoring-worker` → consumer group

Override these values with environment variables such as `KAFKA_INPUT_TOPIC`, `KAFKA_OUTPUT_TOPIC`, `KAFKA_DLQ_TOPIC`, and `KAFKA_GROUP_ID`.

Example local execution after Kafka is running:

```powershell
python scripts/run_streaming_worker.py
```

The worker is intentionally a production-style reference implementation, not a claim of exactly-once processing or durable idempotency. Those guarantees depend on the external state store, producer semantics, offset strategy, and deployment architecture.
