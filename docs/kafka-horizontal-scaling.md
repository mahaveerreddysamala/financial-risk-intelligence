# Kafka Horizontal Scaling

## Phase 36

The streaming worker is designed to scale horizontally through Kafka consumer-group partition assignment.

### Partitioning contract

`transaction.created` events use `customer_id` as the Kafka message key. Kafka hashes that key to a partition, so events for the same customer are routed to the same partition. This preserves per-customer event ordering while different customers can be distributed across partitions.

Other event types use `event_id` as their routing key.

### Local deployment

The Docker Compose stack creates three partitions for `financial-risk-events`. The worker uses the same consumer group ID across replicas:

```text
financial-risk-scoring-worker
```

Scale the worker to two replicas with:

```powershell
docker compose up -d --build --scale worker=2
```

Kafka assigns different partitions to different members of the same consumer group. With three input partitions and two workers, both workers can receive assignments while the remaining partition stays assigned to one worker.

### Verification

Check the partition count:

```powershell
docker compose exec kafka /opt/kafka/bin/kafka-topics.sh --bootstrap-server kafka:9092 --describe --topic financial-risk-events
```

Check consumer-group assignments:

```powershell
docker compose exec kafka /opt/kafka/bin/kafka-consumer-groups.sh --bootstrap-server kafka:9092 --describe --group financial-risk-scoring-worker
```

List the worker replicas:

```powershell
docker compose ps worker
```

### State boundary

Customer feature history and completed-event idempotency remain in Redis, so worker replicas share the same external state. Redis read/modify/write atomicity is intentionally addressed as a separate Phase 37 concern.

### Production boundary

The repository validates partition-aware horizontal scaling locally with one Kafka broker and three application partitions. Production deployments still require broker replication, partition sizing based on throughput, consumer rebalance tuning, durable Redis configuration, and operational capacity planning.
