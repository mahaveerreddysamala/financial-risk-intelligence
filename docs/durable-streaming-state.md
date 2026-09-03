# Durable Streaming State

Phase 35 adds a Redis-backed state boundary for the real-time Kafka worker.

## What is durable

Customer transaction history used by the streaming feature service is stored in Redis so worker restarts do not erase prior-only behavioral and velocity context.

Completed event IDs are also stored in Redis with bounded retention so a restarted worker can continue duplicate-event suppression.

## Runtime behavior

The worker uses `REDIS_URL` when configured. Without it, the existing in-memory implementations remain available for tests and local transport-agnostic runs.

The feature service continues to prepare the current transaction before committing it to history. Redis therefore stores only successfully processed transaction history, preserving the existing leakage boundary.

## Docker Compose

The portfolio Compose stack adds a Redis service with a health check. The streaming worker waits for Redis health and receives:

- `REDIS_URL=redis://redis:6379/0`
- `REDIS_KEY_PREFIX=financial-risk`

This is a local portfolio deployment boundary. Redis persistence, replication, authentication, TLS, backup/recovery, eviction policy, and high availability require explicit production configuration.
