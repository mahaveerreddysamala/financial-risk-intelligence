# Phase 39: Online Community Updates

Phase 39 adds an incremental graph-community tracker for streaming financial transactions.

## Design

`OnlineCommunityTracker` maintains connected components with a union-find data structure. Each transaction contributes typed customer, account, device, IP, and merchant entities. Shared entities connect previously observed components without rebuilding the entire graph for every event.

The tracker is intentionally aligned with the deterministic connected-component community semantics used by Phase 38. It is a lightweight online state layer, not a replacement for a distributed graph database.

## Capabilities

- Incremental community updates as transactions arrive
- Shared-entity joins merge existing customer communities
- Disconnected customers remain isolated
- Deterministic community IDs derived from the current community membership
- Typed community-member inspection for explainability and investigations
- Clear input validation for incomplete transaction records

## Production boundary

The tracker is currently an in-process state component for reproducible portfolio validation. Production deployments would need durable/distributed graph state, partition-aware ownership, state recovery, retention policies, and potentially a graph database or distributed graph-processing engine.
