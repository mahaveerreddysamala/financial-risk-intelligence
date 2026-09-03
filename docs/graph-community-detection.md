# Graph Community Detection

## Purpose

Phase 38 extends the existing entity-network layer with deterministic community detection for fraud-ring style analysis. The implementation operates on transaction relationships among customers, accounts, devices, IPs, and merchants.

## Graph Model

Each transaction connects typed entity nodes:

- `customer:<id>`
- `account:<id>`
- `device:<id>`
- `ip:<id>`
- `merchant:<id>`

Repeated relationships increase the edge weight. This preserves signal that repeated reuse of the same device, IP, account, or merchant is stronger than a one-off relationship.

## Community Detection

`detect_communities()` uses NetworkX weighted greedy modularity optimization. Community IDs are normalized into deterministic integers so repeated execution on the same input produces stable feature values.

## Derived Features

`add_community_features()` adds:

- `community_id` — deterministic community assignment for each customer
- `community_customer_count` — number of unique customers in the community
- `customer_weighted_network_degree` — weighted customer graph degree
- `community_risk_signal` — compact network signal combining community size and customer weighted degree

These features complement the existing network reuse features rather than replacing supervised fraud probability or the established ensemble risk score.

## Validation

Run from the repository root:

```powershell
conda activate portfolio311
pip install -r requirements.txt
ruff check src tests scripts
pytest
```

Phase 38 tests cover typed graph construction, repeated-edge weighting, deterministic community grouping, customer-level feature generation, and validation of missing inputs.

## Production Boundary

The graph is currently constructed and analyzed in-process for deterministic portfolio validation. Production-scale implementations could move graph computation to a distributed graph platform or scheduled graph-processing service and could maintain incrementally updated communities. This phase does not claim an online incremental community detector or distributed graph database deployment.
