# Graph-Based Financial Crime Intelligence

Phase 7 adds graph-style relationship intelligence on top of the existing transaction and network signals. It intentionally uses pandas group-by operations rather than requiring a graph database, keeping the feature layer reproducible and easy to scale later.

## Entity relationships

```text
Customer ── Account
    │          │
    ├── Device ── IP
    │
    └── Merchant
```

The feature builder measures how connected a transaction is to shared entities:

- `shared_device_accounts`: distinct accounts observed on the same device
- `shared_ip_accounts`: distinct accounts observed on the same IP
- `customer_device_degree`: distinct devices used by the customer
- `customer_ip_degree`: distinct IPs used by the customer
- `merchant_customer_degree`: distinct customers observed at the merchant
- `network_entity_degree`: combined relationship degree
- `network_risk_score`: logarithmically damped shared-entity signal

These signals help surface behaviors associated with coordinated activity, account sharing, mule networks, and fraud rings.

## Interpretation

A high network score means the transaction is connected to more shared entities. It is a prioritization signal, not proof of criminal activity. For a production system, graph relationships should be time-aware and governed by entity-resolution, privacy, and investigative controls.

## Scaling path

The current implementation is deliberately database-free. The same entity relationships can later be represented in a graph technology such as Neo4j, Amazon Neptune, or a Spark graph-processing layer when graph traversal, community detection, or connected-component analysis becomes necessary.
