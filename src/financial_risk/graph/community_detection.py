"""Deterministic community-level network intelligence for financial crime."""
from __future__ import annotations

from collections.abc import Iterable

import networkx as nx
import numpy as np
import pandas as pd

ENTITY_COLUMNS = ("customer_id", "account_id", "device_id", "ip_id", "merchant_id")
ENTITY_TYPES = {
    "customer_id": "customer",
    "account_id": "account",
    "device_id": "device",
    "ip_id": "ip",
    "merchant_id": "merchant",
}


def _node(entity_type: str, value: object) -> str:
    return f"{entity_type}:{value}"


def build_entity_graph(
    df: pd.DataFrame,
    *,
    entity_columns: Iterable[str] = ENTITY_COLUMNS,
) -> nx.Graph:
    """Build a heterogeneous undirected entity graph from transactions."""
    columns = tuple(entity_columns)
    required = {"transaction_id", *columns}
    missing = required.difference(df.columns)
    if missing:
        raise ValueError(f"Missing required columns: {sorted(missing)}")

    graph = nx.Graph()
    for row in df[list(columns)].itertuples(index=False, name=None):
        nodes = [
            _node(ENTITY_TYPES[column], value)
            for column, value in zip(columns, row, strict=True)
            if pd.notna(value)
        ]
        for node in nodes:
            graph.add_node(node)
        for left, right in zip(nodes, nodes[1:]):
            if graph.has_edge(left, right):
                graph[left][right]["weight"] += 1
            else:
                graph.add_edge(left, right, weight=1)
    return graph


def detect_communities(graph: nx.Graph) -> dict[str, int]:
    """Assign deterministic integer community IDs using modularity optimization."""
    if graph.number_of_nodes() == 0:
        return {}

    communities = nx.community.greedy_modularity_communities(graph, weight="weight")
    ordered = sorted(
        communities,
        key=lambda members: (
            -sum(graph.degree(node, weight="weight") for node in members),
            min(members),
        ),
    )
    assignments: dict[str, int] = {}
    for community_id, members in enumerate(ordered):
        for node in sorted(members):
            assignments[node] = community_id
    return assignments


def add_community_features(df: pd.DataFrame) -> pd.DataFrame:
    """Add community membership, size, and weighted-degree risk features."""
    required = {"customer_id", *ENTITY_COLUMNS}
    missing = required.difference(df.columns)
    if missing:
        raise ValueError(f"Missing required columns: {sorted(missing)}")

    graph = build_entity_graph(df)
    assignments = detect_communities(graph)
    result = df.copy()

    result["community_id"] = result["customer_id"].map(
        lambda value: assignments.get(_node("customer", value), -1)
    ).astype("int64")

    community_sizes = result.groupby("community_id")["customer_id"].nunique()
    result["community_customer_count"] = result["community_id"].map(community_sizes).astype("int64")

    customer_degrees = {
        node.removeprefix("customer:"): float(degree)
        for node, degree in graph.degree(weight="weight")
        if node.startswith("customer:")
    }
    result["customer_weighted_network_degree"] = result["customer_id"].map(customer_degrees).fillna(0.0)

    # Larger communities with more cross-entity reuse produce a stronger signal.
    result["community_risk_signal"] = (
        np.log1p(result["community_customer_count"].clip(lower=1).astype(float))
        + np.log1p(result["customer_weighted_network_degree"].clip(lower=0.0))
    ).astype(float)
    return result
