"""Incremental entity-community tracking for streaming financial risk events."""
from __future__ import annotations

from collections.abc import Iterable
from dataclasses import dataclass, field
from hashlib import sha256

import pandas as pd

from financial_risk.graph.community_detection import ENTITY_COLUMNS, ENTITY_TYPES


@dataclass
class OnlineCommunityTracker:
    """Maintain connected entity communities as transactions arrive incrementally.

    Each transaction connects its typed entities. Community membership is based on
    connected components, matching the deterministic batch community semantics
    without rebuilding the full graph for every event.
    """

    parent: dict[str, str] = field(default_factory=dict)
    size: dict[str, int] = field(default_factory=dict)

    @staticmethod
    def _node(entity_type: str, value: object) -> str:
        return f"{entity_type}:{value}"

    def _ensure(self, node: str) -> None:
        if node not in self.parent:
            self.parent[node] = node
            self.size[node] = 1

    def _find(self, node: str) -> str:
        self._ensure(node)
        root = node
        while self.parent[root] != root:
            root = self.parent[root]
        while self.parent[node] != node:
            next_node = self.parent[node]
            self.parent[node] = root
            node = next_node
        return root

    def _union(self, left: str, right: str) -> None:
        left_root = self._find(left)
        right_root = self._find(right)
        if left_root == right_root:
            return
        if (self.size[left_root], left_root) < (self.size[right_root], right_root):
            left_root, right_root = right_root, left_root
        self.parent[right_root] = left_root
        self.size[left_root] += self.size[right_root]

    def add_transaction(
        self,
        transaction: dict[str, object],
        *,
        entity_columns: Iterable[str] = ENTITY_COLUMNS,
    ) -> int:
        """Add one transaction and return its customer's current community ID."""
        columns = tuple(entity_columns)
        missing = {"customer_id", *columns}.difference(transaction)
        if missing:
            raise ValueError(f"Missing required columns: {sorted(missing)}")

        nodes = [
            self._node(ENTITY_TYPES[column], transaction[column])
            for column in columns
            if pd.notna(transaction[column])
        ]
        if not nodes:
            raise ValueError("transaction must contain at least one graph entity")

        for node in nodes:
            self._ensure(node)
        first = nodes[0]
        for node in nodes[1:]:
            self._union(first, node)
        return self.community_id(transaction["customer_id"])

    def community_id(self, customer_id: object) -> int:
        """Return a stable integer community ID for a customer node."""
        root = self._find(self._node("customer", customer_id))
        members = sorted(
            candidate for candidate in self.parent if self._find(candidate) == root
        )
        digest = sha256(members[0].encode("utf-8")).hexdigest()
        return int(digest[:8], 16)

    def community_members(self, customer_id: object) -> list[str]:
        """Return sorted typed entity members in the customer's current community."""
        root = self._find(self._node("customer", customer_id))
        return sorted(
            candidate for candidate in self.parent if self._find(candidate) == root
        )
