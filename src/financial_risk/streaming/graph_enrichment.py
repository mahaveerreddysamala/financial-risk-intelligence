"""Streaming graph/community enrichment for transaction risk events."""
from __future__ import annotations

from dataclasses import dataclass
from math import log1p
from typing import Any

from financial_risk.graph.online_communities import ENTITY_COLUMNS, OnlineCommunityTracker
from financial_risk.models.graph_risk import GraphRiskAdjustment, integrate_graph_risk
from financial_risk.streaming.events import EventEnvelope


@dataclass(frozen=True)
class GraphEnrichmentResult:
    """Auditable graph enrichment emitted for one transaction event."""

    event_id: str
    transaction_id: str
    community_id: int
    community_customer_count: int
    community_entity_count: int
    network_risk: float
    community_risk: float
    graph_risk: GraphRiskAdjustment

    def to_dict(self) -> dict[str, Any]:
        """Return a JSON-serializable enrichment payload."""
        return {
            "event_id": self.event_id,
            "transaction_id": self.transaction_id,
            "community_id": self.community_id,
            "community_customer_count": self.community_customer_count,
            "community_entity_count": self.community_entity_count,
            "network_risk": self.network_risk,
            "community_risk": self.community_risk,
            "graph_risk": {
                "base_score": self.graph_risk.base_score,
                "network_score": self.graph_risk.network_score,
                "community_score": self.graph_risk.community_score,
                "adjusted_score": self.graph_risk.adjusted_score,
                "risk_band": self.graph_risk.decision.level,
                "action": self.graph_risk.decision.action,
            },
        }


class StreamingGraphEnricher:
    """Incrementally maintain graph state and enrich transaction events.

    The enricher updates a union-find community state per event. It derives
    bounded network and community risk signals from the current connected
    component, then passes those signals through the Phase 45 graph-risk
    decision function.
    """

    def __init__(self, tracker: OnlineCommunityTracker | None = None) -> None:
        self.tracker = tracker or OnlineCommunityTracker()
        self._seen_events: set[str] = set()
        self._entity_seen: dict[str, set[str]] = {column: set() for column in ENTITY_COLUMNS}
        self._customer_seen: set[str] = set()

    @staticmethod
    def _validate_signal(value: object, name: str) -> float:
        try:
            signal = float(value)
        except (TypeError, ValueError) as exc:
            raise ValueError(f"{name} must be numeric") from exc
        if not 0.0 <= signal <= 1.0:
            raise ValueError(f"{name} must be between 0 and 1")
        return signal

    def _network_risk(self, transaction: dict[str, object]) -> float:
        """Estimate bounded entity-reuse risk using state accumulated so far."""
        reuse_scores: list[float] = []
        for column in ("device_id", "ip_id", "merchant_id"):
            value = transaction.get(column)
            if value is None:
                continue
            key = str(value)
            prior = sum(1 for existing in self._entity_seen[column] if existing == key)
            # Membership is maintained as a set, so prior is 0/1. Community size
            # below captures broader reuse while this signal captures direct reuse.
            reuse_scores.append(min(1.0, prior + 0.0))
        direct_reuse = sum(reuse_scores) / len(reuse_scores) if reuse_scores else 0.0
        return round(direct_reuse, 6)

    def _community_risk(self, customer_count: int, entity_count: int) -> float:
        """Convert connected-component size into a bounded community signal."""
        if customer_count <= 1 and entity_count <= 5:
            return 0.0
        customer_component = min(1.0, log1p(max(customer_count - 1, 0)) / log1p(20))
        entity_component = min(1.0, log1p(max(entity_count - 5, 0)) / log1p(50))
        return round(0.7 * customer_component + 0.3 * entity_component, 6)

    def enrich_transaction(
        self,
        transaction: dict[str, object],
        *,
        fraud_probability: float,
        anomaly_score: float,
        velocity_score: float,
        event_id: str,
    ) -> GraphEnrichmentResult:
        """Update graph state and enrich one transaction with graph risk signals."""
        if not event_id.strip():
            raise ValueError("event_id must not be empty")
        if event_id in self._seen_events:
            raise ValueError(f"duplicate event_id: {event_id}")
        for name, value in {
            "fraud_probability": fraud_probability,
            "anomaly_score": anomaly_score,
            "velocity_score": velocity_score,
        }.items():
            self._validate_signal(value, name)

        required = {"transaction_id", "customer_id", *ENTITY_COLUMNS}
        missing = required.difference(transaction)
        if missing:
            raise ValueError(f"Missing required graph fields: {sorted(missing)}")

        network_risk = self._network_risk(transaction)
        community_id = self.tracker.add_transaction(transaction)
        members = self.tracker.community_members(transaction["customer_id"])
        customer_members = {
            member for member in members if member.startswith("customer:")
        }

        self._seen_events.add(event_id)
        self._customer_seen.add(str(transaction["customer_id"]))
        for column in ENTITY_COLUMNS:
            value = transaction.get(column)
            if value is not None:
                self._entity_seen[column].add(str(value))

        community_customer_count = len(customer_members)
        community_entity_count = len(members)
        community_risk = self._community_risk(
            community_customer_count,
            community_entity_count,
        )
        graph_risk = integrate_graph_risk(
            fraud_probability,
            anomaly_score,
            network_risk,
            velocity_score,
            community_risk,
        )
        return GraphEnrichmentResult(
            event_id=event_id,
            transaction_id=str(transaction["transaction_id"]),
            community_id=community_id,
            community_customer_count=community_customer_count,
            community_entity_count=community_entity_count,
            network_risk=network_risk,
            community_risk=community_risk,
            graph_risk=graph_risk,
        )

    def enrich_event(self, event: EventEnvelope) -> EventEnvelope:
        """Consume a transaction event and return a graph-enriched event envelope."""
        if event.event_type != "transaction.created":
            raise ValueError(f"unsupported event_type: {event.event_type}")
        payload = event.payload
        transaction_fields = {
            key: payload[key]
            for key in {"transaction_id", "customer_id", *ENTITY_COLUMNS}
            if key in payload
        }
        fraud_probability = self._validate_signal(payload.get("fraud_probability"), "fraud_probability")
        anomaly_score = self._validate_signal(payload.get("anomaly_score"), "anomaly_score")
        velocity_score = self._validate_signal(payload.get("velocity_risk"), "velocity_risk")
        result = self.enrich_transaction(
            transaction_fields,
            fraud_probability=fraud_probability,
            anomaly_score=anomaly_score,
            velocity_score=velocity_score,
            event_id=event.event_id,
        )
        return EventEnvelope(
            event_id=event.event_id,
            event_type="transaction.graph_enriched",
            schema_version=1,
            occurred_at=event.occurred_at,
            payload={**payload, "graph_enrichment": result.to_dict()},
        )
