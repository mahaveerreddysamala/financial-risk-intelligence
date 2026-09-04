# Phase 42: Champion / Challenger Selection

Phase 42 turns model benchmarking into a reproducible model-selection decision.

## Selection policy

The operating-point results are ranked by business impact first:

1. Lowest realized decision cost
2. Lowest Brier score
3. Highest F1
4. Highest lift
5. Highest precision
6. Highest recall

The policy produces a deterministic champion and an ordered challenger list. This prevents selecting a model solely from ROC-AUC when the deployed decision policy is cost-sensitive and investigators operate on ranked cases.

## Current benchmark decision

Using the Phase 41 20K synthetic benchmark, LightGBM is the current candidate champion because it achieved the lowest realized cost (4141), lowest Brier score (0.0213), and highest F1 (0.0968) among the evaluated operating points.

This is a portfolio selection decision, not a production or regulatory approval.

## Production boundary

The selector is intentionally model-agnostic and does not deploy or promote artifacts. A later phase can connect this decision to the existing model registry / MLflow lifecycle with approval gates and rollback controls.
