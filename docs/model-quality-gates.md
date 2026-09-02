# Model Quality Gates and Promotion

Phase 18 adds an explicit quality gate between model evaluation and MLflow model promotion.

## Promotion flow

```text
Train
  |
  v
Evaluate
  |
  v
Quality Gates
  +-- minimum test PR-AUC
  +-- minimum test recall
  +-- minimum test precision
  +-- optional maximum feature PSI
  |
  +---- FAIL --> do not promote
  |
  +---- PASS --> register model + assign champion alias
```

## Default portfolio thresholds

The defaults are intentionally configurable and are **portfolio operating assumptions**, not regulatory or production fraud-policy thresholds:

| Gate | Default |
|---|---:|
| Minimum test PR-AUC | 0.05 |
| Minimum test recall | 0.05 |
| Minimum test precision | 0.02 |
| Maximum feature PSI | 0.20 |

Every required metric must be present. A missing metric fails the gate rather than silently passing.

When drift evidence is supplied, the gate uses the worst observed feature PSI. The promotion decision therefore considers both model discrimination and feature-distribution stability.

## Promotion boundary

`promote_if_approved()` refuses to register a model when any gate fails. When all gates pass, it delegates registration and alias assignment to the Phase 16 MLflow integration. The default alias is `champion`.

This phase does not claim automated production deployment, approval authority, or regulatory compliance. The thresholds are examples for the portfolio implementation and should be calibrated with business loss, investigator capacity, and governance requirements in a real system.
