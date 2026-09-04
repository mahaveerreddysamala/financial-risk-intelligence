# Phase 43: Automated Model Promotion

Phase 43 turns champion selection into an auditable promotion workflow.

## Flow

```text
Operating-point results
        |
        v
Champion selection
        |
        v
Quality gates
(PR-AUC, recall, precision, optional PSI)
        |
   +----+----+
   |         |
 fail      pass
   |         |
 block   registration
             |
             v
       champion alias
```

## Governance contract

- Champion selection remains business-first: realized cost, calibration, F1, lift, precision, and recall.
- Promotion is blocked when any configured quality gate fails.
- PSI can be supplied as an optional drift safeguard.
- A passing gate with no model URI is a safe dry-run state; no registry mutation occurs.
- Actual registration is dependency-injected so CI tests do not require a live MLflow service.
- Production promotion can use the existing MLflow `register_model_version` helper with the `champion` alias.

This phase establishes the control boundary; deployment to a managed MLflow environment remains an infrastructure concern.
