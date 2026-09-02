# Calibration & Cost-Sensitive Decisioning

Phase 8 separates model probability quality from business decisioning.

## Probability calibration

Calibration diagnostics report Brier score, mean predicted fraud probability, observed fraud rate, and reliability bins. A calibrated probability is intended to be more useful for downstream risk policies than an arbitrary classifier score.

The calibration utility supports sigmoid and isotonic calibration through scikit-learn's `CalibratedClassifierCV`. Calibration should use a validation/calibration period distinct from the final test period.

## Cost-sensitive decisions

The portfolio policy models three actions:

- `approve`: a fraudulent transaction is missed, creating a false-negative cost
- `review`: an investigation is initiated, creating a review cost
- `hold`: a legitimate transaction is unnecessarily blocked, creating a false-positive cost

For fraud probability `p`:

```text
Approve expected cost = p * false_negative_cost
Review expected cost  = review_cost
Hold expected cost    = (1 - p) * false_positive_cost
```

The minimum expected-cost action is selected. This makes the operating policy explicit and allows the business to change decision behavior without retraining the classifier.

## Example policy

The default portfolio policy uses:

```text
false_positive_cost = 5
false_negative_cost = 100
review_cost = 3
```

These values are illustrative portfolio assumptions, not financial-institution loss estimates or regulatory requirements.

## Important distinction

Model probability and operational risk are not interchangeable. Calibration addresses whether probabilities are reliable; cost-sensitive decisioning addresses what the organization should do given those probabilities and business costs.
