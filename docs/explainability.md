# SHAP Explainability & Reason Codes

Phase 6 adds transaction-level explainability to the XGBoost fraud model.

## Design

The fitted XGBoost pipeline is transformed using the same preprocessing configuration used during model training. SHAP TreeExplainer then estimates each transformed feature's contribution to the fraud prediction. The largest absolute contributions are surfaced as analyst-facing reason codes.

The output is intentionally deterministic for a fixed model and input row. Each reason code includes:

- the contributing feature
- a human-readable explanation
- the SHAP contribution value

## Interpretation

Positive SHAP values move the prediction toward fraud; negative values move it away from fraud. Reason codes are explanations of model behavior, not proof that a transaction is fraudulent.

For categorical variables, one-hot encoded feature names may identify the observed category. Numeric behavioral features are rendered with contextual labels such as transaction velocity, historical amount deviation, or shared-device reuse.

## Example

```text
Transaction Risk: HIGH

Top Model Drivers:
1. Amount deviation from customer baseline is elevated
2. 1-hour transaction velocity is elevated (9)
3. Shared-device account count is elevated (4)
4. International transaction indicator is present
5. Customer 7-day transaction count is elevated (18)
```

The reason-code layer is designed to feed a later investigation API and GenAI investigation copilot. Downstream narrative generation should preserve the underlying SHAP evidence and avoid inventing facts.

## Limitations

SHAP values explain the fitted model, not causality. This portfolio implementation is intended for model transparency and investigator prioritization; it does not establish regulatory explainability compliance or production suitability by itself.
