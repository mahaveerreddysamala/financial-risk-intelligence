"""SHAP explainability and deterministic fraud reason-code utilities."""
from __future__ import annotations

from dataclasses import dataclass

import pandas as pd
import shap

from financial_risk.models.baseline import CATEGORICAL_FEATURES, NUMERIC_FEATURES

MODEL_FEATURES = NUMERIC_FEATURES + CATEGORICAL_FEATURES


@dataclass(frozen=True)
class ReasonCode:
    feature: str
    reason: str
    shap_value: float


def _feature_label(feature: str) -> str:
    labels = {
        "amount": "transaction amount",
        "is_international": "international transaction indicator",
        "is_night": "night-time transaction indicator",
        "shared_device_account_count": "shared-device account count",
        "customer_txn_count_7d": "customer 7-day transaction count",
        "customer_avg_amount_30d": "customer 30-day average amount",
        "customer_std_amount_30d": "customer 30-day amount variability",
        "customer_unique_merchants_7d": "customer 7-day merchant diversity",
        "customer_unique_devices_30d": "customer 30-day device diversity",
        "customer_international_rate_30d": "customer 30-day international rate",
        "customer_night_txn_rate_30d": "customer 30-day night transaction rate",
        "amount_vs_customer_avg": "amount vs. customer historical average",
        "amount_zscore": "amount deviation from customer baseline",
        "txn_count_5m": "5-minute transaction velocity",
        "txn_count_1h": "1-hour transaction velocity",
        "txn_count_24h": "24-hour transaction velocity",
        "merchant_category": "merchant category",
        "payment_method": "payment method",
        "channel": "transaction channel",
        "country": "transaction country",
    }
    return labels.get(feature, feature.replace("_", " "))


def _reason_for_value(feature: str, value: object, positive: bool) -> str:
    label = _feature_label(feature)
    if feature == "amount_vs_customer_avg":
        return f"{label} is unusually high" if positive else f"{label} is unusually low"
    if feature == "amount_zscore":
        return f"{label} is elevated" if positive else f"{label} is lower than normal"
    if feature in {"txn_count_5m", "txn_count_1h", "txn_count_24h"}:
        return f"{label} is elevated ({value})" if positive else f"{label} is lower ({value})"
    if feature in {"shared_device_account_count", "customer_unique_devices_30d", "customer_unique_merchants_7d"}:
        return f"{label} is elevated ({value})" if positive else f"{label} is low ({value})"
    if feature in {"is_international", "is_night"}:
        indicator = label.replace(" indicator", "")
        return f"{indicator} is present" if positive else f"{indicator} is absent"
    return f"{label} ({value}) increases fraud risk" if positive else f"{label} ({value}) reduces fraud risk"


def _raw_feature_name(transformed_name: str) -> str:
    """Map a preprocessed feature name back to its raw model feature."""
    raw_name = transformed_name.split("__", 1)[-1]
    if raw_name in MODEL_FEATURES:
        return raw_name
    for feature in CATEGORICAL_FEATURES:
        prefix = f"{feature}_"
        if raw_name.startswith(prefix):
            return feature
    return raw_name


def explain_xgboost(model, transactions: pd.DataFrame, top_n: int = 5) -> list[list[ReasonCode]]:
    """Return top SHAP reason codes for each transaction."""
    if top_n < 1:
        raise ValueError("top_n must be at least 1")
    missing = [feature for feature in MODEL_FEATURES if feature not in transactions.columns]
    if missing:
        raise ValueError(f"Missing model features: {missing}")

    preprocessor = model.named_steps["preprocess"]
    classifier = model.named_steps["model"]
    transformed = preprocessor.transform(transactions[MODEL_FEATURES])
    feature_names = list(preprocessor.get_feature_names_out())
    dense = transformed.toarray() if hasattr(transformed, "toarray") else transformed
    explainer = shap.TreeExplainer(classifier)
    shap_values = explainer.shap_values(dense)
    if isinstance(shap_values, list):
        shap_values = shap_values[0]

    expanded_values = transactions[MODEL_FEATURES].reset_index(drop=True)
    results: list[list[ReasonCode]] = []
    for row_idx in range(len(transactions)):
        ranked = sorted(
            enumerate(shap_values[row_idx]),
            key=lambda item: abs(float(item[1])),
            reverse=True,
        )[:top_n]
        reasons: list[ReasonCode] = []
        for feature_idx, contribution in ranked:
            transformed_name = feature_names[feature_idx]
            raw_feature = _raw_feature_name(transformed_name)
            value = expanded_values.iloc[row_idx].get(raw_feature, transformed_name)
            reasons.append(
                ReasonCode(
                    feature=raw_feature,
                    reason=_reason_for_value(raw_feature, value, float(contribution) >= 0),
                    shap_value=float(contribution),
                )
            )
        results.append(reasons)
    return results


def reason_code_table(reason_codes: list[ReasonCode]) -> pd.DataFrame:
    """Convert reason codes into a compact analyst-facing table."""
    return pd.DataFrame(
        [
            {"feature": item.feature, "reason": item.reason, "shap_value": item.shap_value}
            for item in reason_codes
        ]
    )
