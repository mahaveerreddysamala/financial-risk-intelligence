"""Composable feature engineering pipeline."""

from __future__ import annotations

import pandas as pd

from financial_risk.features.behavioral import add_behavioral_features
from financial_risk.features.geographic import add_geographic_features
from financial_risk.features.network import add_network_features
from financial_risk.features.velocity import add_velocity_features


def build_feature_table(df: pd.DataFrame) -> pd.DataFrame:
    """Build the complete leakage-aware feature table in a deterministic order."""
    result = add_behavioral_features(df)
    result = add_velocity_features(result)
    result = add_geographic_features(result)
    return add_network_features(result)
