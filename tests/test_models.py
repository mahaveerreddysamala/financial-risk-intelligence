from __future__ import annotations

import pandas as pd

from financial_risk.models.split import temporal_split


def test_temporal_split_is_chronological() -> None:
    df = pd.DataFrame(
        {
            "timestamp": pd.to_datetime(
                ["2025-01-01", "2025-10-01", "2025-11-15", "2025-12-01"]
            ),
            "is_fraud": [0, 1, 0, 1],
        }
    )
    train, validation, test = temporal_split(df)
    assert len(train) == 1
    assert len(validation) == 1
    assert len(test) == 2
    assert train["timestamp"].max() < validation["timestamp"].min()
    assert validation["timestamp"].max() < test["timestamp"].min()
