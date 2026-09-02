"""Temporal train/validation/test split utilities."""
from __future__ import annotations

import pandas as pd


def temporal_split(
    df: pd.DataFrame,
    time_column: str = "timestamp",
    train_end: str = "2025-09-01",
    validation_end: str = "2025-11-01",
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """Split chronologically so future observations never enter training."""
    if time_column not in df.columns:
        raise ValueError(f"Missing time column: {time_column}")

    result = df.copy()
    result[time_column] = pd.to_datetime(result[time_column], errors="raise")
    train_cutoff = pd.Timestamp(train_end)
    validation_cutoff = pd.Timestamp(validation_end)

    train = result[result[time_column] < train_cutoff].copy()
    validation = result[
        (result[time_column] >= train_cutoff) & (result[time_column] < validation_cutoff)
    ].copy()
    test = result[result[time_column] >= validation_cutoff].copy()

    if train.empty or validation.empty or test.empty:
        raise ValueError("Temporal split must produce non-empty train, validation, and test sets")

    return train, validation, test
