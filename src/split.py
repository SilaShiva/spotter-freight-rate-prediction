"""Data split strategy.

The deployment target (data/validation.csv) is strictly future data:
2025-11-01 through 2025-12-31. A random split would let the model peek at the
slow-moving market cycle (market_index peaks in May, troughs in January) and
report optimistic metrics. We therefore use an out-of-time split:

    train  : 2025-01-01 .. 2025-08-31   (8 months, ~38.4k loads)
    valid  : 2025-09-01 .. 2025-10-31   (2 months, ~9.6k loads)

The validation window has the same length (2 months) and the same "next
months after training" relationship to the training window that the real
validation set has. Within the training window, GridSearchCV uses TimeSeriesSplit
so that every CV fold is also strictly forward-looking.
"""
from __future__ import annotations

import pandas as pd

from . import config


def time_split(df: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Split a dated frame into (train, validation) by config.VAL_START."""
    date = pd.to_datetime(df["date"])
    train = df[date < config.VAL_START].reset_index(drop=True)
    valid = df[date >= config.VAL_START].reset_index(drop=True)
    return train, valid
