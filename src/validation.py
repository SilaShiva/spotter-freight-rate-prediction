"""Data validation: profile the raw data and apply cleaning rules.

Findings from the profiling pass (see report/ for the full write-up):

1. `weight` has 300 missing values and 292 negative values (impossible
   physics for a truckload shipment). Negative weights are treated as
   missing and both are imputed with the median weight of the same
   equipment type. Weight is a weak predictor (|r| ~ 0.07 with rate) so
   the imputation risk is low.
2. `market_index` has 374 missing values. It moves slowly day to day, so
   missing values are forward/backward filled *within the same date*
   group using the daily median, then fall back to the global median.
3. `distance` occasionally disagrees wildly with the great-circle distance
   implied by the lat/lon columns (ratios up to 9.6x). The provided
   `distance` column is kept as-is (it is the operationally billed
   distance and the same field is present in validation), but the
   great-circle distance is added as a separate feature so the model can
   reconcile the two.
4. `posted_rate` is strictly positive but heavily right-skewed with
   multiplicative noise; we model log(rate) and invert at prediction time.
"""
from __future__ import annotations

import numpy as np
import pandas as pd

from . import config


def great_circle_miles(lat1, lon1, lat2, lon2) -> np.ndarray:
    """Vectorised great-circle distance in miles."""
    lat1, lon1, lat2, lon2 = map(np.radians, [lat1, lon1, lat2, lon2])
    dlat = lat2 - lat1
    dlon = lon2 - lon1
    a = np.sin(dlat / 2) ** 2 + np.cos(lat1) * np.cos(lat2) * np.sin(dlon / 2) ** 2
    return 2 * 3958.8 * np.arcsin(np.sqrt(np.clip(a, 0, 1)))


def profile(df: pd.DataFrame) -> dict:
    """Return a small dict of data-quality counts used in the report."""
    return {
        "rows": len(df),
        "missing_weight": int(df["weight"].isna().sum()),
        "negative_weight": int((df["weight"] < 0).sum()),
        "missing_market_index": int(df["market_index"].isna().sum()),
        "non_positive_rate": int((df[config.TARGET] <= 0).sum())
        if config.TARGET in df.columns
        else 0,
    }


def clean(df: pd.DataFrame, is_train: bool) -> pd.DataFrame:
    """Apply the cleaning rules. Works on both train and validation frames."""
    df = df.copy()

    # --- weight: negative -> missing, then median impute per equipment type ---
    df.loc[df["weight"] < 0, "weight"] = np.nan
    if is_train:
        medians = df.groupby("equipment")["weight"].transform("median")
        df["weight"] = df["weight"].fillna(medians)
        df["weight"] = df["weight"].fillna(df["weight"].median())
        # persist equipment medians so validation uses *train* statistics
        global _WEIGHT_MEDIANS
        _WEIGHT_MEDIANS = df.groupby("equipment")["weight"].median().to_dict()
    else:
        medians = df["equipment"].map(_WEIGHT_MEDIANS)
        df["weight"] = df["weight"].fillna(medians)
        df["weight"] = df["weight"].fillna(df["weight"].median())

    # --- market_index: daily median within the frame, then global median ---
    if df["market_index"].isna().any():
        daily = df.groupby("date")["market_index"].transform("median")
        df["market_index"] = df["market_index"].fillna(daily)
        df["market_index"] = df["market_index"].fillna(df["market_index"].median())

    # --- quote_signal is fully populated in both files, but be defensive ---
    if df["quote_signal"].isna().any():
        df["quote_signal"] = df["quote_signal"].fillna(df["quote_signal"].median())

    # --- sanity: rate must be positive to be log-transformed ---
    if is_train and (df[config.TARGET] <= 0).any():
        raise ValueError("posted_rate contains non-positive values after cleaning")

    return df


# module-level state shared between the train and validation passes; a full
# production system would persist this with the model artifact instead.
_WEIGHT_MEDIANS: dict[str, float] = {}
