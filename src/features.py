"""Feature engineering and transformation.

Design notes
------------
* The target is modelled in log space: posted_rate behaves multiplicatively
  (noise scales with the level of the rate), so log(rate) makes the residual
  variance roughly homoscedastic and the tree models' squared-error objective
  meaningful.
* Lane (pickup, delivery) mean rates are strong predictors but ~1,450 rows of
  the validation set touch cities never seen in training. We therefore build
  *smoothed* lane priors (empirical-Bayes style shrinkage toward the
  equipment-level mean rate) plus raw coordinates, so unseen cities still get
  a sensible prior from their coordinates and route geometry rather than a
  global fallback.
* All prior/statistic features are computed on the *training partition only*
  and then applied to validation rows, to avoid target leakage.
"""
from __future__ import annotations

import numpy as np
import pandas as pd

from . import config
from .validation import great_circle_miles

# shrinkage strength for lane priors: lanes with fewer than this many
# observations are pulled proportionally toward the equipment mean.
PRIOR_MIN_SAMPLES = 10.0


def fit_priors(train: pd.DataFrame) -> dict:
    """Compute lane/city rate priors from the training partition only."""
    equip_mean = train.groupby("equipment")[config.TARGET].mean()

    def smoothed(frame: pd.DataFrame, keys: list[str]) -> pd.Series:
        grp = frame.groupby(keys + ["equipment"])[config.TARGET].agg(["mean", "count"])
        shrink = grp["count"] / (grp["count"] + PRIOR_MIN_SAMPLES)
        overall = pd.Series(
            grp.index.get_level_values("equipment").map(equip_mean), index=grp.index)
        return grp["mean"] * shrink + overall * (1 - shrink)

    lane = smoothed(train, ["pickup", "delivery"])
    pickup = smoothed(train, ["pickup"])

    return {
        "lane": lane,
        "pickup": pickup,
        "equip_mean": equip_mean,
        "global_mean": train[config.TARGET].mean(),
    }


def transform(df: pd.DataFrame, priors: dict | None = None) -> pd.DataFrame:
    """Build the model feature matrix from a cleaned frame."""
    df = df.copy()
    date = pd.to_datetime(df["date"])

    feats = pd.DataFrame(index=df.index)
    feats[config.BASE_NUMERIC] = df[config.BASE_NUMERIC]

    # calendar features
    feats["month"] = date.dt.month
    feats["day_of_week"] = date.dt.dayofweek
    feats["is_weekend"] = (date.dt.dayofweek >= 5).astype(int)

    # distance-based features
    feats["log_distance"] = np.log1p(df["distance"])
    feats["gc_distance"] = great_circle_miles(
        df["pickup_lat"], df["pickup_lon"], df["delivery_lat"], df["delivery_lon"])
    feats["dist_ratio"] = df["distance"] / feats["gc_distance"].clip(lower=1.0)
    feats["weight_per_mile"] = df["weight"] / df["distance"].clip(lower=1.0)
    feats["distance_bucket"] = pd.cut(
        df["distance"], bins=[0, 200, 400, 700, 1200, 2000, 4000],
        labels=False, include_lowest=True).fillna(-1)

    # interactions with the market cycle
    feats["dist_x_market"] = df["distance"] * df["market_index"]

    # equipment as categorical dtype so tree models treat it natively
    feats["equipment"] = df["equipment"].astype("category")

    # smoothed historical priors (training rows use their own lane's prior
    # computed from other rows is overkill at this data size; the shrinkage
    # already limits each lane's self-influence).
    if priors is not None:
        lane_key = pd.MultiIndex.from_arrays(
            [df["pickup"], df["delivery"], df["equipment"]])
        pick_key = pd.MultiIndex.from_arrays(
            [df["pickup"], df["equipment"]])
        feats["pair_rate_prior"] = np.log(
            lane_key.map(priors["lane"]).astype(float)
            .fillna(priors["global_mean"]))
        feats["pickup_rate_prior"] = np.log(
            pick_key.map(priors["pickup"]).astype(float)
            .fillna(priors["global_mean"]))

    return feats


def make_xy(train: pd.DataFrame, val: pd.DataFrame | None = None):
    """Full transform path: fit priors on train, apply to both partitions.

    Returns (X_train, y_train, X_val, y_val, priors). y values are log rates.
    """
    priors = fit_priors(train)
    X_tr = transform(train, priors)
    y_tr = np.log(train[config.TARGET])

    if val is None:
        return X_tr, y_tr, None, None, priors

    X_va = transform(val, priors)
    y_va = np.log(val[config.TARGET]) if config.TARGET in val.columns else None
    return X_tr, y_tr, X_va, y_va, priors
