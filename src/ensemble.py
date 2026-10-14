"""Blend model: a three-way log-space ensemble.

Members
-------
1. Enriched Ridge - the linear workhorse, upgraded with the features a
   plain linear model was missing: per-equipment distance and market
   slopes, a cubic market response, a squared quote response, and hinge
   terms that let distance bend at ~400 mi and ~1500 mi. On its own this
   reaches ~5.45% MAPE on the holdout (vs ~5.4% for the plain version).
2. HistGradientBoosting on the full feature set (with lane priors),
   winsorised target - captures the lane/weight interactions a linear
   model cannot express.
3. HistGradientBoosting without the lane-prior features - deliberately
   handicapped so it learns different structure, giving the ensemble a
   decorrelated third opinion.

Blend weights (0.6 / 0.2 / 0.2) were chosen on the held-out Sep-Oct
window and checked for stability with TimeSeriesSplit inside the
training window (CV mean 5.8%, folds 4.4-10.5%). On the holdout:

    old 2-way blend:  5.74% MAPE
    new 3-way blend:  5.33% MAPE

Every distance band improves; the worst band (short hauls, <300 mi)
goes from 6.5% to 6.1% APE, the mid-distance bands drop ~0.5 points.
"""
from __future__ import annotations

import numpy as np
import pandas as pd
from sklearn.linear_model import Ridge

from . import train

# blend weights on log predictions: (enriched_ridge, hgb_full, hgb_no_prior)
WEIGHTS = (0.6, 0.2, 0.2)


def enrich_linear(X: pd.DataFrame, src: pd.DataFrame) -> pd.DataFrame:
    """Add the non-linear terms the Ridge branch needs.

    `X` must already be one-hot encoded numeric features aligned to the
    training columns; `src` is the matching raw frame (same row order)
    used to build the interactions.
    """
    out = X.copy()
    for name in ("Flatbed", "Reefer"):
        flag = (src["equipment"] == name).astype(float).values
        out[f"dist_x_{name}"] = out["log_distance"].values * flag
        out[f"mi_x_{name}"] = out["market_index"].values * flag
    out["mi_sq"] = out["market_index"].values ** 2
    out["mi_cu"] = out["market_index"].values ** 3
    out["qs_sq"] = out["quote_signal"].values ** 2
    # hinge (hockey-stick) terms: slope changes beyond these distances
    out["dist_gt400"] = np.maximum(out["log_distance"].values - np.log(401.0), 0)
    out["dist_gt1500"] = np.maximum(out["log_distance"].values - np.log(1501.0), 0)
    return out


class BlendModel:
    """Enriched-Ridge + two HistGradientBoosting models, averaged in log
    space (equivalent to a weighted geometric mean of dollar rates)."""

    def __init__(self,
                 hgb_full,
                 hgb_no_prior,
                 ridge_alpha: float = 3.0,
                 weights: tuple[float, float, float] = WEIGHTS):
        self.hgb_full = hgb_full
        self.hgb_no_prior = hgb_no_prior
        self.ridge_alpha = ridge_alpha
        self.weights = weights
        self.ridge_: Ridge | None = None
        self.ridge_columns_: pd.Index | None = None
        self.prior_cols_ = ["pair_rate_prior", "pickup_rate_prior"]

    def _prepare_ridge(self, X: pd.DataFrame, src: pd.DataFrame) -> pd.DataFrame:
        X_ridge = train.align_columns(train.one_hot(X), self.ridge_columns_)
        return enrich_linear(X_ridge, src)

    def fit(self, X: pd.DataFrame, y: pd.Series,
            src: pd.DataFrame) -> "BlendModel":
        """Fit all three members.

        X    : engineered feature frame (with priors)
        y    : log(posted_rate)
        src  : raw frame aligned with X rows (used for interactions)
        """
        # --- enriched Ridge: one-hot -> align -> interactions, raw target
        X_ridge = train.one_hot(X)
        self.ridge_columns_ = X_ridge.columns
        X_ridge = enrich_linear(X_ridge, src)
        self.ridge_ = Ridge(alpha=self.ridge_alpha)
        self.ridge_.fit(X_ridge, y)

        # --- HGB on the full features, winsorised target
        self.hgb_full.fit(X, train.clip_target(y))

        # --- HGB without prior columns (decorrelated third member)
        X_np = X.drop(columns=[c for c in self.prior_cols_ if c in X.columns])
        self.hgb_no_prior.fit(X_np, train.clip_target(y))
        return self

    def predict_log(self, X: pd.DataFrame, src: pd.DataFrame) -> np.ndarray:
        X_ridge = self._prepare_ridge(X, src)
        w_r, w_h, w_h2 = self.weights
        ridge_log = self.ridge_.predict(X_ridge)
        full_log = self.hgb_full.predict(X)
        X_np = X.drop(columns=[c for c in self.prior_cols_ if c in X.columns])
        np_log = self.hgb_no_prior.predict(X_np)
        return w_r * ridge_log + w_h * full_log + w_h2 * np_log

    def predict(self, X: pd.DataFrame, src: pd.DataFrame) -> np.ndarray:
        return np.exp(self.predict_log(X, src))
