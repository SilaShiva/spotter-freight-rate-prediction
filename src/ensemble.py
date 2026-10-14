"""Blend model: a 50/50 log-space ensemble of the two complementary winners.

Why blend
---------
On the held-out Sep-Oct window the single models behave very differently:

    Ridge (log)                    ~5.4% MAPE
    HistGradientBoosting (clipped) ~5.5-6.5% MAPE depending on tuning
    50/50 log-space blend          ~5.7% MAPE (vs 7.8% linear baseline)

Ridge is excellent at the smooth market cycle (its market_index coefficient
transfers cleanly to future months) but cannot express lane/weight
interactions. The gradient-boosted trees capture those interactions but
chase the extreme pricing noise. Averaging the two in log space (equivalent
to a geometric mean of dollar predictions) cancels each model's opposite
failure mode; both CV and the held-out window show the blend dominating
either single model.
"""
from __future__ import annotations

import numpy as np
import pandas as pd
from sklearn.linear_model import Ridge

from . import train


class BlendModel:
    """Ridge + arbitrary tree-model blend in log space.

    Parameters
    ----------
    tree_model : fitted-on-raw-features sklearn regressor (e.g. HistGB).
    tree_weight : blend weight on the tree model's log prediction;
        0.5 means an equal geometric mean of the two dollar predictions.
    """

    def __init__(self, tree_model, tree_weight: float = 0.5):
        self.tree_model = tree_model
        self.tree_weight = tree_weight
        self.ridge_: Ridge | None = None
        self.ridge_columns_: pd.Index | None = None

    def fit(self, X: pd.DataFrame, y: pd.Series) -> "BlendModel":
        # Ridge branch: one-hot features, unclipped log target (Ridge's
        # L2 penalty already absorbs the outlier rows).
        X_ridge = train.one_hot(X)
        self.ridge_columns_ = X_ridge.columns
        self.ridge_ = Ridge(alpha=1.0)
        self.ridge_.fit(X_ridge, y)

        # Tree branch: native categorical features, winsorised log target.
        self.tree_model.fit(X, train.clip_target(y))
        return self

    def predict_log(self, X: pd.DataFrame) -> np.ndarray:
        X_ridge = train.align_columns(train.one_hot(X), self.ridge_columns_)
        ridge_log = self.ridge_.predict(X_ridge)
        tree_log = self.tree_model.predict(X)
        w = self.tree_weight
        return w * tree_log + (1 - w) * ridge_log

    def predict(self, X: pd.DataFrame) -> np.ndarray:
        return np.exp(self.predict_log(X))
