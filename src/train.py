"""Model selection and hyper-parameter tuning with GridSearchCV.

Candidate models
----------------
1. Ridge on log(rate)            - log-linear baseline; rates are almost
   exactly log-linear in distance, so a regularised linear model is a
   surprisingly strong, well-calibrated learner here.
2. HistGradientBoostingRegressor - captures the residual non-linearities
   (lane effects, weight/distance interactions) that the linear model misses.
3. RandomForestRegressor         - bagged-trees robustness check.

All regression happens on log(posted_rate): the raw rate has multiplicative
noise (spread scales with the level), and extreme rows (roughly 1% of loads
price far above or below any feature pattern) destabilise a squared-error fit.
We therefore winsorise the training target at the 0.5/99.5 percentiles.

Scoring is MAPE on the original dollar rate (log predictions are inverted
inside the scorer) because that is the business metric and it weights all
lanes equally regardless of rate level.

The grid search uses TimeSeriesSplit(5) inside the *training window only*,
so every fold is forward-looking and no information from the held-out
Sep-Oct window influences tuning. The winning configuration of each family
is then evaluated once on the held-out window.
"""
from __future__ import annotations

import numpy as np
import pandas as pd
from sklearn.ensemble import HistGradientBoostingRegressor, RandomForestRegressor
from sklearn.linear_model import Ridge
from sklearn.metrics import make_scorer, mean_absolute_percentage_error
from sklearn.model_selection import GridSearchCV, TimeSeriesSplit

from . import config

TARGET_CLIP = (0.005, 0.995)  # winsorisation quantiles for the log target


def mape_original_scale(y_true_log, y_pred_log):
    y_true = np.exp(np.asarray(y_true_log))
    y_pred = np.exp(np.asarray(y_pred_log))
    return mean_absolute_percentage_error(y_true, y_pred)


MAPE_SCORER = make_scorer(mape_original_scale, greater_is_better=False)


def clip_target(y: pd.Series) -> pd.Series:
    lo, hi = y.quantile(TARGET_CLIP[0]), y.quantile(TARGET_CLIP[1])
    return y.clip(lo, hi)


CANDIDATES = {
    "ridge": {
        "estimator": Ridge(),
        "params": {"alpha": [0.1, 0.3, 1.0, 3.0, 10.0]},
        "one_hot": True,  # Ridge needs numeric input: equipment is one-hot encoded
    },
    "hist_gradient_boosting": {
        "estimator": HistGradientBoostingRegressor(
            random_state=config.RANDOM_STATE, early_stopping=False),
        "params": {
            "learning_rate": [0.03, 0.06],
            "max_iter": [300, 600],
            "max_leaf_nodes": [31, 63],
            "min_samples_leaf": [80, 240],
            "l2_regularization": [0.0, 1.0],
        },
        "one_hot": False,  # native categorical support
    },
    "random_forest": {
        "estimator": RandomForestRegressor(
            random_state=config.RANDOM_STATE, n_jobs=-1),
        "params": {
            "n_estimators": [300],
            "max_depth": [None, 20],
            "min_samples_leaf": [1, 5, 20],
        },
        "one_hot": True,
    },
}


def one_hot(X: pd.DataFrame) -> pd.DataFrame:
    """One-hot the equipment column; used by the models without native
    categorical support."""
    X_num = X.astype({c: float for c in X.columns if c != "equipment"})
    return pd.get_dummies(X_num, columns=["equipment"], drop_first=True)


def align_columns(X: pd.DataFrame, columns: pd.Index) -> pd.DataFrame:
    """Reindex to the training columns, filling unseen one-hot levels with 0."""
    return X.reindex(columns=columns, fill_value=0)


def run_grid_search(X: pd.DataFrame, y: pd.Series, n_jobs: int = -1,
                    verbose: int = 1) -> tuple[dict, pd.DataFrame]:
    """Grid-search every candidate family; return (best, results_frame).

    `best` is a dict with the winning model name, params and CV MAPE per
    family; each family's best estimator is also returned for the final
    comparison table.
    """
    tscv = TimeSeriesSplit(n_splits=5)
    y_clipped = clip_target(y)

    all_rows = []
    winners: dict[str, dict] = {}
    for name, spec in CANDIDATES.items():
        X_fit = one_hot(X) if spec["one_hot"] else X
        n_combos = int(np.prod([len(v) for v in spec["params"].values()]))
        print(f"\n=== Grid search: {name} ({n_combos} parameter combos x 5 folds) ===")
        gs = GridSearchCV(
            spec["estimator"], spec["params"],
            scoring=MAPE_SCORER, cv=tscv, n_jobs=n_jobs,
            refit=True, return_train_score=True, verbose=verbose,
        )
        gs.fit(X_fit, y_clipped)

        rows = pd.DataFrame(gs.cv_results_)
        rows["model"] = name
        all_rows.append(rows)

        mean_mape = -gs.best_score_
        print(f"{name}: best CV MAPE = {mean_mape:.4f} with {gs.best_params_}")
        winners[name] = {
            "name": name,
            "model": gs.best_estimator_,
            "params": gs.best_params_,
            "cv_mape": mean_mape,
            "one_hot": spec["one_hot"],
        }

    results = pd.concat(all_rows, ignore_index=True)
    best = min(winners.values(), key=lambda w: w["cv_mape"])
    return best, results
