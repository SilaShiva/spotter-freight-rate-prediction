"""Model validation: held-out metrics, baselines, residual diagnostics."""
from __future__ import annotations

import json

import numpy as np
import pandas as pd
from sklearn.metrics import mean_absolute_error, mean_absolute_percentage_error, r2_score

from . import config


def evaluate(y_true: pd.Series, y_pred: np.ndarray) -> dict:
    y_true = np.asarray(y_true, dtype=float)
    y_pred = np.asarray(y_pred, dtype=float)
    ape = np.abs(y_true - y_pred) / y_true
    return {
        "mape": float(mean_absolute_percentage_error(y_true, y_pred)),
        "mae": float(mean_absolute_error(y_true, y_pred)),
        "rmse": float(np.sqrt(np.mean((y_true - y_pred) ** 2))),
        "r2": float(r2_score(y_true, y_pred)),
        "median_ape": float(np.median(ape)),
        "p90_ape": float(np.percentile(ape, 90)),
    }


def baseline_predictions(train: pd.DataFrame, val: pd.DataFrame) -> np.ndarray:
    """Per-equipment linear rate = a*distance + b sanity baseline."""
    coefs = {}
    for eq, sub in train.groupby("equipment"):
        coefs[eq] = tuple(np.polyfit(sub["distance"], sub[config.TARGET], 1))
    fallback = (1.0, train[config.TARGET].mean())
    return np.array([
        coefs.get(eq, fallback)[0] * d + coefs.get(eq, fallback)[1]
        for eq, d in zip(val["equipment"], val["distance"])
    ])


def residual_report(y_true: pd.Series, y_pred: np.ndarray,
                    val: pd.DataFrame) -> pd.DataFrame:
    """Mean APE by equipment type and distance band - hunts systematic bias."""
    frame = val.copy()
    frame["actual"] = np.asarray(y_true)
    frame["predicted"] = np.asarray(y_pred)
    frame["ape"] = np.abs(frame["actual"] - frame["predicted"]) / frame["actual"]
    frame["distance_band"] = pd.cut(frame["distance"], [0, 300, 700, 1500, 4000])
    rows = [("equipment", str(level), float(ape))
            for level, ape in frame.groupby("equipment", observed=True)["ape"].mean().items()]
    rows += [("distance_band", str(level), float(ape))
             for level, ape in frame.groupby("distance_band", observed=True)["ape"].mean().items()]
    return pd.DataFrame(rows, columns=["slice", "level", "mean_ape"])


def save_metrics(payload: dict, path) -> None:
    path.write_text(json.dumps(payload, indent=2, default=str))
