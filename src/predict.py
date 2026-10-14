"""Prediction export helpers."""
from __future__ import annotations

import numpy as np
import pandas as pd

from . import config
from .features import transform

# Lane coordinates and per-day market context come from the development data:
# coordinates from the train/test city table, market_index / quote_signal from
# the validation set's December rows (same market the chart lane lives in).
LANE_COORDS = {
    "Lexington": (36.99152, -84.99876),
    "Fort Wayne": (41.31561, -85.36206),
}


def _enrich_december(december: pd.DataFrame, validation: pd.DataFrame) -> pd.DataFrame:
    """Give the 31 December rows the full feature frame: coordinates,
    market_index and quote_signal (daily medians from December validation rows)."""
    dec = validation[validation["date"].dt.month == 12]
    daily_mi = dec.groupby("date")["market_index"].median()
    daily_qs = dec.groupby("date")["quote_signal"].median()
    out = december.copy()
    out["pickup_lat"], out["pickup_lon"] = zip(
        *out["pickup"].map(LANE_COORDS))
    out["delivery_lat"], out["delivery_lon"] = zip(
        *out["delivery"].map(LANE_COORDS))
    out["market_index"] = out["date"].map(daily_mi)
    out["quote_signal"] = out["date"].map(daily_qs)
    # any date without validation coverage falls back to the December medians
    out["market_index"] = out["market_index"].fillna(dec["market_index"].median())
    out["quote_signal"] = out["quote_signal"].fillna(dec["quote_signal"].median())
    return out


def write_december_predictions(model, december: pd.DataFrame, priors,
                               validation: pd.DataFrame) -> pd.DataFrame:
    """Score the 31 fixed December chart lanes and fill predicted_rate.

    Keeps the exact column layout score.py validates:
    pickup,delivery,distance,equipment,weight,date,predicted_rate
    """
    enriched = _enrich_december(december, validation)
    X = transform(enriched, priors)
    preds = np.round(model.predict(X, enriched), 2)
    out = december.copy()
    out["predicted_rate"] = preds
    cols = ["pickup", "delivery", "distance", "equipment", "weight", "date",
            "predicted_rate"]
    out = out[cols]
    out["date"] = out["date"].dt.strftime("%Y-%m-%d")
    return out
