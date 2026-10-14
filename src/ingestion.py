"""Data ingestion: load raw CSVs, parse types, and run schema checks.

The ingestion stage is deliberately strict: if a column is missing or a type
cannot be parsed we fail loudly here rather than three stages downstream.
"""
from __future__ import annotations

import pandas as pd

from . import config


def load_train_test() -> pd.DataFrame:
    df = pd.read_csv(config.TRAIN_TEST_CSV, parse_dates=["date"])
    _check_schema(df, required=config.BASE_NUMERIC + config.CATEGORICAL + [
        "load_id", "pickup", "delivery", "date", config.TARGET])
    return df


def load_validation() -> pd.DataFrame:
    df = pd.read_csv(config.VALIDATION_CSV, parse_dates=["date"])
    _check_schema(df, required=config.BASE_NUMERIC + config.CATEGORICAL + [
        "load_id", "pickup", "delivery", "date"])
    if df["load_id"].duplicated().any():
        raise ValueError("validation set contains duplicate load_id values")
    return df.sort_values("load_id").reset_index(drop=True)


def load_december_inputs() -> pd.DataFrame:
    df = pd.read_csv(config.DECEMBER_INPUTS_CSV, parse_dates=["date"])
    _check_schema(df, required=[
        "pickup", "delivery", "distance", "equipment", "weight", "date"])
    return df.sort_values("date").reset_index(drop=True)


def load_predictions_template() -> pd.DataFrame:
    return pd.read_csv(config.PREDICTIONS_TEMPLATE_CSV)


def _check_schema(df: pd.DataFrame, required: list[str]) -> None:
    missing = [c for c in required if c not in df.columns]
    if missing:
        raise ValueError(f"missing expected columns: {missing}")
    if len(df) == 0:
        raise ValueError("dataset is empty")
