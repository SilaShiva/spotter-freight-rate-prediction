"""Central configuration for the freight rate prediction pipeline.

All paths, feature definitions, split boundaries and model search spaces
live here so every stage of the pipeline reads from a single source of truth.
"""
from __future__ import annotations

from pathlib import Path

# ---------------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------------
PROJECT_ROOT = Path(__file__).resolve().parents[1]
DATA_DIR = PROJECT_ROOT / "data"
ARTIFACTS_DIR = PROJECT_ROOT / "artifacts"
REPORT_DIR = PROJECT_ROOT / "report"

TRAIN_TEST_CSV = DATA_DIR / "train-test.csv"
VALIDATION_CSV = DATA_DIR / "validation.csv"
PREDICTIONS_TEMPLATE_CSV = DATA_DIR / "validation-predictions-template.csv"
DECEMBER_INPUTS_CSV = DATA_DIR / "december-chart-inputs.csv"

VALIDATION_PREDICTIONS_CSV = DATA_DIR / "validation_predictions.csv"
DECEMBER_PREDICTIONS_CSV = DATA_DIR / "december_chart_inputs.csv"  # completed copy, same name as spec

MODEL_PATH = ARTIFACTS_DIR / "model.joblib"
METRICS_PATH = ARTIFACTS_DIR / "metrics.json"
CV_RESULTS_PATH = ARTIFACTS_DIR / "cv_results.csv"

# ---------------------------------------------------------------------------
# Data split
# ---------------------------------------------------------------------------
# The validation set covers 2025-11-01 .. 2025-12-31, i.e. two future months.
# We therefore hold out the last two months of the development data
# (Sep + Oct 2025) as the out-of-time validation window: it mirrors the
# deployment shift (future months, unseen market conditions) far better than a
# random split, which would leak the slow-moving market trend into training.
VAL_START = "2025-09-01"

RANDOM_STATE = 42

# ---------------------------------------------------------------------------
# Features
# ---------------------------------------------------------------------------
BASE_NUMERIC = [
    "pickup_lat",
    "pickup_lon",
    "delivery_lat",
    "delivery_lon",
    "distance",
    "weight",
    "market_index",
    "quote_signal",
]

CATEGORICAL = ["equipment"]

ENGINEERED = [
    "month",
    "day_of_week",
    "is_weekend",
    "distance_bucket",
    "weight_per_mile",
    "log_distance",
    "dist_x_market",
    "pair_rate_prior",   # smoothed historical mean rate for the (pickup, delivery) lane
    "pickup_rate_prior",  # smoothed historical mean rate for the origin city
]

TARGET = "posted_rate"

# December chart lane parameters (from score.py expectations)
DECEMBER_LANE = {
    "pickup": "Lexington",
    "delivery": "Fort Wayne",
    "distance": 360.0,
    "equipment": "Dry Van",
    "weight": 32_000.0,
}
