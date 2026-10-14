"""Report figures: exploratory charts + residual diagnostics for the PDF.

    python -c "import sys; sys.path.insert(0,'.'); \
        from src.reporting import main; main()"

Writes PNGs into report/figures/.
"""
from __future__ import annotations

import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from . import config, features, ingestion, split
from .validation import clean

FIG_DIR = config.REPORT_DIR / "figures"
COLOR = "#064A56"


def _style(axis) -> None:
    axis.spines[["top", "right"]].set_visible(False)
    axis.spines[["left", "bottom"]].set_color("#9DAFB3")
    axis.grid(axis="y", color="#D9E2E4", linewidth=0.8)


def fig_rate_distributions(df: pd.DataFrame) -> None:
    fig, axes = plt.subplots(1, 3, figsize=(12, 3.4), dpi=150)
    df["rate_per_mile"] = df["posted_rate"] / df["distance"]

    axes[0].hist(df["posted_rate"], bins=80, color=COLOR)
    axes[0].set_title("Posted rate ($)")
    axes[1].hist(np.log(df["posted_rate"]), bins=80, color=COLOR)
    axes[1].set_title("log(posted rate)")
    axes[2].hist(df["rate_per_mile"].clip(0, 6), bins=80, color=COLOR)
    axes[2].set_title("Rate per mile ($/mi, clipped at 6)")
    for ax in axes:
        _style(ax)
    fig.suptitle("Rate distribution: right-skewed, log-stabilised by the transform",
                 fontsize=11, y=1.04)
    fig.tight_layout()
    fig.savefig(FIG_DIR / "rate_distributions.png", bbox_inches="tight")
    plt.close(fig)


def fig_market_cycle(df: pd.DataFrame) -> None:
    monthly = df.groupby(df["date"].dt.to_period("M")).agg(
        rate=("posted_rate", "mean"), mi=("market_index", "mean"))
    fig, ax1 = plt.subplots(figsize=(9, 3.6), dpi=150)
    ax1.plot(monthly.index.astype(str), monthly["rate"], color=COLOR,
             marker="o", label="mean posted rate")
    ax1.set_ylabel("Mean posted rate ($)", color=COLOR)
    ax2 = ax1.twinx()
    ax2.plot(monthly.index.astype(str), monthly["mi"], color="#C0392B",
             marker="s", label="mean market_index")
    ax2.set_ylabel("Mean market_index", color="#C0392B")
    ax1.tick_params(axis="x", rotation=45)
    ax1.spines[["top"]].set_visible(False)
    ax2.spines[["top"]].set_visible(False)
    ax1.set_title("Market cycle: rates track market_index (peak in May, trough in January)")
    fig.tight_layout()
    fig.savefig(FIG_DIR / "market_cycle.png", bbox_inches="tight")
    plt.close(fig)


def fig_rate_vs_distance(df: pd.DataFrame) -> None:
    fig, ax = plt.subplots(figsize=(9, 3.8), dpi=150)
    for eq, sub in df.groupby("equipment"):
        ax.scatter(sub["distance"], sub["posted_rate"], s=4, alpha=0.25, label=eq)
    ax.set_xlabel("Distance (mi)")
    ax.set_ylabel("Posted rate ($)")
    ax.set_title("Rate vs distance by equipment (log-linear core + extreme outliers)")
    ax.legend(frameon=False)
    _style(ax)
    fig.tight_layout()
    fig.savefig(FIG_DIR / "rate_vs_distance.png", bbox_inches="tight")
    plt.close(fig)


def fig_data_quality(df: pd.DataFrame) -> None:
    fig, axes = plt.subplots(1, 2, figsize=(10, 3.4), dpi=150)
    axes[0].hist(df["weight"].clip(-50_000, 50_000), bins=60, color=COLOR)
    axes[0].axvline(0, color="#C0392B", linestyle="--")
    axes[0].set_title("Weight distribution (negative values = data errors)")
    ratio = df["distance"] / (
        2 * 3958.8 * np.arcsin(np.sqrt(np.clip(
            np.sin(np.radians(df["delivery_lat"] - df["pickup_lat"]) / 2) ** 2
            + np.cos(np.radians(df["pickup_lat"])) * np.cos(np.radians(df["delivery_lat"]))
            * np.sin(np.radians(df["delivery_lon"] - df["pickup_lon"]) / 2) ** 2,
            0, 1))))
    axes[1].hist(ratio.clip(0.8, 2.5), bins=60, color=COLOR)
    axes[1].axvline(1.19, color="#C0392B", linestyle="--",
                    label="typical road factor ~1.19")
    axes[1].set_title("Billed distance / great-circle distance")
    axes[1].legend(frameon=False)
    for ax in axes:
        _style(ax)
    fig.tight_layout()
    fig.savefig(FIG_DIR / "data_quality.png", bbox_inches="tight")
    plt.close(fig)


def fig_validation_scatter(y_true: pd.Series, y_pred: np.ndarray) -> None:
    fig, ax = plt.subplots(figsize=(5.4, 5.0), dpi=150)
    ax.scatter(y_true, y_pred, s=5, alpha=0.25, color=COLOR)
    lims = [min(y_true.min(), y_pred.min()), max(y_true.max(), y_pred.max())]
    ax.plot(lims, lims, color="#C0392B", linestyle="--", linewidth=1)
    ax.set_xlabel("Actual rate ($)")
    ax.set_ylabel("Predicted rate ($)")
    ax.set_title("Held-out window (Sep–Oct): predicted vs actual")
    _style(ax)
    fig.tight_layout()
    fig.savefig(FIG_DIR / "validation_scatter.png", bbox_inches="tight")
    plt.close(fig)


def fig_model_comparison(metrics: dict) -> None:
    fig, ax = plt.subplots(figsize=(7.4, 3.2), dpi=150)
    names = ["Per-equipment\nlinear baseline", "3-model blend\n(this model)"]
    vals = [metrics["baseline_linear_mape"] * 100, metrics["mape"] * 100]
    bars = ax.barh(names, vals, color=["#9DAFB3", COLOR])
    ax.bar_label(bars, fmt="%.1f%%", padding=4)
    ax.set_xlabel("MAPE on Sep–Oct holdout (%)")
    ax.set_xlim(0, max(vals) * 1.2)
    _style(ax)
    fig.tight_layout()
    fig.savefig(FIG_DIR / "model_comparison.png", bbox_inches="tight")
    plt.close(fig)


def main() -> None:
    FIG_DIR.mkdir(parents=True, exist_ok=True)
    raw = ingestion.load_train_test()
    cleaned = clean(raw, is_train=True)

    fig_rate_distributions(cleaned)
    fig_market_cycle(cleaned)
    fig_rate_vs_distance(cleaned.sample(6000, random_state=config.RANDOM_STATE))
    fig_data_quality(cleaned)

    # held-out diagnostics need a fitted blend; reuse saved metrics if present
    if config.METRICS_PATH.exists() and config.MODEL_PATH.exists():
        import joblib
        from src.evaluate import evaluate as eval_fn

        model = joblib.load(config.MODEL_PATH)
        tr, va = split.time_split(cleaned)
        priors = features.fit_priors(tr)
        X_va = features.transform(va, priors)
        pred = model.predict(X_va, va)
        fig_validation_scatter(va[config.TARGET], pred)
        fig_model_comparison(json.loads(config.METRICS_PATH.read_text()))
    print(f"figures written to {FIG_DIR}")


if __name__ == "__main__":
    main()
