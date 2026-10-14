"""End-to-end pipeline entry point.

    python -m src.run_pipeline

Stages
------
1. ingestion      - load the raw CSVs with schema checks
2. validation     - profile data-quality issues and clean
3. split          - out-of-time split (train Jan-Aug, hold out Sep-Oct)
4. features       - engineering + leakage-safe lane priors
5. model select   - GridSearchCV (TimeSeriesSplit) over 3 model families
6. ensemble       - fit the 3-model blend (enriched Ridge + 2 HGB) on the
                     training window, evaluate
7. export         - refit on all development data, write
                     data/validation_predictions.csv (12,000 loads) and
                     data/december_chart_inputs.csv (31 December lanes)
"""
from __future__ import annotations

import json

import joblib
import numpy as np
import pandas as pd

from . import config, evaluate, features, ingestion, ensemble, split
from .train import run_grid_search
from .validation import clean, profile


def main() -> None:
    config.ARTIFACTS_DIR.mkdir(exist_ok=True)

    # ------------------------------------------------------------------ load
    print("[1/7] ingesting data ...")
    raw = ingestion.load_train_test()
    val_real = ingestion.load_validation()
    december = ingestion.load_december_inputs()
    template = ingestion.load_predictions_template()
    print(f"      train-test: {len(raw):,} rows | validation: {len(val_real):,} rows")

    # ------------------------------------------------------------- validate
    print("[2/7] profiling & cleaning ...")
    issues = profile(raw)
    print(f"      data-quality issues: {json.dumps(issues)}")
    raw = clean(raw, is_train=True)

    # ----------------------------------------------------------------- split
    print("[3/7] out-of-time split ...")
    tr, va = split.time_split(raw)
    print(f"      train: {tr['date'].min().date()}..{tr['date'].max().date()} "
          f"({len(tr):,}) | holdout: {va['date'].min().date()}.."
          f"{va['date'].max().date()} ({len(va):,})")

    # -------------------------------------------------------------- features
    print("[4/7] feature engineering (priors fit on training partition only) ...")
    priors_tr = features.fit_priors(tr)
    X_tr = features.transform(tr, priors_tr)
    y_tr = np.log(tr[config.TARGET])
    X_va = features.transform(va, priors_tr)
    y_va = va[config.TARGET]

    # ---------------------------------------------------------- model select
    print("[5/7] GridSearchCV over candidate families (TimeSeriesSplit, "
          "training window only) ...")
    best, cv_results = run_grid_search(X_tr, y_tr)
    cv_results.to_csv(config.CV_RESULTS_PATH, index=False)
    print(f"      best single family: {best['name']} | CV MAPE {best['cv_mape']:.4f}")

    # -------------------------------------------------------------- ensemble
    print("[6/7] fitting 3-model blend and evaluating on the held-out window ...")
    from sklearn.ensemble import HistGradientBoostingRegressor

    def make_hgb():
        return HistGradientBoostingRegressor(
            random_state=config.RANDOM_STATE, learning_rate=0.03, max_iter=400,
            max_leaf_nodes=63, min_samples_leaf=240, l2_regularization=1.0,
            early_stopping=False)

    blend = ensemble.BlendModel(make_hgb(), make_hgb())
    blend.fit(X_tr, y_tr, tr)
    val_pred = blend.predict(X_va, va)

    metrics = evaluate.evaluate(y_va, val_pred)
    baseline = evaluate.baseline_predictions(tr, va)
    metrics["baseline_linear_mape"] = float(np.mean(np.abs(y_va - baseline) / y_va))
    metrics["best_single_model"] = best["name"]
    metrics["best_single_cv_mape"] = best["cv_mape"]
    metrics["cleaning"] = issues
    print(f"      held-out MAPE: {metrics['mape']:.4f} "
          f"(linear baseline {metrics['baseline_linear_mape']:.4f})")

    residual = evaluate.residual_report(y_va, val_pred, va)
    print("\n      mean APE by slice:")
    print(residual.to_string(index=False))

    # --------------------------------------------------------------- export
    print("\n[7/7] refitting on ALL development data (Jan-Oct) and exporting ...")
    priors_full = features.fit_priors(raw)
    X_all = features.transform(raw, priors_full)
    y_all = np.log(raw[config.TARGET])
    final_blend = ensemble.BlendModel(make_hgb(), make_hgb())
    final_blend.fit(X_all, y_all, raw)
    joblib.dump(final_blend, config.MODEL_PATH)

    # validation predictions (must be keyed by load_id, template order);
    # the real validation frame needs the same cleaning as the dev data
    val_real = clean(val_real, is_train=False)
    X_val_real = features.transform(val_real, priors_full)
    val_real_pred = np.round(final_blend.predict(X_val_real, val_real), 2)
    id_to_rate = dict(zip(val_real["load_id"], val_real_pred))
    out_val = template.copy()
    out_val["predicted_rate"] = out_val["load_id"].map(id_to_rate)
    if out_val["predicted_rate"].isna().any():
        missing = out_val.loc[out_val["predicted_rate"].isna(), "load_id"].head()
        raise ValueError(f"template load_ids missing from validation set: {list(missing)}")
    if (out_val["predicted_rate"] <= 0).any():
        raise ValueError("generated non-positive predictions")
    out_val.to_csv(config.VALIDATION_PREDICTIONS_CSV, index=False)
    print(f"      wrote {config.VALIDATION_PREDICTIONS_CSV.name} ({len(out_val):,} rows)")

    # december chart predictions
    from .predict import write_december_predictions
    out_dec = write_december_predictions(final_blend, december, priors_full,
                                         val_real)
    out_dec.to_csv(config.DECEMBER_PREDICTIONS_CSV, index=False)
    print(f"      wrote {config.DECEMBER_PREDICTIONS_CSV.name} "
          f"({len(out_dec)} rows, rate range "
          f"${out_dec.predicted_rate.min():.0f}-${out_dec.predicted_rate.max():.0f})")

    evaluate.save_metrics(metrics, config.METRICS_PATH)
    print(f"      metrics saved to {config.METRICS_PATH}")
    print("done.")


if __name__ == "__main__":
    main()
