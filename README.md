# Freight Rate Prediction — Spotter ML Engineer Assessment

Predicts truckload `posted_rate` from lane, distance, equipment, weight,
calendar and market features, then prices every load in the validation set
(12,000 loads, Nov–Dec 2025) and the fixed December chart lane
(Lexington → Fort Wayne, Dry Van, 360 mi, 32,000 lb, 31 days).

Final model: **3-model blend in log space** (enriched Ridge + two
HistGradientBoosting variants) — **5.3 % MAPE** on the out-of-time holdout
(Sep–Oct 2025), versus **7.8 %** for a per-equipment linear baseline.

## Repository layout

```
├── data/                        # provided inputs (unmodified)
│   ├── train-test.csv           # 48,000 labeled loads, Jan–Oct 2025
│   ├── validation.csv           # 12,000 loads to score, Nov–Dec 2025
│   ├── validation-predictions-template.csv
│   └── december-chart-inputs.csv
├── src/
│   ├── config.py                # paths, split boundary, feature lists
│   ├── ingestion.py             # strict loaders + schema checks
│   ├── validation.py            # data profiling + cleaning rules
│   ├── split.py                 # out-of-time split strategy
│   ├── features.py              # engineering + leakage-safe lane priors
│   ├── train.py                 # GridSearchCV over 3 model families
│   ├── ensemble.py              # the final Ridge+HGB blend
│   ├── evaluate.py              # metrics, baselines, residual diagnostics
│   ├── predict.py               # December export helper
│   └── run_pipeline.py          # end-to-end entry point
├── artifacts/                   # produced by the pipeline (metrics, model)
├── report/                      # assessment PDF report
├── score.py                     # provided scorer (unchanged)
└── requirements.txt
```

## Setup

```bash
python -m pip install -r requirements.txt
```

## Run

```bash
# train, tune, evaluate, and produce both prediction files
python -m src.run_pipeline
```

The first run executes the full GridSearchCV and caches the winning
model to `artifacts/best_single_model.joblib`; subsequent runs reuse
it and skip stage 5 (delete that file to force a fresh search).

Outputs:

| File | Description |
|---|---|
| `data/validation_predictions.csv` | `load_id,predicted_rate` for all 12,000 validation loads |
| `data/december_chart_inputs.csv`  | the December input file with `predicted_rate` filled (31 rows) |
| `artifacts/metrics.json`          | held-out metrics, cleaning counts, best params |
| `artifacts/best_single_model.joblib` | cached GridSearchCV winner (skips stage 5 on re-runs) |
| `artifacts/cv_results.csv`        | full GridSearchCV results |

Then run the provided scorer:

```bash
python score.py --predictions data/validation_predictions.csv \
    --december-predictions data/december_chart_inputs.csv
```

which validates both files and renders `scorer_results/candidate_december.png`.

## Approach in one page

1. **Ingestion & validation** — strict schema checks; cleaning rules handle
   292 negative weights (→ missing → per-equipment median imputation),
   300 missing weights, 374 missing `market_index` (daily-median fill).
2. **Split** — out-of-time: train Jan–Aug 2025, hold out Sep–Oct 2025,
   mirroring the future-month shift to the real validation window.
   GridSearchCV inside the training window uses `TimeSeriesSplit(5)`.
3. **Features** — log-distance, distance bands, great-circle distance and
   its ratio to billed distance, weight-per-mile, calendar features,
   market interactions, and smoothed (empirical-Bayes) lane/city rate
   priors fit on the training partition only. Priors + coordinates keep
   the 1,447 validation rows touching unseen cities predictable.
4. **Model selection** — `GridSearchCV` over Ridge, HistGradientBoosting
   and RandomForest with MAPE (on dollar rates) as the scorer; the target
   is modeled as winsorised `log(posted_rate)` because pricing noise is
   multiplicative and ~1 % of loads are extreme outliers.
5. **Tuning** — 5-fold expanding-window CV; best CV MAPE per family is
   reported in `artifacts/cv_results.csv`.
6. **Validation** — held-out Sep–Oct evaluation, per-equipment and
   per-distance-band residual analysis, linear-baseline comparison.
7. **Final model** — a three-way log-space blend, weights 0.6/0.2/0.2:
   an **enriched Ridge** (per-equipment distance and market slopes, cubic
   market response, distance hinge terms at ~400 and ~1,500 mi), an
   **HistGradientBoosting** on the full features, and a second HGB
   trained without lane priors for a decorrelated third opinion. The
   linear member nails the smooth market cycle, the trees capture
   lane/weight interactions, and the blend beats every single model and
   the previous 2-way blend (5.3 % vs 5.7 %), improving all four
   distance bands. Refit on all of Jan–Oct before scoring
   November/December.

See `report/freight-rate-report.pdf` for the full write-up.
