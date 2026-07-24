# How Fast Does the Model Go Stale? — Drift Analysis (Roadmap #4 evidence)

**Date:** 2026-07-24
**Branch:** `sulu-drift-analysis`
**Artifact:** [scripts/experiment_data_drift.py](scripts/experiment_data_drift.py) (measurement only)

## Why this exists

Roadmap #4 is "retrain on fresher 2022–2025 data." But there is **no free source of
2022–2025 Tunisair delay labels** — public feeds (e.g. OpenSky) give *actual* flight times
but not the *scheduled* times needed to compute a delay, and historical schedules are paid.
So instead of chasing data we can't get, we **measured how quickly the current model decays**
on the data we do have — which tells us how much a refresh would actually buy, and how often
we'd need to retrain.

**Method:** train one model on **2016** only, then apply it unchanged to each following
half-year (2017 H1 → 2018 H2). Holding the model fixed and sliding the test window forward
isolates the effect of the gap between training and prediction.

## Results

| Test window | Gap (months) | Mean delay | AUC ≥15 | AUC ≥60 | Brier ≥15 | Brier ≥60 | p90 coverage |
|---|---|---|---|---|---|---|---|
| 2017 H1 | 0 | 51.8 | 0.768 | 0.673 | 0.223 | 0.162 | 80% |
| 2017 H2 | 6 | 55.2 | 0.778 | 0.707 | 0.205 | 0.156 | 85% |
| 2018 H1 | 12 | 52.8 | 0.746 | 0.671 | 0.234 | 0.169 | 79% |
| 2018 H2 | 18 | **64.2** | 0.766 | 0.715 | 0.220 | **0.182** | 83% |

## What it means (three findings)

1. **Ranking skill is durable.** ROC-AUC barely moves out to 18 months — the model keeps
   telling risky flights from safe ones. So the **risk *ordering* (and therefore the 🟢/🟡/🔴
   category) ages well**; you do *not* need frequent retraining just to keep the ranking useful.

2. **The decay is regime-driven, not clock-driven.** There's no smooth month-by-month rot.
   The clear signal is the **delay regime shifting**: 2018 H2 runs at a 64-min mean vs ~52
   earlier, and that is exactly where the fixed model's **probability accuracy degrades**
   (Brier ≥60 worst) — its calibration was set on a calmer period.

3. **The severe tail is chronically under-covered.** The p90 range catches only ~79–85% of
   flights, not 90%, across every window — the model persistently under-estimates how bad the
   bad days get (consistent with the classifier/quantile experiment).

## Recommendation (retraining cadence)

- **Refit the ranking model rarely** (≈annually) — its skill is stable; that's not the weak spot.
- **Recalibrate probabilities + refresh the quantiles frequently** (each season / whenever a new
  batch of recent flights lands). Calibration and tail coverage are what drift when the regime
  shifts, and the app already supports **time-aware calibration** — the fix is to keep re-running
  it on the most recent slice, not to rebuild everything.
- **A 2022–2025 refresh matters mainly for *level*, not ranking.** Within 2016–2018 the pure
  time-gap effect is modest; the real regime change is **post-COVID** — 2022–2025 operations
  almost certainly sit at a different delay level than 2016–2018, a far bigger shift than anything
  inside our window. So fresh data would mostly correct **calibration and the tail range**, and
  keep the *typical delay* honest — while the risk ranking would likely survive largely intact.

**Bottom line for #4:** fresh data is worth getting for calibration/level realism, but it is
**not** a magic accuracy unlock — the booking-time ceiling is still the unpredictable severe tail.
The larger accuracy gain remains **roadmap #6 (day-of "closer to departure" signals)**.

## Reproduce
```
.venv/bin/python scripts/experiment_data_drift.py
```
