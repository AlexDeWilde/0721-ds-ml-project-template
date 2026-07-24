# Calibrated Risk Classifier + Flight-Specific Range — Experiment Write-up

**Date:** 2026-07-24
**Branch:** `sulu-risk-classifier`
**Roadmap item:** #5 — "Reframe as a calibrated risk classifier + quantile intervals"
**Artifact:** [scripts/experiment_risk_classifier.py](scripts/experiment_risk_classifier.py) (measurement only — the app is unchanged)

---

## 1. The idea

The app *sells* a risk **category** (🟢/🟡/🔴) and a **typical-delay range**, but under the hood it:
1. predicts a single **mean** delay in minutes and then thresholds that shaky mean into the category, and
2. shows the departure **route's** empirical p25/p50/p75 — the same range for every date, time and weather.

The mean is a poor target here (it's dragged around by the rare severe-delay tail; dataset mean is 48.7 min vs a median of 14). This experiment asks: **is it better to predict the risk category directly (as a calibrated probability) and the range as flight-specific quantiles?**

Same chronological hold-out and feature set as the app model (booking features + weather).

---

## 2. Results

### A. Risk — does a purpose-built classifier separate risk better than the mean-regressor?

Higher **ROC-AUC / PR-AUC** = better separation; lower **Brier** + reliability pairs near the diagonal = trustworthy probabilities.

| Target | Model | ROC-AUC | PR-AUC | Brier |
|---|---|---|---|---|
| P(delay ≥ 15 min) | regressor's minutes as a score (today) | 0.757 | 0.761 | — (no probabilities) |
| | **calibrated classifier** | **0.811** | **0.832** | 0.18 |
| P(delay ≥ 60 min) | regressor's minutes as a score (today) | 0.750 | 0.460 | — |
| | **calibrated classifier** | **0.767** | **0.504** | 0.16 |

**A purpose-trained classifier separates risk better** — clearly for the "will it be ≥15 min late?" question (AUC 0.76 → 0.81, PR-AUC 0.76 → 0.83), modestly for the severe ≥60 case — **and** it produces a real probability we can show ("≈30% chance of 60+ min"), which the regressor cannot.

### B. Calibration needs to be *time-aware* (an important gotcha)

Calibrating the classifier with ordinary cross-validation left it **under-confident on the hold-out** — e.g. it said 65% when the real rate was 82%. That's **temporal drift**: the validation months (summer-heavy, later 2018) run later than the training period expects.

**Fix that works:** calibrate on the **most recent slice** of the training data instead of random folds. Reliability then hugs the diagonal and Brier improves:

| | Brier (CV calib.) | Brier (time-aware) | reliability after time-aware fix |
|---|---|---|---|
| P(≥15) | 0.185 | **0.179** | 0.44→0.50, 0.57→0.57, 0.75→0.79, 0.85→0.89, 0.92→0.95 |
| P(≥60) | 0.167 | **0.155** | 0.23→0.26, 0.37→0.39, 0.53→0.66 |

### C. Range — flight-specific quantiles vs the route-empirical range used today

Coverage should match the quantile level; pinball loss lower = better.

| Quantile | Method | Coverage | Pinball |
|---|---|---|---|
| p50 | quantile regression | 46.2% | **27.64** |
| | route-empirical (app now) | 47.7% | 28.17 |
| p75 | quantile regression | 68.6% | **34.01** |
| | route-empirical (app now) | 67.7% | 35.39 |
| p90 | quantile regression | 85.8% | **28.19** |
| | route-empirical (app now) | 84.4% | 29.58 |

**Quantile regression beats the route-empirical range on pinball loss at every level**, with similar-or-better coverage — and it's **flight-specific** (varies by date, time and weather), which the current route-level range is not.

---

## 3. Verdict

**Go — reframe the app around a calibrated classifier + quantile range.**

- **Risk category** should come from a **calibrated classifier** (better separation, and it yields an honest probability to show), calibrated **time-aware** (on the most recent training slice), not by thresholding a mean.
- **The range** should be **flight-specific quantile regression** (lower pinball, varies with the inputs) rather than the static route-level range.

**Deeper insight for later:** *both* methods still slightly **under-cover** the tail on the future hold-out — the later months are simply worse than 2016–2018 training expects. No booking-time model fixes that; it's the strongest argument yet for **roadmap #4 (fresher 2022–2025 data)**.

---

## 4. Suggested integration (a later step, once agreed)

1. In `scripts/build_app_data.py`, additionally train and save: two **time-aware calibrated** classifiers (P≥15, P≥60) and three **quantile regressors** (p50/p75/p90).
2. In `app/delay_core.py`, derive the risk band from the calibrated probabilities and show the probability; build the range from the per-flight quantiles.
3. Keep the weather ladder — it already runs the model under scenarios; the classifier version would show *risk probability* per weather scenario, which is even clearer.

Nothing committed to the app; this document + the script are the deliverables.

## 5. Reproduce
```
.venv/bin/python scripts/experiment_risk_classifier.py
```
