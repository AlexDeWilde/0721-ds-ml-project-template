# Session Handoff — Tunisair Flight Delay Prediction

> **Purpose:** This document is a complete handoff of a Claude Code working session on `04_flight_delay_eda_modeling.ipynb`. It captures what was prompted, what was decided, what was built, the current project-board state, and exactly what to do next. Pull this repo, read this file top to bottom, and you can continue seamlessly in your own VS Code + Claude session.

**Last updated:** 2026-07-24 (accuracy roadmap #1, #2 & #5; weather ladder + calibrated-classifier reframe shipped into the app)
**Working branch:** `sulu-dayof-mode` (#6 day-of/closer-to-departure mode; merging to `main`). Prior merged: `sulu-flight-history-features` (#1/#2), `sulu-risk-classifier` (#5), `sulu-actual-weather-badge` (tiered weather), `sulu-drift-analysis` (#4 evidence).
**Notebook:** [04_flight_delay_eda_modeling.ipynb](04_flight_delay_eda_modeling.ipynb)
**Slide deck:** [slides_draft.md](slides_draft.md) (content source of truth) + `Tunisair Flight Delay Deck.html` / `Tunisair Flight Delay Deck v2.html` (rendered)
**Data product:** [app/](app/) — Streamlit "Tunisair Delay-Alert" (see `app/README.md`)

---

## ⭐ CURRENT STATE — 2026-07-24 (accuracy roadmap #1 & #2; weather ladder in the app)

Session focus: **improving prediction accuracy & UX** (the roadmap below). Both #1 and #2 were prototyped as measured, leakage-safe experiments *before* touching the app; #2 was then shipped into the Streamlit app as a user-facing feature. Branch: `sulu-flight-history-features`.

### Roadmap #1 — flight-number (FLTID) history as features → *documented finding, NOT wired in*
- Script [scripts/experiment_fltid_features.py](scripts/experiment_fltid_features.py); write-up [FLTID_FEATURE_EXPERIMENT.md](FLTID_FEATURE_EXPERIMENT.md).
- Leakage-safe eval (per-FLTID stats fit on the train split only, shrunk toward global). **Finding:** the kitchen-sink of 5 stats *overfits* (net worse); a **single strongly-shrunk per-FLTID median delay** gives a small real gain (**136.78 → 136.22, −0.55 min / 0.4%**, stable across shrinkage K=75–150).
- **Decision:** kept as evidence; **not** integrated (tiny gain). Cheap to add later (median-only feature).

### Roadmap #2 — weather → *shipped into the app as a "weather-sensitivity ladder"*
- Scripts: [scripts/fetch_weather.py](scripts/fetch_weather.py) (caches ERA5 hourly weather, Open-Meteo, no key), [scripts/weather_core.py](scripts/weather_core.py) (naming + join + scenario table), [scripts/experiment_weather_features.py](scripts/experiment_weather_features.py); write-up [WEATHER_SCENARIO_EXPERIMENT.md](WEATHER_SCENARIO_EXPERIMENT.md).
- **Two findings:** (a) weather does **not** lower RMSE (−0.4 min; the severe-delay tail dominates, same as #1); (b) but the model's weather **response is strong & monotonic** (calm→heavy-rain+wind ≈ +27 min; clean gust dose-response). So weather is valuable as a **sensitivity ladder**, not a sharper number — which validated the product idea.
- **Shipped:** the app model is retrained **with 4 weather features** (`wx_wind_gust/precip/snow/adverse`; neutral-filled for the ~23% of flights at airports without weather coverage). Holdout RMSE unchanged (**137.12**). New committed reference [app/reference/weather_scenarios.csv](app/reference/weather_scenarios.csv) (15 airports × 12 months × calm/rough/severe bands, **named** conditions + odds; precomputed so the app needs no network). `delay_core.weather_ladder()` runs the model under each scenario; the headline "Average expected delay" for covered airports is the **weather-weighted outlook** (rungs bracket it; monotonic display clamp; uncovered airports show no ladder).
- **Coverage / data limits (in the UI):** weather only for the **top-15 departure airports (77% of flights)**; ERA5 has **no fog/thunderstorm** codes (a future **METAR** upgrade — needs IATA→ICAO mapping).

### App UX changes (Streamlit)
- **Weather-sensitivity card** with **line-style SVG weather icons** per rung (sun/cloud/rain/snow/wind/storm/fog), minutes tinted by risk, probabilities, and a weighted-outlook line.
- **Reconciled the two headline numbers** (a user-confusion fix): "Model risk estimate" → **"Average expected delay"** (burgundy, enlarged; it's the *mean*, pulled up by the skewed tail — dataset mean 48.7 vs median 14), and the typical range now shows the **median too** (`p25–median–p75`). "Typical delay on this route" right-aligned.
- **Slider fix:** Streamlit 1.60's slider is **react-aria + emotion** (not baseweb) — the red came from `theme.primaryColor` on emotion `target` classes `.e23vpic5` (fill) / `.e23vpic3` (thumb). Now white track + white thumb (with ring) + white value text. Logo size tweaked.

### Commits on branch (not yet on `main` at time of writing → then merged)
`411b591` experiments + README rewrite · `4fb13c4` weather ladder integrated · `f2449aa` UI polish. README rewritten from the stale template into a full Tunisair file guide.

### Gotchas this session
- **Untracked files got wiped once** mid-session (the env cleaned untracked files) — recovered from context; **commit early** to avoid recurrence.
- **No browser in the sandbox** (no root; Chromium libs missing) — the slider fix was verified by reading Streamlit's compiled `Slider.js` to find the real emotion classes, not by rendering.
- `data/weather_cache/` is **gitignored** — regenerate with `scripts/fetch_weather.py`.

### Roadmap #5 — calibrated risk classifier + quantile range → *shipped into the app*
- Experiment [scripts/experiment_risk_classifier.py](scripts/experiment_risk_classifier.py); write-up [RISK_CLASSIFIER_EXPERIMENT.md](RISK_CLASSIFIER_EXPERIMENT.md).
- **The app no longer predicts a single mean-minutes number.** `build_app_data.py` now trains **two time-aware calibrated classifiers** (P(delay≥15), P(delay≥60); base on earliest 80% by date, isotonic calibration on the recent 20% — beats temporal drift) and **three quantile regressors** (p50/p75/p90). Hold-out: **AUC 0.80 (≥15) / 0.76 (≥60)**, Brier 0.18/0.16. Model is now **~2 MB** (was 13 MB RF).
- `delay_core.py`: risk band comes from calibrated **probabilities** (High if P≥60 ≥ 0.33; Moderate if P≥15 ≥ 0.50) not from thresholding a mean; the range is **flight-specific quantiles** (clamped monotonic). Weather ladder shows per-scenario median + P(≥60); headline = weather-weighted outlook. UI shows "Typical delay ~q50 (up to q75 · bad day q90)" + "Chance of a delay: X% ≥15 min · Y% ≥60 min".
- **Insight for later:** both classifier & quantiles still slightly **under-cover the tail** on the future hold-out → strongest argument yet for **fresher data (#4)**.

### App weather sourcing — tiered by horizon, feeds the prediction (shipped)
`delay_core.resolve_weather()` picks the best weather for the chosen date and the app feeds it into the model:
- **RECORDED** — actual ERA5 for historical (2016–2018) dates (`weather_core.actual_weather`).
- **FORECAST** — live Open-Meteo forecast for dates **within ~16 days** of today (`delay_core.forecast_condition`, day-cached per airport, fails soft offline).
- **SEASONAL** — the airport's typical (climatological) weather for that month, from `weather_scenarios.csv`, for any date beyond the forecast horizon (months out / next year).

When weather is **known** (recorded/forecast) the risk & quantile range are predicted **under that specific weather** (not just the weather-weighted outlook), and the matching ladder rung is highlighted; a corner **badge** (icon + RECORDED/FORECAST/SEASONAL tag) shows the condition. **Honest limit:** genuine day-specific forecasts only exist ~16 days out; beyond that it's a seasonal norm, labelled as such. Branch `sulu-actual-weather-badge` (commits `86d8243`→`78ff2d1`→`e33b5b7`).

### Roadmap #4 — drift analysis *(measured; can't get fresh labels)*
`scripts/experiment_data_drift.py` · write-up `DATA_DRIFT_EXPERIMENT.md`. No free 2022–25 Tunisair delay labels exist (public feeds give actual times but not scheduled). So we measured decay on existing data (train 2016, apply to later half-years): **ranking (AUC) is durable to 18 months; decay is regime-driven** (2018 H2 mean 64 vs ~52), not clock-driven; the **severe tail is chronically under-covered** (p90 ≈ 80%). **Cadence recommendation:** refit ranking ~annually; recalibrate probabilities + refresh quantiles each season (time-aware). A post-COVID refresh mainly fixes level/calibration, not ranking.

### Roadmap #6 — "closer to departure" (day-of) mode *(shipped)*
`build_app_data.py` now also trains a **day-of model set** (calibrated classifiers + quantiles) that adds the aircraft's **prior-leg delay** (`prev_leg_delay`, + `hours_since_prior_leg`, `has_prior_leg`; leakage-safe — prior leg departs strictly earlier). Hold-out lift is large: **AUC 0.80→0.89 (≥15) and 0.76→0.91 (≥60)** — most big delays are *propagated*. The app has a **"When are you checking?"** toggle: *At booking* (default) vs *Day of travel*, which adds an **"inbound aircraft currently delayed by"** slider and predicts under the day-of model (booking-time ladder/alternatives hidden; a contrast note shows the booking estimate). Model now ~4.5 MB. Branch `sulu-dayof-mode`.

### Immediate next options
Wire #1's FLTID median in (cheap); add **congestion (#3)**; **METAR** for fog/thunderstorm naming; auto-fetch the inbound-leg delay live (day-of) instead of the manual slider; seasonal recalibration job. The notebook **Appendix** summarises #1/#2/#5.

---

## ⭐ CURRENT STATE — 2026-07-23 (Phases 1–8 complete)

**Phases 1–8 are essentially COMPLETE and all merged to `main`** (`d148d1b`). Board #4: **29 of 31 Done**; only **#25** (Phase 7.2 — export the deck to PDF) and **#27** (Phase 7.4 — rehearse/present) remain, both human/presentation tasks. Alex also merged his stakeholder deck (`Tunisair Flight Delay Deck v2.html`) — the slide-deck coordination resolved cleanly (different files, no conflict).

### Headline results (verified from the run notebook)
- Baseline held-out RMSE **141.95** min.
- **Operational model** (tuned RF + engineered features, uses `prev_leg_delay`): **108.64** (−24%). Saved `models/delay_model.joblib` (gitignored, 350 MB — regenerate by running the notebook). Powers the app's concept; assumes prior-leg delay is known at prediction time (true operationally).
- **Submittable model** (no `prev_leg_delay`): **130.92** → `zindi_submission.csv`.
- **Two-track finding:** Zindi's test set is entire hidden calendar months, so `prev_leg_delay` (the #1 feature, ~38% importance) is unobservable there. Not leakage — a real airline knows the prior leg's delay; the competition just hides whole months. Error concentrates in the severe-delay tail, sparse long-haul routes, and wet-lease aircraft.

### Phase 8 data product (built & merged) — `app/`
Streamlit **"Tunisair Delay-Alert"**: `app/streamlit_app.py` (UI), `app/delay_core.py` (logic — shared `is_ramadan()`), `app/reference/*.csv` (lookups), `scripts/build_app_data.py` (rebuilds model+tables), `models/app_booking_model.joblib` (14 MB, committed). It's a **booking-time** model (inputs: flight number/route + date; uses only route/date/time-of-day) → risk category 🟢/🟡/🔴 + honest typical-delay range + plain-language advice + lower-risk alternative departure times. Booking-time RMSE ~137 (~4% better than baseline) — reliable for the *category*, honest that exact minutes aren't knowable at booking. Ramadan flag is year-general via `hijridate`; future-date caveat shown. Branded UI (white Tunisair logo `app/assets/logo.png`, red bg, white result card, disclaimer + authors). **Run:** `uv sync && uv run streamlit run app/streamlit_app.py`. New deps: `streamlit`, `hijridate`.

### 🎯 NEXT (discussed, NOT yet built): improve the app's prediction accuracy
Prioritized:
1. **Flight-number track record as model features** (cheapest, highest ROI — each FLTID's historical delay median/%-late/variance; currently used only for the displayed range, not as a model input).
2. **Weather** — forecast for near-term dates, seasonal climatology for far-off (Open-Meteo feasibility already positive: adverse weather lifts severe-rate ~45%).
3. **Schedule-derived congestion / hub load** features (knowable in advance).
4. **Fresher data** (2022–2025; current data is pre-COVID 2016–2018).
5. **Reframe as a calibrated risk classifier** (`predict_proba`) + quantile intervals; evaluate with classification/calibration metrics, not just RMSE.
6. **"Closer to departure" mode** using day-of signals — bridges toward the 108.64 operational accuracy.
*(Offered to prototype #1 on a fresh branch — the fastest way to demonstrate a real gain.)*

### Housekeeping / gotchas
- Remote branches: only `main` and `adw0721` (Alex's active branch) — all `sulu-*` feature branches merged & deleted.
- Big operational models (`models/delay_model*.joblib`) are **gitignored**; the small `app_booking_model.joblib` is committed. `zindi_submission.csv` is committed.
- Editing files on disk while the notebook/app is open in VS Code caused save-conflicts once — reload in VS Code after external edits.
- The notebook is large (~50k tokens with outputs); the notebook-edit tool can't read it — edit its cells by scripting the `.ipynb` JSON if needed.

---

## Update — 2026-07-22, Phase 6 complete (Sulu, branch `sulu-phase6-error-analysis`)

**Phase 6 (error analysis → iteration → final evaluation) is DONE in the notebook.** Board #21, #22, #23 → Done.

- **6.1 Error analysis:** model under-predicts the **severe-delay tail** (severe band 180+ min under-predicted ~145 min each — this *is* the RMSE); worst on sparse long-haul routes and wet-lease aircraft.
- **6.2 Iteration:** added `dep_congestion`, `leg_of_day` (best new feature), `ac_type_freq`, `is_wet_lease`, `dep_is_ramadan`. New best model **RF + new features = 108.64 min** (was 109.53). Log-transform target **rejected** (worse: 117.7). Weather was feasibility-tested (real Open-Meteo data: adverse weather lifts the severe rate ~45%) and documented as the **#1 future-work** item (not built — Day-3 time).
- **6.3 Final eval — KEY FINDING (two-track):** the Zindi test set is **whole hidden months**, so `prev_leg_delay` (top feature, ~38% importance) is **unavailable for test**. Not leakage — a real airline knows the prior leg's delay; Zindi just hides whole months. Decision:
  - **Operational model** (with `prev_leg_delay`): **RMSE 108.64** — headline for stakeholders, saved `models/delay_model.joblib`.
  - **Submittable model** (no `prev_leg_delay`): **RMSE 130.92** — the Zindi entry, produces `zindi_submission.csv`, saved `models/delay_model_submittable.joblib`.
  - Dropping the feature costs **+22.29 min** — the propagation insight, a headline point (not a flaw). See ISSUES.md.

**New artifacts:** `zindi_submission.csv`, `models/delay_model.joblib`, `models/delay_model_submittable.joblib`.

**Slides action (Phase 7.2):** headline 108.64 *with the stated assumption* (prior-leg delay known at prediction time), also state the submittable 130.92, and present the propagation gap as the key insight. The current deck predates this finding.

**Next up:** board **#24 (Phase 7.1)** finalize notebook, then **#25 (7.2)** update the deck with the two-track story. (Phase 8 Streamlit app can reuse `models/delay_model.joblib`.)

> ⚠️ **Notebook is large** (~50k tokens with outputs) — the notebook-edit tool can't read it, so this session's cells were inserted by editing the `.ipynb` JSON directly. Also: editing the file on disk while it's open in VS Code caused one cell to be dropped once (recovered) — **close or reload the notebook in VS Code before/after external edits** to avoid save conflicts.

---

## 1. Project at a glance

- **Challenge:** [Zindi Flight Delay Prediction Challenge](https://zindi.africa/competitions/flight-delay-prediction-challenge) (Tunisair).
- **Task type:** Regression — predict `target` (flight delay duration in **minutes**).
- **Metric:** RMSE (minutes), matching Zindi.
- **Stakeholder framing:** an airline operations team (and, downstream, a travel app surfacing delay risk to passengers).
- **Golden rule:** use **only pre-departure information** (schedule, route, aircraft). Never use anything known only after the flight happens.
- **Data files** (in `data/`, lowercase names): `train.csv` (107,833 rows), `test.csv` (unlabeled — Zindi submission set, **no `target` column**), `sample_submission.csv`.

### Raw columns in `train.csv`
`ID, DATOP, FLTID, DEPSTN, ARRSTN, STD, STA, STATUS, AC, target`

- `DATOP` = date of operation
- `FLTID` = flight number
- `DEPSTN` / `ARRSTN` = departure / arrival airport (IATA codes)
- `STD` / `STA` = scheduled time of departure / arrival
- `STATUS` = operational status (**LEAKY — dropped**, see below)
- `AC` = aircraft tail identifier
- `target` = delay in minutes (the label)

---

## 2. How this team wants to work (IMPORTANT — read before continuing)

These conventions were established during the session and should be respected going forward:

1. **Stepwise collaboration.** Walk phase by phase. Present the next step and wait for explicit go-ahead before major actions. The team makes the decisions; Claude proposes and explains.
2. **One small code cell at a time, each with a plain-language markdown explanation above it.** Every code cell in the notebook is preceded by a markdown cell structured as: *Before this step* → *What this code does* → *What we want to get out of it*. Explanations are written for **non-technical readers**.
3. **Follow the GitHub project board sequentially and keep statuses live.** Move issues to their correct status (Backlog / Ready / In Progress / Done / Blocked) in real time as work starts and finishes. After any notebook edit, re-check the **whole** board — one change can complete/advance several issues at once.
4. **Notebook execution:** the team runs cells themselves in VS Code and reports output back. (Exception: on request, Claude ran computations via the venv to write accurate takeaways.)
5. **Section order matches the board 1:1.** The notebook has one `## Phase X.Y: <exact board issue title>` header per board issue, in Phase 1.1 → 8.4 order, including placeholder headers for work not yet started.

---

## 3. GitHub project board reference

- **Project:** "@AlexDeWilde's ML Project", **project #4**, owner **AlexDeWilde** — the single canonical board.
  URL: https://github.com/users/AlexDeWilde/projects/4
- **Repo the issues live on:** `AlexDeWilde/0721-ds-ml-project-template` (project #4 is linked to this repo).
- **31 issues** across 8 phases.

> **History note (corrected 2026-07-21):** an earlier version of this handoff pointed at a *second* board, `sulugambari` project #5. That board has since been **deleted** — it was a duplicate created during Sulu's session and its IDs no longer resolve. There is now only **one** board: #4, owned by AlexDeWilde and linked to the repo. Both teammates work the same #4 board via project-level access (see §8 for how sulugambari gets in).

### Useful CLI snippets for board updates
```bash
# List all items with their status
gh project item-list 4 --owner AlexDeWilde --format json --limit 50

# Project id:      PVT_kwHOELY1284BeADV
# Status field id: PVTSSF_lAHOELY1284BeADVzhYddlc
# Status option ids:
#   Backlog=f75ad846  Ready=61e4505c  In progress=47fc9ee4  Done=98236657  Blocked!=df73e18b

# Example: move an item to Done
gh project item-edit --project-id "PVT_kwHOELY1284BeADV" \
  --id "<ITEM_ID>" \
  --field-id "PVTSSF_lAHOELY1284BeADVzhYddlc" \
  --single-select-option-id "98236657"
```
To get an item's ID: `gh project item-list 4 --owner AlexDeWilde --format json` and look up by issue number.

---

## 4. Current board status (as of end of session)

| # | Phase | Status |
|---|---|---|
| 1 | 1.1 Download dataset & load | ✅ Done |
| 2 | 1.2 airportsdata enrichment | ✅ Done |
| 3 | 1.3 Explore raw schema | ✅ Done |
| 4 | 1.4 Target outliers/impossible values | ✅ Done |
| 5 | 1.5 Temporal range & ordering | ✅ Done |
| 6 | 1.6 Leakage audit | ✅ Done |
| 7 | 1.7 Visualize delay patterns | ✅ Done |
| 8 | 2.1 Define leakage-safe feature set | ✅ Done |
| 9 | 2.2 Engineer temporal features | ✅ Done |
| 10 | 2.3 Engineer route/airport features | ✅ Done |
| 11 | 2.4 Cascading-delay feature | ✅ Done |
| 12 | 2.5 Chronological train/test split | ✅ Done |
| 13 | 3.1 Baseline model + RMSE | ✅ Done |
| 14 | 3.2 Document business framing | ✅ Done |
| 15–19 | 4.1–4.5 Model iteration (LinReg, RF, GBM/XGB, tuning, compare) | ✅ Done |
| 20 | 5.1 Draft slide deck | ✅ Done (see `slides_draft.md`) |
| **21** | **6.1 Residual/error analysis** | ⚪ **Backlog — next up** |
| 22 | 6.2 Iterate features/model | ⚪ Backlog |
| 23 | 6.3 Final evaluation on Zindi test set | ⚪ Backlog |
| 24–27 | 7.1–7.4 Finalize notebook/slides/data-product slide/rehearse | ⚪ Backlog |
| 28–31 | 8.1–8.4 Streamlit delay-alert web app | ⚪ Backlog |

**Milestone 1 (baseline, Day 2 12:00): DONE.** **Milestone 2 (slides draft, Day 3 12:00): DONE** (`slides_draft.md`). **Milestone 3 (model + error analysis, Day 3 16:00):** model selected; error analysis (Phase 6) still outstanding.

> **Board note:** update board items #11–20 to Done to match the notebook (the CLI snippets above still apply).

---

## 5. What was built in the notebook, phase by phase

The notebook now runs top-to-bottom in board order. Summary of completed sections:

### Phase 1.1 — Load data
`pd.read_csv` of train/test; `.head()` sanity view.

### Phase 1.2 — Airport enrichment (`airportsdata`)
Joins `DEPSTN`/`ARRSTN` to country, lat, lon, timezone → new columns `dep_country, dep_lat, dep_lon, dep_tz, arr_country, arr_lat, arr_lon, arr_tz`.
- **Manual override:** `SXF` (Berlin-Schönefeld) was decommissioned in 2020 and is missing from `airportsdata`; hardcoded via `AIRPORT_OVERRIDES` (see `ISSUES.md`).
- All airport codes matched (0 unmatched).

### Phase 1.3 — Explore raw schema
`.info()`, missingness (**0% missing everywhere**), target `.describe()`, and two histograms (full range, and capped at 1500 min).
- **Finding:** `target` is heavily **right-skewed** — median 14 min, most flights 0–20 min, long tail out to 3451 min.

### Phase 1.4 — Target outliers / impossible values
Counts: 0 negative delays; 1,185 (1.10%) > 500 min; 197 (0.18%) > 1000 min; 44 (0.04%) > 2000 min.
- **DECISION (team):** **Keep all rows, no drop or cap.** Extreme delays are rare but real (weather/mechanical/crew), not errors. RMSE rewards capturing the tail, so dropping/capping would discard real signal.

### Phase 1.5 — Temporal range & ordering
Parsed `STD` (colon-separated, direct parse), `STA` (**dot-separated** e.g. `12.55.00` — regex-fixed to colons then parsed), and `DATOP` to datetimes.
- `train` DATOP range: 2016-01-01 → 2018-12-31; **not** pre-sorted.
- `test` DATOP range: 2016-05-01 → 2018-09-29.
- **Test does NOT come after train in time — ranges overlap.** Plus `test.csv` has no `target`. → The provided train/test is **not** a ready-made chronological split; we must carve our own validation split out of `train` by date, and reserve `test` purely for Zindi submission predictions.
- **Data-quality finding:** naive `STA - STD` has 0 negatives but a max of ~11,992 hours (~500 days). A same-tz vs cross-tz split showed the extremes are in the **same-timezone** group (offset 0), so they're **date-entry errors in a few `STA` values**, NOT timezone artifacts. Cross-tz durations are clean (median 2.3h, max ~8.9h) because Tunisair's network spans only small offsets (~0–2h). **Logged in `ISSUES.md`; must be cleaned/clipped before building any duration feature in Phase 2.3.**

### Phase 1.6 — Leakage audit
Table documenting which columns are known pre-departure. `STATUS` flagged leaky (values `ATA`/`SCH`/`DEP`/`RTR`/`DEL` describe actual operational state). `target` is the label. All else (schedule/route/aircraft) is safe.

### Phase 1.7 — Visualize delay patterns (5 plots, median delay used due to skew)
| Factor | Strength | Finding |
|---|---|---|
| Hour of day | **Strong** | Near-0 early morning, cascades to 20–30 min afternoon peaks |
| Route | **Strong** | Algiers/Orly/Marseille ~23–30 min; Djerba & `TUN->TUN` ~0 |
| Month/season | **Solid** | Summer (Jul–Sep) ~18–20 min vs February ~7 min |
| Day of week | Weak | Fairly flat, slight Sunday peak (~18 min) |
| Aircraft | Weak | Narrow 11–23 min band, no standout plane |

**Note:** `TUN->TUN` (same dep/arr airport) exists — likely maintenance/positioning flights; flagged, harmless to label.
**Implication:** prioritize **departure hour, route, month/season** as features.

### Phase 2.1 — Leakage-safe feature set ✅ DONE
The leaky `STATUS` column is dropped from both train and test, **and** a markdown cell now documents the final leakage-safe feature set (keep/exclude table with justification) — bringing the notebook in line with board issue #8. No code beyond the `STATUS` drop; the feature-set decision is the deliverable.

### Phase 2.2 — Temporal features ✅ DONE
Added `holidays` as a dependency (`uv add holidays`) for the Tunisia holiday flag. Three cells, all built and verified in VS Code:
1. `dep_hour`, `dep_dow`, `dep_month` on **both** train and test (formalizing the Phase 1.7 train-only EDA helpers).
2. `dep_is_holiday` (1/0) via `holidays.Tunisia(...)`. **Gotcha logged:** match on `STD.dt.date` — `.dt.normalize().isin(tn_holidays)` silently returns all-False (Timestamp-vs-`date` key mismatch).
3. Spot-check against raw rows (ordinary + holiday flights, with readable `weekday_name`/`holiday_name` helpers) — issue #9 DoD.

### Phase 2.3 — Route/airport features ✅ DONE
Four cells, all built and verified (values checked in the venv), on **both** train and test. Notably none of these need `STA`, so the corrupted-`STA` cleanup was NOT required here.
1. `gc_distance_km` — great-circle (haversine) distance from dep/arr lat/lon.
2. `tz_diff_hours` — arrival minus departure UTC offset, **DST-aware** (offsets evaluated at each flight's `STD` date; helper `tz_offset_hours` groups by tz name for a vectorized `tz_localize`). Verified e.g. `TUN->CDG` = 0 in winter / +1 in summer.
3. `is_domestic` (1/0) + `country_pair` (e.g. `TN->FR`). Domestic ≈ 17.5% of rows; ~195 country pairs.
4. Sanity-check table vs known routes (TUN->DJE/ALG/IST, MXP->TUN, TUN->CDG/JED) + takeaway — issue #10 DoD.

**Scheduled-duration feature was deliberately skipped** (would need `STA` cleaned first) — logged as a Phase 6.2 candidate in ISSUES.md; `gc_distance_km` already proxies flight length.

### Phase 2.4 — Cascading-delay feature ✅ DONE
Per aircraft (`AC`), the **prior leg's** delay as a feature, built on both train and test. Columns: `prev_leg_delay`, `hours_since_prior_leg`, `has_prior_leg`. Sorted by `AC` then `STD`; first-leg rows (68 in train) get a neutral fill and `has_prior_leg=0`.
- **Leakage guard verified:** 0 rows where the prior leg departs at/after the current one. Using a strictly-earlier leg's `target` is legitimate (known before the current departure).
- **Signal:** `corr(prev_leg_delay, target) = 0.373` on real prior legs; median gap between legs 3.58 h. This turns out to be the single strongest predictor in every model.

### Phase 2.5 — Chronological split ✅ DONE
Time-based 80/20 split (NOT random): cutoff `2018-06-01`. `train_split` = 86,266 rows (2016-01-01 → 2018-06-01), `valid_split` = 21,567 rows (2018-06-01 → 2018-12-31); verified train fully precedes validation. `test.csv` (9,333 rows, no `target`) reserved for Zindi submission only.

### Phase 3.1 / 3.2 — Baseline + framing ✅ DONE
- **Baseline (constant mean, 44.9 min):** held-out validation **RMSE 141.95 min**. Per-route-mean baseline: **139.44 min**. This is the number to beat. *(Note: the earlier ~117 figure was the whole-dataset std, not a held-out score — superseded by 141.95.)*
- Phase 3.2 documents stakeholder / prediction / metric (RMSE) / regression framing in a markdown cell.

### Phase 4.1–4.5 — Modeling ✅ DONE
Time-aware CV + held-out validation RMSE:
| Model | Held-out RMSE (min) | vs baseline |
|---|---|---|
| **Random Forest (tuned) — SELECTED** | **109.53** | **−22.8%** |
| Random Forest (default) | 110.69 | −22.0% |
| XGBoost (tuned) | 111.10 | −21.7% |
| XGBoost (default) | 112.39 | −20.8% |
| Linear Regression | 131.96 | −7.0% |
| Baseline (per-route / constant) | 139.44 / 141.95 | — |
- Features frequency-encode `route`/`country_pair` for the tree models. Top features everywhere: `prev_leg_delay`, `hours_since_prior_leg`, `gc_distance_km`; XGB also leans on `is_domestic`.
- **Selected model: tuned Random Forest** (`max_depth=16, max_features=0.5, min_samples_leaf=3, n_estimators=391`), Phase 4.5 leaderboard + bar chart in the notebook.

### Phase 5.1 — Slide draft ✅ DONE
Stakeholder deck outline lives in [slides_draft.md](slides_draft.md) (verified against notebook outputs), rendered to `Tunisair Flight Delay Deck.html`. Slides for error analysis (11) and the data product (13) are intentional placeholders pending Phase 6 / Phase 8.

---

## 6. Data-quality issues log

All quirks are tracked in [ISSUES.md](ISSUES.md). Current entries (newest first):
1. **Corrupted `STA` dates** — a few flights have implausible durations (max ~11,992h) from wrong date components in `STA`. Doesn't affect `target`, but will poison duration features. **Resolved for now:** Phase 2.3 built no duration feature (distance/tz/country need no `STA`), so no cleanup was required; a scheduled-duration feature is a Phase 6.2 candidate that would need the cleanup first.
2. **Notebook order vs board** — notebook was reorganized so sections mirror the board 1:1. Reordering cleared cell outputs — **re-run the notebook top to bottom.**
3. **`SXF` missing from airportsdata** — manually overridden.

---

## 7. What's next (pick up here)

Modeling and the slide draft are done. The next block of work is **Phase 6 (error analysis)** and then the **data product (Phase 8)**.

- **#21 (6.1) — Error analysis (next up).** Residuals of the **selected tuned Random Forest** on the validation split, broken down by route / season / hour / aircraft / delay magnitude. Expect the long tail (rare severe delays) to dominate RMSE — quantify where the model is weakest.
- **#22 (6.2) — Iterate** on features/model from what 6.1 reveals. Candidate features are catalogued in [ISSUES.md](ISSUES.md) (Tier 1: airport congestion, turnaround slack, leg-of-day sequence, aircraft type / wet-lease flag).
- **#23 (6.3) — Final evaluation.** Refit the selected model on all labelled data, predict on the reserved `test.csv` (9,333 rows), and produce a Zindi submission. Save the fitted model with `joblib`.
- **Phase 7** — finalize the notebook (PEP 8, reproducible), export the deck to PDF, add the data-product slide, rehearse the 10-min talk.
- **Phase 8** — build the Streamlit delay-alert web app (data-product MVP): input a flight → predicted delay + risk category, with an alternative-flight suggestion.

**Once error analysis lands, refresh `slides_draft.md`:** fill Slide 11 (error analysis) and Slide 13 (data product), and refresh the results table if 6.2 changes the numbers.

---

## 8. Environment & repo notes

- **Package manager:** `uv`. Install new deps with `uv add <package>` (updates `pyproject.toml` + `uv.lock` for the team). Added so far: `airportsdata`, `holidays` (Tunisia holiday flag, Phase 2.2). e.g. `uv add xgboost` when you reach Phase 4.3.
- **Venv:** `.venv/` (Python 3.13). Run scripts with `.venv/bin/python`.
- **Notebook merge conflicts:** enable nbdime once — `uvx nbdime config-git --enable`; resolve with `uvx nbdime mergetool`. Clear cell outputs before committing to reduce conflicts (`Notebook: Clear All Outputs`).
- **Uncommitted at session end:** modified `04_flight_delay_eda_modeling.ipynb`, new `ISSUES.md`, new `HANDOFF.md`, and the lowercase `data/*.csv` files (the original uppercase `Train.csv`/`Test.csv`/`SampleSubmission.csv` show as deleted). There is also a stray empty `xx` file marked deleted. Review with `git status` before committing.

---

## 9. Chronological log of this session

What Sulu prompted (paraphrased) and what was done, in order:

1. **"Read the whole repo, understand what we've done, walk us through phase by phase, don't act without permission, populate notebook 4 with commented Python for each step."** → Read repo, memory, notebook; summarized status.
2. **"Implement the cleaning changes in notebook 4 / populate the Python code."** → Added Clean-the-Data cells: parse `STD`, parse `STA` (dot-fix), parse `DATOP`, drop `STATUS`, sanity checks.
3. **"What is the airport-enrichment code doing? Explain like we're beginners."** → Explained `enrich_airport_columns` line by line.
4. **"Add plain-language explanations to all the code in notebook 4."** → Added *Before / What / Goal* markdown above every code cell.
5. **"What's next?"** → Proposed Train-Test Split; began the temporal ordering check.
6. **"Always write one small code cell at a time with a plain-language explanation above each."** → Adopted as standing convention.
7. **Added target histograms** (full + capped at 1500 min); wrote right-skew takeaway.
8. **"Follow the board tasks sequentially and update statuses in real time."** → Adopted; synced board; agreed to work strict Phase 1.1 → 8.4 order.
9. **Phase 1.4 outlier analysis** → team decided **keep all rows, no cap**.
10. **"Is notebook 4 organized in the same sequence/naming as the board?"** → It wasn't; **reorganized the notebook to mirror the board 1:1** (one `## Phase X.Y` header per issue, placeholders included). Logged in `ISSUES.md`.
11. **Phase 1.5** → added STA/STD consistency check and train-vs-test date comparison; concluded test overlaps train and is unlabeled.
12. **"Are you considering timezone differences?"** → Added same-tz vs cross-tz duration diagnostic; **ran it in the venv on request**; found extremes are date-entry errors, not tz artifacts. Logged in `ISSUES.md`.
13. **Reinforced board-sync habit** after #8 was briefly left stale.
14. **Phase 1.7** → built 5 delay-pattern plots with per-plot takeaways + a ranked summary; marked #7 Done.
15. **"Sync the board to current statuses."** → Verified #1–7 Done, #8 In Progress, set #9 Ready.
16. **"Create this handoff file."** → This document.

### Session 2 (2026-07-22, Alex — branch `adw0721`)

17. **Ran the notebook end-to-end and continued Phase 2.4 → 5.1.** Built the cascading-delay feature (with verified leakage guard), the chronological split, the baseline, three models, hyperparameter tuning, and model selection (tuned Random Forest, 109.53 min RMSE).
18. **"What do the Milestone-2 slides need to present?"** → Derived expectations from `01_assignment.md` and produced [slides_draft.md](slides_draft.md) with placeholders for un-built work.
19. **"Verify against the notebook and update the draft."** → Repeatedly re-checked notebook outputs and synced `slides_draft.md` as Phases 2.4–4.5 completed; corrected the baseline figure (141.95, not the old ~117 std).
20. **Rendered the deck** to `Tunisair Flight Delay Deck.html`.
21. **"Update handoff and issues before I commit."** → This update (board status, Phase 2.4–5.1 build notes, next steps).

---

*When you resume: read this file, run the notebook top-to-bottom to regenerate outputs, sync the board (#11–20 → Done), then start **Phase 6.1 error analysis** on the selected tuned Random Forest.*
