# Slides Draft — Milestone 2

> **Deliverable:** Milestone 2 "Slides draft" (target: Day 3, 12:00). This is the *draft* deck, not the final one.
> **Final deck** (Milestone 4 / Day 4) will build on this: a PDF stakeholder deck supporting a **10-minute presentation** for a **non-technical audience**, covering findings, recommendations, and future work, plus a slide on a potential data product (per `01_assignment.md`, "Final Deliverables").
>
> **Project:** Tunisair Flight Delay Prediction (Zindi). Predict flight delay in **minutes** (regression). Metric: **RMSE**.
> **Audience:** airline operations team; downstream, a travel app surfacing delay risk to passengers.
>
> **Legend:** ✅ = content available now from the repo · 🟡 = partial · ⛳ `[PLACEHOLDER]` = not yet produced (blocked on modeling / error analysis / data-product work still in the backlog).
>
> **Target length:** ~12–14 slides for a 10-minute talk. Keep one idea per slide, minimal text, visuals over tables.
>
> ---
> **Verification status (2026-07-22, updated):** All numbers below were checked against the *run* notebook `04_flight_delay_eda_modeling.ipynb`.
> - **EDA (Phase 1) — complete & verified.** Target: mean **48.7 min**, median **14 min**, std **117 min**, max **3451 min**; 0% missing; 107,833 flights.
> - **Feature engineering — now complete through Phase 2.5.** Temporal (2.2), route/airport (2.3), **cascading-delay (2.4)**, and the **chronological 80/20 split (2.5)** are all built and verified.
> - **Baseline (3.1) + three models (4.1–4.3) — DONE, with real held-out results.** Best model so far: **Random Forest, 110.69 min RMSE** vs a constant-mean baseline of **141.95 min**.
> - **Still NOT started:** hyperparameter tuning (4.4), formal model selection (4.5), **error analysis (Phase 6)**, and the **data product / Streamlit app (Phase 8)**. Slides 11 and 13 remain placeholders.
> - **Note on the old ~117 baseline:** earlier docs (`milestone_1.md`) quoted RMSE ~117 — that was the *whole-dataset* standard deviation. The real baseline on the *held-out chronological validation set* is **141.95 min**; use that figure going forward.

---

## Slide 1 — Title ✅
- **Title:** Predicting Tunisair Flight Delays
- Subtitle: Knowing delay risk *before the plane leaves the gate*
- Team names, date, "Milestone 2 — Draft"

## Slide 2 — The problem & who cares ✅
- Delays are costly and disruptive: knock-on schedule chaos for the **airline ops team**, uncertainty for **passengers**.
- Framing: if we can estimate a flight's likely delay using only what's known *before departure*, ops can plan and passengers can be warned.
- Stakeholders: (1) Tunisair operations, (2) a passenger-facing travel app (downstream).

## Slide 3 — The question & how we measure success ✅
- **Prediction:** how many minutes will this flight be delayed?
- **Why regression, not "delayed yes/no":** magnitude matters to ops planning.
- **Success metric: RMSE (minutes)** — matches the Zindi leaderboard; penalizes large misses, which is what hurts ops most.
- Non-technical gloss: "on average, how many minutes off are our predictions?"

## Slide 4 — The data ✅ *(verified)*
- Tunisair scheduled flights, **2016–2018** (DATOP 2016-01-01 → 2018-12-31), **107,833 flights** (train).
- Each flight: date, flight number, departure/arrival airports, scheduled times, aircraft.
- Enriched each airport with country, location, and timezone (via `airportsdata`); ~17.5% of flights are domestic.
- **0% missing values** across the dataset.
- Note (plain language): we predict using only pre-departure schedule/route/aircraft info.

## Slide 5 — Our golden rule: no cheating with the future (leakage) ✅
- Simple message: **we only use information available before the flight takes off.**
- Example we caught and removed: an operational-`STATUS` column that encodes what *actually* happened — using it would be cheating.
- Why it matters to stakeholders: a model that "cheats" looks great in testing but is useless in real life.

## Slide 6 — What drives delays (EDA headline) ✅
The strongest, most intuitive delay drivers we found:
- **Time of day (strong):** near-zero early morning, building to **20–30 min afternoon peaks** — delays cascade through the day.
- **Route (strong):** Algiers / Paris-Orly / Marseille run ~23–30 min; Djerba and internal hops near 0.
- **Season (solid):** summer (Jul–Sep) ~18–20 min vs February ~7 min.
- Weaker: day of week (slightly worse Sundays), specific aircraft (no standout).
- *Visual suggestion:* the delay-by-hour and delay-by-route charts from the notebook (Phase 1.7).

## Slide 7 — A note on data quality (builds trust) ✅ *(verified)*
- Delays are **heavily right-skewed:** median **14 min**, 75% of flights ≤ 43 min, with a long tail out to **3451 min**. Rare severe delays: 1.10% > 500 min, 0.18% > 1000 min.
- **Decision:** we kept the extreme delays — they're real (weather/mechanical/crew), and predicting them well is exactly what RMSE rewards.
- We found and flagged a few corrupted arrival timestamps; they don't affect the delay label and are quarantined from features.
- *(Keep this light for a non-technical audience — one sentence each.)*

## Slide 8 — Baseline: the number to beat ✅ *(verified)*
- A baseline is the simplest thing that could work — our yardstick.
- We tested honestly on a **held-out, time-based validation set** (train on the earliest 80% of flights, test on the most recent 20% — never predicting the past from the future).
- **Baseline (predict the average delay for every flight): RMSE = 141.95 min.**
- Slightly smarter baseline (predict each route's average): RMSE = 139.44 min.
- Everything we build has to beat ~142 minutes.
- *(Speaker note: earlier we quoted ~117 min — that was the spread of the whole dataset, not a held-out test. 141.95 is the honest number.)*

## Slide 9 — How we built a better model ✅ *(verified)*
- Engineered features from schedule/route/aircraft — **all leakage-safe** (only pre-departure info): departure hour/day/month, holiday flag; route distance, timezone difference, domestic-vs-international; and a per-aircraft **cascading-delay** signal (how late the *same plane's previous flight* was).
- **Leakage guardrail on the cascade feature verified:** 0 flights use a later leg to predict an earlier one.
- Tested **three algorithms** with time-aware cross-validation: Linear Regression, Random Forest, XGBoost.
- *Visual suggestion:* the feature-importance chart — it makes the headline (Slide 10a) land.

## Slide 10 — Results: we beat the baseline ✅ *(verified)*
Held-out validation RMSE (lower is better):

| Model | RMSE (min) | vs baseline |
|---|---|---|
| Baseline (constant mean) | 141.95 | — |
| Baseline (per-route mean) | 139.44 | −2.5 |
| Linear Regression | 131.96 | −10 |
| **Random Forest (best)** | **110.69** | **−31.3 (~22%)** |
| XGBoost | 112.39 | −29.6 |
- **Headline:** our best model (Random Forest) cuts prediction error by roughly **22%** versus guessing the average.
- *Visual suggestion:* horizontal bar chart of the five RMSE values.
- ⛳ `[PLACEHOLDER: numbers may improve after hyperparameter tuning (Phase 4.4) and formal model selection (4.5), which aren't done yet — refresh this table before the final deck.]`

## Slide 10a — The single biggest driver: knock-on delays ✅ *(verified — strong story slide)*
- Across every model, **the strongest predictor of a flight's delay is how late the same aircraft's previous flight was** (`prev_leg_delay`): correlation 0.37 with delay, and the #1 feature in both Linear Regression and Random Forest.
- Plain message: **delays cascade through the day along each aircraft's chain of flights.**
- Domestic-vs-international and route distance also matter.
- *This is the most stakeholder-friendly insight in the deck — consider leading with it.*

## Slide 11 — Where the model struggles (error analysis) ⛳
- ⛳ `[PLACEHOLDER: error-analysis findings — residuals by route / season / aircraft / delay magnitude. This is Milestone 3 work (Phase 6), not expected complete for the draft. Insert placeholder chart + "to come".]`

## Slide 12 — Recommendations 🟡 *(model-backed, pending error analysis)*
- Messages we can now stand behind (EDA **and** model):
  - **Protect early-leg punctuality.** The prior leg's delay is the #1 driver — an on-time morning keeps the whole day's chain on time. Build recovery buffers into aircraft rotations.
  - Focus operational attention on **afternoon departures, summer months, and the high-delay routes** (Algiers, Orly, Marseille).
  - A Random-Forest model predicts delay ~22% more accurately than the status quo — usable for advance planning.
- ⛳ `[PLACEHOLDER: sharpen with error-analysis findings (where the model is weakest) once Phase 6 is done.]`

## Slide 13 — Data product idea (required deliverable) 🟡
- Concept: a **delay-alert tool** — enter a flight (or have ops/the app pull the schedule) and get a predicted delay + risk level, before departure.
- Users: ops planners (staffing/gate decisions) and passengers (via a travel app).
- Planned MVP in this project: a small **Streamlit** app wrapping the model (Phase 8).
- ⛳ `[PLACEHOLDER: screenshot / mockup of the app once built; confirm exact scope of the MVP.]`

## Slide 14 — Future work & close ✅ / 🟡
- Next feature ideas (from `ISSUES.md`): airport **congestion** at scheduled hour, **turnaround slack**, leg-of-day sequence, Ramadan flag, aircraft type / wet-lease flag, weather.
- Concept drift over 2016–2018 (fleet re-assignment) → validated with a time-based split.
- Thank you / questions.

---

## What's still missing before this becomes the *final* deck

Confirmed against the run notebook (2026-07-22). The draft is now well ahead of the Milestone-2 bar — EDA, baseline, and three trained models are all real. Remaining gaps to fill for Milestone 4:

1. **Hyperparameter tuning + formal model selection** (Slides 9–10) — Phase 4.4/4.5 are still empty headers; the current "best = Random Forest" is from untuned models, so RMSEs may improve.
2. **Error analysis** (Slide 11) — Phase 6 not started; where does the model fail (which routes/seasons/delay magnitudes)? Tied to Milestone 3.
3. **Data-product specifics** (Slide 13) — the Streamlit delay-alert app (Phase 8) isn't built; needs a mockup/screenshot and a firm MVP scope.
4. **Final visuals** — export the actual charts from the notebook as images: EDA plots (Phase 1.7, cells 46/49/52/55/58) for Slides 6–7, feature-importance for Slide 10a, and an RMSE bar chart for Slide 10.

## Notes / assumptions
- All EDA, baseline, and model figures verified directly from notebook outputs (cells 97/99/109/117/123). Only error-analysis and data-product content remains un-computed.
- Model results are from **untuned** models on a single chronological split; treat as strong-but-preliminary and refresh after Phase 4.4/4.5.
- Board/handoff show error analysis (Milestone 3) downstream of this draft, so those placeholders are the expected state at Milestone 2, not a shortfall.
