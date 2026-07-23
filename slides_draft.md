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
> **This file started as the Phase 5.1 deliverable** ("Draft stakeholder slide deck") and has since been **extended through Phase 6** (error analysis, feature iteration, two-track final evaluation) as part of Phase 7.2 — it is now the near-final content, pending the data-product slide and PDF render.
>
> **Verification status (2026-07-22, latest):** All numbers below were checked against the *run* notebook `04_flight_delay_eda_modeling.ipynb`.
> - **EDA (Phase 1) — complete & verified.** Target: mean **48.7 min**, median **14 min**, std **117 min**, max **3451 min**; 0% missing; 107,833 flights.
> - **Feature engineering — complete through Phase 2.5.** Temporal (2.2), route/airport (2.3), **cascading-delay (2.4)**, and the **chronological 80/20 split (2.5)** all built and verified.
> - **Modeling & error analysis — now COMPLETE (Phases 3.1–6.3).** Baseline, three algorithms, tuning, selection, error analysis, feature iteration, and the two-track final evaluation are all done. **Selected operational model: Random Forest + engineered features — held-out RMSE 108.64 min, 23.5% better than the 141.95-min baseline.**
> - **Two-track result (key, Phase 6.3):** the operational model's top feature is `prev_leg_delay` (how late the same aircraft's previous flight was) — known in a real ops room. **Zindi hides whole test months**, so that feature is unavailable there; a **submittable model without it scores 130.92 min** and produces `zindi_submission.csv`. Both are honest — the deck leads with 108.64 *and its assumption*.
> - **Still NOT started:** the **data product / Streamlit app (Phase 8)**. Slide 13 remains a placeholder.
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
- Tested **three algorithms** with time-aware cross-validation (Linear Regression, Random Forest, XGBoost), then **tuned** the front-runners and selected the best.
- *Visual suggestion:* the feature-importance chart — it makes the headline (Slide 10a) land.

## Slide 10 — Results: we beat the baseline ✅ *(verified — Phases 4 & 6 complete)*
Held-out validation RMSE (lower is better):

| Model | RMSE (min) | vs baseline |
|---|---|---|
| Baseline (constant mean) | 141.95 | — |
| Baseline (per-route mean) | 139.44 | −1.8% |
| Linear Regression | 131.96 | −7.0% |
| XGBoost (tuned) | 111.10 | −21.7% |
| Random Forest (tuned) | 109.53 | −22.8% |
| **Random Forest + engineered features — selected** | **108.64** | **−23.5%** |
- **Headline:** our chosen model cuts prediction error by roughly **24%** versus guessing the average (108.64 vs 141.95 min).
- Tree-based models clearly beat linear; tuning plus a second round of features (airport congestion, leg-of-day, aircraft type) took us from 109.53 → 108.64.
- *Visual suggestion:* horizontal bar chart of the RMSE values (the notebook produces one in Phase 4.5).

## Slide 10a — The single biggest driver: knock-on delays ✅ *(verified — strong story slide)*
- Across every model, **the strongest predictor of a flight's delay is how late the same aircraft's previous flight was** (`prev_leg_delay`): the #1 feature by far (~38% of the Random Forest's importance).
- Plain message: **delays cascade through the day along each aircraft's chain of flights.**
- Domestic-vs-international and route distance also matter.
- *This is the most stakeholder-friendly insight in the deck — consider leading with it.*

## Slide 10b — The honesty slide: what a competition can't see ✅ *(NEW — Phase 6.3, memorable point)*
- Our best signal, `prev_leg_delay`, is worth **~22 RMSE minutes** on its own — dropping it takes the model from 108.64 to **130.92 min**.
- **A real airline knows it** (the previous flight has already landed before the next departs), so 108.64 is our true operational capability.
- **The Zindi competition hides entire test months**, so within them that signal is unavailable — hence a separate, honest *submittable* model at 130.92 (`zindi_submission.csv`).
- Plain message: *"Our model is so good at reading delay already flowing through the system that its top clue is one a real operations team has — but a leaderboard deliberately hides."* Not a weakness — a sign the model learned the right thing.

## Slide 11 — Where the model struggles (error analysis) ✅ *(verified — Phase 6.1 complete)*
- **The model plays it safe on extremes.** It slightly *over*-predicts on-time flights and heavily *under*-predicts severe delays: for flights delayed 180+ min (real avg ~382 min) it predicts only ~237 — missing by ~145 min. Those ~2,000 severe flights drive most of the error.
- **Worst routes:** infrequent long-haul routes (Abidjan, Dakar, Bamako, Paris-CDG, Amsterdam) — too few examples to learn from.
- **Worst aircraft:** the **wet-lease / subcontracted planes** run later than the mainline fleet the model expects.
- **Season:** error is highest in **August and December** (peak travel).
- Plain message: the model is reliable for everyday delays; its misses cluster on **rare, severe disruptions** whose causes (weather, ATC) aren't in the data.
- *Visual suggestion:* the "mean residual by delay magnitude" bar chart from the notebook (Phase 6.1).

## Slide 12 — Recommendations ✅ *(model- and error-analysis-backed)*
- Messages we can now stand behind (EDA, model, **and** error analysis):
  - **Protect early-leg punctuality.** The prior leg's delay is the #1 driver — an on-time morning keeps the whole day's chain on time. Build recovery buffers into aircraft rotations.
  - Focus operational attention on **afternoon departures, summer months (Aug), and the high-delay routes** (Algiers, Orly, Marseille).
  - **Add manual oversight for the model's known blind spots:** rare long-haul routes and **wet-lease flights**, where predictions are least reliable — don't let the tool run unchecked there.
  - Treat the model as a **planning aid for typical delays**, not a guarantee on rare severe ones — its residual error is concentrated in disruptions (weather/ATC) it can't yet see.
  - A model predicting delay ~24% more accurately than the status quo is usable for advance staffing, gate, and passenger-comms decisions.

## Slide 13 — Data product: "Tunisair Delay-Alert" ✅ *(the required data-product slide)*

**One-liner:** a pre-departure delay-alert tool — pick a flight, get its **predicted delay + risk level before the plane leaves the gate**, and (for passengers) **lower-delay alternative flights** on the same route.

**Who uses it & the decision it supports:**
- **Ops planners** → advance staffing, gate, and turnaround decisions on flights flagged high-risk.
- **Passengers** (via a travel app) → set expectations, and see calmer alternatives when rebooking.

**What you put in → what you get out:**
- *In:* a scheduled flight (route, time, aircraft) — and, for ops, the aircraft's prior-leg status they already track.
- *Out:* predicted delay in minutes, a **risk badge** (🟢 Low < 15 min · 🟡 Moderate 15–60 · 🔴 High 60+), and a short list of same-route flights with lower expected delay.

```
┌──────────────────────────────────────────────┐
│  Tunisair Delay-Alert                          │
│  Route: TUN → ORY   Date: 2018-08-12  14:30    │
│  Aircraft: TU 320IMU                           │
│  ───────────────────────────────────────────  │
│  Predicted delay:  47 min      🔴 HIGH RISK     │
│  ───────────────────────────────────────────  │
│  Calmer alternatives on TUN → ORY:             │
│    • 08:10 departure   ~12 min   🟢             │
│    • 11:25 departure   ~21 min   🟡             │
└──────────────────────────────────────────────┘
```

**Why it works — and why the product beats the leaderboard:** the tool runs the **operational model (108.64 min)**, which uses the prior-leg delay. A real ops team *has* that data in real time (the previous flight has landed) — so the product delivers the model's full strength, the very signal the Zindi test format hides. *The "hidden feature" from Slide 10b is exactly what makes this product valuable.*

**Scope (honest):** MVP is a **Streamlit demo on the historical dataset** (Phase 8) — a proof of concept, not a live/real-time system. ⛳ `[Add app screenshot once Phase 8 is built.]`

## Slide 14 — Future work & close ✅ *(weather now evidence-backed)*
- **#1 next lever — weather.** We ran a quick feasibility check with real historical weather (Open-Meteo) at the busiest airports: **adverse weather (gusts/precip) raises the severe-delay rate from ~5% to ~7%** — exactly the errors the model misses today. Worth a full build with airport-level weather (ideally METAR visibility/ceiling).
- Other candidate features (some already added in Phase 6.2 — congestion, leg-of-day, aircraft type): **turnaround slack**, a **Ramadan** refinement, and richer **wet-lease** handling.
- Concept drift over 2016–2018 (fleet re-assignment) → validated with a time-based split.
- Thank you / questions.

---

## What's still missing before this becomes the *final* deck

Updated against the run notebook (2026-07-22, after Phase 6). EDA, baseline, a fully tuned/selected model, **error analysis, and the two-track final evaluation are all done**. Remaining gaps for Milestone 4:

1. ~~Error analysis~~ ✅ done (Slide 11 filled from Phase 6.1).
2. ~~Final Zindi-test evaluation~~ ✅ done — `zindi_submission.csv` produced (submittable model, 130.92 min); see the two-track story (Slide 10b).
3. **Data-product specifics** (Slide 13) — the Streamlit delay-alert app (Phase 8) isn't built; needs a mockup/screenshot and a firm MVP scope. The trained model is saved (`models/delay_model.joblib`) ready to wrap.
4. **Final visuals** — export the actual charts from the notebook as images: EDA plots (Phase 1.7) for Slides 6–7, feature-importance and the RMSE leaderboard bar chart (Phase 4.5) for Slide 10, and the **residual-by-magnitude** chart (Phase 6.1) for Slide 11.
5. **Render to PDF** — the HTML deck (`Tunisair Flight Delay Deck.html`) predates Phase 6 and must be re-synced with these slides, then exported to PDF for submission.

## Notes / assumptions
- All EDA, baseline, model, and error-analysis figures verified directly from notebook outputs. The only remaining un-built piece is the **data product / Streamlit app (Phase 8)**.
- **Selected operational model:** Random Forest + engineered features, **108.64 min** held-out RMSE, measured on a single chronological validation split. Its headline number assumes `prev_leg_delay` is available at prediction time (true operationally; not for the Zindi hidden months).
- **Submittable model** (no `prev_leg_delay`): **130.92 min**, used for the Zindi entry (`zindi_submission.csv`).
- See `ISSUES.md` for the full record of the two-track decision and the month-block test-set finding.
