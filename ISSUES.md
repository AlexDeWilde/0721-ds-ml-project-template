# Issues We Came Across Along The Way

A running log of data-quality quirks, gotchas, and decisions made during the project, so the reasoning behind them doesn't get lost. Newest entries at the top.

---

## RESOLVED: Zindi test = whole hidden months → `prev_leg_delay` unavailable → two-track models

**Phase:** 2.4 / 6.3 — feature engineering vs. final evaluation
**Status:** Resolved (Phase 6.3, 2026-07-22) — supersedes the earlier "recompute the test cascade" plan

**Root cause found.** Inspecting the split by month shows Zindi holds out **entire calendar months** (e.g. May 2016 = 94% test, September 2018 = 97% test; all other months ~0% test). So within a test month a flight's previous leg on the same aircraft is — **98.9% of the time — also a hidden test flight**, whose delay we don't have. `prev_leg_delay` (the model's #1 feature, ~38% importance) therefore **cannot be observed for the Zindi test set**. Chained/fixed-point prediction of it was tried and **does not converge** (oscillates ±600 min), so it is not a viable reconstruction.

**Not leakage.** In a real ops room the prior leg has already landed before the current flight departs, so `prev_leg_delay` is legitimately known at prediction time — it is unavailable *here* only because of the month-block test format. `hours_since_prior_leg` / `has_prior_leg` need only schedule times and ARE computable for test.

**Decision — two-track (with the team):**
- **Operational model** (with `prev_leg_delay`): held-out **RMSE 108.64 min** (−23.5% vs baseline). The honest capability for the stakeholder deck, stated with its assumption. Saved as `models/delay_model.joblib` for the Phase 8 app.
- **Submittable model** (no `prev_leg_delay`): held-out **RMSE 130.92 min** (−7.8% vs baseline). Used for the Zindi leaderboard. Produces `zindi_submission.csv`; saved as `models/delay_model_submittable.joblib`.
- Dropping `prev_leg_delay` costs **+22.29 min** — this quantifies how much of delay is propagation, and is a headline *insight*, not a defect.

**Slides action (Phase 7.2):** headline 108.64 *with the stated assumption*; also state the submittable 130.92 so the story is defensible. Frame the gap as the propagation insight.

## Baseline is 141.95 min (held-out), not ~117

**Phase:** 3.1 — Baseline
**Status:** Resolved (figure corrected)

`milestone_1.md` quotes a baseline RMSE of ~117 min. That was the **standard deviation of the whole `target`** — i.e. the RMSE a constant-mean predictor would get *on the full dataset*, not a held-out score. The honest baseline, measured on the chronological **validation split**, is **141.95 min** (constant mean) / **139.44 min** (per-route mean). Use 141.95 as "the number to beat" in the deck and any write-up; the tuned Random Forest scores 109.53 (−22.8%).

## Candidate features (future — Phase 6.2 refinement backlog)

**Phase:** 6.2 — feature ideas, not yet built
**Status:** Open (backlog)

Brainstormed additional delay drivers, tiered by ROI-vs-effort and leakage-safety. Real-world delay is dominated by **congestion** and **knock-on propagation**, so the free schedule-derived proxies (Tier 1) are the priority.

**Tier 1 — derivable from existing data, free, guaranteed leakage-safe (highest priority):**
- **Airport congestion at the scheduled hour** — count of Tunisair flights scheduled to depart/arrive from the same airport in the same hour (or 15-min) window; proxy for gate/runway/ground-crew contention.
- **Turnaround slack** — gap between a tail's previous-leg `STA` and its next `STD`; tight turnarounds are the main propagation mechanism (partners the Phase 2.4 cascade feature).
- **Leg-of-day sequence** — is this the tail's 1st/3rd/6th flight of the day (delay accumulates down the chain).
- **System load** — total flights that day; hub load at TUN specifically.
- **Aircraft type + wet-lease flag** — see the fleet-swap note below.

**Tier 2 — cheap external joins, plausible value:**
- **Ramadan flag** — Tunisia-specific, likely high value; moving 30-day window not captured by month/holiday; derivable from the hijri/`holidays` machinery already added.
- **School-holiday calendars** (Tunisian + French — France is a large share of routes) — leisure-demand surges.
- **Departure-airport weather** — real but partly already encoded in season/hour (see the weather note; mind the arrival-forecast leakage trap).
- **Aircraft age** — registration → delivery date; older frames → more mechanical delay.
- **Airport structural attributes** — slot-controlled/curfew (e.g. Orly), runway count, hub vs spoke.

**Tier 3 — real but low practical ROI here:**
- **French ATC strikes** — the one worth a look (heavy France exposure); messy data, effectively a forecast at departure time.
- **Economic/political/demand shifts, airport upgrades over time, crew flu/sickness** — slow-moving and largely already absorbed by season/year; collide with the year-extrapolation risk; better surfaced via Phase 6.1 residual analysis (e.g. residuals by airport over time) than engineered explicitly.

---

## Aircraft-route assignments swap heavily over time (fleet re-assignment / drift)

**Phase:** 2.4 / 6.2 — feature-engineering note
**Status:** Open (feature ideas for later; does NOT block the planned cascade feature)

Investigated whether the aircraft serving a route is stable across the 2016–2018 horizon (raised because route↔aircraft coupling and mid-horizon equipment upgrades could shift a route's delay behavior). Measured on train+test combined (117,166 rows; 135 airports, 761 routes, 70 tail IDs):

- **Swaps are the norm.** On busy routes (≥200 flights): the dominant *tail* changed year-to-year on **96%** of routes (expected — tails rotate constantly), and the dominant aircraft *type* changed on **52%** (real re-fleeting, e.g. `AMS->TUN` ran 737→737→A320, `BEG->TUN` 737→A320→A320).
- The literal "same physical aircraft relabeled/upgraded" case is tiny: only **3 of 66** physical tails ever change their type code.

**Implications:**
1. **The Phase 2.4 cascade feature is safe.** It is keyed on the physical tail (`AC`) and follows that plane's own prior-leg delay regardless of route, so route-level equipment swaps do not corrupt it.
2. **Concept drift:** route-level equipment changes mid-horizon are a further argument *for* the chronological train/validation split (Phase 2.5) over a random one — a time split tests robustness to this drift instead of leaking future equipment mixes into training.
3. **Two cheap candidate features (parse-for-free — `AC` already embeds the type, e.g. `TU 320IMW` → type `320`, tail `IMW`):**
   - **Aircraft type** (24 values, more stable/generalizable than the 70 tails; 44% of routes are single-type).
   - **Wet-lease / ACMI flag** — the two-letter-prefixed types (`5K`, `GJ`, `UG`, `D4`, `X9`, …) are subcontracted operations by other carriers and plausibly carry a different delay profile than Tunisair mainline.

---

## A few flights have corrupted STA dates (implausible durations)

**Phase:** 1.5 — Temporal investigation
**Status:** Deferred (never blocked modeling; only matters if a duration feature is added in Phase 6.2)

Computing naive flight duration as `STA - STD` gave a sensible median (~2.3h) but a max of ~11,992 hours (≈500 days). Splitting by same-timezone vs. cross-timezone flights showed the extreme values sit in the **same-timezone** group (offset = 0), so they cannot be a timezone artifact — they're date-entry errors in a small number of `STA` values (the arrival's date component is wrong). Cross-timezone durations are clean and tight (max ~8.9h), because Tunisair's network spans only small offsets (~0–2h).

**Impact:** does NOT affect the `target` label (which is clean), but WOULD poison any duration-based feature. **Decision deferred:** clean or clip these rows before engineering a duration feature in Phase 2.3 (drop/clip vs. recompute the date from `DATOP` — to be decided by the team).

---

## Notebook section order didn't match the project board sequence

**Phase:** Cross-cutting (discovered while working Phase 1.4)
**Status:** Resolved

While building out `04_flight_delay_eda_modeling.ipynb`, sections were added in a pragmatic, EDA-driven order (e.g. Airport Enrichment ended up placed after the Leakage Audit and Clean-the-Data sections, and the Phase 1.5 temporal-ordering check got buried inside the Train-Test Split section) rather than following the GitHub Project board's strict Phase 1.1 → 8.4 issue sequence and titles.

**Decision:** reorganized the notebook so its section headers and order physically mirror the board 1:1 — one `## Phase X.Y: <exact issue title>` header per board issue, in board order, including placeholder headers for phases not yet started so the full sequence is visible even before the work exists. Reordering existing cells required deleting and reinserting them, which cleared their previously-computed outputs — re-run the notebook top to bottom to regenerate them.

---

## SXF (Berlin-Schoenefeld) missing from airportsdata

**Phase:** 1.2 — Airport enrichment
**Status:** Resolved

`airportsdata`'s IATA table has no entry for `SXF`. Berlin-Schoenefeld was decommissioned in 2020 (merged into Berlin Brandenburg/BER), so current lookup tables don't carry it — even though it's a valid code in this 2016-2018 dataset.

**Decision:** manually override it in the enrichment step with Schoenefeld's real-world coordinates/timezone (`country: DE, lat: 52.3667, lon: 13.5033, tz: Europe/Berlin`), rather than leaving it as `NaN`. See the `AIRPORT_OVERRIDES` dict in `04_flight_delay_eda_modeling.ipynb`.
