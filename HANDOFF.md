# Session Handoff — Tunisair Flight Delay Prediction

> **Purpose:** This document is a complete handoff of a Claude Code working session on `04_flight_delay_eda_modeling.ipynb`. It captures what was prompted, what was decided, what was built, the current project-board state, and exactly what to do next. Pull this repo, read this file top to bottom, and you can continue seamlessly in your own VS Code + Claude session.

**Session date:** 2026-07-21
**Branch:** `ml_project_sulu`
**Driven by:** Sulu (sulugambari)
**Continuing:** Alex (AlexDeWilde)
**Notebook:** [04_flight_delay_eda_modeling.ipynb](04_flight_delay_eda_modeling.ipynb)

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

- **Project:** "Flight Delay Prediction", **project #5**, owner **sulugambari**.
  URL: https://github.com/users/sulugambari/projects/5
- **Repo the issues live on:** `AlexDeWilde/0721-ds-ml-project-template`
  (The project is under sulugambari's account because GitHub blocks linking a project to a repo owned by a different user; both have admin.)
- **31 issues** across 8 phases.

### Useful CLI snippets for board updates
```bash
# List all items with their status
gh project item-list 5 --owner sulugambari --format json --limit 50

# Status field id: PVTSSF_lAHOAoc7qc4BeAE5zhYde-I
# Project id:      PVT_kwHOAoc7qc4BeAE5
# Status option ids:
#   Backlog=0be3c144  Ready=2dda5a79  In Progress=b9ef4cc9  Done=38c33c8a  Blocked=1a4144a7

# Example: move an item to Done
gh project item-edit --project-id "PVT_kwHOAoc7qc4BeAE5" \
  --id "<ITEM_ID>" \
  --field-id "PVTSSF_lAHOAoc7qc4BeAE5zhYde-I" \
  --single-select-option-id "38c33c8a"
```
To get an item's ID: `gh project item-list 5 --owner sulugambari --format json` and look up by issue number.

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
| **8** | **2.1 Define leakage-safe feature set** | 🟡 **In Progress** |
| 9 | 2.2 Engineer temporal features | 🔵 Ready |
| 10 | 2.3 Engineer route/airport features | 🔵 Ready |
| 11 | 2.4 Cascading-delay feature | ⚪ Backlog |
| 12 | 2.5 Chronological train/test split | ⚪ Backlog |
| 13 | 3.1 Baseline model + RMSE | ⚪ Backlog |
| 14 | 3.2 Document business framing | ⚪ Backlog |
| 15–19 | 4.1–4.5 Model iteration (LinReg, RF, GBM/XGB, tuning, compare) | ⚪ Backlog |
| 20 | 5.1 Draft slide deck | ⚪ Backlog |
| 21–23 | 6.1–6.3 Error analysis, iterate, final eval | ⚪ Backlog |
| 24–27 | 7.1–7.4 Finalize notebook/slides/data-product slide/rehearse | ⚪ Backlog |
| 28–31 | 8.1–8.4 Streamlit delay-alert web app | ⚪ Backlog |

**Milestone 1 (baseline) target: Day 2, 12:00.** All of Phase 1 (EDA) is complete.

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

### Phase 2.1 — Leakage-safe feature set (IN PROGRESS)
Done: the leaky `STATUS` column is dropped from both train and test.
**Still to do:** write the documented final feature list with justification (see "What's next").

---

## 6. Data-quality issues log

All quirks are tracked in [ISSUES.md](ISSUES.md). Current entries (newest first):
1. **Corrupted `STA` dates** — a few flights have implausible durations (max ~11,992h) from wrong date components in `STA`. Doesn't affect `target`, but will poison duration features. **Decision deferred to Phase 2.3** (drop/clip vs. recompute date from `DATOP`).
2. **Notebook order vs board** — notebook was reorganized so sections mirror the board 1:1. Reordering cleared cell outputs — **re-run the notebook top to bottom.**
3. **`SXF` missing from airportsdata** — manually overridden.

---

## 7. What's next (pick up here)

**Immediate:** the notebook's Phase 1.7 cells (and anything after the reorg) need a fresh **Run All** in VS Code to regenerate outputs/charts. Confirm the charts match the written takeaways.

**Next board item — #8 (Phase 2.1), to finish:** add a markdown cell documenting the final leakage-safe feature set with justification:
- **Keep:** `DEPSTN`, `ARRSTN` (route), `AC` (aircraft), `STD`-derived time fields (hour, day-of-week, month), airport enrichment (`dep_*`/`arr_*`), plus engineered features from Phases 2.2–2.4.
- **Exclude:** `STATUS` (leaky, dropped), `ID`/`FLTID` (identifiers), `target` (label). Handle `STA` carefully (schedule is known pre-departure, but corrupted dates make raw duration risky).
- Then move **#8 → Done**.

**Then, in board order:**
- **#9 (2.2)** Engineer temporal features: hour, day-of-week, month, Tunisia holiday flag from `STD`.
- **#10 (2.3)** Route/airport features: country pair, great-circle distance, timezone difference — **clean the corrupted `STA` dates first** if building duration.
- **#11 (2.4)** Cascading-delay feature: per aircraft (`AC`), prior leg's delay — **must be leakage-safe (only prior legs, never future).**
- **#12 (2.5)** Chronological train/validation split by a `DATOP` date cutoff (NOT random). Reserve `test.csv` for Zindi submission only.
- **#13 (3.1)** Baseline model (e.g. mean/per-route median delay) + RMSE → the number to beat (Milestone 1).
- **#14 (3.2)** Document business framing (already drafted in the Phase 3.2 notebook cell).
- **#15–19 (Phase 4)** LinReg → Random Forest → Gradient Boosting/XGBoost, each with cross-validation (use `TimeSeriesSplit` to respect time), then tune the best, then compare/select.
- **#21–23 (Phase 6)** Error analysis (residuals by route/season/aircraft/magnitude), iterate, final eval on held-out set + save model with `joblib`.
- **Phases 5, 7, 8** — slide deck, deliverables, and the Streamlit delay-alert web app (data-product MVP).

---

## 8. Environment & repo notes

- **Package manager:** `uv`. Install new deps with `uv add <package>` (updates `pyproject.toml` + `uv.lock` for the team). e.g. `uv add xgboost` when you reach Phase 4.3.
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

---

*Generated at the end of the session so work can continue in a fresh Claude Code chat. When you resume, start by reading this file, then run the notebook top-to-bottom, then continue with board item #8.*
