# Issues We Came Across Along The Way

A running log of data-quality quirks, gotchas, and decisions made during the project, so the reasoning behind them doesn't get lost. Newest entries at the top.

---

## A few flights have corrupted STA dates (implausible durations)

**Phase:** 1.5 — Temporal investigation
**Status:** Open (to handle before Phase 2.3)

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
