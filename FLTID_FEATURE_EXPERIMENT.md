# Flight-Number History as Model Features — Experiment Write-up

**Date:** 2026-07-23
**Branch:** `sulu-flight-history-features`
**Roadmap item:** #1 — "Flight-number track record as model features" (the cheapest, highest-ROI idea from the accuracy-improvement roadmap in `HANDOFF.md`)
**Artifact:** [scripts/experiment_fltid_features.py](scripts/experiment_fltid_features.py) — a self-contained measurement script (it does **not** change the app)

---

## 1. What we set out to test

Our booking-time app currently predicts a flight's delay risk from **route, date, and time of day** only. Each flight number's own track record (how late *that specific flight* usually runs) is used only to draw the "typical delay range" shown to the traveller — it is **never fed to the model**.

The question for today: **if we give the model each flight number's historical delay behaviour as an input, does it predict better?**

The intuition is appealing — "TU 0712 is always late" feels like it should help. The job of the experiment was to check whether that intuition survives an honest test, rather than assuming it does.

---

## 2. How we tested it honestly

Three things mattered for the test to be trustworthy:

1. **Same yardstick as the app.** We used the identical chronological hold-out the app's build script uses: train on the earliest ~80% of flights (Jan 2016 → Jun 2018), then measure error on the most recent ~20% (Jun 2018 → Dec 2018). We predict the *future* from the *past*, never the reverse.

2. **No cheating with the answer key ("leakage").** A flight's historical delay stats are built from past delays — i.e. from the very thing we're trying to predict. If we computed those stats using the whole dataset and then "tested" on part of it, the model would essentially be peeking at the answers, and the score would look great but be a lie. So the per-flight stats were computed **only from the training portion**, then looked up for the validation flights. This is the single most important discipline in the whole exercise.

3. **Not trusting thin evidence ("shrinkage").** Some flight numbers appear hundreds of times; others only a handful. A flight seen 3 times gives a very noisy "usual delay." So each flight's stat is blended toward the overall average, weighted by how many times we've actually seen it. Rarely-seen flights lean on the global average; frequently-seen flights are trusted on their own record. Flights never seen in training fall back to the global average.

We tried five candidate stats per flight number: **median delay, mean delay, % of flights that ran late (≥15 min), spread (standard deviation), and how many times the flight was seen.**

---

## 3. Results

All numbers are RMSE in minutes on the held-out recent flights (**lower is better**). The constant-mean baseline (141.95) is the "predict the average every time" floor we've always measured against.

| Model | Hold-out RMSE | Change vs. current app |
|---|---|---|
| Constant-mean baseline | 141.95 | — |
| **Current booking features** | **136.78** | — (this is what the app uses today) |
| Current + **all 5** flight-history stats | 136.84 | **−0.06 min (slightly worse)** |
| Current + **flight median delay only** | **136.22** | **+0.55 min better (0.40%)** |

Two clear findings came out of this:

### Finding A — Throwing in all five stats does *not* help (and slightly hurts)
When we added all five flight-history stats, they immediately became the model's **top five most-important inputs** — the model clearly *wanted* to lean on them. And yet the error got marginally *worse*. This is the textbook signature of **overfitting**: the stats look powerful on paper (in the training data) but don't carry forward to new months. The extra stats are largely redundant with each other, and the noisy ones (spread, % late) actively mislead.

The feature-importance ranking illustrates the trap — the model pours its attention into the flight-history stats even though they don't earn their keep:

| Feature | Importance | |
|---|---|---|
| flight mean delay | 0.202 | ← flight-history stat |
| flight times-seen | 0.135 | ← flight-history stat |
| flight % late | 0.130 | ← flight-history stat |
| flight median delay | 0.111 | ← flight-history stat |
| flight delay spread | 0.099 | ← flight-history stat |
| departure month | 0.093 | |
| departure hour | 0.047 | |
| day of week | 0.043 | |
| distance (km) | 0.040 | |
| route frequency | 0.034 | |
| country-pair frequency | 0.031 | |
| is domestic | 0.015 | |
| in Ramadan | 0.015 | |
| is holiday | 0.004 | |

### Finding B — A single, well-smoothed flight-median delay is a real (but small) win
When we kept **only** the flight's median historical delay — strongly smoothed toward the global average — the error dropped to **136.22**, a genuine **0.55-minute (0.40%) improvement**. Crucially, this gain was **stable** across a wide range of smoothing strengths (it held between roughly 136.2–136.3 for smoothing weights from 75 to 150), which tells us it's a real effect and not a lucky setting. Less smoothing, or less model regularisation, made things worse.

---

## 4. Why the improvement is small (and what that tells us)

Most flights are on time or a little late (median delay ~14 min); the error score is dominated by the **rare severe delays** — the 180-minutes-plus events caused by weather, mechanical issues, or knock-on disruption. A flight's *typical* (median) delay simply cannot anticipate those exceptional bad days, so even a perfect per-flight median can only nudge the overall error. This matches what Phase 6 error analysis already found: the RMSE lives in the severe-delay tail.

In short: **the idea works, but its ceiling is low by nature.** The larger accuracy gains on the roadmap are the ones that speak to *why a specific day is bad* — weather (#2) and a "closer to departure" mode that uses day-of signals (#6).

---

## 5. Recommendation & decision point

The gain is real, leakage-safe, and cheap, and it has a nice side benefit: it would make the **flight-number input actually influence the model's prediction** (today the flight number only changes the displayed range, not the risk).

Two reasonable paths from here:

- **A — Wire it into the app (recommended).** Add just the flight-median-delay feature to the build script, save a small flight-delay reference table, and read it at prediction time. Low complexity, honest improvement.
- **B — Stop here** and move to a higher-payoff roadmap item (weather, or the day-of mode), keeping this experiment on file as the documented finding.

**Nothing has been committed or merged yet, and the app is unchanged.** This document and the experiment script are the deliverables of this piece of work.

---

## 6. Reproduce it

```
.venv/bin/python scripts/experiment_fltid_features.py
```

Prints the comparison table, the verdict, and the feature-importance ranking above.
