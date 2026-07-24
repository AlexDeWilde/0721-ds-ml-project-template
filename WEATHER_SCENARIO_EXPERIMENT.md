# Weather as a Per-Flight Scenario Ladder — Experiment Write-up

**Date:** 2026-07-24
**Branch:** `sulu-flight-history-features`
**Roadmap item:** #2 — "Weather (forecast for near-term, climatology for far-off)"
**Artifacts (all measurement only — the app is unchanged):**
- [scripts/fetch_weather.py](scripts/fetch_weather.py) — downloads & caches historical weather
- [scripts/weather_core.py](scripts/weather_core.py) — shared weather helpers (naming, joining)
- [scripts/experiment_weather_features.py](scripts/experiment_weather_features.py) — the experiment

---

## 1. The idea we tested (and how it changed)

The original roadmap idea was "add weather as a model feature." But a booking-time app genuinely **cannot know the weather** weeks or months in advance, so a single weather-adjusted number would be dishonest.

The better framing — which this experiment is built around — is to show the traveller **how sensitive their specific flight is to weather**, as a small ladder of plausible, *named* scenarios:

```
  Typical (calm) weather   ~65% of days   ~38 min
  Snow                     ~30% of days   ~46 min
  Drizzle + strong wind     ~5% of days   ~59 min
  → weather-weighted outlook              ~42 min
```

Two flights with the same "typical" delay can behave very differently when the weather turns. That difference — *is this flight weather-robust or weather-fragile?* — is exactly what a booking tool should surface.

The mechanism: train **one** model on *actual historical weather*, then at prediction time run it several times with the weather inputs swapped to different scenario values (holding route/date/time fixed) — a "what-if" query. No live forecast is needed for the far-off case.

---

## 2. Where the weather data comes from

We used the free **Open-Meteo historical archive (ERA5 reanalysis)** — no API key. For the **15 busiest departure airports (77.3% of all flights)** we pulled 2016–2018 **hourly** departure weather — wind gusts, wind speed, precipitation, snowfall, temperature, and a WMO **weather code** (which lets us *name* the condition) — each in the airport's own local time so it lines up with the scheduled departure. The data is cached locally so nothing re-hits the network.

Joining this to flights matched **100%** of flights at those airports. As a sanity check, it reproduced our earlier feasibility finding: **adverse-weather departures average 58 min of delay vs 42 min in calm conditions** (median 19 vs 11).

---

## 3. The two findings (they point in opposite directions — read both)

### Finding A — Weather does NOT improve the single-number accuracy (RMSE)

| Model (15 covered airports) | Hold-out RMSE | Change |
|---|---|---|
| Constant-mean baseline | 132.65 | — |
| Base booking features | 126.81 | — |
| Base **+ weather features** | 127.24 | **−0.43 min (worse)** |

Just like the "throw in all the flight-history stats" experiment (#1), adding weather features did **not** lower the error. Same reason: RMSE is dominated by the **severe-delay tail** (180-minute-plus events), and historical/typical weather can't predict *which* specific future day will have an exceptional disruption.

*(A note on honesty: an early version included temperature, which looked useful — but temperature was mostly standing in for "season," which the model already gets from the month. We removed it so the measured weather effect reflects genuine weather, not a hidden season signal.)*

### Finding B — But the model's weather RESPONSE is strong, monotonic, and large

This is the finding that matters for the scenario-ladder feature. Holding everything else fixed and changing only the weather, the model's average predicted delay moves cleanly and substantially:

| Departure weather | Avg predicted delay | vs. calm |
|---|---|---|
| Calm / clear | 36.6 min | — |
| Rain (3 mm) | 47.8 min | +11.2 |
| Snow (3 cm) | 47.8 min | +11.2 |
| Strong wind (50 km/h) | 51.6 min | +15.0 |
| Heavy rain (8 mm) + wind | 63.1 min | +26.5 |

And the response to wind gusts is a clean, monotonic dose-response (worse wind → more delay, levelling off at the extreme):

```
  gust  5 km/h → 37.8 min
  gust 20 km/h → 40.5 min
  gust 35 km/h → 42.4 min
  gust 50 km/h → 51.6 min
  gust 65 km/h → 54.4 min
```

Wind gusts are by far the strongest weather signal (importance 0.103), with precipitation second and snow rare in this mostly-Mediterranean network.

**Bottom line:** weather fails as a way to sharpen the headline number, but **succeeds as a way to communicate flight-specific weather sensitivity.** The scenario ladder is the right product — exactly the "if fine → less delay, if stormy → more delay" idea.

---

## 4. Naming the weather (as requested)

Each scenario is labelled with the actual condition driving it, derived from the WMO weather code plus wind strength — e.g. **Clear, Overcast, Drizzle, Rain, Heavy rain, Snow, Strong wind, Gale-force wind**, and combinations like **"Drizzle + strong wind."** Scenarios are **anchored to each airport and month's own history**, so a "rough day" means a plausibly rough day *for that place and season*, and each carries its real frequency (% of days).

**Data limitation to flag:** the ERA5 weather code reliably reports rain/snow/cloud/wind but **essentially never emits fog or thunderstorm codes** — a known limitation of reanalysis data (our Phase 6 notes already flagged that fog/visibility needs METAR airport observations). So today we can name wind, rain, and snow honestly; **fog and thunderstorm labels would require adding METAR data** (a clear future upgrade, needs an airport-code mapping).

---

## 5. Worked examples (real routes, from the experiment)

**IST → TUN, January** — a weather-exposed winter route; the ladder is informative:
| Scenario | Frequency | Predicted delay |
|---|---|---|
| Drizzle | 65% of days | ~38 min |
| Snow | 30% of days | ~46 min |
| Drizzle + strong wind | 5% of days | ~59 min |
| **Weather-weighted outlook** | | **~42 min** |

**DJE → ORY, August** — a dry-summer route; the ladder is (correctly) almost flat:
| Scenario | Frequency | Predicted delay |
|---|---|---|
| Clear | 94% of days | ~66 min |
| Strong wind | 6% of days | ~64 min |
| **Weather-weighted outlook** | | **~66 min** |

The Djerba example is a feature, not a bug: this flight's delay is driven by operations/propagation, not weather, and the ladder honestly says so ("not very weather-sensitive").

**One caveat:** at the individual-flight level the model's weather response is occasionally flat or slightly non-monotonic (a random-forest quirk), even though it is clean and monotonic on average. If we ship the ladder, we should enforce a sensible ordering and phrase the numbers as ranges, not false precision.

---

## 6. Recommendation

**Go — but as a *scenario ladder*, not as a accuracy tweak.** The measured evidence says:
- Do **not** expect weather to improve the headline RMSE.
- **Do** show a per-flight, named, probability-weighted weather ladder — the model's weather response is real (11–27 min spread), monotonic on average, and genuinely useful to a traveller.

Suggested build (a later branch/step, once we decide):
1. Compute the per-airport/month scenario table (conditions, frequencies, representative values) offline and commit it as a small reference file — the app then needs no live network for the far-off case.
2. When a flight is within the ~16-day forecast horizon, additionally call the live forecast and **highlight the scenario row it points to** ("forecast says snow Friday → the ~46 min case").
3. Consider a future METAR upgrade to add **fog** and **thunderstorm** naming (the two aviation-critical conditions ERA5 misses).

**Nothing has been committed or merged; the app is unchanged.** These scripts and this document are the deliverables.

---

## 7. Reproduce it

```
.venv/bin/python scripts/fetch_weather.py            # one-time: cache ERA5 (data/weather_cache/, gitignored)
.venv/bin/python scripts/experiment_weather_features.py
```
