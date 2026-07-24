# Tunisair Delay-Alert (Streamlit MVP)

A booking-time delay-risk demo for the Tunisair Flight Delay project (Phase 8 data product).
Pick a flight — by **flight number** or by **route** — and a **date**, and get:

- a **delay-risk category** (🟢 Low / 🟡 Moderate / 🔴 High) backed by **calibrated probabilities**
  (chance of a 15+ / 60+ min delay),
- a **flight-specific delay range** (typical / usual-up-to / bad-day, from quantile models),
- a **weather-sensitivity ladder** — how the flight tends to run under that airport's plausible
  weather for the month (calm / rough / severe, named, with how often each occurs),
- **action suggestions**, and
- **lower-risk alternative departure times** on the same route.

## Run it

```bash
uv sync                                   # installs streamlit + deps
uv run streamlit run app/streamlit_app.py
```

Then open the local URL Streamlit prints (usually http://localhost:8501).

## What powers it

- **`models/app_booking_model.joblib`** — the *booking-time* models: two **calibrated classifiers**
  (P(delay ≥ 15 min), P(delay ≥ 60 min)) and three **quantile regressors** (p50/p75/p90). They use
  only what a traveller knows before a flight: route, date, time of day, **plus departure weather**.
  They still do **not** use signals you can't know at booking (the aircraft's earlier delays,
  congestion). Because you also can't know the weather, the app doesn't guess a forecast — it queries
  the models under each plausible named weather scenario (the *weather-sensitivity ladder*).
  Calibration is **time-aware** (calibrated on the most recent training slice) to resist temporal drift.
- **`app/reference/*.csv`** — small lookup tables (route frequencies, per-flight typical departure
  hour, per-route delay quantiles, and **per-airport/month weather scenarios**) so the app can build
  features and scenarios from typed inputs.

The weather scenarios are precomputed from free **ERA5** (Open-Meteo) records for the busiest ~15
departure airports (~77% of flights), cached by `scripts/fetch_weather.py`; other airports show the
estimate without a weather ladder. Fog and thunderstorms aren't represented in ERA5, so they aren't
shown yet (a future METAR upgrade). See `WEATHER_SCENARIO_EXPERIMENT.md` for the evidence behind this.

Regenerate all of the above from the training data with:

```bash
uv run python scripts/build_app_data.py
```

## Honest scope

- **Good at:** ranking risk — on a held-out future period it separates 15-min-late flights from
  on-time ones at ROC-AUC ~0.80 (60+ min delays ~0.76), with calibrated probabilities.
- **Not good at:** exact minutes — *how many* minutes a specific flight slips is driven by day-of
  factors no booking-time tool can see. That's why the app shows a **probability + range**, not a
  false-precise number.
- This is a **demo on 2016–2018 historical data**, not a live system.

An airline **operations** team, with the aircraft's real-time prior-leg status, can predict far more
accurately (held-out RMSE ~109 min via the notebook's operational model) — but that data isn't
available at booking time, which is exactly what this passenger-facing tool is designed around.

## Files

| File | Purpose |
|---|---|
| `app/streamlit_app.py` | The Streamlit UI |
| `app/delay_core.py` | Feature engineering, risk logic, predictions, alternatives (UI-free, testable) |
| `app/reference/` | Small lookup tables the app loads (incl. `weather_scenarios.csv`) |
| `scripts/build_app_data.py` | Rebuilds the model + reference tables (incl. weather scenarios) |
| `scripts/fetch_weather.py` / `weather_core.py` | Fetch/cache ERA5 weather + build scenario table |
