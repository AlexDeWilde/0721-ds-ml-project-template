# Tunisair Delay-Alert (Streamlit MVP)

A booking-time delay-risk demo for the Tunisair Flight Delay project (Phase 8 data product).
Pick a flight — by **flight number** or by **route** — and a **date**, and get:

- a **delay-risk category** (🟢 Low / 🟡 Moderate / 🔴 High),
- an honest **typical-delay range** (the middle-half of similar historical flights),
- **action suggestions**, and
- **lower-risk alternative departure times** on the same route.

## Run it

```bash
uv sync                                   # installs streamlit + deps
uv run streamlit run app/streamlit_app.py
```

Then open the local URL Streamlit prints (usually http://localhost:8501).

## What powers it

- **`models/app_booking_model.joblib`** — a compact *booking-time* Random Forest. It uses only
  what a traveller knows before a flight: route, date, and time of day (day-of-week, month, holiday
  and Ramadan flags, great-circle distance, domestic/international, route frequency). It deliberately
  does **not** use day-of signals (the aircraft's earlier delays, congestion, weather).
- **`app/reference/*.csv`** — small lookup tables (route frequencies, per-flight typical departure
  hour, per-route delay quantiles) so the app can build features from typed inputs.

Regenerate all of the above from the training data with:

```bash
uv run python scripts/build_app_data.py
```

## Honest scope

- **Good at:** the risk *category* — it separates low- from high-risk flights well.
- **Not good at:** exact minutes. Held-out RMSE is ~137 min vs ~142 for guessing the average —
  *how many* minutes a specific flight slips is driven by day-of factors no booking-time tool can see.
  That's why the app shows a **range**, not a false-precise number.
- This is a **demo on 2016–2018 historical data**, not a live system.

An airline **operations** team, with the aircraft's real-time prior-leg status, can predict far more
accurately (held-out RMSE ~109 min via the notebook's operational model) — but that data isn't
available at booking time, which is exactly what this passenger-facing tool is designed around.

## Files

| File | Purpose |
|---|---|
| `app/streamlit_app.py` | The Streamlit UI |
| `app/delay_core.py` | Feature engineering, risk logic, predictions, alternatives (UI-free, testable) |
| `app/reference/` | Small lookup tables the app loads |
| `scripts/build_app_data.py` | Rebuilds the model + reference tables from `data/train.csv` |
