"""Build the artifacts the Streamlit delay-alert app needs.

Trains a compact *booking-time* model (uses only what a traveller knows before a
flight: route, date, and time of day) and writes small reference tables so the
app can engineer features from typed inputs and predict a delay-risk category.

Run:  uv run python scripts/build_app_data.py
Outputs (all small and committable):
  models/app_booking_model.joblib   compact model + metadata
  app/reference/route_freq.csv      route -> how common it is (training count)
  app/reference/country_pair_freq.csv
  app/reference/flight_schedule.csv FLTID -> route + typical departure hour
  app/reference/route_delay_stats.csv route -> actual-delay p25/p50/p75 (the range)
"""
import json
import os
import sys

import airportsdata
import holidays
import joblib
import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import root_mean_squared_error

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import weather_core as wc  # noqa: E402  (ERA5 weather join + scenario table)

# Reuse the app's Ramadan logic so training and inference define the feature identically.
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "app"))
from delay_core import is_ramadan  # noqa: E402  (Hijri-based, works for any year)

# --- Feature config (must match the app's runtime) ---
# Booking-time features a traveller knows before the flight...
BOOKING_FEATURES = [
    "dep_hour", "dep_dow", "dep_month", "dep_is_holiday", "dep_is_ramadan",
    "gc_distance_km", "is_domestic", "route_freq", "country_pair_freq",
]
# ...plus departure weather (roadmap #2). At training we use ACTUAL historical (ERA5)
# weather; at booking we query the model under named weather scenarios (the "ladder").
WEATHER_FEATURES = wc.WEATHER_FEATURES  # wx_wind_gust, wx_precip, wx_snow, wx_adverse
MODEL_FEATURES = BOOKING_FEATURES + WEATHER_FEATURES
RISK_BANDS = {"low": 15, "moderate": 60}  # predicted < 15 -> Low; < 60 -> Moderate; else High
AIRPORT_OVERRIDES = {"SXF": {"country": "DE", "lat": 52.3667, "lon": 13.5033, "tz": "Europe/Berlin"}}

_airports = airportsdata.load("IATA")
_tn_holidays = holidays.Tunisia(years=range(2016, 2031))


def airport_latlon_country(code):
    a = _airports.get(code) or AIRPORT_OVERRIDES.get(code)
    if not isinstance(a, dict):
        return None
    return a["lat"], a["lon"], a["country"]


def haversine_km(lat1, lon1, lat2, lon2):
    r = 6371.0
    p1, p2 = np.radians(lat1), np.radians(lat2)
    dp, dl = np.radians(lat2 - lat1), np.radians(lon2 - lon1)
    a = np.sin(dp / 2) ** 2 + np.cos(p1) * np.cos(p2) * np.sin(dl / 2) ** 2
    return 2 * r * np.arcsin(np.sqrt(a))


def build_features(df):
    """Engineer the booking-time features on a dataframe with STD/DEPSTN/ARRSTN."""
    df = df.copy()
    df["dep_hour"] = df["STD"].dt.hour
    df["dep_dow"] = df["STD"].dt.dayofweek
    df["dep_month"] = df["STD"].dt.month
    df["dep_is_holiday"] = df["STD"].dt.date.map(lambda d: d in _tn_holidays).astype(int)
    # Ramadan on unique days only (Hijri conversion is per-date), then map back — fast.
    _days = df["STD"].dt.normalize()
    _ram = {d: is_ramadan(d) for d in _days.drop_duplicates()}
    df["dep_is_ramadan"] = _days.map(_ram).astype(int)
    dep = df["DEPSTN"].map(airport_latlon_country)
    arr = df["ARRSTN"].map(airport_latlon_country)
    df["gc_distance_km"] = [
        haversine_km(d[0], d[1], a[0], a[1]) if d and a else np.nan
        for d, a in zip(dep, arr)
    ]
    df["is_domestic"] = [
        int(d[2] == a[2]) if d and a else 0 for d, a in zip(dep, arr)
    ]
    df["route"] = df["DEPSTN"] + "->" + df["ARRSTN"]
    df["country_pair"] = [
        f"{d[2]}->{a[2]}" if d and a else "??->??" for d, a in zip(dep, arr)
    ]
    return df


def main():
    train = pd.read_csv("data/train.csv")
    train["STD"] = pd.to_datetime(train["STD"])
    train = build_features(train)

    # Frequency encodings learned from the full training set.
    route_freq = train["route"].value_counts()
    cp_freq = train["country_pair"].value_counts()
    train["route_freq"] = train["route"].map(route_freq).astype(int)
    train["country_pair_freq"] = train["country_pair"].map(cp_freq).astype(int)

    # Join ACTUAL historical departure weather (ERA5). Covered airports (the busiest ~15)
    # get real weather; everywhere else we fill NEUTRAL "typical calm" values so the model
    # still trains on every row and, at booking, an uncovered airport just gets a
    # typical-weather prediction (and no scenario ladder).
    wx = wc.load_weather_cache()
    covered = sorted(wc.covered_airports())
    train = wc.attach_weather(train, wx)
    cov_rows = train["wx_wind_gust"].notna()
    neutral = {
        "wx_wind_gust": round(float(train.loc[cov_rows, "wx_wind_gust"].median()), 1),
        "wx_precip": 0.0, "wx_snow": 0.0, "wx_adverse": 0,
    }
    for f in WEATHER_FEATURES:
        train[f] = train[f].fillna(neutral[f])

    # Compact model (kept small so it can live in git).
    model = RandomForestRegressor(
        n_estimators=120, min_samples_leaf=50, max_features="sqrt",
        n_jobs=-1, random_state=42,
    )
    model.fit(train[MODEL_FEATURES], train["target"])

    # Honest performance check on a chronological hold-out (last 20%).
    ts = train.sort_values("STD").reset_index(drop=True)
    cut = int(len(ts) * 0.8)
    tr, va = ts.iloc[:cut], ts.iloc[cut:]
    hold = RandomForestRegressor(
        n_estimators=120, min_samples_leaf=50, max_features="sqrt",
        n_jobs=-1, random_state=42,
    ).fit(tr[MODEL_FEATURES], tr["target"])
    rmse = root_mean_squared_error(va["target"], hold.predict(va[MODEL_FEATURES]))
    base = root_mean_squared_error(va["target"], [tr["target"].mean()] * len(va))

    joblib.dump(
        {"model": model, "features": MODEL_FEATURES, "risk_bands": RISK_BANDS,
         "holdout_rmse": round(float(rmse), 2), "baseline_rmse": round(float(base), 2),
         "neutral_weather": neutral, "weather_features": WEATHER_FEATURES,
         "weather_airports": covered},
        "models/app_booking_model.joblib",
    )

    # Per-airport/month weather scenarios (calm/rough/severe, named, with odds) for the
    # app's what-if ladder — precomputed so the app needs no network at run time.
    wc.build_scenario_table(wx).to_csv("app/reference/weather_scenarios.csv", index=False)

    # Reference tables for the app.
    route_freq.rename_axis("route").reset_index(name="route_freq").to_csv(
        "app/reference/route_freq.csv", index=False)
    cp_freq.rename_axis("country_pair").reset_index(name="country_pair_freq").to_csv(
        "app/reference/country_pair_freq.csv", index=False)

    # FLTID -> route + typical (modal) departure hour, for the flight-number lookup + dropdowns.
    # Exclude same-airport rows (DEPSTN == ARRSTN are positioning/maintenance, not passenger flights).
    passenger = train[train["DEPSTN"] != train["ARRSTN"]]
    sched = (
        passenger.groupby(["FLTID", "DEPSTN", "ARRSTN"])
        .agg(typical_hour=("dep_hour", lambda s: int(s.mode().iloc[0])),
             n_flights=("dep_hour", "size"))
        .reset_index()
        .sort_values("n_flights", ascending=False)
    )
    sched.to_csv("app/reference/flight_schedule.csv", index=False)

    # Route -> empirical actual-delay quantiles, for the honest "typical range".
    stats = (
        train.groupby("route")["target"]
        .agg(p25=lambda s: s.quantile(0.25), p50="median",
             p75=lambda s: s.quantile(0.75), n="size")
        .round(1).reset_index()
    )
    stats.to_csv("app/reference/route_delay_stats.csv", index=False)

    print("Built app artifacts:")
    print(f"  booking model: holdout RMSE {rmse:.2f} vs baseline {base:.2f} "
          f"({(base - rmse) / base * 100:.1f}% better) | features {len(MODEL_FEATURES)} "
          f"(+{len(WEATHER_FEATURES)} weather)")
    print(f"  routes: {len(route_freq)} | flights(FLTID rows): {len(sched)}")
    print(f"  weather: {len(covered)} covered airports, "
          f"{cov_rows.mean()*100:.0f}% of flights had real weather | neutral fill {neutral}")


if __name__ == "__main__":
    main()
