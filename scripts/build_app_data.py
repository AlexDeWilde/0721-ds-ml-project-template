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

import airportsdata
import holidays
import joblib
import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import root_mean_squared_error

# --- Feature config: booking-time features only (must match the app's runtime) ---
BOOKING_FEATURES = [
    "dep_hour", "dep_dow", "dep_month", "dep_is_holiday", "dep_is_ramadan",
    "gc_distance_km", "is_domestic", "route_freq", "country_pair_freq",
]
RISK_BANDS = {"low": 15, "moderate": 60}  # predicted < 15 -> Low; < 60 -> Moderate; else High
RAMADAN = [("2016-06-06", "2016-07-05"), ("2017-05-27", "2017-06-24"),
           ("2018-05-16", "2018-06-14")]
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


def is_ramadan(ts):
    return any(pd.Timestamp(s) <= ts <= pd.Timestamp(e) + pd.Timedelta(days=1) for s, e in RAMADAN)


def build_features(df):
    """Engineer the booking-time features on a dataframe with STD/DEPSTN/ARRSTN."""
    df = df.copy()
    df["dep_hour"] = df["STD"].dt.hour
    df["dep_dow"] = df["STD"].dt.dayofweek
    df["dep_month"] = df["STD"].dt.month
    df["dep_is_holiday"] = df["STD"].dt.date.map(lambda d: d in _tn_holidays).astype(int)
    df["dep_is_ramadan"] = df["STD"].map(is_ramadan).astype(int)
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

    # Compact model (kept small so it can live in git).
    model = RandomForestRegressor(
        n_estimators=120, min_samples_leaf=50, max_features="sqrt",
        n_jobs=-1, random_state=42,
    )
    model.fit(train[BOOKING_FEATURES], train["target"])

    # Honest performance check on a chronological hold-out (last 20%).
    ts = train.sort_values("STD").reset_index(drop=True)
    cut = int(len(ts) * 0.8)
    tr, va = ts.iloc[:cut], ts.iloc[cut:]
    hold = RandomForestRegressor(
        n_estimators=120, min_samples_leaf=50, max_features="sqrt",
        n_jobs=-1, random_state=42,
    ).fit(tr[BOOKING_FEATURES], tr["target"])
    rmse = root_mean_squared_error(va["target"], hold.predict(va[BOOKING_FEATURES]))
    base = root_mean_squared_error(va["target"], [tr["target"].mean()] * len(va))

    joblib.dump(
        {"model": model, "features": BOOKING_FEATURES, "risk_bands": RISK_BANDS,
         "holdout_rmse": round(float(rmse), 2), "baseline_rmse": round(float(base), 2)},
        "models/app_booking_model.joblib",
    )

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
          f"({(base - rmse) / base * 100:.1f}% better)")
    print(f"  routes: {len(route_freq)} | flights(FLTID rows): {len(sched)}")


if __name__ == "__main__":
    main()
