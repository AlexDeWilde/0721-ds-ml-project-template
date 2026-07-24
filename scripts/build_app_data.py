"""Build the artifacts the Streamlit delay-alert app needs.

Trains *booking-time* models (using only what a traveller knows before a flight:
route, date, time of day, plus departure weather) and writes small reference tables
so the app can engineer features from typed inputs.

As of the roadmap-#5 reframe, the app no longer predicts a single mean-minutes number.
It uses:
  * two TIME-AWARE CALIBRATED classifiers — P(delay >= 15 min) and P(delay >= 60 min) —
    which separate risk better than thresholding a mean and yield honest probabilities;
  * three QUANTILE regressors — p50/p75/p90 — for a flight-specific delay range.
See RISK_CLASSIFIER_EXPERIMENT.md for the evidence.

Run:  uv run python scripts/build_app_data.py
Outputs (all small and committable):
  models/app_booking_model.joblib   classifiers + quantile models + metadata
  app/reference/route_freq.csv, country_pair_freq.csv, flight_schedule.csv,
  route_delay_stats.csv, weather_scenarios.csv
"""
import os
import sys

import airportsdata
import holidays
import joblib
import numpy as np
import pandas as pd
from sklearn.ensemble import (GradientBoostingRegressor,
                              HistGradientBoostingClassifier)
from sklearn.isotonic import IsotonicRegression
from sklearn.metrics import brier_score_loss, roc_auc_score

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import weather_core as wc  # noqa: E402  (ERA5 weather join + scenario table)

# Reuse the app's Ramadan logic so training and inference define the feature identically.
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "app"))
from delay_core import is_ramadan  # noqa: E402  (Hijri-based, works for any year)

# --- Feature config (must match the app's runtime) ---
BOOKING_FEATURES = [
    "dep_hour", "dep_dow", "dep_month", "dep_is_holiday", "dep_is_ramadan",
    "gc_distance_km", "is_domestic", "route_freq", "country_pair_freq",
]
WEATHER_FEATURES = wc.WEATHER_FEATURES  # wx_wind_gust, wx_precip, wx_snow, wx_adverse
MODEL_FEATURES = BOOKING_FEATURES + WEATHER_FEATURES

# Risk thresholds (minutes) the classifiers predict the probability of exceeding.
LATE_MIN, SEVERE_MIN = 15, 60
# Map those probabilities to a risk band shown as 🟢/🟡/🔴.
RISK_PROB_THRESHOLDS = {"high_p_severe": 0.33, "moderate_p_late": 0.50}
QUANTILES = [0.50, 0.75, 0.90]
CALIB_FRAC = 0.8  # base model on earliest 80% by date, isotonic calibration on the recent 20%
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
    _days = df["STD"].dt.normalize()
    _ram = {d: is_ramadan(d) for d in _days.drop_duplicates()}
    df["dep_is_ramadan"] = _days.map(_ram).astype(int)
    dep = df["DEPSTN"].map(airport_latlon_country)
    arr = df["ARRSTN"].map(airport_latlon_country)
    df["gc_distance_km"] = [
        haversine_km(d[0], d[1], a[0], a[1]) if d and a else np.nan
        for d, a in zip(dep, arr)
    ]
    df["is_domestic"] = [int(d[2] == a[2]) if d and a else 0 for d, a in zip(dep, arr)]
    df["route"] = df["DEPSTN"] + "->" + df["ARRSTN"]
    df["country_pair"] = [f"{d[2]}->{a[2]}" if d and a else "??->??" for d, a in zip(dep, arr)]
    return df


def fit_calibrated(X, y_binary):
    """Time-aware calibrated classifier: fit the base model on the earliest CALIB_FRAC of
    the (date-sorted) rows, then isotonically calibrate on the most recent slice. Returns
    (base_model, isotonic) — applied at inference as iso.transform(base.predict_proba[:,1])."""
    n = len(X)
    c = int(n * CALIB_FRAC)
    base = HistGradientBoostingClassifier(
        max_depth=4, learning_rate=0.08, max_iter=300, random_state=42)
    base.fit(X.iloc[:c], y_binary[:c])
    iso = IsotonicRegression(out_of_bounds="clip")
    iso.fit(base.predict_proba(X.iloc[c:])[:, 1], y_binary[c:])
    return base, iso


def predict_proba(clf, X):
    base, iso = clf
    return iso.transform(base.predict_proba(X)[:, 1])


def main():
    train = pd.read_csv("data/train.csv")
    train["STD"] = pd.to_datetime(train["STD"])
    train = build_features(train)

    route_freq = train["route"].value_counts()
    cp_freq = train["country_pair"].value_counts()
    train["route_freq"] = train["route"].map(route_freq).astype(int)
    train["country_pair_freq"] = train["country_pair"].map(cp_freq).astype(int)

    # Join ACTUAL historical departure weather (ERA5); neutral 'typical calm' fill where the
    # airport has no coverage, so every row trains and uncovered airports still get a prediction.
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

    # Date-sorted view for time-aware calibration and honest evaluation.
    ts = train.sort_values("STD").reset_index(drop=True)
    X, y = ts[MODEL_FEATURES], ts["target"].to_numpy()
    y_late, y_severe = (y >= LATE_MIN).astype(int), (y >= SEVERE_MIN).astype(int)

    # --- Shipped models: calibrated classifiers + quantile regressors (fit on ALL data) ---
    clf_late = fit_calibrated(X, y_late)
    clf_severe = fit_calibrated(X, y_severe)
    quantiles = {}
    for q in QUANTILES:
        quantiles[q] = GradientBoostingRegressor(
            loss="quantile", alpha=q, n_estimators=300, max_depth=3,
            learning_rate=0.05, random_state=42).fit(X, y)

    # --- Honest metrics on a future hold-out (never seen by base or calibration) ---
    cut = int(len(ts) * 0.8)
    tr, va = ts.iloc[:cut], ts.iloc[cut:]
    metrics = {}
    for name, thr in [("late", LATE_MIN), ("severe", SEVERE_MIN)]:
        c = fit_calibrated(tr[MODEL_FEATURES], (tr["target"].to_numpy() >= thr).astype(int))
        p = predict_proba(c, va[MODEL_FEATURES])
        yb = (va["target"].to_numpy() >= thr).astype(int)
        metrics[f"auc_{name}"] = round(float(roc_auc_score(yb, p)), 3)
        metrics[f"brier_{name}"] = round(float(brier_score_loss(yb, p)), 3)

    # Risk-band distribution sanity check on the full data.
    p_late_all = predict_proba(clf_late, X)
    p_sev_all = predict_proba(clf_severe, X)
    bands = np.where(p_sev_all >= RISK_PROB_THRESHOLDS["high_p_severe"], "High",
                     np.where(p_late_all >= RISK_PROB_THRESHOLDS["moderate_p_late"],
                              "Moderate", "Low"))
    dist = pd.Series(bands).value_counts(normalize=True).round(3).to_dict()

    joblib.dump(
        {"features": MODEL_FEATURES, "clf_late": clf_late, "clf_severe": clf_severe,
         "quantiles": quantiles, "quantile_levels": QUANTILES,
         "late_min": LATE_MIN, "severe_min": SEVERE_MIN,
         "risk_prob_thresholds": RISK_PROB_THRESHOLDS, "metrics": metrics,
         "neutral_weather": neutral, "weather_features": WEATHER_FEATURES,
         "weather_airports": covered},
        "models/app_booking_model.joblib",
    )

    # Weather scenarios for the what-if ladder (precomputed; app needs no network).
    wc.build_scenario_table(wx).to_csv("app/reference/weather_scenarios.csv", index=False)

    # Reference tables for the app.
    route_freq.rename_axis("route").reset_index(name="route_freq").to_csv(
        "app/reference/route_freq.csv", index=False)
    cp_freq.rename_axis("country_pair").reset_index(name="country_pair_freq").to_csv(
        "app/reference/country_pair_freq.csv", index=False)

    passenger = train[train["DEPSTN"] != train["ARRSTN"]]
    sched = (
        passenger.groupby(["FLTID", "DEPSTN", "ARRSTN"])
        .agg(typical_hour=("dep_hour", lambda s: int(s.mode().iloc[0])),
             n_flights=("dep_hour", "size"))
        .reset_index()
        .sort_values("n_flights", ascending=False)
    )
    sched.to_csv("app/reference/flight_schedule.csv", index=False)

    # Route -> empirical count (kept for the "how much history" context note in the app).
    stats = (
        train.groupby("route")["target"]
        .agg(p25=lambda s: s.quantile(0.25), p50="median",
             p75=lambda s: s.quantile(0.75), n="size")
        .round(1).reset_index()
    )
    stats.to_csv("app/reference/route_delay_stats.csv", index=False)

    print("Built app artifacts:")
    print(f"  risk classifiers (hold-out): late  AUC {metrics['auc_late']} Brier {metrics['brier_late']}"
          f" | severe AUC {metrics['auc_severe']} Brier {metrics['brier_severe']}")
    print(f"  quantile regressors: p50/p75/p90 | features {len(MODEL_FEATURES)} "
          f"(+{len(WEATHER_FEATURES)} weather)")
    print(f"  risk-band mix: {dist}")
    print(f"  routes: {len(route_freq)} | flights(FLTID rows): {len(sched)} | "
          f"weather: {len(covered)} airports, {cov_rows.mean()*100:.0f}% real")


if __name__ == "__main__":
    main()
