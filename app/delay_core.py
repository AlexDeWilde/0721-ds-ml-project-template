"""Core logic for the Tunisair Delay-Alert app.

Turns booking-time inputs (flight number, departure/arrival airport, date) into a
delay-risk category, an honest expected-delay range, and action suggestions.
Kept UI-free so it can be unit-tested and reused.
"""
from __future__ import annotations

import functools
from pathlib import Path

import airportsdata
import holidays
import joblib
import numpy as np
import pandas as pd
from hijridate import Gregorian

_REF = Path(__file__).parent / "reference"
_MODEL_PATH = Path(__file__).parent.parent / "models" / "app_booking_model.joblib"

AIRPORT_OVERRIDES = {"SXF": {"country": "DE", "lat": 52.3667, "lon": 13.5033, "tz": "Europe/Berlin"}}


def is_ramadan(ts):
    """True if the date falls in Ramadan (Hijri month 9). Works for any year via the
    Hijri calendar — the training pipeline and the app use this same function."""
    ts = pd.Timestamp(ts)
    return Gregorian(ts.year, ts.month, ts.day).to_hijri().month == 9

# Risk bands (minutes of predicted delay). Global delay quantiles are the fallback
# range when a route has too little history.
GLOBAL_RANGE = (0.0, 14.0, 43.0)  # p25, p50, p75 of the whole training target


@functools.lru_cache(maxsize=1)
def _airports():
    return airportsdata.load("IATA")


@functools.lru_cache(maxsize=1)
def _tn_holidays():
    return holidays.Tunisia(years=range(2016, 2031))


@functools.lru_cache(maxsize=1)
def load_bundle():
    """Load model + reference tables once."""
    bundle = joblib.load(_MODEL_PATH)
    route_freq = pd.read_csv(_REF / "route_freq.csv").set_index("route")["route_freq"].to_dict()
    cp_freq = (pd.read_csv(_REF / "country_pair_freq.csv")
               .set_index("country_pair")["country_pair_freq"].to_dict())
    schedule = pd.read_csv(_REF / "flight_schedule.csv")
    delay_stats = pd.read_csv(_REF / "route_delay_stats.csv").set_index("route")
    # Weather scenarios for the what-if ladder (roadmap #2); optional so the app still
    # runs if the file is absent (e.g. an older build).
    scen_path = _REF / "weather_scenarios.csv"
    weather_scenarios = pd.read_csv(scen_path) if scen_path.exists() else pd.DataFrame()
    return {
        "model": bundle["model"], "features": bundle["features"],
        "risk_bands": bundle["risk_bands"], "holdout_rmse": bundle["holdout_rmse"],
        "baseline_rmse": bundle["baseline_rmse"], "route_freq": route_freq,
        "cp_freq": cp_freq, "schedule": schedule, "delay_stats": delay_stats,
        "neutral_weather": bundle.get("neutral_weather", {}),
        "weather_features": bundle.get("weather_features", []),
        "weather_airports": set(bundle.get("weather_airports", [])),
        "weather_scenarios": weather_scenarios,
    }


def _airport(code):
    a = _airports().get(code) or AIRPORT_OVERRIDES.get(code)
    return a if isinstance(a, dict) else None


def _haversine_km(lat1, lon1, lat2, lon2):
    r = 6371.0
    p1, p2 = np.radians(lat1), np.radians(lat2)
    dp, dl = np.radians(lat2 - lat1), np.radians(lon2 - lon1)
    a = np.sin(dp / 2) ** 2 + np.cos(p1) * np.cos(p2) * np.sin(dl / 2) ** 2
    return float(2 * r * np.arcsin(np.sqrt(a)))


def make_features(dep, arr, date, hour, bundle):
    """Build the model's booking-time feature row from typed inputs."""
    ts = pd.Timestamp(date) + pd.Timedelta(hours=int(hour))
    d, a = _airport(dep), _airport(arr)
    if d and a:
        dist = _haversine_km(d["lat"], d["lon"], a["lat"], a["lon"])
        domestic = int(d["country"] == a["country"])
        country_pair = f"{d['country']}->{a['country']}"
    else:
        dist, domestic, country_pair = 0.0, 0, "??->??"
    route = f"{dep}->{arr}"
    row = {
        "dep_hour": int(hour),
        "dep_dow": ts.dayofweek,
        "dep_month": ts.month,
        "dep_is_holiday": int(ts.date() in _tn_holidays()),
        "dep_is_ramadan": int(is_ramadan(ts)),
        "gc_distance_km": round(dist, 1),
        "is_domestic": domestic,
        "route_freq": bundle["route_freq"].get(route, 0),
        "country_pair_freq": bundle["cp_freq"].get(country_pair, 0),
    }
    # Weather features default to NEUTRAL "typical calm" — the headline prediction assumes
    # an ordinary day. The weather ladder overrides these to ask the what-if questions.
    row.update(bundle.get("neutral_weather", {}))
    return pd.DataFrame([row])[bundle["features"]]


def risk_category(pred_minutes, bundle):
    """Map a predicted delay to a (label, emoji) risk band."""
    low, moderate = bundle["risk_bands"]["low"], bundle["risk_bands"]["moderate"]
    if pred_minutes < low:
        return "Low", "🟢"
    if pred_minutes < moderate:
        return "Moderate", "🟡"
    return "High", "🔴"


def expected_range(route, bundle):
    """Honest 'typical delay' range = empirical p25/p50/p75 of similar historical flights."""
    stats = bundle["delay_stats"]
    if route in stats.index and stats.loc[route, "n"] >= 20:
        r = stats.loc[route]
        return float(r["p25"]), float(r["p50"]), float(r["p75"]), int(r["n"])
    return (*GLOBAL_RANGE, 0)


def predict(dep, arr, date, hour, bundle):
    """Full prediction for one flight: minutes, risk band, and the typical range."""
    x = make_features(dep, arr, date, hour, bundle)
    pred = float(bundle["model"].predict(x)[0])
    label, emoji = risk_category(pred, bundle)
    p25, p50, p75, n = expected_range(f"{dep}->{arr}", bundle)
    return {"pred_minutes": pred, "risk": label, "emoji": emoji,
            "p25": p25, "p50": p50, "p75": p75, "route_n": n}


def weather_ladder(dep, arr, date, hour, bundle):
    """A per-flight 'what if the weather is …' ladder for the departure airport.

    At booking we can't know the weather, so instead of guessing we show how the SAME
    flight is predicted to behave under each plausible, named weather scenario for this
    airport+month (calm / rough / severe), with how often each occurs. Returns None when
    the departure airport has no weather coverage (then the app simply omits the ladder).
    """
    if dep not in bundle.get("weather_airports", set()):
        return None
    scen = bundle["weather_scenarios"]
    if scen.empty:
        return None
    month = pd.Timestamp(date).month
    bands = scen[(scen["airport"] == dep) & (scen["month"] == month)]
    if bands.empty:
        return None

    base_row = make_features(dep, arr, date, hour, bundle).iloc[0].to_dict()
    wx_feats = bundle["weather_features"]
    order = {"calm": 0, "rough": 1, "severe": 2}
    rungs = []
    for _, b in bands.sort_values("band", key=lambda s: s.map(order)).iterrows():
        row = dict(base_row)
        for f in wx_feats:
            row[f] = b[f]
        pred = float(bundle["model"].predict(pd.DataFrame([row])[bundle["features"]])[0])
        rungs.append({"band": b["band"], "label": b["label"], "prob": float(b["prob"]),
                      "raw_pred": pred})

    # Honest weighted expectation uses the raw predictions...
    weighted = sum(r["prob"] * r["raw_pred"] for r in rungs)
    # ...but for DISPLAY, clamp so a more-severe rung never shows fewer minutes than a
    # calmer one (random forests can wobble on a weak signal).
    running = float("-inf")
    for r in rungs:
        running = max(running, r["raw_pred"])
        r["pred_minutes"] = running
        label, emoji = risk_category(running, bundle)
        r["risk"], r["emoji"] = label, emoji
    spread = rungs[-1]["pred_minutes"] - rungs[0]["pred_minutes"]
    return {"rungs": rungs, "weighted": weighted, "spread": spread,
            "sensitive": spread >= 5.0}


def action_suggestions(risk):
    """Plain-language, traveller-friendly advice for each risk level."""
    if risk == "Low":
        return [
            "Good news — flights like this usually leave close to on time.",
            "You can book with confidence and plan your day as normal.",
        ]
    if risk == "Moderate":
        return [
            "There's a fair chance this flight runs a little late.",
            "If you have a connection or something important afterwards, give yourself some extra time.",
            "It's worth checking your flight status again the day before you travel.",
        ]
    return [
        "Flights like this are often delayed by quite a while.",
        "If your plans are flexible, one of the calmer options below could save you a long wait.",
        "Try not to book tight connections around this flight.",
        "Leave extra time, and check for updates before you head to the airport.",
    ]


def alternatives(dep, arr, date, bundle, top=4):
    """Calmer departure *times* on the same route, scored and sorted lowest-risk first.

    For a fixed route and date only the departure hour changes the prediction, so we
    rank the route's distinct scheduled departure times and surface the calmest ones.
    """
    route_flights = bundle["schedule"].query("DEPSTN == @dep and ARRSTN == @arr")
    if route_flights.empty:
        return pd.DataFrame(columns=["hour", "FLTID", "pred_minutes", "risk", "emoji"])
    # One representative flight per distinct departure hour (busiest FLTID for that hour).
    by_hour = (route_flights.sort_values("n_flights", ascending=False)
               .drop_duplicates("typical_hour"))
    rows = []
    for _, f in by_hour.iterrows():
        p = predict(dep, arr, date, int(f["typical_hour"]), bundle)
        rows.append({"hour": int(f["typical_hour"]), "FLTID": f["FLTID"],
                     "pred_minutes": round(p["pred_minutes"], 1),
                     "risk": p["risk"], "emoji": p["emoji"]})
    out = pd.DataFrame(rows).sort_values(["pred_minutes", "hour"])
    return out.head(top).reset_index(drop=True)
