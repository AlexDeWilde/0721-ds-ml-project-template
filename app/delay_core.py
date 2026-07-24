"""Core logic for the Tunisair Delay-Alert app.

Turns booking-time inputs (flight number, departure/arrival airport, date) into a
delay-risk category, calibrated probabilities, an honest flight-specific delay range,
a weather-sensitivity ladder, and action suggestions. Kept UI-free so it can be
unit-tested and reused.

Since the roadmap-#5 reframe the model is not a single mean-minutes regressor:
  * two calibrated classifiers give P(delay >= 15 min) and P(delay >= 60 min);
  * three quantile regressors give the p50/p75/p90 delay range for THIS flight.
See RISK_CLASSIFIER_EXPERIMENT.md.
"""
from __future__ import annotations

import functools
import json
import sys
import urllib.parse
import urllib.request
from pathlib import Path

import airportsdata
import holidays
import joblib
import numpy as np
import pandas as pd
from hijridate import Gregorian

_REF = Path(__file__).parent / "reference"
_MODEL_PATH = Path(__file__).parent.parent / "models" / "app_booking_model.joblib"

# Optional: the shared weather helper (lets us show the ACTUAL recorded weather for a
# historical date). Absent in a bare deploy -> the app just omits the corner icon.
sys.path.insert(0, str(Path(__file__).parent.parent / "scripts"))
try:
    import weather_core as _wc
except Exception:  # noqa: BLE001
    _wc = None

AIRPORT_OVERRIDES = {"SXF": {"country": "DE", "lat": 52.3667, "lon": 13.5033, "tz": "Europe/Berlin"}}


def is_ramadan(ts):
    """True if the date falls in Ramadan (Hijri month 9). Works for any year via the
    Hijri calendar — the training pipeline and the app use this same function."""
    ts = pd.Timestamp(ts)
    return Gregorian(ts.year, ts.month, ts.day).to_hijri().month == 9


@functools.lru_cache(maxsize=1)
def _airports():
    return airportsdata.load("IATA")


@functools.lru_cache(maxsize=1)
def _tn_holidays():
    return holidays.Tunisia(years=range(2016, 2031))


@functools.lru_cache(maxsize=1)
def load_bundle():
    """Load models + reference tables once."""
    b = joblib.load(_MODEL_PATH)
    route_freq = pd.read_csv(_REF / "route_freq.csv").set_index("route")["route_freq"].to_dict()
    cp_freq = (pd.read_csv(_REF / "country_pair_freq.csv")
               .set_index("country_pair")["country_pair_freq"].to_dict())
    schedule = pd.read_csv(_REF / "flight_schedule.csv")
    route_stats = pd.read_csv(_REF / "route_delay_stats.csv").set_index("route")
    scen_path = _REF / "weather_scenarios.csv"
    weather_scenarios = pd.read_csv(scen_path) if scen_path.exists() else pd.DataFrame()
    return {
        "features": b["features"], "clf_late": b["clf_late"], "clf_severe": b["clf_severe"],
        "quantiles": b["quantiles"], "quantile_levels": b["quantile_levels"],
        "late_min": b["late_min"], "severe_min": b["severe_min"],
        "risk_prob_thresholds": b["risk_prob_thresholds"], "metrics": b["metrics"],
        "neutral_weather": b.get("neutral_weather", {}),
        "weather_features": b.get("weather_features", []),
        "weather_airports": set(b.get("weather_airports", [])),
        "weather_scenarios": weather_scenarios,
        "route_freq": route_freq, "cp_freq": cp_freq,
        "schedule": schedule, "route_stats": route_stats,
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


def make_features(dep, arr, date, hour, bundle, weather=None):
    """Build the model's booking-time feature row from typed inputs. `weather` overrides the
    neutral 'typical calm' weather defaults (used by the weather-sensitivity ladder)."""
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
    row.update(bundle.get("neutral_weather", {}))
    if weather:
        row.update(weather)
    return pd.DataFrame([row])[bundle["features"]]


def _proba(clf, x):
    """Apply a (base, isotonic) calibrated classifier to one feature row -> probability."""
    base, iso = clf
    return float(iso.transform(base.predict_proba(x)[:, 1])[0])


def _quantiles(bundle, x):
    """Flight-specific p50/p75/p90 in minutes, clamped monotonic non-decreasing & >= 0."""
    out, prev = [], 0.0
    for q in bundle["quantile_levels"]:
        v = max(float(bundle["quantiles"][q].predict(x)[0]), prev, 0.0)
        out.append(v)
        prev = v
    return out  # [p50, p75, p90]


def risk_from_probs(p_late, p_severe, bundle):
    """Map calibrated probabilities to a (label, emoji) risk band."""
    t = bundle["risk_prob_thresholds"]
    if p_severe >= t["high_p_severe"]:
        return "High", "🔴"
    if p_late >= t["moderate_p_late"]:
        return "Moderate", "🟡"
    return "Low", "🟢"


def predict(dep, arr, date, hour, bundle, weather=None):
    """Full prediction for one flight (optionally under a specific weather scenario)."""
    x = make_features(dep, arr, date, hour, bundle, weather)
    p_late = _proba(bundle["clf_late"], x)
    p_severe = _proba(bundle["clf_severe"], x)
    risk, emoji = risk_from_probs(p_late, p_severe, bundle)
    q50, q75, q90 = _quantiles(bundle, x)
    route = f"{dep}->{arr}"
    rs = bundle["route_stats"]
    route_n = int(rs.loc[route, "n"]) if route in rs.index else 0
    return {"p_late": p_late, "p_severe": p_severe, "risk": risk, "emoji": emoji,
            "q50": q50, "q75": q75, "q90": q90, "route_n": route_n}


def weather_ladder(dep, arr, date, hour, bundle):
    """Per-flight 'what if the weather is …' ladder for the departure airport.

    Runs the SAME models under each plausible, named weather scenario for this
    airport+month (calm/rough/severe) with how often each occurs, and returns a
    probability-weighted headline. None when the airport has no weather coverage."""
    if dep not in bundle.get("weather_airports", set()):
        return None
    scen = bundle["weather_scenarios"]
    if scen.empty:
        return None
    month = pd.Timestamp(date).month
    bands = scen[(scen["airport"] == dep) & (scen["month"] == month)]
    if bands.empty:
        return None

    wf = bundle["weather_features"]
    order = {"calm": 0, "rough": 1, "severe": 2}
    rungs = []
    for _, b in bands.sort_values("band", key=lambda s: s.map(order)).iterrows():
        r = predict(dep, arr, date, hour, bundle, weather={f: b[f] for f in wf})
        rungs.append({"band": b["band"], "label": b["label"], "prob": float(b["prob"]), **r})

    # Probability-weighted headline (the ladder decomposes this).
    def wavg(key):
        return sum(x["prob"] * x[key] for x in rungs)
    wp_late, wp_severe = wavg("p_late"), wavg("p_severe")
    wrisk, wemoji = risk_from_probs(wp_late, wp_severe, bundle)
    weighted = {"p_late": wp_late, "p_severe": wp_severe, "q50": wavg("q50"),
                "q75": wavg("q75"), "q90": wavg("q90"), "risk": wrisk, "emoji": wemoji}

    # Monotonic display of the per-rung median (RF/GBM can wobble on a weak signal).
    running = 0.0
    for r in rungs:
        running = max(running, r["q50"])
        r["disp_q50"] = running
    spread = rungs[-1]["disp_q50"] - rungs[0]["disp_q50"]
    return {"rungs": rungs, "weighted": weighted, "spread": spread, "sensitive": spread >= 5.0}


def actual_condition(dep, date, hour, bundle):
    """The actual recorded departure weather for the chosen date+hour (historical data),
    as {gust, precip, snow, code, label}. None for uncovered airports or non-historical
    dates — the app then shows no weather icon."""
    if _wc is None or dep not in bundle.get("weather_airports", set()):
        return None
    ts = pd.Timestamp(date) + pd.Timedelta(hours=int(hour))
    return _wc.actual_weather(dep, ts)


_FORECAST_URL = "https://api.open-meteo.com/v1/forecast"
_forecast_cache: dict = {}  # (airport, fetch_date) -> hourly dataframe (or None)


def _forecast_hourly(dep):
    """Fetch (and day-cache) Open-Meteo's ~16-day hourly forecast for a departure airport,
    in local time. Returns a dataframe indexed by hour, or None on any failure/offline."""
    a = _airport(dep)
    if a is None:
        return None
    key = (dep, pd.Timestamp.now().normalize())
    if key in _forecast_cache:
        return _forecast_cache[key]
    q = urllib.parse.urlencode({
        "latitude": a["lat"], "longitude": a["lon"],
        "hourly": "wind_gusts_10m,precipitation,snowfall,weather_code,temperature_2m",
        "forecast_days": 16, "timezone": a.get("tz", "UTC"),
    })
    try:
        with urllib.request.urlopen(f"{_FORECAST_URL}?{q}", timeout=8) as r:
            h = json.load(r)["hourly"]
        df = pd.DataFrame(h)
        df["time"] = pd.to_datetime(df["time"])
        df = df.set_index("time")
    except Exception:  # noqa: BLE001 - offline / API error -> no forecast badge
        df = None
    _forecast_cache[key] = df
    return df


def forecast_condition(dep, date, hour):
    """The live forecast weather for the chosen date+hour, if it falls within Open-Meteo's
    ~16-day horizon. Returns {gust, precip, snow, code, label} or None (out of horizon /
    offline / unknown airport). Works for any airport with coordinates."""
    if _wc is None:
        return None
    df = _forecast_hourly(dep)
    if df is None:
        return None
    hour_ts = (pd.Timestamp(date) + pd.Timedelta(hours=int(hour))).floor("h")
    if hour_ts not in df.index:
        return None
    r = df.loc[hour_ts]
    gust, precip = float(r["wind_gusts_10m"]), float(r["precipitation"])
    snow, code = float(r["snowfall"]), int(r["weather_code"] or 0)
    return {"gust": gust, "precip": precip, "snow": snow, "code": code,
            "label": _wc.condition_name(gust, precip, snow, code)}


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
    """Calmer departure *times* on the same route, ranked by lowest chance of a delay.

    For a fixed route and date only the departure hour changes the prediction, so we
    rank the route's distinct scheduled departure times and surface the calmest ones."""
    route_flights = bundle["schedule"].query("DEPSTN == @dep and ARRSTN == @arr")
    if route_flights.empty:
        return pd.DataFrame(columns=["hour", "FLTID", "p_late", "q50", "risk", "emoji"])
    by_hour = (route_flights.sort_values("n_flights", ascending=False)
               .drop_duplicates("typical_hour"))
    rows = []
    for _, f in by_hour.iterrows():
        p = predict(dep, arr, date, int(f["typical_hour"]), bundle)
        rows.append({"hour": int(f["typical_hour"]), "FLTID": f["FLTID"],
                     "p_late": p["p_late"], "q50": round(p["q50"], 1),
                     "risk": p["risk"], "emoji": p["emoji"]})
    out = pd.DataFrame(rows).sort_values(["p_late", "hour"])
    return out.head(top).reset_index(drop=True)
