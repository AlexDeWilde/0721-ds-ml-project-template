"""Shared weather helpers: load the ERA5 cache, name conditions, and join to flights.

Kept separate so the experiment script, a future app-build step, and the app itself
all name weather the same way.
"""
from pathlib import Path

import pandas as pd

CACHE_DIR = Path(__file__).parent.parent / "data" / "weather_cache"

# Model-facing weather features (units: gust km/h, precip mm, snow cm).
# NOTE: temperature is deliberately excluded — it barely moves RMSE and mostly acts as
# a season proxy for dep_month, which would inflate the apparent "weather" effect and
# isn't a scenario axis travellers care about. wx_temp is still computed for reference.
WEATHER_FEATURES = ["wx_wind_gust", "wx_precip", "wx_snow", "wx_adverse"]

# Thresholds for the "adverse departure" flag and for naming a windy scenario.
GUST_STRONG = 40.0   # km/h — breezy/strong enough to start affecting ops
GUST_GALE = 62.0     # km/h — gale
PRECIP_WET = 0.5     # mm in the hour
SNOW_ANY = 0.1       # cm in the hour

# WMO weather codes that are inherently disruptive regardless of wind.
_ADVERSE_CODES = {45, 48, 56, 57, 65, 66, 67, 75, 77, 82, 85, 86, 95, 96, 99}


def wmo_label(code):
    """Human name for a WMO weather code (Open-Meteo `weather_code`)."""
    c = int(code)
    if c == 0:
        return "Clear"
    if c in (1, 2):
        return "Partly cloudy"
    if c == 3:
        return "Overcast"
    if c in (45, 48):
        return "Fog"
    if c in (51, 53, 55, 56, 57):
        return "Drizzle"
    if c in (61, 80):
        return "Light rain"
    if c in (63, 81):
        return "Rain"
    if c in (65, 82):
        return "Heavy rain"
    if c in (66, 67):
        return "Freezing rain"
    if c in (71, 85):
        return "Light snow"
    if c == 73:
        return "Snow"
    if c in (75, 77, 86):
        return "Heavy snow"
    if c == 95:
        return "Thunderstorm"
    if c in (96, 99):
        return "Thunderstorm w/ hail"
    return "Unsettled"


def is_adverse(gust, precip, snow, code):
    """Whether these conditions count as a disruptive ('adverse') departure hour."""
    return int(
        gust >= GUST_STRONG or precip >= PRECIP_WET or snow >= SNOW_ANY
        or int(code) in _ADVERSE_CODES
    )


def condition_name(gust, precip, snow, code):
    """A single human label for the dominant weather, blending sky condition + wind.

    Precipitation/thunder/snow/fog take priority (they are what `weather_code` captures);
    strong wind is surfaced when it is the notable feature, or appended to a wet label.
    """
    sky = wmo_label(code)
    if gust >= GUST_GALE:
        windy = "Gale-force wind"
    elif gust >= GUST_STRONG:
        windy = "Strong wind"
    else:
        windy = None
    precipitating = sky not in ("Clear", "Partly cloudy", "Overcast", "Fog")
    if precipitating:
        return f"{sky} + strong wind" if gust >= GUST_STRONG else sky
    if sky == "Fog":
        return "Fog"
    # Dry sky: wind is the story if it's up, otherwise just the sky state.
    if windy:
        return windy
    return sky


def adversity_index(gust, precip, snow, code):
    """Transparent 0-ish..high score to RANK hours from calm to severe (for scenario anchors).

    Not fed to the model — only used to pick representative calm / rough / severe days.
    """
    score = gust / 10.0 + precip * 4.0 + snow * 8.0
    if int(code) in (95, 96, 99):
        score += 6.0   # thunderstorms are disruptive out of proportion to rain amount
    if int(code) in (45, 48):
        score += 2.0   # fog
    return score


def load_weather_cache():
    """Concatenate all cached airport parquet files into one long dataframe."""
    files = sorted(CACHE_DIR.glob("*.parquet"))
    if not files:
        raise FileNotFoundError(f"No weather cache in {CACHE_DIR}. Run scripts/fetch_weather.py first.")
    wx = pd.concat((pd.read_parquet(f) for f in files), ignore_index=True)
    wx["time"] = pd.to_datetime(wx["time"])
    return wx


def covered_airports():
    return {f.stem for f in CACHE_DIR.glob("*.parquet")}


def band_of(gust, precip, snow, code):
    """Three severity bands for the scenario ladder, by how disruptive the hour is."""
    if int(code) in (95, 96, 99) or gust >= GUST_GALE or precip >= 5.0 or snow >= 1.0:
        return "severe"
    if is_adverse(gust, precip, snow, code):
        return "rough"
    return "calm"


def build_scenario_table(wx=None):
    """Per (airport, month) weather scenarios for the app's what-if ladder.

    For each airport+month, split its historical hours into calm/rough/severe bands,
    and for each band emit its empirical frequency, a representative (median-adversity)
    set of weather values, and a human label. Returned tidy so it can be written to a
    small CSV the app reads offline — the app never needs the raw weather again.
    """
    if wx is None:
        wx = load_weather_cache()
    w = wx.copy()
    w["gust"] = w["wind_gusts_10m"].astype(float)
    w["precip"] = w["precipitation"].astype(float)
    w["snow"] = w["snowfall"].astype(float)
    w["code"] = w["weather_code"].fillna(0).astype(int)
    w["month"] = w["time"].dt.month
    w["band"] = [band_of(g, p, s, c) for g, p, s, c in zip(w.gust, w.precip, w.snow, w.code)]

    rows = []
    for (airport, month), grp in w.groupby(["airport", "month"]):
        for band in ("calm", "rough", "severe"):
            b = grp[grp["band"] == band]
            if b.empty:
                continue
            adv = [adversity_index(g, p, s, c) for g, p, s, c in zip(b.gust, b.precip, b.snow, b.code)]
            adv = pd.Series(adv, index=b.index)
            rep = b.loc[(adv - adv.median()).abs().idxmin()]
            rows.append({
                "airport": airport, "month": int(month), "band": band,
                "prob": round(len(b) / len(grp), 4),
                "label": condition_name(rep.gust, rep.precip, rep.snow, rep.code),
                "wx_wind_gust": round(float(rep.gust), 1),
                "wx_precip": round(float(rep.precip), 2),
                "wx_snow": round(float(rep.snow), 2),
                "wx_adverse": is_adverse(rep.gust, rep.precip, rep.snow, rep.code),
            })
    out = pd.DataFrame(rows)
    out["_ord"] = out["band"].map({"calm": 0, "rough": 1, "severe": 2})
    out = out.sort_values(["airport", "month", "_ord"]).drop(columns="_ord")
    return out.reset_index(drop=True)


def attach_weather(flights, wx=None):
    """Add wx_* feature columns to flights (needs DEPSTN + STD). Rows whose departure
    airport is not in the cache get NaN weather (caller decides how to handle)."""
    if wx is None:
        wx = load_weather_cache()
    w = wx.copy()
    w["gust"] = w["wind_gusts_10m"].astype(float)
    w["precip"] = w["precipitation"].astype(float)
    w["snow"] = w["snowfall"].astype(float)
    w["temp"] = w["temperature_2m"].astype(float)
    w["code"] = w["weather_code"].fillna(0).astype(int)
    w["wx_adverse"] = [is_adverse(g, p, s, c) for g, p, s, c in
                       zip(w["gust"], w["precip"], w["snow"], w["code"])]
    w = w.rename(columns={"gust": "wx_wind_gust", "precip": "wx_precip",
                          "snow": "wx_snow", "temp": "wx_temp"})
    w = w[["airport", "time", *WEATHER_FEATURES, "wx_temp", "code"]]

    df = flights.copy()
    df["_hour"] = df["STD"].dt.floor("h")
    merged = df.merge(w, left_on=["DEPSTN", "_hour"], right_on=["airport", "time"], how="left")
    return merged.drop(columns=["_hour", "airport", "time"])
