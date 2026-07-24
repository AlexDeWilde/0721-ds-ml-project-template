"""Fetch & cache historical (ERA5) hourly departure weather for the busiest airports.

Uses the free Open-Meteo historical archive (no API key). One call per airport pulls
the full 2016-2018 hourly range in the airport's LOCAL timezone, so the returned
timestamps line up directly with each flight's scheduled departure time (STD, which is
local). Cached to data/weather_cache/<IATA>.parquet (gitignored) so downstream
scripts and the app-build step never re-hit the network.

Run:  .venv/bin/python scripts/fetch_weather.py
"""
import json
import time
import urllib.parse
import urllib.request
from pathlib import Path

import airportsdata
import pandas as pd

ARCHIVE_URL = "https://archive-api.open-meteo.com/v1/archive"
START, END = "2016-01-01", "2018-12-31"
# Aviation-relevant hourly variables ERA5 exposes. weather_code (WMO) lets us NAME the
# condition (fog / snow / thunderstorm / ...); the numeric vars drive the model features.
HOURLY_VARS = ["wind_gusts_10m", "wind_speed_10m", "precipitation", "snowfall",
               "temperature_2m", "weather_code"]
CACHE_DIR = Path("data/weather_cache")
AIRPORT_OVERRIDES = {"SXF": {"lat": 52.3667, "lon": 13.5033, "tz": "Europe/Berlin"}}

# Top-15 departure airports = 77.3% of training flights (see the experiment write-up).
TOP_AIRPORTS = ["TUN", "DJE", "ORY", "MIR", "MRS", "LYS", "NCE", "ALG",
                "MXP", "IST", "FRA", "BRU", "CMN", "FCO", "TOE"]


def _airport(code, airports):
    return airports.get(code) or AIRPORT_OVERRIDES.get(code)


def fetch_airport(code, airports):
    a = _airport(code, airports)
    if not a:
        raise ValueError(f"no lat/lon for {code}")
    q = urllib.parse.urlencode({
        "latitude": a["lat"], "longitude": a["lon"],
        "start_date": START, "end_date": END,
        "hourly": ",".join(HOURLY_VARS),
        "timezone": a["tz"],  # return timestamps in local time to match STD
    })
    url = f"{ARCHIVE_URL}?{q}"
    last_err = None
    for attempt in range(1, 5):  # retry transient network/SSL hiccups
        try:
            with urllib.request.urlopen(url, timeout=90) as r:
                payload = json.load(r)
            break
        except Exception as e:  # noqa: BLE001 - transient network errors vary
            last_err = e
            print(f"    {code}: attempt {attempt} failed ({type(e).__name__}), retrying...")
            time.sleep(3 * attempt)
    else:
        raise RuntimeError(f"{code}: giving up after retries: {last_err}")
    h = payload["hourly"]
    df = pd.DataFrame(h)
    df["time"] = pd.to_datetime(df["time"])
    df.insert(0, "airport", code)
    return df


def main():
    CACHE_DIR.mkdir(parents=True, exist_ok=True)
    airports = airportsdata.load("IATA")
    for i, code in enumerate(TOP_AIRPORTS, 1):
        out = CACHE_DIR / f"{code}.parquet"
        if out.exists():
            print(f"[{i:>2}/{len(TOP_AIRPORTS)}] {code}: cached, skipping")
            continue
        df = fetch_airport(code, airports)
        df.to_parquet(out, index=False)
        print(f"[{i:>2}/{len(TOP_AIRPORTS)}] {code}: {len(df):,} hourly rows "
              f"({df['time'].min().date()}..{df['time'].max().date()})")
        time.sleep(1)  # be polite to the free API
    print(f"\nDone. Cache: {CACHE_DIR}/  ({len(list(CACHE_DIR.glob('*.parquet')))} airports)")


if __name__ == "__main__":
    main()
