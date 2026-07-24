"""Fetch historical ERA5 weather for departure airports and cache it to CSV.

Why this lives in a script (not the notebook): it calls a live API. Keeping it
here means the notebook loads a static cache and stays reproducible offline --
the same decision recorded for the Phase 6.2 weather feasibility check.

Source: Open-Meteo ERA5 reanalysis archive (free, no key). One request per
airport for its full date span, fetched in the airport's OWN local timezone so
the hourly timestamps line up directly with the local scheduled-departure hour
(STD) in the flight data. We then keep only the (airport, date, hour) combos
that actually occur as a departure, so the cache is a few MB, not millions of
idle hourly rows.

Usage:
    python scripts/fetch_weather.py            # all departure airports
    python scripts/fetch_weather.py --limit 3  # smoke-test the busiest 3
Output: data/weather_hourly_Alex.csv  (columns: airport,date,hour,<weather vars>)
Resumable: re-running skips airports already present in the output file.
"""
from __future__ import annotations

import argparse
import json
import time
import urllib.error
import urllib.parse
import urllib.request
from pathlib import Path

import airportsdata
import pandas as pd

ARCHIVE_URL = "https://archive-api.open-meteo.com/v1/archive"
HOURLY_VARS = [
    "temperature_2m",
    "precipitation",
    "snowfall",
    "wind_speed_10m",
    "wind_gusts_10m",
    "cloud_cover",
]
OUT_PATH = Path("data/weather_hourly_Alex.csv")

# Same override the notebook uses: SXF is gone from airportsdata (see ISSUES.md).
AIRPORT_OVERRIDES = {
    "SXF": {"lat": 52.3667, "lon": 13.5033, "tz": "Europe/Berlin"},
}


def load_needed_keys() -> pd.DataFrame:
    """Every (airport, date, hour) that occurs as a scheduled departure."""
    frames = []
    for name in ("train", "test"):
        df = pd.read_csv(f"data/{name}.csv", usecols=["DEPSTN", "STD"])
        std = pd.to_datetime(df["STD"], errors="coerce")  # STD is colon-separated
        frames.append(
            pd.DataFrame(
                {
                    "airport": df["DEPSTN"],
                    "date": std.dt.date.astype("string"),
                    "hour": std.dt.hour.astype("Int64"),
                }
            )
        )
    keys = pd.concat(frames, ignore_index=True).dropna().drop_duplicates()
    return keys


def airport_meta() -> dict:
    airports = airportsdata.load("IATA")

    def lookup(code):
        rec = airports.get(code) or AIRPORT_OVERRIDES.get(code)
        if not rec:
            return None
        return {"lat": rec["lat"], "lon": rec["lon"], "tz": rec.get("tz") or "UTC"}

    return lookup


def fetch_airport(lat, lon, tz, start, end, retries=4):
    params = {
        "latitude": lat,
        "longitude": lon,
        "start_date": start,
        "end_date": end,
        "hourly": ",".join(HOURLY_VARS),
        "timezone": tz,
    }
    url = ARCHIVE_URL + "?" + urllib.parse.urlencode(params)
    for attempt in range(retries):
        try:
            with urllib.request.urlopen(url, timeout=60) as r:
                data = json.load(r)
            h = data["hourly"]
            out = pd.DataFrame(h)
            t = pd.to_datetime(out["time"])
            out["date"] = t.dt.date.astype("string")
            out["hour"] = t.dt.hour
            return out.drop(columns=["time"])
        except urllib.error.HTTPError as e:
            # 429 = rate limit: the archive API weights big requests heavily, so
            # a burst trips the per-minute budget. Wait it out, then retry.
            wait = 60 if e.code == 429 else 2 ** attempt
            print(f"    retry {attempt + 1}/{retries} HTTP {e.code} (wait {wait}s)")
            time.sleep(wait)
        except Exception as e:  # noqa: BLE001 -- retry any transient failure
            wait = 2 ** attempt
            print(f"    retry {attempt + 1}/{retries} {type(e).__name__} (wait {wait}s)")
            time.sleep(wait)
    raise RuntimeError(f"failed after {retries} attempts: {lat},{lon}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--limit", type=int, default=None, help="only the N busiest airports")
    args = ap.parse_args()

    keys = load_needed_keys()
    lookup = airport_meta()

    # Busiest first so a --limit smoke test covers the most flights.
    order = keys["airport"].value_counts().index.tolist()
    if args.limit:
        order = order[: args.limit]

    done = set()
    if OUT_PATH.exists():
        done = set(pd.read_csv(OUT_PATH, usecols=["airport"])["airport"].unique())
        print(f"Resuming: {len(done)} airports already cached in {OUT_PATH}")

    header_written = OUT_PATH.exists()
    skipped = []
    for i, code in enumerate(order, 1):
        if code in done:
            continue
        meta = lookup(code)
        if not meta:
            skipped.append(code)
            continue
        sub = keys[keys["airport"] == code]
        start, end = sub["date"].min(), sub["date"].max()
        print(f"[{i}/{len(order)}] {code}: {len(sub):,} flight-hours, {start}..{end}")
        wx = fetch_airport(meta["lat"], meta["lon"], meta["tz"], start, end)
        # Keep only the (date, hour) combos this airport actually needs.
        merged = sub.merge(wx, on=["date", "hour"], how="left")
        merged.to_csv(OUT_PATH, mode="a", header=not header_written, index=False)
        header_written = True
        time.sleep(1.2)  # heavy 3-year requests are rate-limited; pace them out

    if skipped:
        print("No coordinates (skipped):", skipped)
    total = pd.read_csv(OUT_PATH)
    print(f"\nDone. {OUT_PATH}: {len(total):,} rows, {total['airport'].nunique()} airports.")


if __name__ == "__main__":
    main()
