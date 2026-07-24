"""Experiment: does departure-airport weather improve the booking-time model, and
is the model's weather response strong/monotonic enough to drive a per-flight
"what-if" scenario ladder (calm / rough / severe, with named conditions)?

Roadmap item #2. Measurement only — does NOT touch the app.

Design (see FLTID_FEATURE_EXPERIMENT.md and the weather write-up):
  * Train on ACTUAL historical weather (ERA5, cached by scripts/fetch_weather.py).
  * The app can't know the weather months out, so instead of guessing it we expose
    the flight's SENSITIVITY: run the trained model under a few plausible, named
    weather scenarios anchored to that airport+month's own history.
  * Restrict the whole comparison to the 15 cached airports (77.3% of flights) so
    base vs base+weather is apples-to-apples.

Leakage-safe: departure weather (forecast or historical) is known before departure.

Run:  .venv/bin/python scripts/experiment_weather_features.py
"""
import os
import sys

import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import root_mean_squared_error

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import weather_core as W  # noqa: E402
from experiment_fltid_features import BASE_FEATURES, build_base_features  # noqa: E402

WEATHER_FEATURES = W.WEATHER_FEATURES


def rmse_for(features, tr, va):
    model = RandomForestRegressor(
        n_estimators=120, min_samples_leaf=50, max_features="sqrt",
        n_jobs=-1, random_state=42,
    ).fit(tr[features], tr["target"])
    return root_mean_squared_error(va["target"], model.predict(va[features])), model


def band_of(gust, precip, snow, code):
    """Three severity bands for the scenario ladder, by how disruptive the hour is."""
    if int(code) in (95, 96, 99) or gust >= W.GUST_GALE or precip >= 5.0 or snow >= 1.0:
        return "severe"
    if W.is_adverse(gust, precip, snow, code):
        return "rough"
    return "calm"


def scenario_table(wx, airport, month):
    """For one airport+month, the plausible weather bands with their empirical
    frequency, a representative (median) condition per band, and a human label."""
    h = wx[(wx["airport"] == airport) & (wx["time"].dt.month == month)].copy()
    if h.empty:
        return []
    h["gust"] = h["wind_gusts_10m"].astype(float)
    h["precip"] = h["precipitation"].astype(float)
    h["snow"] = h["snowfall"].astype(float)
    h["temp"] = h["temperature_2m"].astype(float)
    h["code"] = h["weather_code"].fillna(0).astype(int)
    h["band"] = [band_of(g, p, s, c) for g, p, s, c in zip(h.gust, h.precip, h.snow, h.code)]
    out = []
    for band in ("calm", "rough", "severe"):
        b = h[h["band"] == band]
        if b.empty:
            continue
        # Representative = the hour closest to this band's median adversity (a real,
        # in-distribution combo we can name honestly).
        adv = np.array([W.adversity_index(g, p, s, c)
                        for g, p, s, c in zip(b.gust, b.precip, b.snow, b.code)])
        rep = b.iloc[int(np.argmin(np.abs(adv - np.median(adv))))]
        feats = {"wx_wind_gust": rep.gust, "wx_precip": rep.precip, "wx_snow": rep.snow,
                 "wx_temp": rep.temp, "wx_adverse": W.is_adverse(rep.gust, rep.precip, rep.snow, rep.code)}
        out.append({
            "band": band, "prob": len(b) / len(h),
            "label": W.condition_name(rep.gust, rep.precip, rep.snow, rep.code),
            "gust": rep.gust, "precip": rep.precip, "feats": feats,
        })
    return out


def predict_with_weather(model, base_row, feats):
    row = {**base_row, **feats}
    x = pd.DataFrame([row])[BASE_FEATURES + WEATHER_FEATURES]
    return float(model.predict(x)[0])


def main():
    train = pd.read_csv("data/train.csv")
    train["STD"] = pd.to_datetime(train["STD"])
    train["FLTID"] = train["FLTID"].str.strip()

    cov = W.covered_airports()
    train = train[train["DEPSTN"].isin(cov)].reset_index(drop=True)
    train = build_base_features(train)
    train["route_freq"] = train["route"].map(train["route"].value_counts()).astype(int)
    train["country_pair_freq"] = train["country_pair"].map(train["country_pair"].value_counts()).astype(int)

    wx = W.load_weather_cache()
    train = W.attach_weather(train, wx)
    train = train.dropna(subset=WEATHER_FEATURES).reset_index(drop=True)
    print(f"flights (covered airports, weather joined): {len(train):,}")

    ts = train.sort_values("STD").reset_index(drop=True)
    cut = int(len(ts) * 0.8)
    tr, va = ts.iloc[:cut].copy(), ts.iloc[cut:].copy()

    naive = root_mean_squared_error(va["target"], [tr["target"].mean()] * len(va))
    base_rmse, _ = rmse_for(BASE_FEATURES, tr, va)
    wx_rmse, wx_model = rmse_for(BASE_FEATURES + WEATHER_FEATURES, tr, va)

    print("\n=== RMSE on chronological hold-out (covered airports only) ===")
    print(f"{'constant-mean baseline':<30} {naive:7.2f}")
    print(f"{'base features':<30} {base_rmse:7.2f}")
    print(f"{'base + weather':<30} {wx_rmse:7.2f}  "
          f"({base_rmse - wx_rmse:+.2f} min, {(base_rmse - wx_rmse)/base_rmse*100:+.2f}%)")

    imp = pd.Series(wx_model.feature_importances_, index=BASE_FEATURES + WEATHER_FEATURES)
    print("\nweather feature importances:")
    for f in WEATHER_FEATURES:
        print(f"  {f:<16} {imp[f]:.3f}")

    # --- GO/NO-GO: is the weather response monotonic and a real number of minutes? ---
    print("\n=== weather response (partial dependence over the validation set) ===")
    print("Set the whole validation set's departure weather to each scenario, hold all")
    print("else fixed, and average the model's predicted delay:")
    scenarios = [
        ("Calm/clear", {"wx_wind_gust": 12, "wx_precip": 0.0, "wx_snow": 0.0, "wx_temp": 20, "wx_adverse": 0}),
        ("Strong wind", {"wx_wind_gust": 50, "wx_precip": 0.0, "wx_snow": 0.0, "wx_temp": 15, "wx_adverse": 1}),
        ("Rain", {"wx_wind_gust": 25, "wx_precip": 3.0, "wx_snow": 0.0, "wx_temp": 12, "wx_adverse": 1}),
        ("Heavy rain + wind", {"wx_wind_gust": 55, "wx_precip": 8.0, "wx_snow": 0.0, "wx_temp": 10, "wx_adverse": 1}),
        ("Snow", {"wx_wind_gust": 30, "wx_precip": 2.0, "wx_snow": 3.0, "wx_temp": -1, "wx_adverse": 1}),
    ]
    base_only = va[BASE_FEATURES].copy()
    calm_mean = None
    for name, feats in scenarios:
        x = base_only.copy()
        for k, v in feats.items():
            x[k] = v
        x = x[BASE_FEATURES + WEATHER_FEATURES]
        mean_pred = float(wx_model.predict(x).mean())
        if calm_mean is None:
            calm_mean = mean_pred
        print(f"  {name:<20} avg predicted delay {mean_pred:6.1f} min  "
              f"({mean_pred - calm_mean:+5.1f} vs calm)")

    # --- Example named scenario ladders for a few real flights ---
    # Mix of weather-exposed winter northern airports (snow shows up) and a dry
    # Mediterranean summer route (correctly comes out weather-insensitive).
    print("\n=== example per-flight scenario ladders (named conditions) ===")
    examples = [("IST", "TUN", 9, 1), ("NCE", "TUN", 7, 2),
                ("TUN", "ORY", 12, 1), ("DJE", "ORY", 15, 8)]
    schedule = train  # already has base features; pick a matching row for base values
    for dep, arr, hour, month in examples:
        m = schedule[(schedule.DEPSTN == dep) & (schedule.ARRSTN == arr)
                     & (schedule.dep_month == month)]
        if m.empty:
            m = schedule[(schedule.DEPSTN == dep) & (schedule.dep_month == month)]
        if m.empty:
            continue
        base_row = m.iloc[0][BASE_FEATURES].to_dict()
        base_row["dep_hour"] = hour
        bands = scenario_table(wx, dep, month)
        if not bands:
            continue
        print(f"\n  {dep}->{arr}  month {month:02d}, dep ~{hour:02d}:00")
        exp = 0.0
        for b in bands:
            pred = predict_with_weather(wx_model, base_row, b["feats"])
            exp += b["prob"] * pred
            print(f"    {b['label']:<26} {b['prob']*100:4.0f}% of days   "
                  f"~{pred:5.1f} min  (gust {b['gust']:.0f}, precip {b['precip']:.1f}mm, "
                  f"snow {b['feats']['wx_snow']:.1f}cm)")
        print(f"    {'weather-weighted outlook':<26} {'':4}         ~{exp:5.1f} min")


if __name__ == "__main__":
    main()
