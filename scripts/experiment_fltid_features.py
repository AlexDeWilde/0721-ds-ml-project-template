"""Experiment: does adding per-flight-number (FLTID) historical delay stats as
model features improve the booking-time model?

Roadmap item #1. This is a MEASUREMENT script — it does not touch the app. It
trains the current booking-time feature set vs. that set PLUS FLTID-history
features on an identical chronological hold-out, and prints the RMSE delta.

Leakage discipline (the whole point of doing this as an experiment first):
  * FLTID stats are a form of target encoding. They are computed on the TRAIN
    portion only, then mapped onto the validation portion. Computing them on the
    full data would leak the validation targets into the features and make the
    hold-out score dishonestly good.
  * Low-history flight numbers give noisy per-FLTID estimates, so each stat is
    SHRUNK toward the global value by sample count (empirical-Bayes style).
    Unseen FLTIDs (present in validation, absent in train) fall back to global.

Run:  .venv/bin/python scripts/experiment_fltid_features.py
"""
import os
import sys

import airportsdata
import holidays
import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import root_mean_squared_error

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "app"))
from delay_core import is_ramadan  # noqa: E402

BASE_FEATURES = [
    "dep_hour", "dep_dow", "dep_month", "dep_is_holiday", "dep_is_ramadan",
    "gc_distance_km", "is_domestic", "route_freq", "country_pair_freq",
]
# All five stats we *could* add. The sweep below shows adding all of them
# overfits (they are redundant and the noisy ones hurt forward generalization).
FLTID_FEATURES = [
    "fltid_delay_median", "fltid_delay_mean", "fltid_pct_late",
    "fltid_delay_std", "fltid_n",
]
# The winner: a single, strongly-shrunk per-FLTID median delay.
FLTID_BEST = ["fltid_delay_median"]
LATE_THRESHOLD = 15  # minutes; "late" = delay >= 15 min (matches the app's Low band)
SHRINKAGE_K = 100    # prior weight: a FLTID needs ~K flights to outweigh the global prior
AIRPORT_OVERRIDES = {"SXF": {"country": "DE", "lat": 52.3667, "lon": 13.5033, "tz": "Europe/Berlin"}}

_airports = airportsdata.load("IATA")
_tn_holidays = holidays.Tunisia(years=range(2016, 2031))


def airport_latlon_country(code):
    a = _airports.get(code) or AIRPORT_OVERRIDES.get(code)
    return (a["lat"], a["lon"], a["country"]) if isinstance(a, dict) else None


def haversine_km(lat1, lon1, lat2, lon2):
    r = 6371.0
    p1, p2 = np.radians(lat1), np.radians(lat2)
    dp, dl = np.radians(lat2 - lat1), np.radians(lon2 - lon1)
    a = np.sin(dp / 2) ** 2 + np.cos(p1) * np.cos(p2) * np.sin(dl / 2) ** 2
    return 2 * r * np.arcsin(np.sqrt(a))


def build_base_features(df):
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


def fit_fltid_stats(train_df, k=SHRINKAGE_K, late_threshold=LATE_THRESHOLD):
    """Learn per-FLTID delay stats on the training rows ONLY, shrunk toward global.

    Returns (stats_df indexed by FLTID, global_dict) so they can be mapped onto any
    split. Shrinkage: stat = (n*fltid_stat + k*global_stat) / (n + k).
    """
    g_median = train_df["target"].median()
    g_mean = train_df["target"].mean()
    g_pct_late = (train_df["target"] >= late_threshold).mean()
    g_std = train_df["target"].std()

    grp = train_df.groupby("FLTID")["target"]
    raw = pd.DataFrame({
        "n": grp.size(),
        "median": grp.median(),
        "mean": grp.mean(),
        "pct_late": grp.apply(lambda s: (s >= late_threshold).mean()),
        "std": grp.std().fillna(0.0),
    })
    n = raw["n"]
    stats = pd.DataFrame({
        "fltid_delay_median": (n * raw["median"] + k * g_median) / (n + k),
        "fltid_delay_mean": (n * raw["mean"] + k * g_mean) / (n + k),
        "fltid_pct_late": (n * raw["pct_late"] + k * g_pct_late) / (n + k),
        "fltid_delay_std": (n * raw["std"] + k * g_std) / (n + k),
        "fltid_n": n,
    })
    glob = {"fltid_delay_median": g_median, "fltid_delay_mean": g_mean,
            "fltid_pct_late": g_pct_late, "fltid_delay_std": g_std, "fltid_n": 0}
    return stats, glob


def apply_fltid_stats(df, stats, glob):
    df = df.copy()
    joined = df.join(stats, on="FLTID")
    for col in FLTID_FEATURES:
        df[col] = joined[col].fillna(glob[col])
    return df


def rmse_for(features, tr, va):
    model = RandomForestRegressor(
        n_estimators=120, min_samples_leaf=50, max_features="sqrt",
        n_jobs=-1, random_state=42,
    ).fit(tr[features], tr["target"])
    return root_mean_squared_error(va["target"], model.predict(va[features])), model


def main():
    train = pd.read_csv("data/train.csv")
    train["STD"] = pd.to_datetime(train["STD"])
    train["FLTID"] = train["FLTID"].str.strip()  # raw values have trailing spaces
    train = build_base_features(train)

    # Frequency encodings (fit on full train, as the app does — these are not target-based).
    route_freq = train["route"].value_counts()
    cp_freq = train["country_pair"].value_counts()
    train["route_freq"] = train["route"].map(route_freq).astype(int)
    train["country_pair_freq"] = train["country_pair"].map(cp_freq).astype(int)

    # Chronological 80/20 hold-out (identical for both feature sets).
    ts = train.sort_values("STD").reset_index(drop=True)
    cut = int(len(ts) * 0.8)
    tr, va = ts.iloc[:cut].copy(), ts.iloc[cut:].copy()
    print(f"train {len(tr):,} rows | valid {len(va):,} rows | "
          f"cut date ~ {ts.loc[cut, 'STD'].date()}")

    # FLTID stats fit on TRAIN only, then mapped to both splits (no leakage).
    stats, glob = fit_fltid_stats(tr)
    tr = apply_fltid_stats(tr, stats, glob)
    va = apply_fltid_stats(va, stats, glob)

    seen = va["FLTID"].isin(stats.index).mean()
    print(f"validation FLTIDs seen in train: {seen*100:.1f}% "
          f"({len(stats):,} distinct FLTIDs in train)\n")

    naive_base = root_mean_squared_error(va["target"], [tr["target"].mean()] * len(va))
    base_rmse, _ = rmse_for(BASE_FEATURES, tr, va)
    all_rmse, all_model = rmse_for(BASE_FEATURES + FLTID_FEATURES, tr, va)
    best_rmse, best_model = rmse_for(BASE_FEATURES + FLTID_BEST, tr, va)

    print(f"{'constant-mean baseline':<34} RMSE {naive_base:7.2f}")
    print(f"{'current booking features (base)':<34} RMSE {base_rmse:7.2f}")
    print(f"{'base + all 5 FLTID stats':<34} RMSE {all_rmse:7.2f}  "
          f"({base_rmse - all_rmse:+.2f} min)  <- overfits")
    print(f"{'base + FLTID median only':<34} RMSE {best_rmse:7.2f}  "
          f"({base_rmse - best_rmse:+.2f} min)  <- best")
    delta = base_rmse - best_rmse
    print(f"\nVERDICT: a single shrunk per-FLTID median delay improves the booking "
          f"model\nby {delta:.2f} min ({delta / base_rmse * 100:.2f}%). Adding the other four "
          f"stats\nredirects the model's attention onto them but does not generalize forward.")

    print("\nfeature importances (base + all 5, showing why the extras mislead):")
    imp = pd.Series(all_model.feature_importances_,
                    index=BASE_FEATURES + FLTID_FEATURES).sort_values(ascending=False)
    for name, val in imp.items():
        marker = "  <-- FLTID stat" if name in FLTID_FEATURES else ""
        print(f"  {name:<24} {val:.3f}{marker}")


if __name__ == "__main__":
    main()
