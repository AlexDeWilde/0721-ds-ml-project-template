"""Experiment: how fast does the booking-time model go stale? (roadmap #4 evidence)

We can't get 2022-2025 Tunisair delay data (no free source has the scheduled times needed
to label delays), so we instead MEASURE the decay directly on the data we have. A single
model is trained on 2016 and then applied to successive future half-years (2017 H1 -> 2018 H2).
Holding the model fixed and moving the test window forward isolates the effect of the gap
between training and prediction.

What to watch:
  * ROC-AUC  — ranking skill (does it still tell risky flights from safe ones?).
  * Brier    — probability accuracy (calibration + sharpness); rises as the model drifts.
  * quantile coverage — does the p90 range still cover ~90% of flights, or under-cover
    because later periods run worse than training expects?
  * mean delay per window — the regime shift that drives the drift.

Verdict feeds a retraining-cadence recommendation.

Run:  .venv/bin/python scripts/experiment_data_drift.py
"""
import os
import sys

import numpy as np
import pandas as pd
from sklearn.ensemble import (GradientBoostingRegressor,
                              HistGradientBoostingClassifier)
from sklearn.isotonic import IsotonicRegression
from sklearn.metrics import brier_score_loss, roc_auc_score

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import weather_core as W  # noqa: E402
from experiment_fltid_features import BASE_FEATURES, build_base_features  # noqa: E402

WEATHER = W.WEATHER_FEATURES
FEATURES = BASE_FEATURES + WEATHER
LATE, SEVERE = 15, 60


def load_featured():
    df = pd.read_csv("data/train.csv")
    df["STD"] = pd.to_datetime(df["STD"])
    df = build_base_features(df)
    df = W.attach_weather(df, W.load_weather_cache())
    return df.sort_values("STD").reset_index(drop=True)


def prep(train_df, test_df):
    """Freq-encode + weather-neutral-fill using the TRAIN window only, apply to both."""
    train_df, test_df = train_df.copy(), test_df.copy()
    for col, name in [("route", "route_freq"), ("country_pair", "country_pair_freq")]:
        freq = train_df[col].value_counts()
        train_df[name] = train_df[col].map(freq).fillna(0).astype(int)
        test_df[name] = test_df[col].map(freq).fillna(0).astype(int)
    gust = train_df["wx_wind_gust"].median()
    neutral = {"wx_wind_gust": gust, "wx_precip": 0.0, "wx_snow": 0.0, "wx_adverse": 0}
    for f in WEATHER:
        train_df[f] = train_df[f].fillna(neutral[f])
        test_df[f] = test_df[f].fillna(neutral[f])
    return train_df, test_df


def fit_calibrated(X, y, frac=0.8):
    c = int(len(X) * frac)
    base = HistGradientBoostingClassifier(max_depth=4, learning_rate=0.08, max_iter=300,
                                          random_state=42).fit(X.iloc[:c], y[:c])
    iso = IsotonicRegression(out_of_bounds="clip").fit(base.predict_proba(X.iloc[c:])[:, 1], y[c:])
    return base, iso


def main():
    df = load_featured()
    train_df = df[(df["STD"] >= "2016-01-01") & (df["STD"] < "2017-01-01")]
    print(f"Training window: 2016 ({len(train_df):,} flights). Model held fixed; test window moves forward.\n")

    trp, _ = prep(train_df, train_df)
    Xtr = trp[FEATURES]
    ytr_late, ytr_sev = (trp["target"].to_numpy() >= LATE).astype(int), (trp["target"].to_numpy() >= SEVERE).astype(int)
    clf_late = fit_calibrated(Xtr, ytr_late)
    clf_sev = fit_calibrated(Xtr, ytr_sev)
    quant = {q: GradientBoostingRegressor(loss="quantile", alpha=q, n_estimators=300, max_depth=3,
                                          learning_rate=0.05, random_state=42).fit(Xtr, trp["target"])
             for q in (0.50, 0.75, 0.90)}

    def proba(clf, X):
        base, iso = clf
        return iso.transform(base.predict_proba(X)[:, 1])

    windows = [("2017 H1", "2017-01-01", "2017-07-01"), ("2017 H2", "2017-07-01", "2018-01-01"),
               ("2018 H1", "2018-01-01", "2018-07-01"), ("2018 H2", "2018-07-01", "2019-01-01")]
    train_end = pd.Timestamp("2016-12-31")

    print(f"{'test window':<10}{'gap(mo)':>8}{'mean dly':>9}{'AUC15':>7}{'AUC60':>7}"
          f"{'Brier15':>8}{'Brier60':>8}{'cov p90':>8}")
    for name, lo, hi in windows:
        te = df[(df["STD"] >= lo) & (df["STD"] < hi)]
        _, tep = prep(train_df, te)
        Xte, yte = tep[FEATURES], tep["target"].to_numpy()
        gap = (pd.Timestamp(lo) - train_end).days / 30.4
        yl, ys = (yte >= LATE).astype(int), (yte >= SEVERE).astype(int)
        pl, ps = proba(clf_late, Xte), proba(clf_sev, Xte)
        cov90 = (yte <= quant[0.90].predict(Xte)).mean()
        print(f"{name:<10}{gap:>8.1f}{yte.mean():>9.1f}{roc_auc_score(yl, pl):>7.3f}"
              f"{roc_auc_score(ys, ps):>7.3f}{brier_score_loss(yl, pl):>8.3f}"
              f"{brier_score_loss(ys, ps):>8.3f}{cov90*100:>7.0f}%")

    print("\nRanking skill (AUC) is the most durable; probability accuracy (Brier) and p90 "
          "coverage decay as the gap grows and the delay regime shifts — that's the staleness a "
          "data refresh (or periodic recalibration) would fix.")


if __name__ == "__main__":
    main()
