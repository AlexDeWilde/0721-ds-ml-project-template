"""Experiment: should the app predict a *calibrated risk probability* + *flight-specific
quantile range* instead of a single mean-minutes number?

Roadmap #5. Measurement only — does NOT touch the app.

The app currently: (1) an RF *regressor* predicts mean delay minutes, thresholded into
Low/Moderate/High bands; (2) the "typical range" is the departure route's empirical
p25/p50/p75. Two problems this tests:
  * The mean is dominated by the rare severe-delay tail, so the point number is hard to
    trust and the category comes from thresholding a shaky mean rather than being learned.
  * The range is route-level (same for every date/time), not flight-specific.

We compare, on the SAME chronological hold-out and feature set as the app model:
  A. RISK: purpose-trained, calibrated classifiers for P(delay>=15) and P(delay>=60)
     vs. using the regressor's predicted minutes as a risk score.
     Metrics: ROC-AUC & PR-AUC (separation), Brier + reliability bins (calibration).
  B. RANGE: quantile-regression p50/p75/p90 (flight-specific) vs the route-empirical
     quantiles the app uses now. Metrics: coverage (should match the quantile level)
     and pinball loss (lower is better).

Run:  .venv/bin/python scripts/experiment_risk_classifier.py
"""
import os
import sys

import numpy as np
import pandas as pd
from sklearn.calibration import CalibratedClassifierCV
from sklearn.ensemble import (GradientBoostingRegressor,
                              HistGradientBoostingClassifier,
                              RandomForestRegressor)
from sklearn.metrics import (average_precision_score, brier_score_loss,
                             mean_pinball_loss, roc_auc_score)

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import weather_core as W  # noqa: E402
from experiment_fltid_features import BASE_FEATURES, build_base_features  # noqa: E402

WEATHER = W.WEATHER_FEATURES
FEATURES = BASE_FEATURES + WEATHER
THRESHOLDS = [15, 60]  # the app's Low/Moderate/High cut points (minutes)


def load_split():
    train = pd.read_csv("data/train.csv")
    train["STD"] = pd.to_datetime(train["STD"])
    train["FLTID"] = train["FLTID"].str.strip()
    train = build_base_features(train)
    ts = train.sort_values("STD").reset_index(drop=True)
    cut = int(len(ts) * 0.8)
    tr, va = ts.iloc[:cut].copy(), ts.iloc[cut:].copy()
    # Frequency encodings fit on the train split only (not target-based).
    for col, name in [("route", "route_freq"), ("country_pair", "country_pair_freq")]:
        freq = tr[col].value_counts()
        tr[name] = tr[col].map(freq).fillna(0).astype(int)
        va[name] = va[col].map(freq).fillna(0).astype(int)
    # Weather (real for covered airports, neutral 'typical calm' fill elsewhere) — as the app.
    wx = W.load_weather_cache()
    tr, va = W.attach_weather(tr, wx), W.attach_weather(va, wx)
    gust = tr["wx_wind_gust"].median()
    neutral = {"wx_wind_gust": gust, "wx_precip": 0.0, "wx_snow": 0.0, "wx_adverse": 0}
    for f in WEATHER:
        tr[f] = tr[f].fillna(neutral[f])
        va[f] = va[f].fillna(neutral[f])
    return tr, va


def reliability(y_true, p, bins=10):
    """Return (mean_predicted, observed_freq, count) per probability bin."""
    edges = np.linspace(0, 1, bins + 1)
    idx = np.clip(np.digitize(p, edges) - 1, 0, bins - 1)
    rows = []
    for b in range(bins):
        m = idx == b
        if m.sum():
            rows.append((p[m].mean(), y_true[m].mean(), int(m.sum())))
    return rows


def main():
    tr, va = load_split()
    print(f"train {len(tr):,} | valid {len(va):,} | features {len(FEATURES)}")
    ytr, yva = tr["target"].to_numpy(), va["target"].to_numpy()

    # Reference regressor = today's app model (mean minutes -> risk score by thresholding).
    reg = RandomForestRegressor(n_estimators=120, min_samples_leaf=50, max_features="sqrt",
                                n_jobs=-1, random_state=42).fit(tr[FEATURES], ytr)
    pred_min = reg.predict(va[FEATURES])

    print("\n=== A. RISK SEPARATION & CALIBRATION ===")
    print(f"{'target':<10}{'model':<26}{'ROC-AUC':>8}{'PR-AUC':>8}{'Brier':>8}")
    for thr in THRESHOLDS:
        ytr_b, yva_b = (ytr >= thr).astype(int), (yva >= thr).astype(int)
        base_rate = yva_b.mean()
        # (i) regressor's predicted minutes AS a risk score (rank-only; no probabilities)
        auc_r = roc_auc_score(yva_b, pred_min)
        ap_r = average_precision_score(yva_b, pred_min)
        print(f"P(delay>={thr:<3}) {'regressor score':<26}{auc_r:>8.3f}{ap_r:>8.3f}{'  n/a':>8}"
              f"   (base rate {base_rate:.2%})")
        # (ii) purpose-trained, isotonically-calibrated classifier
        clf = CalibratedClassifierCV(
            HistGradientBoostingClassifier(max_depth=4, learning_rate=0.08,
                                           max_iter=300, random_state=42),
            method="isotonic", cv=3,
        ).fit(tr[FEATURES], ytr_b)
        p = clf.predict_proba(va[FEATURES])[:, 1]
        print(f"{'':<10}{'calibrated classifier':<26}{roc_auc_score(yva_b, p):>8.3f}"
              f"{average_precision_score(yva_b, p):>8.3f}{brier_score_loss(yva_b, p):>8.3f}")
        rel = reliability(yva_b, p)
        cal = " ".join(f"{mp:.2f}->{of:.2f}" for mp, of, _ in rel)
        print(f"{'':<10}reliability (pred->observed): {cal}")

    print("\n=== B. RANGE: flight-specific quantiles vs route-empirical quantiles ===")
    print(f"{'quantile':<10}{'method':<26}{'coverage':>9}{'pinball':>9}  (coverage should ~= quantile)")
    # Route-empirical quantiles learned on train, mapped to validation by route.
    route_q = tr.groupby("route")["target"]
    for q in (0.50, 0.75, 0.90):
        gbr = GradientBoostingRegressor(loss="quantile", alpha=q, n_estimators=300,
                                        max_depth=3, learning_rate=0.05,
                                        random_state=42).fit(tr[FEATURES], ytr)
        pq = gbr.predict(va[FEATURES])
        cov = (yva <= pq).mean()
        pin = mean_pinball_loss(yva, pq, alpha=q)
        print(f"p{int(q*100):<9}{'quantile regression':<26}{cov:>8.1%}{pin:>9.2f}")
        # route-empirical: map each validation flight to its route's train quantile
        rq = route_q.quantile(q)
        global_q = np.quantile(ytr, q)
        pr = va["route"].map(rq).fillna(global_q).to_numpy()
        cov_r = (yva <= pr).mean()
        pin_r = mean_pinball_loss(yva, pr, alpha=q)
        print(f"{'':<10}{'route-empirical (app now)':<26}{cov_r:>8.1%}{pin_r:>9.2f}")

    print("\nInterpretation: higher ROC/PR-AUC = better risk separation; lower Brier + "
          "reliability pairs near the diagonal = trustworthy probabilities; quantile coverage "
          "closer to the target with lower pinball = a better, flight-specific range.")


if __name__ == "__main__":
    main()
