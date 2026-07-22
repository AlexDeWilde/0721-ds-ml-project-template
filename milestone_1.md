# Milestone 1: Baseline Model

- **Value:** helps ops teams and passengers know Tunisair delays before takeoff.
- **Target:** flight arrival delay in minutes (regression).
- **Metric:** RMSE, matching the Zindi leaderboard.
- **Validation:** a chronological 80/20 split of `train` (cutoff 2018-06-01) — train on 2016-01→mid-2018, score on the last ~7 months of 2018. Not random, so the score honestly reflects predicting future flights.
- **Baseline:** predict the training mean delay (44.9 min) for every flight.
- **Baseline score:** RMSE ≈ **142 min** on the held-out future (a per-route mean is marginally better, ≈139 min).

> Note: an earlier *in-sample* estimate put this at ~117 min. The honest chronological validation is higher because delays trended upward through 2018 (validation mean ~64 min vs training ~45 min) — a constant guess from the calmer past under-predicts the busier future. This is the number every real model must beat.
