# Tunisair Flight Delay Prediction

Predicting flight **delay in minutes** for Tunisair, framed for two audiences: an airline **operations team** and, downstream, a **traveller-facing app** that surfaces delay risk at booking time. Built for the [Zindi Flight Delay Prediction Challenge](https://zindi.africa/competitions/flight-delay-prediction-challenge).

- **Task:** regression — predict `target` (delay in minutes). **Metric:** RMSE (minutes).
- **Data:** 107,833 historical flights (2016–2018): date, flight number, route, scheduled times, aircraft.
- **Golden rule:** use **only pre-departure information** (schedule, route, aircraft, weather) — never anything known only after the flight happens.

## Headline results

| Model | Held-out RMSE (min) | Notes |
|---|---|---|
| Constant-mean baseline | 141.95 | the number to beat |
| **Operational model** (tuned RF, uses prior-leg delay) | **108.64** | −24%; for the ops team; assumes prior-leg delay known at prediction time |
| Submittable model (no prior-leg delay) | 130.92 | the Zindi entry → `zindi_submission.csv` |
| Booking-time model (app) | — | reframed as a **calibrated risk classifier** (ROC-AUC 0.80 for ≥15 min, 0.76 for ≥60 min) + **quantile range**; powers the Streamlit app |
| Day-of model (app) | — | adds the inbound aircraft's current delay → ROC-AUC **0.89 / 0.91**; the app's "Day of travel" mode |

**Key insight:** error is dominated by the **severe-delay tail** (180 min+), driven mostly by delay *propagation* (a plane inheriting its previous leg's delay). Features that don't speak to the tail (weather, per-flight history) barely move RMSE — but are still valuable for *communicating risk* to travellers (see the experiment write-ups).

---

## Repository guide

Every file, what it contains, and why it matters.

### 📋 Project planning & narrative

| File | Contains | Significance |
|---|---|---|
| [01_assignment.md](01_assignment.md) | The project brief: timeline, milestones, deliverables. | Ground truth for what the project must deliver. |
| [02_kanban_board.md](02_kanban_board.md) | How the GitHub project board is set up. | The board (project #4) drives the work order; notebook sections mirror it 1:1. |
| [milestone_1.md](milestone_1.md) | Milestone-1 checkpoint notes. | Record of the baseline milestone. |
| [HANDOFF.md](HANDOFF.md) | **Full session handoff** — decisions, board state, results, what's next. | **Start here** to resume the project. The single source of truth for status. |
| [ISSUES.md](ISSUES.md) | Data-quality issues log (corrupted `STA` dates, `SXF` airport override, notebook/board ordering). | Prevents re-discovering the same data traps; lists candidate features. |
| [slides_draft.md](slides_draft.md) | Stakeholder slide-deck content (source of truth). | The 10-minute talk; rendered to the HTML decks below. |

### 📓 Notebooks

| File | Contains | Significance |
|---|---|---|
| [04_flight_delay_eda_modeling.ipynb](04_flight_delay_eda_modeling.ipynb) | **The project notebook** — Phases 1–8: load → clean → EDA → features → baseline → modelling → error analysis → final eval. | The core analysis; runs top-to-bottom in board order with plain-language explanations above every cell. |
| [03_eda-and-modeling.ipynb](03_eda-and-modeling.ipynb) | The original template example (coffee-quality dataset). | Reference scaffolding only — not part of the Tunisair analysis. |
| [03_eda-and-modeling Alex.ipynb](03_eda-and-modeling%20Alex.ipynb) | A teammate's working copy of the template example. | Kept for history; not part of the final deliverable. |

### 📊 Stakeholder decks

| File | Contains | Significance |
|---|---|---|
| [Tunisair Flight Delay Deck.html](Tunisair%20Flight%20Delay%20Deck.html) | Rendered stakeholder deck (v1). | Presentation deliverable. |
| [Tunisair Flight Delay Deck v2.html](Tunisair%20Flight%20Delay%20Deck%20v2.html) | Rendered stakeholder deck (v2, Alex). | Updated presentation deliverable. |

### 🖥️ The data product — `app/` (Streamlit "Tunisair Delay-Alert")

A **booking-time** web app: type a flight → get a delay-risk category (🟢/🟡/🔴) backed by **calibrated probabilities** (chance of a 15+/60+ min delay), a **flight-specific quantile range** (typical / up-to / bad-day), a **weather-sensitivity ladder** (how the flight runs under that airport's plausible weather — calm/rough/severe, named), plain-language advice, and calmer alternative departure times. It resolves weather by horizon — **recorded** (historical) → **live forecast** (≤16 days, Open-Meteo) → **seasonal** typical (further out) — and feeds any known weather into the risk & range. A **"Day of travel"** mode additionally takes the **inbound aircraft's current delay** (delay propagation, the strongest signal) for a far sharper estimate near departure.

| File | Contains | Significance |
|---|---|---|
| [app/streamlit_app.py](app/streamlit_app.py) | The Streamlit UI (branded, red/white Tunisair theme). | What the user sees and interacts with. |
| [app/delay_core.py](app/delay_core.py) | UI-free prediction logic: feature engineering, risk bands, expected range, `is_ramadan()`, alternatives. | Testable core; shared definitions so training and inference match exactly. |
| [app/reference/route_freq.csv](app/reference/route_freq.csv) | Route → training frequency. | Frequency encoding the model needs at inference. |
| [app/reference/country_pair_freq.csv](app/reference/country_pair_freq.csv) | Country-pair → frequency. | Same, for country pairs. |
| [app/reference/flight_schedule.csv](app/reference/flight_schedule.csv) | Flight number → route + typical departure hour. | Powers the flight-number lookup, dropdowns, and alternatives. |
| [app/reference/route_delay_stats.csv](app/reference/route_delay_stats.csv) | Route → empirical delay p25/p50/p75. | The honest "typical range" shown to travellers. |
| [app/reference/weather_scenarios.csv](app/reference/weather_scenarios.csv) | Per airport/month calm/rough/severe weather bands (named, with odds & representative values). | Powers the **weather-sensitivity ladder** (roadmap #2); precomputed so the app needs no network. |
| [app/assets/logo.png](app/assets/logo.png) | White Tunisair logo. | Branding. |
| [app/README.md](app/README.md) | How to run the app. | App-specific docs. |

### 🔧 Scripts — `scripts/`

| File | Contains | Significance |
|---|---|---|
| [scripts/build_app_data.py](scripts/build_app_data.py) | Trains the calibrated classifiers + quantile regressors + writes the `app/reference/*.csv` tables. | **Rebuilds all app artifacts** — run after any feature change. |
| [scripts/experiment_risk_classifier.py](scripts/experiment_risk_classifier.py) | Experiment for **roadmap #5** (calibrated classifier + quantile range) with time-aware calibration. | Proved the reframe now shipped in the app: better risk separation + honest probabilities + flight-specific range. |
| [scripts/experiment_fltid_features.py](scripts/experiment_fltid_features.py) | Experiment for **roadmap #1** (flight-number history as features), with leakage-safe evaluation. | Measured a real but small win (~0.55 min) from a single smoothed per-flight median; the kitchen-sink of stats overfits. See its write-up. |
| [scripts/fetch_weather.py](scripts/fetch_weather.py) | Downloads & caches ERA5 hourly weather for the 15 busiest airports (Open-Meteo, no key). | One-time data pull for the weather experiment; cache is gitignored. |
| [scripts/weather_core.py](scripts/weather_core.py) | Shared weather helpers: WMO code → human label, condition naming, flight join. | Ensures weather is named/joined consistently across experiment and (future) app. |
| [scripts/experiment_weather_features.py](scripts/experiment_weather_features.py) | Experiment for **roadmap #2** (weather): RMSE impact + the per-flight **named scenario ladder** (calm/rough/severe). | Found weather won't lower RMSE, but the model's weather *response* is strong & monotonic — validating a "what-if the weather is bad" scenario feature. See its write-up. |

### 🧪 Experiment write-ups (accuracy-improvement roadmap)

| File | Contains | Significance |
|---|---|---|
| [FLTID_FEATURE_EXPERIMENT.md](FLTID_FEATURE_EXPERIMENT.md) | Plain-language findings for roadmap #1 (flight-number history). | Evidence behind the pending decision to wire the flight-median feature into the app. |
| [WEATHER_SCENARIO_EXPERIMENT.md](WEATHER_SCENARIO_EXPERIMENT.md) | Plain-language findings for roadmap #2 (weather scenario ladder). | Evidence for the per-flight weather ladder now shipped in the app. |
| [RISK_CLASSIFIER_EXPERIMENT.md](RISK_CLASSIFIER_EXPERIMENT.md) | Plain-language findings for roadmap #5 (calibrated classifier + quantile range). | Evidence for the classifier/quantile reframe now shipped in the app. |

### 📦 Data, models & outputs

| File / Folder | Contains | Significance |
|---|---|---|
| [data/train.csv](data/) | 107,833 labelled historical flights. | Training data. |
| [data/test.csv](data/) | Unlabelled Zindi submission set (no `target`). | Predict on this for the competition entry. |
| [data/sample_submission.csv](data/) | Zindi submission format. | Template for the output file. |
| `data/weather_cache/` | Cached ERA5 hourly weather (parquet). | **Gitignored** — regenerate with `scripts/fetch_weather.py`. |
| [models/app_booking_model.joblib](models/) | Calibrated classifiers + quantile regressors + metadata (~2 MB). | **Committed** — the app loads this. |
| `models/delay_model*.joblib` | Large operational/submittable models (~350 MB). | **Gitignored** — regenerate by running the notebook. |
| [zindi_submission.csv](zindi_submission.csv) | Predictions on `test.csv`. | The Zindi competition deliverable. |

### ⚙️ Configuration & infrastructure

| File / Folder | Contains | Significance |
|---|---|---|
| [pyproject.toml](pyproject.toml) | Dependencies & project config. | Source of truth for the environment (`uv sync`). |
| [uv.lock](uv.lock) | Pinned dependency versions. | Reproducible installs for the whole team. |
| [.python-version](.python-version) | Python 3.13. | Pins the interpreter. |
| [.streamlit/config.toml](.streamlit/config.toml) | Streamlit theme/config. | App appearance. |
| [.gitignore](.gitignore) | Ignore rules (large models, weather cache, caches). | Keeps the repo lean. |
| [assets/](assets/) | Screenshots for the Kanban board guide. | Documentation images. |
| [.github/workflows/](.github/workflows/) | CI workflows (import checks, notifications). | Automated checks. |
| [LICENSE](LICENSE) | License. | Legal. |

---

## Quickstart

```bash
uv sync                                              # install deps + create .venv/

# Run the analysis notebook (regenerates models/, in VS Code — see Setup below)
#   open 04_flight_delay_eda_modeling.ipynb, select the .venv kernel, Run All

# Run the traveller app
uv run streamlit run app/streamlit_app.py

# Rebuild the app's model + reference tables
uv run python scripts/build_app_data.py

# Reproduce the accuracy-improvement experiments
.venv/bin/python scripts/experiment_fltid_features.py
.venv/bin/python scripts/fetch_weather.py            # one-time weather cache
.venv/bin/python scripts/experiment_weather_features.py
```

---

## Setup

> [!NOTE]
> Text in angle brackets like `<repo-name>` is a **placeholder** — replace it (brackets included) with your own value.

### 1. Clone and install

```bash
git clone <copied-ssh-url>
cd <repo-name>
uv sync
```

`uv sync` installs all dependencies into a virtual environment (`.venv/`). Add new packages with `uv add <package>` — it updates `pyproject.toml` and `uv.lock` for the whole team.

### 2. Open the notebook

> [!NOTE]
> Open VS Code from the project root so it auto-detects the `uv sync` environment.

```bash
code .
```

Open `04_flight_delay_eda_modeling.ipynb` and select the Python environment created by `uv sync` as the kernel.

## Handling merge conflicts in notebooks

`.ipynb` files are JSON and conflict messily. `nbdime` (run via `uvx`, no dependency added) makes this manageable.

```bash
uvx nbdime config-git --enable    # enable once
uvx nbdime mergetool              # when a conflict happens
```

> [!TIP]
> Clear outputs before committing (`Notebook: Clear All Outputs` in VS Code) to reduce conflicts.

## References

- [scikit-learn](https://scikit-learn.org/stable/) · [estimator map](https://scikit-learn.org/stable/machine_learning_map.html) — modelling.
- [PEP 8](https://peps.python.org/pep-0008/) — Python style.
- [uv](https://docs.astral.sh/uv/) — package/environment manager.
- [Open-Meteo](https://open-meteo.com/) — free weather API used for the weather experiment.
- [Zindi Flight Delay Challenge](https://zindi.africa/competitions/flight-delay-prediction-challenge) — the competition.
