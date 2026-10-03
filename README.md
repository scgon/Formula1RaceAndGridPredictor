# Formula 1 Race & Grid Predictor

Two independent ML pipelines for predicting F1 race results and qualifying grids using historical data from [fastf1](https://github.com/theOehrly/fastf1), plus a Streamlit web app to explore their predictions.

**Live web app:** https://formula1predictions.streamlit.app

## Project Layout

```
.
├── app.py                   # Streamlit entry point (st.navigation)
├── pipelines/               # the two prediction pipelines (CLI)
│   ├── predict_race.py      # race finish prediction (gain vs direct models)
│   ├── predict_grid.py      # qualifying prediction (anchor vs direct models)
│   └── f1_common.py         # shared machinery for both pipelines (see below)
├── webapp/                  # Streamlit pages
│   ├── page_home.py         # homepage: next race, cached-data status, methodology
│   ├── page_race.py         # race prediction page
│   ├── page_quali.py        # qualifying prediction page
│   └── webapp_common.py     # app-only glue: run_pipeline, live stdout capture, table/chart helpers (no pipeline logic)
├── notebooks/               # interactive inline copies of the pipelines
│   ├── race_predictions.ipynb
│   └── grid_predictions.ipynb
├── data/                    # per-season CSV caches (auto-created, gitignored)
├── cache/                   # fastf1 HTTP/session cache (auto-created, gitignored)
├── requirements.txt  AGENTS.md  LICENSE
```

`f1_common.py` holds everything both pipelines need — fastf1 cache setup, qualifying-lap extraction, practice-lap features, the season CSV cache, the model factory and permutation importance. The pipelines stay independent entry points with their own features, targets and models. The web app builds on the same pipeline functions, so CLI and web results are identical by construction.

## Setup

Requires Python 3.10+ (tested on 3.14). Any environment with the dependencies installed works — I use Homebrew miniconda on macOS, but a plain `venv` (or conda, or any other manager) is fine:

```bash
git clone https://github.com/scgon/Formula1RaceAndGridPredictor.git
cd Formula1RaceAndGridPredictor

python -m venv .venv
source .venv/bin/activate        # Windows: .venv\Scripts\activate

python -m pip install -r requirements.txt
```

If your system's `python` is missing or maps to an old Python, create the venv with `python3 -m venv .venv` (Linux/macOS) or `py -3 -m venv .venv` (Windows launcher). Once the environment is activated, plain `python` refers to it — all commands below assume that.

## Pipelines

| Script | Target | Models | Baseline |
|--------|--------|--------|----------|
| `pipelines/predict_race.py` | Race finish position | **Gain** (finish − grid) vs **Direct** (absolute finish) | Grid order |
| `pipelines/predict_grid.py` | Qualifying position | **Anchor** (change vs last quali) vs **Direct** (absolute position) | Last quali order (persistence) |

Both use `HistGradientBoostingRegressor` with a rolling backtest and identical feature sets across models for fair comparison. Two model profiles are selectable (`--model` on the CLI, *Model profile* in the web app): **fast** (fixed hyperparameters, quickest) and **optimized** (every model tunes its hyperparameters by minimizing cross-validated MAE over its own training rounds — better error on average, slower to run; models fall back to the fast values whenever no candidate clearly beats them).

## Features

**Race (`predict_race.py`):**
- Grid position, quali delta to pole, team quali delta
- FP1–FP3 pace (stint & best lap deltas, lap counts)
- Sprint quali/race results (when available)
- Driver/team form (3-race rolling, season expanding), championship points

**Grid (`predict_grid.py`):**
- FP1–FP2 pace only (day-before-quali constraint; no FP3/sprint race)
- Sprint qualifying position & delta (derived from best SQ laps; Ergast fallback)
- Previous quali position, driver/team quali form, championship points

## Web App

**Try it live:** https://formula1predictions.streamlit.app — or run it locally:

```bash
python -m streamlit run app.py
```

Three pages:

| Page | Content |
|------|---------|
| **Home** | Next race on the calendar, cached-data status, methodology overview |
| **Race prediction** | Gain vs direct prediction table, error-by-round chart, final predicted order diagram, backtest metrics, podium points and predicted-winner tables, feature importance |
| **Qualifying prediction** | Anchor vs direct prediction table, error-by-round chart, final predicted grids diagram, backtest metrics, pole points and predicted-pole tables, feature importance |

Presentation notes:
- Runs are explicit: change any setting and press **Run prediction** — changing a widget never starts the pipeline. **Reload season data** refreshes the underlying data (downloading newly completed rounds, or everything when *Force full re-download* is checked — checking the box alone downloads nothing), and progress is shown live in a log while data loads and while the backtest trains each round's models (round, event, model — and whether hyperparameters are being tuned).
- Review mode (already-completed rounds) sorts rows by the actual result; prediction mode by the model's order. Table cells hold numeric values displayed as `P{n}`, so column sorting works numerically, and exact predictions (error 0) are highlighted green.
- P1/P2/P3 cells are colored gold/silver/bronze; driver and team names are colored with official team colors (from fastf1).
- Scoring: race podium points (+15 exact position, +5 wrong slot, +100 perfect podium) and quali pole points (+15 correct pole), with per-round results plus season-total and average-per-round rows.
- Rounds predicted from a single round of training data (round 2 for races, round 3 for qualifying) are flagged in-app: with so few rows the models cannot make a single tree split, so those predictions effectively reproduce the baseline (grid / last-quali) order.

Each prediction page lets you pick the season, the target round (auto / next on the calendar / any specific round), the first backtest round (auto — recommended — or any specific round; earliest selectable: round 2 for races, round 3 for qualifying) and the model profile (**Fast** — fixed hyperparameters, or **Optimized** — per-model hyperparameter search, slower but usually lower MAE; the chosen hyperparameters are shown after each run) in the sidebar.

## Quick Start (CLI)

```bash
# Race prediction for the next race (after quali is done)
python -u pipelines/predict_race.py --next

# Grid prediction for the next qualifying (day before quali)
python -u pipelines/predict_grid.py --next

# Specific season & round
python -u pipelines/predict_race.py --season 2024 --predict-round 12
python -u pipelines/predict_grid.py --season 2024 --predict-round 12

# Optimized model profile (tunes hyperparameters per model — slower run)
python -u pipelines/predict_race.py --next --model optimized

# Force re-download of season data
python -u pipelines/predict_race.py --refresh
```

**First run** downloads ~45 fastf1 sessions (~several minutes). Subsequent runs use `cache/` and `data/*.csv` (~3–4 min).

Run with `-u` (unbuffered output) and don't pipe long runs through `head` — block buffering makes them look stalled.

## Flags

| Flag | Description |
|------|-------------|
| `--season YEAR` | Season year (default: current, falls back to previous) |
| `--predict-round N` | Round number to predict |
| `--next` | Auto-select next race/quali on calendar |
| `--model {fast,optimized}` | Model profile: `fast` uses fixed hyperparameters, `optimized` tunes them per model by minimizing cross-validated MAE (better error, slower run) |
| `--refresh` | Ignore local cache, re-download all sessions |
| `--min-train-rounds N` | Minimum completed rounds before first backtest prediction (default: 5) |

## Output

Each run prints (with live per-model training progress while the backtest and final models train):
1. **Rolling backtest** — MAE, podium hit rate, pole/winner hit rate, rank correlation for both models vs baseline
2. **Final prediction** — Predicted grid/finish order for the target round with both models
3. **Tuned hyperparameters** — the values the optimized profile picked for the final models (only with `--model optimized`; "fast defaults" means nothing clearly beat them)
4. **Feature importance** — Permutation importance (MAE increase when shuffled)

## Notebooks

`notebooks/race_predictions.ipynb` / `notebooks/grid_predictions.ipynb` are inline copies of the pipelines for interactive exploration. **They are not auto-synced** — edit the pipeline `.py` files and manually update the notebooks cell-by-cell. They resolve `cache/` and `data/` relative to the repo root, whether opened from the repo root or executed in place.

Verify notebooks:
```bash
python -m nbconvert --to notebook --execute --inplace notebooks/race_predictions.ipynb
python -m nbconvert --to notebook --execute --inplace notebooks/grid_predictions.ipynb
```
(Takes 5–10 min each.)

## Requirements

- Python 3.10+ (tested on 3.14)
- Dependencies listed in `requirements.txt`: `fastf1`, `pandas`, `numpy`, `scikit-learn`, `streamlit`, `matplotlib` (notebook charts)
- Install with `python -m pip install -r requirements.txt` — see [Setup](#setup)

## Architecture Notes

- Each pipeline exposes structured results for the web app: `backtest_records(features, min_train_rounds, profile)` (per-round metric dicts) and `final_predictions(features, target, profile)` (models + prediction frames + the hyperparameters the profile picked); `backtest_report()` / `final_report()` print the same data for the CLI.
- Model profiles: `fast` fits the fixed hyperparameters; `optimized` runs a seeded randomized search per model (pooled-MAE scoring over rolling CV folds, the fast values always a candidate, adopted only when they are clearly beaten)
- Deterministic: fixed `random_state=42`, stable data ordering → identical metrics on identical input under both profiles (the search sampling is seeded too)
- Caches (`cache/`, `data/`) live at the repo root and are shared by pipelines, web app and notebooks; `--refresh` or schema changes trigger full re-download
- Race completion buffer: 3 hours after race start (prevents caching partial results during live races)
- Grid pipeline day-before-quali constraint: only FP1/FP2 + sprint quali features allowed

## Common Issues

| Issue | Fix |
|-------|-----|
| `ModuleNotFoundError: fastf1` | The interpreter you're using doesn't have the dependencies — activate the environment from [Setup](#setup), or run `python -m pip install -r requirements.txt` inside it |
| Slow first run | Expected — downloads ~45 sessions; subsequent runs are fast |
| "qualifying has not happened yet" | Run after FP2 for grid, after quali for race; or use `--predict-round` on a past round |
| Metrics shift after code change | Expected — deterministic models mean metric changes = code changes, not noise |

## License

MIT
