# Formula 1 Race & Grid Predictor

Two independent ML pipelines for predicting F1 race results and qualifying grids using historical data from [fastf1](https://github.com/theOehrly/fastf1), plus a third one predicting single weekend milestones (pole, race winner, first retirement, fastest lap), and a Streamlit web app to explore their predictions.

**Live web app:** https://formula1predictions.streamlit.app

## Project Layout

```
.
├── app.py                   # Streamlit entry point (st.navigation)
├── pipelines/               # the three prediction pipelines (CLI)
│   ├── predict_race.py      # race finish prediction (gain vs direct models)
│   ├── predict_grid.py      # qualifying prediction (anchor vs direct models)
│   ├── predict_extras.py    # milestone classifiers (pole, winner, first DNF, fastest lap)
│   └── f1_common.py         # shared machinery for the pipelines (see below)
├── webapp/                  # Streamlit pages
│   ├── page_home.py         # homepage: next race, cached-data status, methodology
│   ├── page_race.py         # race prediction page
│   ├── page_quali.py        # qualifying prediction page
│   ├── page_extras.py       # milestone prediction page (one selected model per run)
│   └── webapp_common.py     # app-only glue: run_pipeline, live stdout capture, table/chart helpers (no pipeline logic)
├── notebooks/               # interactive inline copies of the order pipelines
│   ├── race_predictions.ipynb
│   └── grid_predictions.ipynb
├── scripts/
│   └── refresh_data.py      # regenerates the bundled season CSVs (used by the scheduled workflow)
├── .github/workflows/
│   └── refresh-data.yml     # scheduled job keeping data/*.csv current
├── data/                    # per-season CSV caches (bundled in the repo, refreshed by the workflow)
├── cache/                   # fastf1 HTTP/session cache (auto-created, gitignored)
├── requirements.txt  AGENTS.md  LICENSE
```

`f1_common.py` holds everything the three pipelines share — fastf1 cache setup, qualifying-lap extraction, practice-lap features, the season CSV cache, the model factory, permutation importance and the CLI verbosity gating. The pipelines stay independent entry points with their own features, targets and models. The web app builds on the same pipeline functions, so CLI and web results are identical by construction.

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
| `pipelines/predict_extras.py` | One of: pole sitter, race winner, first retirement, fastest lap, sprint pole sitter, sprint winner (`--milestone`) | One binary classifier for the selected milestone (driver probabilities; top pick = prediction) | Pole: best season quali form · Winner: grid P1 · First DNF: most retirements · Fastest lap: most FLs / fastest qualifier · Sprint pole: past sprint-quali form · Sprint winner: sprint grid P1 |

The two order pipelines use `HistGradientBoostingRegressor`; the milestone pipeline uses `HistGradientBoostingClassifier`. All run a rolling backtest, and feature sets are tailored per model in the milestone pipeline. Two model profiles are selectable (`--model` on the CLI, *Model profile* in the web app): **fast** (fixed hyperparameters, quickest) and **optimized** (every model tunes its hyperparameters by minimizing cross-validated error — MAE for the order models, log loss for the milestone classifiers — over its own training rounds; better error on average, slower to run; models fall back to the fast values whenever no candidate clearly beats them).

The race pipeline runs in three modes: it reviews completed rounds (`post`), predicts after qualifying with both models (`pre`, grid known), and — when qualifying has not happened yet — predicts anyway with the **direct model only** (`prequali`): there is no grid for the gain model to anchor to, so the direct model predicts from practice, sprint and season-form features alone (grid and quali features are simply missing at prediction time). Auto, `--next` and `--predict-round` all use that mode for upcoming rounds whose qualifying has not run.

## Features

**Race (`predict_race.py`):**
- Grid position, quali delta to pole, team quali delta
- FP1–FP3 pace (stint & best lap deltas, lap counts)
- Sprint quali/race results (when available)
- Driver/team form (3-race rolling, season expanding), championship points

**Grid (`predict_grid.py`):**
- FP1–FP2 pace, plus FP3 and sprint-race results once those sessions have run — optional, never required, so predictions work the day before qualifying too
- Sprint qualifying position & delta (derived from best SQ laps; Ergast fallback)
- Previous quali position, driver/team quali form, championship points

**Milestones (`predict_extras.py`):**
- Pole model: the grid pipeline's original day-before-quali set (FP1/FP2, sprint quali, quali form)
- Race milestone models (winner / first retirement / fastest lap): same pre-race information as the race pipeline, plus driver/team reliability (DNF rates) and milestone history (wins, fastest laps, retirements so far)
- Sprint models (sprint pole / sprint winner): pre-sprint information — FP1, past sprint and quali form; the sprint winner additionally uses the weekend's sprint quali result. They exist on sprint weekends only (~5 per season), and their targets are derived from the stored sprint quali/race results
- The pole model and the two sprint models also predict *before* qualifying runs (their information never includes the target round's qualifying; the sprint sessions happen before qualifying on sprint weekends), with sprint results scored as soon as their session has run. The race milestones keep requiring the grid, and the target list for the sprint milestones only offers sprint weekends
- First-retirement and fastest-lap targets come from race lap data, so this pipeline keeps its own season CSVs (`data/extras_season_*.csv`; currently bundled for 2025–2026)

## Web App

**Try it live:** https://formula1predictions.streamlit.app — or run it locally:

```bash
python -m streamlit run app.py
```

The hosted app runs on Streamlit Community Cloud's free shared containers: runs are slower than on your machine, and a cold app can take minutes to fetch data under a single shared fastf1 API rate limit. If the app feels throttled or keeps hitting limits, run the project locally — the command above gives you your own cache and full-speed runs.

Four pages:

| Page | Content |
|------|---------|
| **Home** | Next race on the calendar, cached-data status, methodology overview |
| **Race prediction** | Gain vs direct prediction table, error-by-round chart, final predicted order diagram, backtest metrics, podium points and predicted-winner tables, feature importance. Before qualifying runs it switches to direct-model-only (no grid yet) |
| **Qualifying prediction** | Anchor vs direct prediction table, error-by-round chart, final predicted grids diagram, backtest metrics, pole points and predicted-pole tables, feature importance |
| **Milestones & extras** | Pick one milestone model (pole, winner, first retirement, fastest lap, sprint pole, sprint winner) in the sidebar — only it runs. Full driver-probability table with the actual outcome marked, top-10 probability chart, backtest hit-rate metrics vs the naive baseline, cumulative-hits and model-confidence charts, per-round picks table, milestone points and feature importance. After qualifying (pre-race mode) the pole and sprint calls are already scored; the race milestones stay open until the race |

Presentation notes:
- Runs are explicit: change any setting and press **Run prediction** — changing a widget never starts the pipeline. **Reload season data** refreshes the underlying data (downloading newly completed rounds, or everything when *Force full re-download* is checked — checking the box alone downloads nothing), and progress is shown live in a log while data loads and while the backtest trains each round's models (round, event, model — and whether hyperparameters are being tuned).
- Review mode (already-completed rounds) sorts rows by the actual result; prediction mode by the model's order. Table cells hold numeric values displayed as `P{n}`, so column sorting works numerically, and exact predictions (error 0) are highlighted green.
- P1/P2/P3 cells are colored gold/silver/bronze; driver and team names are colored with official team colors (from fastf1, matched across the sport's team renames), adjusted to stay readable in both the light and dark themes.
- Scoring: race podium points (+15 exact position, +5 wrong slot) and quali pole points (+15 correct pole), with per-round results plus season-total and average-per-round rows.
- Rounds predicted from a single round of training data (round 2 for races, round 3 for qualifying) are flagged in-app: with so few rows the models cannot make a single tree split, so those predictions effectively reproduce the baseline (grid / last-quali) order.

Each prediction page lets you pick the season, the target round (auto / next on the calendar / any specific round), the first backtest round (auto — recommended — or any specific round; earliest selectable: round 2 for races, round 3 for qualifying) and the model profile (**Fast** — fixed hyperparameters, or **Optimized** — per-model hyperparameter search, slower but usually lower MAE; the chosen hyperparameters are shown after each run) in the sidebar.

## Quick Start (CLI)

```bash
# Race prediction for the next race (after its qualifying, or before it
# with the direct model only)
python -u pipelines/predict_race.py --next

# Grid prediction for the next qualifying (any time before it runs)
python -u pipelines/predict_grid.py --next

# Milestone pick for the next race — one model per run (--milestone selects
# which of pole / winner / first_dnf / fastest_lap)
python -u pipelines/predict_extras.py --next
python -u pipelines/predict_extras.py --next --milestone winner

# Specific season & round
python -u pipelines/predict_race.py --season 2024 --predict-round 12
python -u pipelines/predict_grid.py --season 2024 --predict-round 12
python -u pipelines/predict_extras.py --season 2025 --predict-round 12 --milestone first_dnf

# Optimized model profile (tunes hyperparameters per model — slower run)
python -u pipelines/predict_race.py --next --model optimized

# Force re-download of season data
python -u pipelines/predict_race.py --refresh
```

Season CSVs are bundled in `data/`, so runs normally only download rounds completed since the last scheduled refresh. An unbundled season's first run still downloads ~45 fastf1 sessions (~several minutes); subsequent runs use `cache/` and `data/*.csv` (~3–4 min).

Run with `-u` (unbuffered output) and don't pipe long runs through `head` — block buffering makes them look stalled.

## Flags

| Flag | Description |
|------|-------------|
| `--season YEAR` | Season year (default: current, falls back to previous) |
| `--predict-round N` | Round number to predict |
| `--next` | Auto-select next race/quali on calendar |
| `--milestone {pole,winner,first_dnf,fastest_lap,sprint_pole,sprint_win}` | Milestone pipeline: which one of the six models to run (default: pole) |
| `--model {fast,optimized}` | Model profile: `fast` uses fixed hyperparameters, `optimized` tunes them per model by minimizing cross-validated error (better, slower run) |
| `--models {both,gain,direct}` / `{both,anchor,direct}` | Which models the race / qualifying pipeline trains, backtests and predicts (default: both). A pre-quali race target always runs the direct model only — there is no grid to anchor the gain model to |
| `-q`, `--quiet` | Print less: only the backtest summary and the final prediction (no progress lines, per-round backtest table or feature importance) |
| `-v`, `--verbose` | Print more: live per-round backtest metrics, hyperparameter-tuner decisions and importance spread (full probability table in the milestone pipeline) |
| `--refresh` | Ignore local cache, re-download all sessions |
| `--min-train-rounds N` | Minimum completed rounds before first backtest prediction (default: 5) |

## Output

Each run prints (with live per-model training progress while the backtest and final models train):
1. **Rolling backtest** — MAE, podium hit rate, pole/winner hit rate, rank correlation for both models vs baseline (order pipelines); top-1 / top-3 hit rates vs the naive baselines (milestone pipeline)
2. **Final prediction** — Predicted grid/finish order for the target round with both models, or per-milestone top-5 probability tables with the predicted driver and baseline pick
3. **Tuned hyperparameters** — the values the optimized profile picked for the final models (only with `--model optimized`; "fast defaults" means nothing clearly beat them)
4. **Feature importance** — Permutation importance (MAE increase for the order models, log-loss increase for the milestone classifiers, when shuffled)

`--quiet` trims that to the backtest summary and the final prediction; `--verbose` adds live per-round backtest metrics (`round 7 result: gain 3.3 / direct 4.0 / grid 3.1 | podium 2/2 | winner hit/-`), hyperparameter-tuner decisions (`tuner: adopting ... (pooled MAE 0.412 vs fast 0.455)`) and importance spread (`±` std). The web app always runs at the default level.

## Notebooks

`notebooks/race_predictions.ipynb` / `notebooks/grid_predictions.ipynb` are inline copies of the two order pipelines for interactive exploration — they cover the race pipeline's three target modes (including the pre-qualifying direct-model-only prediction) and the qualifying pipeline's relaxed information set (FP1/FP2, plus FP3 and sprint-race results once those sessions have run), with backtest results identical to the CLI. They do not cover the milestone pipeline or the CLI flags — notebooks always run both models. **They are not auto-synced** — edit the pipeline `.py` files and manually update the notebooks cell-by-cell. They resolve `cache/` and `data/` relative to the repo root, whether opened from the repo root or executed in place.

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
- Model profiles: `fast` fits the fixed hyperparameters; `optimized` runs a seeded randomized search per model (pooled CV scoring over rolling folds, the fast values always a candidate, adopted only when they are clearly beaten) — MAE for the regressors, log loss for the milestone classifiers
- Deterministic: fixed `random_state=42`, stable data ordering → identical metrics on identical input under both profiles (the search sampling is seeded too)
- Caches (`cache/`, `data/`) live at the repo root and are shared by pipelines, web app and notebooks; `--refresh` or schema changes trigger full re-download
- The season CSVs (`data/*.csv`) are tracked in the repo and kept current by a scheduled GitHub workflow (`.github/workflows/refresh-data.yml`, running `scripts/refresh_data.py`): fastf1 hard-stops at 500 uncached API calls/hour, and a cold Streamlit Cloud container (disk resets on every restart) would otherwise bulk-download whole seasons on the first run and hit that limit. The refresh script detects outdated CSV schemas (a pipeline feature change rewrites every bundled season it runs for) and collects the extras kind only for the default years or years that already have an extras CSV. The extras pipeline's first collection of a season also downloads race lap data (~25–30 API calls per round), so its CSVs exist for 2025–2026 and the web app's season picker only offers years whose extras CSV is bundled (the CLI stays unrestricted)
- Race completion buffer: 3 hours after race start (prevents caching partial results during live races)
- Grid pipeline information set: FP1/FP2 always, plus sprint quali, FP3 and sprint-race results when those sessions have already run — optional features that are NaN before they happen, so pre-quali predictions work at any time. The extras pole model keeps the grid pipeline's original day-before-quali set (FP1/FP2 + sprint quali); the extras race-milestone models use the race pipeline's pre-race information set
- Extras milestone definitions: pole = quali P1; winner = race P1; first retirement = the retiree(s) with the fewest completed laps (lap-1 ties are kept as a set — a hit is any of them); DNS and DSQ rows are never retirements; sprint pole = sprint-quali P1 and sprint winner = sprint-race P1, both derived from the stored sprint results and existing only on sprint weekends

## Common Issues

| Issue | Fix |
|-------|-----|
| `ModuleNotFoundError: fastf1` | The interpreter you're using doesn't have the dependencies — activate the environment from [Setup](#setup), or run `python -m pip install -r requirements.txt` inside it |
| Slow first run | Expected — downloads ~45 sessions; subsequent runs are fast |
| `RateLimitExceededError: 500 calls/h` | The F1 data API's hourly limit was hit — fetched rounds are cached, so simply re-run later; collection resumes from where it stopped (the refresh script's `--wait-on-limit` does this automatically). On the hosted app the limit is shared with other visitors — running the project locally avoids it |
| Hosted app slow or throttled | Streamlit Community Cloud runs on free shared containers — [run the project locally](#web-app) (`python -m streamlit run app.py`) for full-speed runs and your own cache |
| Pre-quali race prediction looks form-only | Expected — before practice and qualifying run there is no weekend data yet; the race pipeline predicts with the direct model from season form, the grid pipeline from historical form. Re-run after practice (and after FP3 / the sprint) for sharper predictions |
| Metrics shift after code change | Expected — deterministic models mean metric changes = code changes, not noise |

## License

MIT
