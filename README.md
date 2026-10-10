# Formula 1 Race & Grid Predictor

A Python project that predicts Formula 1 race results, qualifying grids, and weekend milestone outcomes using historical data from fastf1 and machine learning models built with scikit-learn.

Live app: https://formula1predictions.streamlit.app

## Overview

This project combines three independent prediction pipelines with a Streamlit web app:

- Race prediction: predict finishing positions for an upcoming or completed race
- Grid prediction: predict qualifying positions for an upcoming or completed qualifying session
- Milestone predictions: predict one selected target per run, such as pole sitter, race winner, first retirement, fastest lap, sprint pole, or sprint winner

The app and the CLI both use the same shared data-loading and training logic, so results stay consistent across interfaces.

## Important project files

```text
.
├── app.py                          # Streamlit entry point
├── pipelines/
│   ├── f1_common.py               # shared data loading, feature building, tuning, reports
│   ├── predict_race.py            # race pipeline
│   ├── predict_grid.py            # qualifying/grid pipeline
│   ├── predict_extras.py          # milestone pipeline
│   └── ...
├── webapp/
│   ├── _bootstrap.py              # stale-module guard for live code pulls
│   ├── page_home.py               # home page
│   ├── page_race.py               # race prediction page
│   ├── page_quali.py              # qualifying prediction page
│   ├── page_extras.py             # milestone prediction page
│   └── webapp_common.py           # app glue, logging, tables, charts, run wrappers
├── notebooks/
│   ├── race_predictions.ipynb     # interactive race pipeline notebook
│   └── grid_predictions.ipynb     # interactive qualifying pipeline notebook
├── scripts/
│   └── refresh_data.py            # refresh tracked season CSVs
├── .github/workflows/
│   └── refresh-data.yml           # scheduled CSV refresh workflow
├── data/                          # bundled season CSVs used by the project
├── cache/                         # fastf1 session cache (gitignored)
├── requirements.txt               # project dependencies
├── AGENTS.md                      # environment and project operational notes
├── TODO.md                        # planned enhancements / future work
├── LICENSE                        # MIT license
└── README.md
```

## Data and caching model

The project uses fastf1 historical data plus a local cache layer:

- `data/*.csv` contains tracked season data bundled with the repo
- `cache/` stores downloaded fastf1 session data for faster future runs
- `scripts/refresh_data.py` regenerates or updates bundled CSVs when needed
- The GitHub workflow in `.github/workflows/refresh-data.yml` refreshes the bundled data on a schedule and on code pushes

This matters because fastf1 has a hard rate limit for uncached API calls. The project treats rate-limit exhaustion as a soft stop: fetched rounds are saved, and the next run resumes from where it left off. The data refresh flow is designed to avoid cold-start bulk downloads from a live app container.

## Requirements

- Python 3.10+ (tested on Python 3.14)
- Dependencies from `requirements.txt`

Install:

```bash
git clone https://github.com/scgon/Formula1RaceAndGridPredictor.git
cd Formula1RaceAndGridPredictor
python -m venv .venv
source .venv/bin/activate
python -m pip install -r requirements.txt
```

On macOS/Linux, the project also works well with a conda/miniconda environment. The repo's operational notes assume the interpreter from Homebrew Miniconda, but any valid Python environment with the dependencies installed is fine.

## Running the project

### Streamlit app

From the repo root:

```bash
python -m streamlit run app.py
```

The app has four pages:

- Home: next race, data status, methodology overview
- Race prediction: finish prediction and backtest analysis
- Qualifying prediction: grid prediction and backtest analysis
- Milestones & extras: pick one milestone model and review driver probabilities

### CLI pipelines

All pipeline commands should be run with `-u` so output is unbuffered and long runs do not look stalled.

```bash
# Race prediction for the next race (or direct model only before qualifying)
python -u pipelines/predict_race.py --next

# Qualifying prediction for the next qualifying session
python -u pipelines/predict_grid.py --next

# Milestone model for the next target round
python -u pipelines/predict_extras.py --next
python -u pipelines/predict_extras.py --next --milestone winner

# Specific season and round
python -u pipelines/predict_race.py --season 2025 --predict-round 12
python -u pipelines/predict_grid.py --season 2025 --predict-round 12
python -u pipelines/predict_extras.py --season 2025 --predict-round 12 --milestone first_dnf

# Use the optimized model profile (slower, usually better fit)
python -u pipelines/predict_race.py --next --model optimized

# Force a re-download of season data
python -u pipelines/predict_race.py --refresh
```

Common flags:

```text
--season YEAR
--predict-round N
--next
--milestone {pole,winner,first_dnf,fastest_lap,sprint_pole,sprint_win}
--model {fast,optimized}
--models {both,gain,direct}      # race pipeline
--models {both,anchor,direct}    # grid pipeline
--refresh
--min-train-rounds N
-q, --quiet
-v, --verbose
```

## Pipeline architecture

### 1) Race pipeline (`pipelines/predict_race.py`)

This pipeline predicts race finishing positions.

Key behavior:

- Uses two models: gain vs direct
- Base comparison is grid order
- Supports three target modes:
  - post: review a completed round
  - pre: predict after qualifying when grid is known
  - prequali: predict before qualifying; the direct model runs alone because there is no grid to anchor the gain model to
- Trains with rolling historical backtests and reports MAE, podium hit rate, rank correlation, etc.

It is designed to work even when qualifying has not happened yet: the prequali mode uses practice and season-form data without a grid.

### 2) Grid pipeline (`pipelines/predict_grid.py`)

This pipeline predicts qualifying positions.

Key behavior:

- Uses anchor vs direct models
- Base comparison is last-quali persistence
- Uses FP1/FP2 pace as the core set, with FP3 and sprint-race results included when available
- Supports regular pre-quali predictions even though some features are still NaN before those sessions occur

### 3) Milestone pipeline (`pipelines/predict_extras.py`)

This pipeline predicts one milestone at a time, chosen with `--milestone`.

Supported milestones:

- pole
- winner
- first_dnf
- fastest_lap
- sprint_pole
- sprint_win

Each milestone has its own classifier and its own feature set. The final prediction is the driver with the highest predicted probability for the selected target.

The pipeline stores its own season CSVs (`data/extras_season_*.csv`) because race-lap data and retirement/fastest-lap definitions are not part of the two main season bundles.

## Model profiles

The project supports two model profiles:

- `fast` — fixed hyperparameters, quick and deterministic
- `optimized` — randomized tuning over a seeded search space before fitting each final model

For the race and grid models, tuning minimizes pooled MAE across rolling validation folds. For the milestone pipeline, tuning minimizes pooled log loss.

The fast profile is the default. The optimized profile is slower but often yields better backtest performance. When no candidate beats the fast defaults clearly, the model falls back to the fast hyperparameters.

All models are built with `HistGradientBoostingRegressor` or `HistGradientBoostingClassifier`, and the project keeps the fit deterministic with a fixed `random_state` and stable data ordering.

## Feature sets and project logic

Race features include items such as:

- grid position and grid delta to pole
- team and driver form
- practice metrics and relevant session data
- sprint qualifying and sprint race information when available
- championship points and season rolling trends

Grid features include:

- FP1/FP2 pace
- optional FP3 and sprint-race data once those sessions exist
- previous qualifying performance
- driver/team form and points

Milestone features vary by target:

- pole model: day-before-qualifying style information set
- race winner / first retirement / fastest lap: pre-race data plus reliability and milestone history
- sprint-pole and sprint-win models: sprint-weekend chronology and sprint prior performance

The project is careful about unsupported/NaN feature columns: if a training split has a column with all NaN values, the training logic drops it rather than crashing, which is critical for early-round training and for pre-session predictions.

## Web app behavior

The Streamlit app is built around the same pipeline logic used by the CLI, so predictions remain aligned.

Pages:

- Home
- Race prediction
- Qualifying prediction
- Milestones & extras

App details:

- Changing sidebar settings does not trigger a run immediately
- The user must press the Run prediction button to execute the pipeline
- Reload season data refreshes cached CSV/session state while showing live progress in the log area
- The app shows live model-training logs for each round and each model
- Review mode sorts by actual result; prediction mode sorts by model prediction
- Driver and team names are color-coded by team, with readable theme-aware contrast adjustments for both dark and light modes
- Prediction pages also include feature-importance charts and point-based summaries for podium/pole scoring

The code includes a stale-module guard inside `webapp/_bootstrap.py` to keep the app healthy after a live code pull on Streamlit Community Cloud, where new code can be pulled into an already-running app process without restarting Python.

## Verification and local sanity checks

A lightweight import smoke test for the CLI entry points is:

```bash
python pipelines/predict_race.py --help
python pipelines/predict_grid.py --help
python pipelines/predict_extras.py --help
```

A representative AppTest check for page rendering and button execution:

```bash
python - <<'PY'
from streamlit.testing.v1 import AppTest

for page, run_key in (
    ("webapp/page_home.py", None),
    ("webapp/page_race.py", "race_run"),
    ("webapp/page_quali.py", "grid_run"),
    ("webapp/page_extras.py", "extras_run"),
):
    at = AppTest.from_file(page, default_timeout=900)
    at.run()
    assert not at.exception and not at.error, page
    if run_key:
        at.sidebar.button(key=run_key).click()
        at.run()
        assert not at.exception and not at.error, page
    print(page, "OK")
PY
```

Notebook verification:

```bash
python -m nbconvert --to notebook --execute --inplace notebooks/race_predictions.ipynb
python -m nbconvert --to notebook --execute --inplace notebooks/grid_predictions.ipynb
```

These notebook runs take longer because the backtests and charts execute end-to-end.

## Refresh workflow and operational notes

- A bundled season normally downloads only newly completed rounds
- A brand-new season can still require a full download of many sessions
- The project is intentionally built to keep bundled CSVs current so live web-app users do not hit the API limit on cold starts
- The refresh script and workflow are designed to handle seasonal schema changes and to resume partial downloads cleanly

Important notes:

- Run with `-u` on CLI jobs
- Do not pipe long jobs through `head`
- For heavy runs under system load, cap threading with `OMP_NUM_THREADS=4`
- The web app runs at the default verbosity level, while CLI jobs support `-q` and `-v`
- Determinism matters: the models are fixed-seed and stable; if metrics change after a code change, the code change affected the model behavior rather than producing random noise

## Future work / backlog

The project includes a `TODO.md` with the active backlog, including ideas such as:

- weather and track-condition features
- prediction uncertainty ranges
- reliability-aware race adjustments
- cross-season training
- better driver-change handling
- ensemble / blended model strategy
- tests and CI setup
- persist fitted models to skip retraining

The README is intentionally kept current with the codebase; the full backlog is tracked in [TODO.md](TODO.md).

## License

MIT
