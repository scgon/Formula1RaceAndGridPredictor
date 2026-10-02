# Formula 1 Race & Grid Predictor

Two independent ML pipelines for predicting F1 race results and qualifying grids using historical data from [fastf1](https://github.com/theOehrly/fastf1), plus a Streamlit web app to explore their predictions.

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
│   └── webapp_common.py     # app-only glue: cached wrappers, stdout capture (no pipeline logic)
├── notebooks/               # interactive inline copies of the pipelines
│   ├── race_predictions.ipynb
│   └── grid_predictions.ipynb
├── data/                    # per-season CSV caches (auto-created, gitignored)
├── cache/                   # fastf1 HTTP/session cache (auto-created, gitignored)
├── requirements.txt
└── LICENSE
```

`f1_common.py` holds everything both pipelines need — fastf1 cache setup, qualifying-lap extraction, practice-lap features, the season CSV cache, the model factory and permutation importance. The pipelines stay independent entry points with their own features, targets and models. The web app builds on the same pipeline functions, so CLI and web results are identical by construction.

## Pipelines

| Script | Target | Models | Baseline |
|--------|--------|--------|----------|
| `pipelines/predict_race.py` | Race finish position | **Gain** (finish − grid) vs **Direct** (absolute finish) | Grid order |
| `pipelines/predict_grid.py` | Qualifying position | **Anchor** (change vs last quali) vs **Direct** (absolute position) | Last quali order (persistence) |

Both use `HistGradientBoostingRegressor` with a rolling backtest and identical feature sets across models for fair comparison.

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

```bash
PY=/opt/homebrew/Caskroom/miniconda/base/bin/python
$PY -m streamlit run app.py
```

Three pages:

| Page | Content |
|------|---------|
| **Home** | Next race on the calendar, cached-data status, methodology overview |
| **Race prediction** | Gain vs direct prediction table, error-by-round & rank-correlation charts, final predicted order diagram, backtest metrics, podium points and predicted-winner tables, feature importance |
| **Qualifying prediction** | Anchor vs direct prediction table, error-by-round & rank-correlation charts, final predicted grids diagram, backtest metrics, pole points and predicted-pole tables, feature importance |

Presentation notes:
- Review mode (already-completed rounds) sorts rows by the actual result; prediction mode by the model's order. Table cells hold numeric values displayed as `P{n}`, so column sorting works numerically.
- P1/P2/P3 cells are colored gold/silver/bronze; driver and team names are colored with official team colors (from fastf1).
- Scoring: race podium points (+15 exact position, +5 wrong slot, +100 perfect podium) and quali pole points (+15 correct pole), with per-round and season totals.

Each prediction page lets you pick the season and target round (auto / next on the calendar / any specific round) in the sidebar. Results are cached in-process, so tweaking widgets does not retrain models; use **Reload season data** to pick up newly completed rounds, or **Force full re-download** for the `--refresh` behaviour.

## Quick Start (CLI)

```bash
# Use the miniconda Python (required — system python lacks deps)
PY=/opt/homebrew/Caskroom/miniconda/base/bin/python

# Race prediction for the next race (after quali is done)
$PY -u pipelines/predict_race.py --next

# Grid prediction for the next qualifying (day before quali)
$PY -u pipelines/predict_grid.py --next

# Specific season & round
$PY -u pipelines/predict_race.py --season 2024 --predict-round 12
$PY -u pipelines/predict_grid.py --season 2024 --predict-round 12

# Force re-download of season data
$PY -u pipelines/predict_race.py --refresh
```

**First run** downloads ~45 fastf1 sessions (~several minutes). Subsequent runs use `cache/` and `data/*.csv` (~3–4 min).

## Flags

| Flag | Description |
|------|-------------|
| `--season YEAR` | Season year (default: current, falls back to previous) |
| `--predict-round N` | Round number to predict |
| `--next` | Auto-select next race/quali on calendar |
| `--refresh` | Ignore local cache, re-download all sessions |
| `--min-train-rounds N` | Minimum completed rounds before first backtest prediction (default: 5) |

## Output

Each run prints:
1. **Rolling backtest** — MAE, podium hit rate, pole/winner hit rate, rank correlation for both models vs baseline
2. **Final prediction** — Predicted grid/finish order for the target round with both models
3. **Feature importance** — Permutation importance (MAE increase when shuffled)

## Notebooks

`notebooks/race_predictions.ipynb` / `notebooks/grid_predictions.ipynb` are inline copies of the pipelines for interactive exploration. **They are not auto-synced** — edit the pipeline `.py` files and manually update the notebooks cell-by-cell. They resolve `cache/` and `data/` relative to the repo root, whether opened from the repo root or executed in place.

Verify notebooks:
```bash
$PY -m nbconvert --to notebook --execute --inplace notebooks/race_predictions.ipynb
$PY -m nbconvert --to notebook --execute --inplace notebooks/grid_predictions.ipynb
```
(Takes 5–10 min each.)

## Requirements

- Python 3.10+ (tested on 3.14 via miniconda)
- `fastf1`, `pandas`, `numpy`, `scikit-learn`, `streamlit`
- All deps pre-installed in the miniconda env at `/opt/homebrew/Caskroom/miniconda/base/bin/python`

## Architecture Notes

- Each pipeline exposes structured results for the web app: `backtest_records()` (per-round metric dicts) and `final_predictions()` (models + prediction frames); `backtest_report()` / `final_report()` print the same data for the CLI.
- Deterministic: fixed `random_state=42`, stable data ordering → identical metrics on identical input
- Caches (`cache/`, `data/`) live at the repo root and are shared by pipelines, web app and notebooks; `--refresh` or schema changes trigger full re-download
- Race completion buffer: 3 hours after race start (prevents caching partial results during live races)
- Grid pipeline day-before-quali constraint: only FP1/FP2 + sprint quali features allowed

## Common Issues

| Issue | Fix |
|-------|-----|
| `ModuleNotFoundError: fastf1` | Use the miniconda python path (`$PY` above), not `python3` |
| Slow first run | Expected — downloads ~45 sessions; subsequent runs are fast |
| "qualifying has not happened yet" | Run after FP2 for grid, after quali for race; or use `--predict-round` on a past round |
| Metrics shift after code change | Expected — deterministic models mean metric changes = code changes, not noise |

## License

MIT
