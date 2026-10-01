# Formula 1 Race & Grid Predictor

Two independent ML pipelines for predicting F1 race results and qualifying grids using historical data from [fastf1](https://github.com/theOehrly/fastf1).

## Pipelines

| Script | Target | Models | Baseline |
|--------|--------|--------|----------|
| `predict_race.py` | Race finish position | **Gain** (finish − grid) vs **Direct** (absolute finish) | Grid order |
| `predict_grid.py` | Qualifying position | **Anchor** (change vs last quali) vs **Direct** (absolute position) | Last quali order (persistence) |

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

## Quick Start

```bash
# Use the miniconda Python (required — system python lacks deps)
PY=/opt/homebrew/Caskroom/miniconda/base/bin/python

# Race prediction for the next race (after quali is done)
$PY -u predict_race.py --next

# Grid prediction for the next qualifying (day before quali)
$PY -u predict_grid.py --next

# Specific season & round
$PY -u predict_race.py --season 2024 --predict-round 12
$PY -u predict_grid.py --season 2024 --predict-round 12

# Force re-download of season data
$PY -u predict_race.py --refresh
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

`race_predictions.ipynb` / `grid_predictions.ipynb` are inline copies of the pipelines for interactive exploration. **They are not auto-synced** — edit the `.py` files and manually update the notebooks cell-by-cell.

Verify notebooks:
```bash
$PY -m nbconvert --to notebook --execute --inplace race_predictions.ipynb
$PY -m nbconvert --to notebook --execute --inplace grid_predictions.ipynb
```
(Takes 5–10 min each.)

## Requirements

- Python 3.10+ (tested on 3.14 via miniconda)
- `fastf1`, `pandas`, `numpy`, `scikit-learn`
- All deps pre-installed in the miniconda env at `/opt/homebrew/Caskroom/miniconda/base/bin/python`

## Architecture Notes

- **No shared code** between pipelines — intentional duplication for independence
- Deterministic: fixed `random_state=42`, stable data ordering → identical metrics on identical input
- Caches (`cache/`, `data/`) are gitignored; `--refresh` or schema changes trigger full re-download
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