# AGENTS.md

## Environment

- **Interpreter**: `/opt/homebrew/Caskroom/miniconda/base/bin/python` (Python 3.14, all deps installed). Do NOT rely on `python3` from PATH — on a fresh shell it can resolve to macOS system Python, which has none of the packages. PyCharm uses this same miniconda SDK.
- **GitHub CLI**: `/opt/homebrew/bin/gh` (authenticated as `scgon`); often not on PATH.
- No tests, lint, or typecheck exist. Verification = a full script run.

## Commands

```bash
PY=/opt/homebrew/Caskroom/miniconda/base/bin/python

$PY -u predict_race.py   # race prediction (gain + direct models, rolling backtest)
$PY -u predict_grid.py   # qualifying/grid prediction (anchor + direct models)
```

Shared flags: `--season YEAR`, `--predict-round N`, `--next` (next race/quali on the calendar), `--refresh` (re-download season data), `--min-train-rounds N`.

- **First run downloads ~45 fastf1 sessions (several minutes)**. After that `cache/` and `data/*.csv` make runs take ~3-4 min. A round that fails mid-download is skipped and retried on the next run.
- Always run with `-u`; do not pipe output through `head` — block buffering makes long jobs look stalled.
- Under system load (e.g. a PyCharm Jupyter kernel is running), cap threads: `OMP_NUM_THREADS=4`.

## Architecture

- Two **independent, self-contained pipelines** — `predict_race.py` and `predict_grid.py` intentionally share no module. Do not refactor them into a common library without being asked.
- Each trains two models on the same features and compares them in a rolling backtest against a naive baseline:
  - `predict_race.py`: **gain** model (target = finish − grid) vs **direct** model (absolute finish), baseline = grid order.
  - `predict_grid.py`: **anchor** model (target = quali pos − last quali pos) vs **direct**, baseline = persistence (last quali order). Day-before-quali constraint: features may only use **FP1/FP2 + sprint qualifying** — never FP3 or the sprint race (sprint runs the same day as GP quali).
- `race_predictions.ipynb` / `grid_predictions.ipynb` inline copies of the same pipeline code. **Editing a `.py` does not update the notebooks — keep both in sync manually**, cell by cell.
- Caches: `cache/` = fastf1 downloads; `data/season_<year>.csv` + `data/quali_season_<year>.csv` = per-season feature tables. A missing/outdated column schema triggers automatic re-download of every round; `--refresh` forces it. Both are gitignored — never commit them.

## fastf1 gotchas

- Sprint-quali (`SQ`) session results are **empty** in fastf1 3.8.x ("not supported by Ergast"). Derive the sprint quali order from best SQ lap times (existing code in `sprint_quali_features`); fallback = Sprint session `GridPosition`.
- Schedule `Session*DateUtc` values are **tz-naive** despite the name.
- Race results `Position` includes retirees (timing position); DNS rows are NaN. 2026 status values are `Finished`/`Lapped`/`Retired`/`Did not start` — not the old `+1 Lap` format.
- A round only counts as "completed" **3 hours after its race start** (`COMPLETION_BUFFER`), so a run during a live race never caches partial results.

## Determinism

- Models are `HistGradientBoostingRegressor(random_state=42)` with fixed data ordering — identical input yields identical backtest metrics. If metrics shift after a code change, the change affected the model; it is not noise.

## Notebook verification

```bash
$PY -m nbconvert --to notebook --execute --inplace race_predictions.ipynb   # or grid_predictions.ipynb
```

Takes ~5-10 min (re-runs backtest + charts). The notebooks must finish with no cell errors.

## Repo

- Remote: `https://github.com/scgon/Formula1RaceAndGridPredictor.git`, branch `main`. Push with the full remote URL flow (`gh` / git credentials are configured).
- `predict_race.py --help` is the cheap import/argparse smoke test (~2s) when you only need to confirm the code loads.
