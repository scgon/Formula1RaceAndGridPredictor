# AGENTS.md

## Environment

- **Interpreter**: `/opt/homebrew/Caskroom/miniconda/base/bin/python` (Python 3.14, all deps installed). Do NOT rely on `python3` from PATH — on a fresh shell it can resolve to macOS system Python, which has none of the packages. PyCharm uses this same miniconda SDK.
- **GitHub CLI**: `/opt/homebrew/bin/gh` (authenticated as `scgon`); often not on PATH.
- No tests, lint, or typecheck exist. Verification = a full script run (pipelines) or an AppTest run (web app).

## File layout

```
.
├── app.py                   # Streamlit entry point (st.navigation, stays at root)
├── pipelines/               # the two prediction pipelines (CLI entry points)
│   ├── predict_race.py      # race finish prediction
│   ├── predict_grid.py      # qualifying prediction
│   └── f1_common.py         # machinery shared by both pipelines + the web app
├── webapp/                  # Streamlit pages (paths in st.Page resolve relative to app.py)
│   ├── page_home.py
│   ├── page_race.py
│   ├── page_quali.py
│   └── webapp_common.py     # app-only glue (cached wrappers, stdout capture)
├── notebooks/               # inline copies of the pipelines (manual sync)
├── data/  cache/            # season CSVs and fastf1 cache (repo root, gitignored)
├── requirements.txt  README.md  LICENSE
```

Every webapp/pipeline script self-bootstraps `sys.path` (`webapp/` + `pipelines/`), so files can be run directly from the repo root without installation. `f1_common.BASE_DIR` points at the repo root, so `cache/` and `data/` are shared by pipelines, web app and notebooks no matter where each is executed from.

## Commands

```bash
PY=/opt/homebrew/Caskroom/miniconda/base/bin/python

$PY -u pipelines/predict_race.py   # race prediction (gain + direct models, rolling backtest)
$PY -u pipelines/predict_grid.py   # qualifying/grid prediction (anchor + direct models)
$PY -m streamlit run app.py        # web app (homepage + one page per pipeline)
```

Shared CLI flags: `--season YEAR`, `--predict-round N`, `--next` (next race/quali on the calendar), `--refresh` (re-download season data), `--min-train-rounds N`.

- **First run downloads ~45 fastf1 sessions (several minutes)**. After that `cache/` and `data/*.csv` make runs take ~3-4 min. A round that fails mid-download is skipped and retried on the next run.
- Always run with `-u`; do not pipe output through `head` — block buffering makes long jobs look stalled.
- Under system load (e.g. a PyCharm Jupyter kernel is running), cap threads: `OMP_NUM_THREADS=4`.

## Architecture

- `pipelines/f1_common.py` — shared machinery for both pipelines and the web app: fastf1 cache setup, UTC/session-time helpers (`utc_now`, `session_utc`, `completed_rounds`), quali-lap extraction (`best_quali_seconds`, `pole_seconds`), practice-lap features (`practice_features(year, rnd, sessions)`), the season CSV cache (`collect_season`, parameterized by filename/columns/result column/loader), the model factory (`make_model(categorical_features)`), `rank_corr`, and permutation importance (`importance_scores` returns a DataFrame, `importance_report` prints it).
- Two **independent pipeline entry points** — `predict_race.py` (gain = finish − grid vs direct = absolute finish; baseline = grid order) and `predict_grid.py` (anchor = change vs last quali vs direct; baseline = last-quali persistence; day-before-quali constraint: only FP1/FP2 + sprint qualifying, never FP3 or the sprint race). Each owns its FEATURES/targets/feature engineering (`build_features`), sprint handling and round loaders.
- Each pipeline exposes structured results alongside the printed reports, for the web app:
  - `backtest_records(features, min_train_rounds)` → list of per-round metric dicts; `backtest_report(records)` prints it. CLI output text is unchanged by this split. Records also carry web-app-only fields the CLI never prints: `gain_points`/`direct_points` (race podium scoring: +15 exact position, +5 wrong slot, +100 perfect podium) or `anchor_points`/`direct_points` (grid pole scoring: +15 correct pole), plus `*_top1` and `actual_top1` winner/pole driver names.
  - `final_predictions(features, target)` → `{"gain"/"anchor", "direct", "*_model", "train"}`; `final_report(...)` prints it. The main frame carries a `direct_pos` column for side-by-side display.
  - `importance_frame(pred, mode)` → permutation-importance DataFrame for one final model.
- **Streamlit app** (`app.py`, `st.navigation`): `webapp/page_home.py`, `webapp/page_race.py`, `webapp/page_quali.py` are the pages (`st.Page` and `st.page_link` paths resolve relative to `app.py`, i.e. the repo root); `webapp/webapp_common.py` is app-only glue (no pipeline logic): `capture_stdout`, `st.cache_resource` wrappers keyed on `(kind, year, data_version, force_refresh, target, mode, ...)` — the sidebar's **Reload season data** bumps `data_version`, the refresh checkbox maps to CLI `--refresh`. Never create Streamlit elements inside `@st.cache_resource` functions (CacheReplayClosureError) — cached functions return their logs, pages display them afterwards. `webapp_common.page_link` wraps `st.page_link` so pages also run standalone (AppTest) where no navigation context exists.
- Web app presentation details:
  - Tables keep **numeric underlying values** and display `P{n}` via pandas Styler formats, so interactive column sorting is numeric (not the "P1, P10, P11" string sort). Cell colors are Styler CSS: gold/silver/bronze for P1/P2/P3 cells (`position_css`), team-colored Driver/Team text (`team_colors(year)` reads the latest completed round's `TeamColor` from fastf1, like the notebooks; `readable_color` darkens light colors).
  - In review (post) mode the prediction table is sorted by the actual result; in prediction (pre) mode by the gain/anchor prediction.
  - The sidebar "First backtest round" selectbox maps to `min_train_rounds` = count of completed rounds before the selection (`first_backtest_control`).
  - Charts are Streamlit-native (no matplotlib): error-by-round and rank-correlation are `st.bar_chart` grouped series; the "final predicted grids" lane diagram (`predicted_order_chart` in webapp_common) is an Altair layered chart rendered via `st.altair_chart(..., theme="streamlit")` — the same engine/theme as native charts, used because text marks and dashed cut rules aren't expressible with the `st.*_chart` shortcuts. Pages also show podium/pole points and predicted winner/pole tables.
- `notebooks/*.ipynb` inline copies of the pipeline logic (pre-`f1_common` layout, logic-equivalent). Their first cell resolves the repo root (`Path.cwd().parent` when executed from `notebooks/`) so they share the root `cache/` and `data/`. **Editing a pipeline `.py` does not update the notebooks — keep both in sync manually**, cell by cell.

## fastf1 gotchas

- Sprint-quali (`SQ`) session results are **empty** in fastf1 3.8.x ("not supported by Ergast"). Derive the sprint quali order from best SQ lap times (existing code in `sprint_quali_features`); fallback = Sprint session `GridPosition`.
- Schedule `Session*DateUtc` values are **tz-naive** despite the name.
- Race results `Position` includes retirees (timing position); DNS rows are NaN. 2026 status values are `Finished`/`Lapped`/`Retired`/`Did not start` — not the old `+1 Lap` format.
- A round only counts as "completed" **3 hours after its race start** (`COMPLETION_BUFFER`), so a run during a live race never caches partial results.

## Determinism

- Models are `HistGradientBoostingRegressor(random_state=42)` (built via `f1_common.make_model`) with fixed data ordering — identical input yields identical backtest metrics. If metrics shift after a code change, the change affected the model; it is not noise.
- After touching pipeline code, verify parity by diffing CLI output before/after on a fixed round, e.g. `$PY -u pipelines/predict_race.py --season 2026 --predict-round 15`.

## Web app verification

```bash
$PY - <<'EOF'
from streamlit.testing.v1 import AppTest
for page in ("webapp/page_home.py", "webapp/page_race.py", "webapp/page_quali.py"):
    at = AppTest.from_file(page, default_timeout=900)
    at.run()
    assert not at.exception and not at.error, page
    print(page, "OK")
EOF
```

Prediction pages re-run the full backtest + importance on first execution (~1-3 min with cached data). A real browser check is `$PY -m streamlit run app.py --server.headless true` and `curl localhost:8501/healthz`.

## Notebook verification

```bash
$PY -m nbconvert --to notebook --execute --inplace notebooks/race_predictions.ipynb   # or grid_predictions.ipynb
```

Takes ~5-10 min (re-runs backtest + charts). The notebooks must finish with no cell errors.

## Repo

- Remote: `https://github.com/scgon/Formula1RaceAndGridPredictor.git`, branch `main`. Push with the full remote URL flow (`gh` / git credentials are configured).
- `pipelines/predict_race.py --help` is the cheap import/argparse smoke test (~2s) when you only need to confirm the code loads.
