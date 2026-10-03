# AGENTS.md

## Environment

- **Interpreter**: `/opt/homebrew/Caskroom/miniconda/base/bin/python` (Python 3.14, all deps installed). Do NOT rely on `python3` from PATH — on a fresh shell it can resolve to macOS system Python, which has none of the packages. PyCharm uses this same miniconda SDK.
- **GitHub CLI**: `/opt/homebrew/bin/gh` (authenticated as `scgon`); often not on PATH.
- No tests, lint, or typecheck exist. Verification = a full script run (pipelines), an AppTest run (web app), or a CLI-output parity diff (pipeline edits).

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
│   └── webapp_common.py     # app-only glue (run_pipeline, stdout capture, table/chart helpers)
├── notebooks/               # inline copies of the pipelines (manual sync)
├── data/  cache/            # season CSVs and fastf1 cache (repo root, gitignored)
├── requirements.txt  README.md  LICENSE
```

Page scripts and `webapp_common` self-bootstrap `sys.path` (`webapp/` + `pipelines/`), so they run from the repo root without installation; `app.py` needs no bootstrap (imports only streamlit). `f1_common.BASE_DIR` points at the repo root, so `cache/` and `data/` are shared by pipelines, web app and notebooks no matter where each is executed from.

## Commands

```bash
PY=/opt/homebrew/Caskroom/miniconda/base/bin/python

$PY -u pipelines/predict_race.py   # race prediction (gain + direct models, rolling backtest)
$PY -u pipelines/predict_grid.py   # qualifying/grid prediction (anchor + direct models)
$PY -m streamlit run app.py        # web app (homepage + one page per pipeline)
```

Shared CLI flags: `--season YEAR`, `--predict-round N`, `--next` (next race/quali on the calendar), `--refresh` (re-download season data), `--min-train-rounds N`, `--model {fast,optimized}` (model profile, see below).

- **First run downloads ~45 fastf1 sessions (several minutes)**. After that `cache/` and `data/*.csv` make runs take ~3-4 min. A round that fails mid-download is skipped and retried on the next run.
- Always run with `-u`; do not pipe output through `head` — block buffering makes long jobs look stalled.
- Under system load (e.g. a PyCharm Jupyter kernel is running), cap threads: `OMP_NUM_THREADS=4`.

## Architecture

- `pipelines/f1_common.py` — shared machinery for both pipelines and the web app: fastf1 cache setup, UTC/session-time helpers (`utc_now`, `session_utc`, `completed_rounds`), quali-lap extraction (`best_quali_seconds`, `pole_seconds`), practice-lap features (`practice_features(year, rnd, sessions)`), the season CSV cache (`collect_season`, parameterized by filename/columns/result column/loader), the model factory, `rank_corr`, and permutation importance (`importance_scores` returns a DataFrame, `importance_report` prints it).
- **Model profiles** — the CLI `--model {fast,optimized}` flag and the web-app *Model profile* selectbox select between: `fast` (default) — `HistGradientBoostingRegressor` with the fixed `FAST_PARAMS`; `optimized` — before every fit, `f1_common.tune_hyperparameters` runs a seeded randomized search over `TUNED_SEARCH_SPACE`, scoring candidates by pooled MAE over rolling CV folds (each fold = one training round predicted by the rounds before it; `TUNE_MAX_SPLITS` recent rounds, `None` + `TUNE_STRIDE` = strided leave-one-round-out). `FAST_PARAMS` is always the first candidate, ties keep it, and a candidate must beat it by `TUNE_MIN_IMPROVEMENT` (relative MAE) or the tuner returns `None` and the fit falls back to the fast values (this also covers <2-round training sets, so degenerate single-round models behave identically under both profiles and the web-app degenerate notes stay accurate). Tuning inside the backtest only ever sees rounds before the predicted round, so optimized backtest metrics stay honest. A fold's training share can be all-NaN in a column (round 1 has no form/sprint data) — the tuner zeroes those columns for that fold (a constant column can't be split on); without that, HistGB raises and the backtest silently skips the round. `train_model` returns the tuned dict; `final_predictions` bundles it as `gain_params`/`anchor_params`/`direct_params` for the CLI/web-app "Tuned hyperparameters" display.
- Two **independent pipeline entry points** — `predict_race.py` (gain = finish − grid vs direct = absolute finish; baseline = grid order) and `predict_grid.py` (anchor = change vs last quali vs direct; baseline = last-quali persistence; day-before-quali constraint: only FP1/FP2 + sprint qualifying, never FP3 or the sprint race). Each owns its FEATURES/targets/feature engineering (`build_features`), sprint handling and round loaders.
- Each pipeline exposes structured results alongside the printed reports, for the web app:
  - `backtest_records(features, min_train_rounds, profile)` → list of per-round metric dicts; `backtest_report(records)` prints it. The record/report split left CLI output text unchanged; `backtest_records`/`final_predictions` have since gained live per-model training-progress prints (see the web-app execution bullet below). Records also carry web-app-only fields the CLI never prints: `gain_points`/`direct_points` (race podium scoring: +15 exact position, +5 wrong slot, +100 perfect podium) or `anchor_points`/`direct_points` (grid pole scoring: +15 correct pole), `*_top1` and `actual_top1` winner/pole driver names, and `*_train_rounds` (rounds each model trained on — a value <2 means the model degenerates to the baseline order, see below).
  - `final_predictions(features, target, profile)` → `{"gain"/"anchor", "direct", "*_model", "train", "*_features", "*_params"}`; `final_report(...)` prints it. The main frame carries a `direct_pos` column for side-by-side display. The `*_features` lists are the features each model actually used (all-NaN training columns dropped, see below); the `*_params` dicts are the hyperparameters the profile picked (None = fast defaults).
  - `importance_frame(pred, mode)` → permutation-importance DataFrame for one final model.
- sklearn's `HistGradientBoosting` **cannot fit a feature column that is entirely NaN** — it raises `ValueError: window shape cannot be larger than input array shape` (still true in sklearn 1.9.x; round-1 rows have all-NaN form features, sprint columns are all-NaN before the first sprint weekend). Each pipeline's `train_model` therefore drops those columns via `usable_features(train)`, `make_model(feature_names)` computes the `team_id` categorical index within the reduced list, and `predict_round`/importance reuse the returned feature list. Rounds with a full training set are unaffected, so default CLI output is unchanged; this is what makes the race backtest able to predict round 2 (trained on round 1 only).
- A model trained on a **single round (~22 rows) cannot make any tree split** (`min_samples_leaf=20` needs ≥40 rows in a parent), so it predicts a constant and its "prediction" collapses to the baseline order (grid order for race — a constant gain added to grid; last-quali order for the grid pipeline — a constant added to the anchor, plus `rank(method="first")` tie-breaking by the grid/driver pre-sort). This affects race round 2 and quali round 3; their backtest rows/metrics are baseline performance, not model skill. The web app flags this via `degenerate_backtest_note` / `degenerate_training_note` (webapp_common), gated on the `*_train_rounds` record fields and the bundle's `train_rounds`.
- **Streamlit app** (`app.py`, `st.navigation`): `webapp/page_home.py`, `webapp/page_race.py`, `webapp/page_quali.py` are the pages (`st.Page` and `st.page_link` paths resolve relative to `app.py`, i.e. the repo root); `webapp/webapp_common.py` is app-only glue (no pipeline logic): `capture_stdout` (optionally streams printed output live into an `st.empty` log element), `run_pipeline` (data load → target resolution → features → backtest → final models → importance, returns a render bundle or `{"error": ...}`), and small UI/table/chart helpers. `webapp_common.page_link` wraps `st.page_link` so pages also run standalone (AppTest) where no navigation context exists.
- Web app execution model — **button-gated**:
  - Changing a widget never starts the pipeline. The sidebar **Run prediction** button executes `run_pipeline` once and stores the bundle in `st.session_state[f"{kind}_result"]` (plus a settings snapshot used for the "settings changed since the last run" warning); every rerun afterwards just renders from the bundle.
  - **Reload season data** bumps `st.session_state[f"{kind}_data_version"]` and refreshes the data layer only (with live progress in a log element). The **Force full re-download** checkbox maps to CLI `--refresh` but takes effect only on the next Reload/Run press — checking it alone downloads nothing.
  - Only `get_schedule` and `team_colors` are `@st.cache_resource`-cached (tiny, element-free) — never create Streamlit elements inside cached functions (CacheReplayClosureError); the uncached collect path is exactly what allows the live download log. `collect_season` prints `fetching round N  <event> ...` before each round download so users can track slow progress, and `backtest_records` / `final_predictions` print one line per model being trained (`backtest round N  <event>: training gain model...`, `training final direct model (tuning hyperparameters)...`) — `run_pipeline` keeps the whole run inside one `capture_stdout(log_box)` so both kinds of progress stream live into the same log element.
- Web app presentation details:
  - Tables keep **numeric underlying values** and display `P{n}` via pandas Styler formats, so interactive column sorting is numeric (not the "P1, P10, P11" string sort). Cell colors are Styler CSS: gold/silver/bronze for P1/P2/P3 cells (`position_css`), green for exact predictions (`zero_error_css` on the error columns in review mode), team-colored Driver/Team text (`team_colors(year)` reads the latest completed round's `TeamColor` from fastf1, like the notebooks; `readable_color` darkens light colors).
  - In review (post) mode the prediction table is sorted by the actual result; in prediction (pre) mode by the gain/anchor prediction.
  - The sidebar "First backtest round" selectbox maps to `min_train_rounds` (completed rounds are calendar-based, no data load needed). Its default, **Auto (recommended)**, passes the pipeline default `MIN_TRAIN_ROUNDS` (5); an explicit round maps to the count of completed rounds before it. Earliest explicit round: 2 for race (round 1 can never be predicted — no training data before it), 3 for qualifying (the anchor model additionally needs each driver's previous quali result).
  - Charts are Streamlit-native (no matplotlib): the prediction-error-by-round chart is a `st.bar_chart(..., stack=False)` grouped series; the "final predicted grids" lane diagram (`predicted_order_chart` in webapp_common) is an Altair layered chart rendered via `st.altair_chart(..., theme="streamlit")` — the same engine/theme as native charts, used because text marks and dashed cut rules aren't expressible with the `st.*_chart` shortcuts. Lane order (bottom → top): gain/anchor model, direct model, actual result / starting (or last-quali) grid — keep the page captions in sync. Cut lines: race podium/points (3.5 / 10.5); quali Q3/Q2 (10.5 / 16.5 — with the 22-car grid, P16 reaches Q2). Pages also show podium/pole points and predicted winner/pole tables, each with bold **Season total** and **Average per round** summary rows.
- `notebooks/*.ipynb` inline copies of the pipeline logic (pre-`f1_common` layout, logic-equivalent). Their first cell resolves the repo root (`Path.cwd().parent` when executed from `notebooks/`) so they share the root `cache/` and `data/`. **Editing a pipeline `.py` does not update the notebooks — keep both in sync manually**, cell by cell.

## fastf1 gotchas

- Sprint-quali (`SQ`) session results are **empty** in fastf1 3.8.x ("not supported by Ergast"). Derive the sprint quali order from best SQ lap times (existing code in `sprint_quali_features`); fallback = Sprint session `GridPosition`.
- Schedule `Session*DateUtc` values are **tz-naive** despite the name.
- Race results `Position` includes retirees (timing position); DNS rows are NaN. 2026 status values are `Finished`/`Lapped`/`Retired`/`Did not start` — not the old `+1 Lap` format.
- A round only counts as "completed" **3 hours after its race start** (`COMPLETION_BUFFER`), so a run during a live race never caches partial results.

## Determinism

- Models are `HistGradientBoostingRegressor(random_state=42)` (built via `f1_common.make_model`) with fixed data ordering — identical input yields identical backtest metrics, under both profiles (the tuner's candidate sampling is seeded too). If metrics shift after a code change, the change affected the model; it is not noise. Default (`--model fast`) metrics are byte-identical to the pre-profile behavior; the only fast-output change since then is the live training-progress lines (prints only, no metric impact).
- After touching pipeline code, verify parity by diffing CLI output before/after on a fixed round, e.g. `$PY -u pipelines/predict_race.py --season 2026 --predict-round 15`.

## Web app verification

```bash
$PY - <<'EOF'
from streamlit.testing.v1 import AppTest
for page, run_key in (("webapp/page_home.py", None),
                      ("webapp/page_race.py", "race_run"),
                      ("webapp/page_quali.py", "grid_run")):
    at = AppTest.from_file(page, default_timeout=900)
    at.run()
    assert not at.exception and not at.error, page
    if run_key:  # prediction pages only render after the Run button is pressed
        at.sidebar.button(key=run_key).click()
        at.run()
        assert not at.exception and not at.error, page
    print(page, "OK")
EOF
```

- Prediction pages execute the full backtest + importance when **Run prediction** is pressed (~1-3 min with cached data under the fast profile; the optimized profile takes several times longer). A real browser check is `$PY -m streamlit run app.py --server.headless true` and `curl localhost:8501/healthz`.
- AppTest gotchas (learned the hard way):
  - Press sidebar buttons via `at.sidebar.button(key="race_run").click()` then `at.run()` — elements render in the same run.
  - `selectbox.value` is the raw widget value and `set_value` must be given that same raw value: for the first-backtest selectbox (label-based options) that's the label string; anywhere a `format_func` was used, it's the underlying int. Passing the wrong one raises a confusing `ValueError: list.index(x)`. AppTest.from_file resolves relative paths against the *calling file's* directory — pass absolute paths.
  - The auto target can **transiently fail right after a quali session whose results Ergast hasn't ingested yet** (the run then reports a clean "Pipeline failed: ... 0 sample(s)" error — no exception; this happens on unmodified HEAD too). When the AppTest run errors like that, select a completed round in the target selectbox instead (the label is `f"Round {rn} — {event} (completed)"`).
  - Chart elements (`st.bar_chart`/`st.altair_chart`) are not exposed on the AppTest element tree — verify charts via absence of exceptions plus the markdown titles around them, not by inspecting chart data.

## Notebook verification

```bash
$PY -m nbconvert --to notebook --execute --inplace notebooks/race_predictions.ipynb   # or grid_predictions.ipynb
```

Takes ~5-10 min (re-runs backtest + charts). The notebooks must finish with no cell errors.

## Repo

- Remote: `https://github.com/scgon/Formula1RaceAndGridPredictor.git`, branch `main`; `git push origin main` works (credentials configured).
- The web app is publicly hosted at https://formula1predictions.streamlit.app — treat the UI as user-facing: changes pushed to the repo can end up visible there.
- `pipelines/predict_race.py --help` is the cheap import/argparse smoke test (~2s) when you only need to confirm the code loads.
