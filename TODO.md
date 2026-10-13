# TODO / Ideas

## Modeling

- [ ] **Weather features** — air/track temp and rainfall from fastf1 `session.weather_data` for quali and race. Wet sessions are where the models most likely mispredict; also worth a "session was wet" flag as a training feature on past rounds.
- [ ] **Prediction uncertainty** — quantile variants of the models (`HistGradientBoostingRegressor(loss="quantile")`); show a P10–P90 predicted range per driver in the app instead of a single position. A cheaper alternative: conformal intervals from the rolling backtest residuals.
- [x] **DNF/retirement model** — done as part of the extras pipeline (`predict_extras.py`: first-retirement classifier + reliability features). Still open: use it to condition the position prediction (retirees currently sit in the results with timing positions) — e.g. sample retirements from the classifier and re-rank the survivors.
- [ ] **Cross-season training (race/grid pipelines)** — pre-train on previous seasons (with a year/era adjustment) so the first backtest rounds aren't stuck near the baseline order. The championship pipeline already trains across complete earlier seasons; the order pipelines are still single-season.
- [ ] **Rookie / driver-change handling** — drivers with no prior rounds get all-NaN form features; carry over last-season/career stats or add an experience feature. Also handles mid-season driver swaps.
- [ ] **Teammate head-to-head features** — rolling quali & race delta vs teammate within the season.
- [ ] **Circuit-type features** — cluster tracks (street / power / technical) and encode as a categorical, so pace signals are weighed per track character.
- [ ] **Monotonic constraints** — `HistGradientBoosting` accepts `monotonic_cst`: a worse grid slot should never improve the predicted finish, worse quali form never help. Cheap to try against the current unconstrained fits.
- [ ] **Recency-weighted training** — recent rounds carry more signal about current form; pass decaying `sample_weight` (by round distance) instead of letting every training round count equally.
- [ ] **Ensemble gain/anchor + direct models** — blend the two models' predicted orders instead of only comparing them side by side (rank-average, or learn per-round blend weights from the backtest).

## Data / features

- [ ] Tyre & stint features from practice — compounds used, stint lengths, long-run pace.
- [ ] Telemetry-derived features — top speed / sector-time deltas from practice laps (fastf1 telemetry).
- [ ] Pit-stop / strategy features for the race pipeline (stop counts, pit-lane times).
- [ ] Track-specific driver/team history — career quali/race record at the same circuit (some drivers and teams consistently out- or under-perform their season form at particular tracks).
- [ ] **Backfill the extras bundle** — the extras CSVs only cover 2025–2026 (lap data costs ~25–30 API calls/round); collect 2018–2024 offline with `--wait-on-limit` and bundle them, so the extras backtest spans regulation eras like the race pipeline's does.

## Web app

- [ ] **Prediction history** — persist each run's final predictions (e.g. a `predictions/` CSV store committed by the refresh workflow), then compare against actuals later: a "how have we done" page.
- [ ] **Season scoreboard** — model vs baseline podium/pole points across all completed rounds. Partially done: the Podium points / Pole points tables already score both models per round (+15 exact slot, +5 wrong slot) — still open: score the baselines with the same helpers and a dedicated cross-season view.
- [x] Feature-importance bar chart in the app (done — every prediction page shows permutation importance as a bar chart).
- [ ] CSV download buttons for the prediction/backtest tables.
- [ ] Predicted-vs-actual movement chart in review mode — a per-driver grid → predicted → actual slope/dumbbell view. The lane diagram already shows predicted vs actual, but its starting-grid lane only appears in prediction mode.
- [ ] Shareable run state — encode the sidebar settings in `st.query_params` so a configured prediction survives a page reload and can be linked.
- [ ] Settings-keyed run cache — keep the last few run bundles in `session_state` (keyed on the settings snapshot) so switching back to previous settings re-renders instantly instead of re-running the pipeline.
- [ ] Colorblind-safe palette option — team colors are the only identity encoding on most tables and charts; offer an accessible fallback palette.

## Engineering

- [x] **Tests** — pytest covers the pure helpers in `f1_common`, all four pipelines (race, grid, championship, extras), bundled-data schemas, CLI `--help` smoke tests, and Streamlit AppTest page checks.
- [x] **CI** — GitHub Actions runs the full pytest suite on pushes and pull requests for Python 3.12 and 3.14.
- [ ] **Notebook sync** — generate the notebooks from the pipeline modules (or make them thin wrappers importing `pipelines/`) to end the manual cell-by-cell sync. The championship and extras pipelines have no inline notebooks at all.
- [ ] **Persist fitted models** — joblib cache keyed on data + params hash so identical re-runs skip retraining (biggest win for the optimized profile).
- [ ] Parallelize the backtest across rounds (folds are independent) with joblib.
- [ ] Lint/format setup (ruff) + pre-commit hooks.
- [x] Type hints on `f1_common`'s public helpers — done: every function in `pipelines/f1_common.py` now carries full parameter and return annotations (CLI parity verified on a post-mode round).
- [x] Pin the fastf1 upper bound — done: `fastf1>=3.8,<5` in requirements.txt (CI + refresh-data workflows install from it, as does the app deploy).
- [ ] Notify on refresh-data failures — the 6-hour scheduled workflow keeps the hosted app fed; GitHub's default failure emails are easy to miss, so add an explicit notify step or a stale-data banner in the app.

## Maybe / exploratory

- [x] Predict championship points instead of raw positions — done as the championship pipeline (`predict_championship.py`): remaining-points + direct models against a run-rate baseline, for drivers and constructors alike.
- [ ] Multi-season backtest report (e.g. 2018–2026) to test race/grid feature robustness across regulation eras. The championship pipeline's leave-one-season-out backtest already spans every complete bundled season; the order pipelines' backtests are single-season.
- [ ] Driver championship-position feature (title pressure / consistency signals) — the champion milestone model already uses `champ_standing_before`/`champ_gap_before`; still open for the race/grid pipelines.
- [ ] Penalty-aware grid prediction — announced engine/gearbox drops aren't in any fastf1 feed; a small hand-curated penalty overlay could correct the pre-race grid before the model runs.
- [ ] Learning-to-rank formulation — pairwise/listwise objective on finishing order instead of per-driver regression (e.g. XGBoost `rank:pairwise`), matched to the rank-correlation metric the backtest already reports.
- [ ] Constructors-title milestone — the champion extras model scores the drivers' title only; a team-level twin (labels derivable from the bundled race CSVs) would mirror it for the constructors'.
- [x] Calibrated "podium probability" per driver — done as the podium milestone classifier of the extras pipeline (`--milestone podium` scores every driver's top-three chance via `predict_proba`); residual-based calibration of the position models remains unexplored.
