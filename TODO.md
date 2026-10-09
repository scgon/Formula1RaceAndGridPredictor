# TODO / Ideas

## Modeling

- [ ] **Weather features** — air/track temp and rainfall from fastf1 `session.weather_data` for quali and race. Wet sessions are where the models most likely mispredict; also worth a "session was wet" flag as a training feature on past rounds.
- [ ] **Prediction uncertainty** — quantile variants of the models (`HistGradientBoostingRegressor(loss="quantile")`); show a P10–P90 predicted range per driver in the app instead of a single position.
- [x] **DNF/retirement model** — done as part of the extras pipeline (`predict_extras.py`: first-retirement classifier + reliability features). Still open: use it to condition the position prediction (retirees currently sit in the results with timing positions).
- [ ] **Cross-season training** — pre-train on previous seasons (with a year/era adjustment) so the first backtest rounds aren't stuck near the baseline order.
- [ ] **Rookie / driver-change handling** — drivers with no prior rounds get all-NaN form features; carry over last-season/career stats or add an experience feature. Also handles mid-season driver swaps.
- [ ] **Teammate head-to-head features** — rolling quali & race delta vs teammate within the season.
- [ ] **Circuit-type features** — cluster tracks (street / power / technical) and encode as a categorical, so pace signals are weighed per track character.
- [ ] **Ensemble gain/anchor + direct models** — blend the two models' predicted orders instead of only comparing them side by side.

## Data / features

- [ ] Tyre & stint features from practice — compounds used, stint lengths, long-run pace.
- [ ] Telemetry-derived features — top speed / sector-time deltas from practice laps (fastf1 telemetry).
- [ ] Pit-stop / strategy features for the race pipeline (stop counts, pit-lane times).

## Web app

- [ ] **Prediction history** — persist each run's final predictions (e.g. a `predictions/` CSV store committed by the refresh workflow), then compare against actuals later: a "how have we done" page.
- [ ] **Season scoreboard page** — model vs baseline podium/pole points across all completed rounds, using the existing scoring helpers.
- [x] Feature-importance bar chart in the app (done — every prediction page shows permutation importance as a bar chart).
- [ ] CSV download buttons for the prediction/backtest tables.
- [ ] Predicted-vs-actual movement chart in review mode (grid → predicted → actual per driver).

## Engineering

- [ ] **Tests** — pytest for the pure helpers in `f1_common` (`rank_corr`, `usable_features`, `completed_rounds`, quali-lap extraction) and `predict_extras` (`_finished_like`, `baseline_pick`, single-class guards) on synthetic frames; no network needed.
- [ ] **CI** — GitHub Actions running the tests + `--help` smoke tests + the AppTest page checks on push.
- [ ] **Notebook sync** — generate the notebooks from the pipeline modules (or make them thin wrappers importing `pipelines/`) to end the manual cell-by-cell sync.
- [ ] **Persist fitted models** — joblib cache keyed on data + params hash so identical re-runs skip retraining (biggest win for the optimized profile).
- [ ] Parallelize the backtest across rounds (folds are independent) with joblib.
- [ ] Lint/format setup (ruff) + pre-commit hooks.
- [ ] Type hints on `f1_common`'s public helpers.

## Maybe / exploratory

- [ ] Predict championship points instead of raw positions — aligns model error with what actually matters.
- [ ] Multi-season backtest report (e.g. 2018–2026) to test feature robustness across regulation eras.
- [ ] Driver championship-position feature (title pressure / consistency signals).
- [ ] Calibrated "podium probability" per driver from backtest residuals.
