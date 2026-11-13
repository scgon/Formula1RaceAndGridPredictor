# Graph Report - Formula1RaceAndGridPredictor  (2026-11-12)

## Corpus Check
- 36 files · ~64,255 words
- Verdict: corpus is large enough that graph structure adds value.
- Unclassified: 28 file(s) not represented in the graph (top: .csv 22, .ipynb 3, (none) 2)
- WARNING: 28 of 64 files seen (44%) have no supported extractor and are missing from this graph entirely. It may look complete while representing only part of the codebase.

## Summary
- 809 nodes · 1560 edges · 34 communities (28 shown, 6 thin omitted)
- Extraction: 97% EXTRACTED · 3% INFERRED · 0% AMBIGUOUS · INFERRED: 42 edges (avg confidence: 0.85)
- Token cost: 0 input · 0 output

## Community Hubs (Navigation)
- opencode
- steps
- pytest
- streamlit
- f1_common.py
- predict_championship.py
- predict_extras.py
- predict_race.py
- train_or_load
- _store_cached_model
- predict_fantasy.py
- DataFrame
- get_prices
- pandas
- refresh_data.py
- pathlib
- test_cli_smoke.py
- test_championship_pipeline.py
- test_webapp_common.py
- test_extras_pipeline.py
- test_f1_common.py
- test_fantasy_pipeline.py
- numpy
- test_webapp_apptest.py
- fresh_modules
- webapp_common.py
- capture_stdout
- team_colors
- season_inputs
- model_profile_input
- _fit_lightness
- readable_color
- AGENTS.md
- Formula 1 Race & Grid Predictor

## God Nodes (most connected - your core abstractions)
1. `main()` - 20 edges
2. `main()` - 18 edges
3. `vprint()` - 17 edges
4. `main()` - 17 edges
5. `main()` - 16 edges
6. `main()` - 16 edges
7. `_seasons()` - 15 edges
8. `session_utc()` - 14 edges
9. `capture_stdout()` - 14 edges
10. `Formula 1 Race & Grid Predictor` - 14 edges
11. `train_or_load()` - 13 edges
12. `backtest_records()` - 13 edges
13. `backtest_records()` - 11 edges
14. `final_predictions()` - 11 edges
15. `_synthetic_training_set()` - 11 edges

## Surprising Connections (you probably didn't know these)
- `missing_rounds()` --indirect_call--> `stale_rounds()`  [INFERRED]
  scripts/refresh_data.py → pipelines/predict_race.py
- `_prequali_note()` --calls--> `vprint()`  [INFERRED]
  pipelines/predict_fantasy.py → pipelines/f1_common.py
- `main()` --calls--> `setup()`  [INFERRED]
  pipelines/predict_extras.py → pipelines/f1_common.py
- `main()` --calls--> `setup()`  [INFERRED]
  pipelines/predict_fantasy.py → pipelines/f1_common.py
- `main()` --calls--> `setup()`  [INFERRED]
  pipelines/predict_grid.py → pipelines/f1_common.py

## Import Cycles
- None detected.

## Communities (34 total, 6 thin omitted)

### Community 16 - "opencode"
Cohesion: 0.08
Nodes (23): name, on, issue_comment, types, pull_request_review_comment, types, jobs, opencode (+15 more)

### Community 4 - "steps"
Cohesion: 0.05
Nodes (38): name, on, schedule, cron, push, branches, paths, workflow_dispatch (+30 more)

### Community 12 - "pytest"
Cohesion: 0.08
Nodes (25): name, on, push, branches, pull_request, branches, workflow_dispatch, concurrency (+17 more)

### Community 17 - "streamlit"
Cohesion: 0.10
Nodes (9): target_selectbox(), snapshot_round_selectbox(), first_backtest_control(), first_backtest_season_control(), tuned_params_expander(), log_expander(), degenerate_training_note(), degenerate_backtest_note() (+1 more)

### Community 15 - "f1_common.py"
Cohesion: 0.11
Nodes (8): set_verbosity(), add_verbosity_args(), apply_verbosity(), setup(), set_model_cache(), add_model_cache_args(), apply_model_cache(), main()

### Community 1 - "predict_championship.py"
Cohesion: 0.05
Nodes (30): vprint(), tune_hyperparameters(), tune_classifier(), format_params(), sprint_points_map(), _finished_like(), bundled_season_years(), read_season_csv() (+22 more)

### Community 2 - "predict_extras.py"
Cohesion: 0.06
Nodes (35): utc_now(), session_utc(), _finished_like(), practice_entrants(), load_upcoming_round_prequali(), collect_season(), build_features(), _sprint_points() (+27 more)

### Community 0 - "predict_race.py"
Cohesion: 0.05
Nodes (44): completed_rounds(), season_team_colors(), best_quali_seconds(), pole_seconds(), clean_practice_laps(), practice_session_frame(), practice_features(), collect_season() (+36 more)

### Community 22 - "train_or_load"
Cohesion: 0.15
Nodes (6): make_model(), make_classifier(), _model_cache_key(), _load_cached_model(), train_or_load(), make_classifier()

### Community 9 - "predict_fantasy.py"
Cohesion: 0.10
Nodes (11): is_finished_like(), calculate_driver_fantasy_points(), calculate_constructor_fantasy_points(), collect_season(), load_upcoming_round_prequali(), load_upcoming_round_checked(), build_features(), _prequali_note() (+3 more)

### Community 13 - "DataFrame"
Cohesion: 0.13
Nodes (10): optimize_team(), load_upcoming_round(), usable_features(), train_model(), predict_drivers(), predict_constructors_from_drivers(), backtest_records(), final_predictions() (+2 more)

### Community 23 - "get_prices"
Cohesion: 0.17
Nodes (6): load_price_table(), _read_json(), _official_gamedays(), official_price_snapshot(), _official_constructor_price(), get_prices()

### Community 10 - "pandas"
Cohesion: 0.14
Nodes (18): practice_entrants(), sprint_quali_features(), sprint_race_features(), _merge_weekend_features(), load_upcoming_round(), collect_season(), build_features(), anchor_value() (+10 more)

### Community 26 - "refresh_data.py"
Cohesion: 0.27
Nodes (5): required_columns(), missing_rounds(), refresh_team_colors(), collect_year(), main()

### Community 14 - "test_championship_pipeline.py"
Cohesion: 0.14
Nodes (18): _raw_season(), _seasons(), test_round_points_include_derived_sprint_points(), test_snapshots_are_inclusive_of_their_round(), test_final_points_only_for_complete_seasons(), test_standings_and_gap_features(), test_usable_features_drops_all_nan_columns(), test_train_uses_only_earlier_seasons_up_to_the_snapshot_round() (+10 more)

### Community 5 - "test_webapp_common.py"
Cohesion: 0.07
Nodes (7): _bundled(), test_the_repo_bundles_season_data(), test_every_bundled_season_has_the_required_schema(), test_bundled_seasons_have_full_grids(), test_team_colors_bundle_is_wellformed(), test_fantasy_price_bundle_is_wellformed(), test_get_schedule_serves_the_last_good_copy_when_the_refresh_fails()

### Community 7 - "test_extras_pipeline.py"
Cohesion: 0.10
Nodes (18): test_finished_like(), _extras_features(), test_target_rows_selects_the_milestones_candidates(), test_actual_drivers_is_a_set_of_achievers(), _milestone_features(), test_train_and_score_a_pole_classifier(), test_train_model_rejects_single_class_training_sets(), _race_season_raw() (+10 more)

### Community 3 - "test_f1_common.py"
Cohesion: 0.06
Nodes (33): test_verbosity_flags_map_to_levels(), _schedule_row(), test_completed_rounds_respects_the_three_hour_buffer(), _quali_results(), test_best_quali_seconds(), test_pole_seconds(), _synthetic_training_set(), test_tune_hyperparameters_is_deterministic() (+25 more)

### Community 6 - "test_fantasy_pipeline.py"
Cohesion: 0.09
Nodes (19): _sample_race_df(), test_calculate_driver_fantasy_points(), test_calculate_constructor_fantasy_points(), _synthetic_season(), test_build_features_shift_and_leakage(), test_train_and_predict_cycle(), test_backtest_records_and_final_predictions(), test_model_selection_trains_only_requested_lane() (+11 more)

### Community 27 - "numpy"
Cohesion: 0.33
Nodes (5): _features_frame(), test_usable_features_drops_all_nan_columns(), test_train_and_predict_anchor_mode(), test_train_and_predict_direct_mode(), test_train_model_requires_training_data()

### Community 20 - "test_webapp_apptest.py"
Cohesion: 0.25
Nodes (8): _completed_target_label(), test_race_page_full_run_on_a_completed_round(), test_quali_page_full_run_on_a_completed_round(), test_championship_page_full_run_on_a_completed_season(), test_championship_page_first_backtest_season_setting(), test_extras_page_full_run_on_a_completed_round(), test_extras_champion_page_full_run_on_a_completed_season(), test_fantasy_page_full_run_on_a_completed_round()

### Community 18 - "webapp_common.py"
Cohesion: 0.12
Nodes (7): completed_round_numbers(), position_css(), zero_error_css(), team_css(), team_by_driver(), style_driver_team_columns(), bold_row_style()

### Community 11 - "capture_stdout"
Cohesion: 0.11
Nodes (13): TargetUnavailable, capture_stdout(), write(), write_raw(), _update_element(), get_schedule(), load_season(), resolve_target() (+5 more)

### Community 29 - "team_colors"
Cohesion: 0.29
Nodes (3): _bundled_team_colors(), team_colors(), expand_team_aliases()

### Community 24 - "_fit_lightness"
Cohesion: 0.18
Nodes (5): theme_is_dark(), _relative_luminance(), _contrast(), _fit_lightness(), probability_css()

### Community 21 - "readable_color"
Cohesion: 0.14
Nodes (7): _parse_hex(), readable_color(), driver_md(), drivers_md(), predicted_order_chart(), ranked_bars_chart(), probability_chart()

### Community 25 - "AGENTS.md"
Cohesion: 0.18
Nodes (10): Environment, File layout, Commands, Tests, Architecture, fastf1 gotchas, Determinism, Web app verification (+2 more)

### Community 8 - "Formula 1 Race & Grid Predictor"
Cohesion: 0.07
Nodes (27): Formula 1 Race & Grid Predictor, Overview, Important project files, Data and caching model, Requirements, Running the project, Streamlit app, CLI pipelines (+19 more)

## Knowledge Gaps
- **6 thin communities (<3 nodes) omitted from report** — run `graphify query` to explore isolated nodes.