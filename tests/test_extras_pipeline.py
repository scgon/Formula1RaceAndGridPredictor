"""Offline unit tests for pipelines/predict_extras.py: the TARGETS specs,
retirement classification, candidate/baseline/actual selection, sprint
weekend detection, the championship standings/pool machinery and an
end-to-end classifier train+score on synthetic data."""

import numpy as np
import pandas as pd
import pytest
from sklearn.ensemble import HistGradientBoostingClassifier

import predict_extras


EXPECTED_MILESTONES = {"pole", "winner", "podium", "first_dnf", "fastest_lap",
                       "sprint_pole", "sprint_win", "champion"}


def test_targets_cover_the_eight_milestones():
    assert set(predict_extras.TARGETS) == EXPECTED_MILESTONES
    required_spec = {"label", "column", "rows_column", "train_columns",
                     "features", "baseline_sort", "baseline_note"}
    for name, spec in predict_extras.TARGETS.items():
        assert required_spec <= set(spec), name
        assert spec["features"], name
        assert isinstance(spec["train_columns"], tuple), name
        assert len({col for col, _ in spec["baseline_sort"]}) == len(spec["baseline_sort"])
        for col, ascending in spec["baseline_sort"]:
            assert isinstance(col, str) and isinstance(ascending, bool), name


def test_prequali_capable_milestones():
    prequali = {name for name, spec in predict_extras.TARGETS.items()
                if spec.get("prequali")}
    pre_scored = {name for name, spec in predict_extras.TARGETS.items()
                  if spec.get("pre_scored")}
    # decided before the race: pole on any weekend, both sprint milestones on
    # sprint weekends (their sessions run before GP qualifying). The
    # champion model also predicts before quali (the standings exist by
    # then), but its call is only scored once the season completes —
    # prequali-capable, not pre-scored.
    assert prequali == {"pole", "sprint_pole", "sprint_win", "champion"}
    assert pre_scored == {"pole", "sprint_pole", "sprint_win"}


def test_champion_is_the_only_season_history_target():
    cross_season = {name for name, spec in predict_extras.TARGETS.items()
                    if spec.get("season_history")}
    assert cross_season == {"champion"}


@pytest.mark.parametrize("status,expected", [
    ("Finished", True), ("Lapped", True), ("Disqualified", True),
    ("+1 Lap", True), ("+3 Laps", True),
    ("Retired", False), ("Did not start", False), ("Damage", False),
    ("nan", False),
])
def test_finished_like(status, expected):
    assert predict_extras._finished_like(status) is expected


def test_usable_features_drops_all_nan_columns():
    names = ["a", "b", "team_id"]
    frame = pd.DataFrame({"a": [1.0, np.nan, 2.0], "b": [np.nan] * 3,
                          "team_id": [0.0, 1.0, 0.0]})
    assert predict_extras.usable_features(frame, names) == ["a", "team_id"]


def test_make_classifier_profiles():
    names = ["a", "b", "team_id"]
    model = predict_extras.make_classifier(names)
    assert isinstance(model, HistGradientBoostingClassifier)
    assert model.random_state == 42
    assert model.categorical_features == [names.index("team_id")]
    assert model.min_samples_leaf == 10  # positives are ~1/20 of the rows
    with pytest.raises(ValueError):
        predict_extras.make_classifier(names, profile="turbo")


def test_baseline_pick_sorts_by_the_milestone_spec():
    rows = pd.DataFrame({
        "driver": ["Alpha", "Bravo", "Charlie", "Delta"],
        "quali_form_season": [3.2, 2.1, np.nan, 4.0],
        "driver_number": ["1", "4", "16", "12"],
    })
    # pole: best season quali form wins, NaN sorts last
    assert predict_extras.baseline_pick(rows, "pole") == "Bravo"


def test_baseline_pick_winner_and_podium_are_grid_p1():
    rows = pd.DataFrame({
        "driver": ["Alpha", "Bravo", "Charlie"],
        "grid": [2.0, 1.0, 3.0],
        "driver_number": ["1", "4", "16"],
    })
    assert predict_extras.baseline_pick(rows, "winner") == "Bravo"
    assert predict_extras.baseline_pick(rows, "podium") == "Bravo"


def _extras_features():
    rows = []
    # round 1: a completed sprint weekend
    for i, driver in enumerate(["A", "B", "C"]):
        rows.append({"round": 1, "driver": driver, "quali_pos": float(i + 1),
                     "grid": float(i + 1), "sprint_quali_pos": float(i + 1),
                     "pole": int(i == 0), "winner": int(i == 1),
                     "sprint_pole": int(i == 2), "sprint_win": 0})
    # round 2: a completed non-sprint round
    for i, driver in enumerate(["A", "B", "C"]):
        rows.append({"round": 2, "driver": driver, "quali_pos": float(i + 1),
                     "grid": float(i + 1), "sprint_quali_pos": np.nan,
                     "pole": int(i == 1), "winner": int(i == 0),
                     "sprint_pole": 0, "sprint_win": 0})
    # round 3: an upcoming round before its qualifying
    for i, driver in enumerate(["A", "B", "C"]):
        rows.append({"round": 3, "driver": driver, "quali_pos": np.nan,
                     "grid": np.nan, "sprint_quali_pos": np.nan,
                     "pole": 0, "winner": 0, "sprint_pole": 0, "sprint_win": 0})
    return pd.DataFrame(rows)


def test_target_rows_selects_the_milestones_candidates():
    features = _extras_features()
    assert set(predict_extras.target_rows(features, 1, "pole")["driver"]) == {"A", "B", "C"}
    assert predict_extras.target_rows(features, 2, "sprint_pole").empty
    # a pre-quali round has no quali positions at all, so a prequali-capable
    # milestone scores every entrant instead
    assert set(predict_extras.target_rows(features, 3, "pole")["driver"]) == {"A", "B", "C"}
    assert set(predict_extras.target_rows(features, 3, "sprint_win")["driver"]) == {"A", "B", "C"}


def test_actual_drivers_is_a_set_of_achievers():
    features = _extras_features()
    assert predict_extras.actual_drivers(features, 1, "winner") == {"B"}
    assert predict_extras.actual_drivers(features, 1, "pole") == {"A"}
    assert predict_extras.actual_drivers(features, 3, "pole") == set()  # unknown yet


def test_is_sprint_weekend_keys_on_the_session_names():
    assert predict_extras.is_sprint_weekend(pd.Series({
        "Session1": "Practice 1", "Session2": "Sprint Qualifying",
        "Session3": "Sprint", "Session4": "Qualifying", "Session5": "Race"}))
    assert not predict_extras.is_sprint_weekend(pd.Series({
        "Session1": "Practice 1", "Session2": "Practice 2",
        "Session3": "Qualifying", "Session4": "Race"}))


def _milestone_features(rounds=4, per_round=10, seed=2):
    spec_cols = predict_extras.TARGETS["pole"]["features"]
    rng = np.random.RandomState(seed)
    rows = []
    for rnd in range(1, rounds + 1):
        for i in range(per_round):
            row = {"round": rnd, "driver_number": str(i + 1), "driver": f"D{i}",
                   "team": f"T{i % 3}", "quali_pos": float(i + 1),
                   "pole": int(i == 0)}
            for col in spec_cols:
                row[col] = rng.rand()
            rows.append(row)
    return pd.DataFrame(rows)


def test_train_and_score_a_pole_classifier():
    features = _milestone_features()
    # one spec feature stays all-NaN: usable_features must drop it for the fit
    nan_col = next(c for c in predict_extras.TARGETS["pole"]["features"]
                   if c != "team_id")
    features[nan_col] = np.nan

    model, train, used, tuned = predict_extras.train_model(features, 4, "pole")
    assert tuned is None  # fast profile
    assert sorted(train["round"].unique()) == [1, 2, 3]
    assert "team_id" in used
    assert nan_col not in used

    rows = predict_extras.target_rows(features, 4, "pole")
    proba = model.predict_proba(rows[used].to_numpy(dtype=float))
    assert proba.shape == (10, 2)
    assert np.allclose(proba.sum(axis=1), 1.0)


def test_train_model_rejects_single_class_training_sets():
    features = _milestone_features()
    features["pole"] = 0  # nobody ever took pole in the training window
    with pytest.raises(ValueError):
        predict_extras.train_model(features, 4, "pole")


# --- championship model ------------------------------------------------------

# The race-CSV schema predict_race.py writes (what champion_history reads).
RACE_CSV_COLUMNS = [
    "round", "event", "driver_number", "driver", "team", "grid", "quali_time",
    "finish", "points", "status", "fp_race_pace_delta", "fp_best_delta",
    "fp_laps", "sprint_quali_pos", "sprint_finish", "sprint_gain", "quali_delta",
]

_RACE_POINTS = [25.0, 18.0, 15.0, 12.0, 10.0, 8.0]


def _race_season_raw(year, champion="AAA", rounds=4, drivers=("AAA", "BBB", "CCC",
                                                              "DDD", "EEE", "FFF")):
    """A synthetic bundled-race-CSV season: `champion` wins every round, the
    rest keep their positions; the last round is a sprint weekend whose
    finishing order scores the era's sprint points."""
    drivers = (champion,) + tuple(d for d in drivers if d != champion)
    rows = []
    for rnd in range(1, rounds + 1):
        sprint = (rnd == rounds)
        for i, driver in enumerate(drivers):
            rows.append({
                "round": rnd, "event": f"GP {rnd}", "driver_number": str(i + 1),
                "driver": driver, "team": "T1" if driver == champion else "T2",
                "grid": float(i + 1), "quali_time": 60.0 + i,
                "finish": float(i + 1), "points": _RACE_POINTS[i],
                "status": "Finished", "fp_race_pace_delta": 0.5 * i,
                "fp_best_delta": 0.1 * i, "fp_laps": 20.0,
                "sprint_quali_pos": float(i + 1) if sprint else np.nan,
                "sprint_finish": float(i + 1) if sprint else np.nan,
                "sprint_gain": 0.0 if sprint else np.nan,
                "quali_delta": float(i),
            })
    return pd.DataFrame(rows, columns=RACE_CSV_COLUMNS)


@pytest.mark.parametrize("year,expected", [
    (2018, {}), (2020, {}),
    (2021, {1: 3.0, 2: 2.0, 3: 1.0}),
    (2023, {1: 8.0, 2: 7.0, 3: 6.0, 4: 5.0, 5: 4.0, 6: 3.0, 7: 2.0, 8: 1.0}),
    (2025, {1: 8.0, 2: 7.0, 3: 6.0, 4: 5.0, 5: 4.0, 6: 3.0, 7: 2.0, 8: 1.0}),
])
def test_sprint_points_scheme_by_era(year, expected):
    assert predict_extras._sprint_points(year) == expected


def _title_features(year=2024):
    """Minimal feature-frame slice: two rounds where AAA won round 1 (25
    points) ahead of BBB (18) and CCC (15); round 2 is a 2024 sprint weekend
    and BBB took sprint P2 (7 extra title points)."""
    rows = []
    for rnd in (1, 2):
        for i, driver in enumerate(("AAA", "BBB", "CCC")):
            rows.append({
                "round": rnd, "driver_number": str(i + 1), "driver": driver,
                "team": "T1" if driver == "AAA" else "T2",
                "points": _RACE_POINTS[i], "finish": float(i + 1),
                "sprint_finish": 2.0 if (rnd == 2 and driver == "BBB") else np.nan,
                "winner": int(i == 0), "podium": 1,
            })
    return pd.DataFrame(rows)


def test_add_championship_columns_standings_and_label():
    df = predict_extras.add_championship_columns(_title_features(), 2024, 3, complete=True)
    r2 = df[df["round"] == 2].set_index("driver")
    # "before round 2" follows the shift(1) discipline: round 1's race points
    # only — a round's own sessions (its sprint included) never inform its
    # own "before" features
    assert r2.loc["AAA", "champ_points_before"] == 25.0
    assert r2.loc["BBB", "champ_points_before"] == 18.0
    assert r2.loc["CCC", "champ_points_before"] == 15.0
    assert list(r2["champ_standing_before"]) == [1.0, 2.0, 3.0]
    assert list(r2["champ_gap_before"]) == [0.0, 7.0, 10.0]
    assert list(df.loc[df["round"] == 2, "rounds_remaining"]) == [1.0] * 3
    # the round-2 title points themselves DO include its sprint result:
    # BBB banked 18 race + 7 sprint points (P2 in 2024) = 25 for round 3
    assert r2.loc["BBB", "title_points"] == 25.0
    # per-driver accumulated team title points: T1 (AAA alone) vs T2 (BBB+CCC)
    assert r2.loc["AAA", "team_title_points_before"] == 25.0
    assert r2.loc["BBB", "team_title_points_before"] == 33.0
    # AAA took the title: the label rides on every AAA row
    assert list(df.loc[df["driver"] == "AAA", "champion"]) == [1, 1]
    assert list(df.loc[df["driver"] != "AAA", "champion"]) == [0, 0, 0, 0]


def test_add_championship_columns_unknown_label_while_the_season_runs():
    df = predict_extras.add_championship_columns(_title_features(), 2024, 3, complete=False)
    assert not df["champion"].any()  # the champion flag stays unknown


def test_add_championship_columns_2021_sprint_points():
    df = predict_extras.add_championship_columns(_title_features(), 2021, 3, complete=True)
    r2 = df[(df["round"] == 2) & (df["driver"] == "BBB")].iloc[0]
    # round 2's sprint P2 was worth 2 title points in 2021 (7 from 2022 on)
    assert r2["title_points"] == 18.0 + 2.0


def test_add_championship_columns_pre_round_one_standings():
    df = predict_extras.add_championship_columns(_title_features(), 2024, 3, complete=True)
    r1 = df[df["round"] == 1]
    assert list(r1["champ_points_before"]) == [0.0, 0.0, 0.0]
    assert list(r1["champ_standing_before"]) == [1.0, 1.0, 1.0]  # everyone level
    assert list(r1["rounds_remaining"]) == [2.0] * 3


def test_champion_history_pools_earlier_seasons(tmp_path, monkeypatch):
    monkeypatch.setattr(predict_extras.common, "DATA_DIR", tmp_path)
    _race_season_raw(2022, champion="AAA").to_csv(tmp_path / "season_2022.csv", index=False)
    _race_season_raw(2023, champion="BBB").to_csv(tmp_path / "season_2023.csv", index=False)
    # the target year, an extras CSV and an older-schema file are all ignored
    _race_season_raw(2024, champion="CCC").to_csv(tmp_path / "season_2024.csv", index=False)
    _race_season_raw(2023).to_csv(tmp_path / "extras_season_2023.csv", index=False)

    pool = predict_extras.champion_history(2024)
    assert sorted(pool["season"].unique()) == [2022, 2023]
    # each pooled season labels its own champion on every one of their rows
    for year, champion in ((2022, "AAA"), (2023, "BBB")):
        season = pool[pool["season"] == year]
        assert set(season.loc[season["champion"] == 1, "driver"]) == {champion}
        assert season.loc[season["driver"] == champion, "champion"].all()
    # the pool carries every champion-model feature and the standings columns
    for col in predict_extras.TARGETS["champion"]["features"]:
        assert col in pool.columns
    # no history at all -> an empty pool (final_predictions surfaces the error)
    assert predict_extras.champion_history(2022).empty


def _champion_pool(seasons=(2022, 2023), champion="AAA"):
    frames = []
    for year in seasons:
        feats = predict_extras.build_features(
            predict_extras._champion_history_frame(_race_season_raw(year, champion)))
        feats = predict_extras.add_championship_columns(feats, year, 4, complete=True)
        feats["season"] = year
        frames.append(feats)
    return pd.concat(frames, ignore_index=True)


def test_champion_train_model_trains_on_the_pool_not_the_target_season():
    pool = _champion_pool()
    # the target season claims a different (wrong) champion on its own rows —
    # the champion model must ignore them entirely and train on the pool
    target = predict_extras.add_championship_columns(_title_features(), 2024, 3,
                                                      complete=True)
    model, train, used, tuned = predict_extras.train_model(target, 2, "champion",
                                                           history=pool)
    assert "season" in train.columns   # the pool, never the target frame
    assert len(train) == len(pool)     # every pool row has a finish
    assert "team_id" not in used        # no cross-season categorical codes
    assert tuned is None
    with pytest.raises(ValueError):
        predict_extras.train_model(target, 2, "champion", history=None)
    with pytest.raises(ValueError):
        predict_extras.train_model(target, 2, "champion", history=pd.DataFrame())


def test_champion_backtest_and_final_prediction():
    pool = _champion_pool()
    # a completed synthetic 2025 season, again dominated by one driver
    raw = _race_season_raw(2025, champion="AAA", rounds=4,
                           drivers=("AAA", "BBB", "CCC", "DDD"))
    target = predict_extras.build_features(
        predict_extras._champion_history_frame(raw))
    target = predict_extras.add_championship_columns(target, 2025, 4, complete=True)

    records = predict_extras.backtest_records(target, 2, milestone="champion",
                                              history=pool)
    assert [rec["round"] for rec in records] == [3, 4]
    for rec in records:
        assert rec["actual"] == "AAA"           # the completed season's champion
        assert rec["baseline"] == "AAA"        # the points leader
        assert rec["train_rounds"] == 8        # 2 pool seasons x 4 rounds
        assert rec["hit"] in (0.0, 1.0)         # a complete season -> scored

    pred = predict_extras.final_predictions(target, 4, milestone="champion",
                                            history=pool)
    assert "error" not in pred
    assert pred["actual"] == ["AAA"]
    assert pred["top1"] == "AAA"
    assert len(pred["table"]) == 4              # every entrant is scored
    assert pred["table"]["probability"].between(0.0, 1.0).all()
    assert pred["train_rounds"] == 8

    # without history the run surfaces a clean error, not a crash
    failed = predict_extras.final_predictions(target, 4, milestone="champion",
                                              history=None)
    assert "error" in failed
