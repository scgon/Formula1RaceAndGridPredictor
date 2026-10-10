"""Offline unit tests for pipelines/predict_extras.py: the TARGETS specs,
retirement classification, candidate/baseline/actual selection, sprint
weekend detection and an end-to-end classifier train+score on synthetic
data."""

import numpy as np
import pandas as pd
import pytest
from sklearn.ensemble import HistGradientBoostingClassifier

import predict_extras


EXPECTED_MILESTONES = {"pole", "winner", "first_dnf", "fastest_lap",
                       "sprint_pole", "sprint_win"}


def test_targets_cover_the_six_milestones():
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


def test_prequali_capable_milestones_are_the_pre_scored_ones():
    prequali = {name for name, spec in predict_extras.TARGETS.items()
                if spec.get("prequali")}
    pre_scored = {name for name, spec in predict_extras.TARGETS.items()
                  if spec.get("pre_scored")}
    # decided before the race: pole on any weekend, both sprint milestones on
    # sprint weekends (their sessions run before GP qualifying)
    assert prequali == pre_scored == {"pole", "sprint_pole", "sprint_win"}


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


def test_baseline_pick_winner_is_grid_p1():
    rows = pd.DataFrame({
        "driver": ["Alpha", "Bravo", "Charlie"],
        "grid": [2.0, 1.0, 3.0],
        "driver_number": ["1", "4", "16"],
    })
    assert predict_extras.baseline_pick(rows, "winner") == "Bravo"


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
