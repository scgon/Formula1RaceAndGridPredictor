"""Offline unit tests for pipelines/predict_race.py: feature engineering
(the one-round shift), the all-NaN feature guard, and an end-to-end
train/predict cycle on synthetic data — including the pre-quali path where
no grid exists and only the direct model can run."""

import numpy as np
import pandas as pd
import pytest

import predict_race


def _raw_season(rounds=2, drivers=("A", "B", "C", "D")):
    rows = []
    for rnd in range(1, rounds + 1):
        for i, driver in enumerate(drivers):
            rows.append({
                "round": rnd,
                "driver_number": str(i + 1),
                "driver": driver,
                "team": ["Team A", "Team B"][i % 2],
                "grid": float(i + 1),
                "finish": float((rnd + i) % len(drivers) + 1),
                "points": float(25 - 5 * i) if i < 3 else 0.0,
                "quali_delta": 0.1 * i,
                "status": "Finished",
            })
    return pd.DataFrame(rows)


def _features_frame(rounds=4, per_round=10, seed=11):
    """A synthetic season with every FEATURES column present; the sprint
    columns stay all-NaN like a non-sprint season (which is exactly the
    case usable_features exists to survive)."""
    rng = np.random.RandomState(seed)
    rows = []
    for rnd in range(1, rounds + 1):
        for i in range(per_round):
            row = {"round": rnd, "driver_number": str(i + 1), "driver": f"D{i}",
                   "team": f"T{i % 3}"}
            for col in predict_race.FEATURES:
                row[col] = np.nan
            row.update({
                "grid": float(i + 1),
                "quali_delta": rng.rand(),
                "team_quali_delta": rng.rand(),
                "fp_race_pace_delta": rng.rand(),
                "fp_best_delta": rng.rand(),
                "fp_laps": float(rng.randint(10, 40)),
                "driver_form_3": rng.rand() * 5,
                "driver_form_season": rng.rand() * 5,
                "driver_last_finish": float(rng.randint(1, per_round + 1)),
                "team_form_3": rng.rand() * 5,
                "team_form_season": rng.rand() * 5,
                "driver_points_before": float(rng.randint(0, 100)),
                "team_points_before": float(rng.randint(0, 200)),
                "team_id": float(i % 3),
                "finish": float((i + rnd) % per_round + 1),
            })
            rows.append(row)
    return pd.DataFrame(rows)


def test_build_features_shifts_history_by_one_round():
    features = predict_race.build_features(_raw_season(rounds=2))
    first = features[features["round"] == 1].set_index("driver_number")
    second = features[features["round"] == 2].set_index("driver_number")

    # the form columns must never see the target round itself
    assert first["driver_last_finish"].isna().all()
    assert first["driver_points_before"].eq(0).all()  # nothing before round 1
    for num in second.index:
        assert second.loc[num, "driver_last_finish"] == first.loc[num, "finish"]
        assert second.loc[num, "driver_points_before"] == first.loc[num, "points"]

    # team aggregates stay within one round
    a = second[second["team"] == "Team A"]
    assert a["team_race_mean"].nunique() == 1
    assert a["team_race_mean"].iloc[0] == pytest.approx(a["finish"].mean())

    # team identity is factorized consistently across the whole season
    assert features.groupby("team")["team_id"].nunique().eq(1).all()


def test_usable_features_drops_all_nan_columns():
    features = _features_frame(rounds=3)
    used = predict_race.usable_features(features)
    assert "sprint_gain" in predict_race.FEATURES  # the synthetic season has it
    assert "sprint_gain" not in used  # ... but entirely NaN, so it must be dropped
    assert "grid" in used
    assert set(used) <= set(predict_race.FEATURES)


def test_train_and_predict_round_gain_mode():
    features = _features_frame(rounds=4)
    model, train, used, tuned = predict_race.train_model(features, target_round=4,
                                                          mode="gain")
    assert tuned is None  # fast profile: fixed hyperparameters
    assert sorted(train["round"].unique()) == [1, 2, 3]
    assert "sprint_quali_pos" not in used  # all-NaN training column dropped
    assert "grid" in used

    predicted = predict_race.predict_round(model, features, 4, mode="gain",
                                           use_features=used)
    assert len(predicted) == 10
    assert sorted(predicted["pred_pos"]) == list(range(1, 11))
    # gain mode anchors the model output on the starting grid
    assert np.allclose(predicted["pred_finish"],
                       predicted["grid"] + predicted["model_output"])
    assert predicted["pred_pos"].iloc[0] == 1


def test_train_and_predict_round_direct_mode():
    features = _features_frame(rounds=4)
    model, _, used, _ = predict_race.train_model(features, 4, mode="direct")
    predicted = predict_race.predict_round(model, features, 4, mode="direct",
                                           use_features=used)
    assert np.allclose(predicted["pred_finish"], predicted["model_output"])
    assert sorted(predicted["pred_pos"]) == list(range(1, 11))


def test_predict_round_prequali_scores_all_entrants():
    """A grid-less (pre-quali) round must still be predictable by the direct
    model over every entrant — the gain model has nothing to anchor to."""
    features = _features_frame(rounds=4)
    prequali = features[features["round"] == 4].copy()
    prequali["grid"] = np.nan
    prequali["round"] = 5
    features = pd.concat([features, prequali], ignore_index=True)

    model, _, used, _ = predict_race.train_model(features, 5, mode="direct")
    predicted = predict_race.predict_round(model, features, 5, mode="direct",
                                            use_features=used)
    assert len(predicted) == 10  # every entrant scored, grid or not
    assert sorted(predicted["pred_pos"]) == list(range(1, 11))


def test_train_model_requires_training_data():
    features = _features_frame(rounds=4)
    with pytest.raises(ValueError):
        predict_race.train_model(features, target_round=1)


def test_podium_points_scoring():
    assert predict_race.podium_points(["A", "B", "C"], ["A", "B", "C"]) == 45
    assert predict_race.podium_points(["A", "B", "C"], ["A", "X", "B"]) == 20
    assert predict_race.podium_points(["A", "B", "C"], ["X", "Y", "Z"]) == 0
