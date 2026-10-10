"""Offline unit tests for pipelines/predict_grid.py: the quali-history shift,
the anchor fallback chain, the all-NaN feature guard and an end-to-end
anchor/direct train+predict cycle on synthetic data."""

import numpy as np
import pandas as pd
import pytest

import predict_grid


def _features_frame(rounds=4, per_round=10, seed=5):
    rng = np.random.RandomState(seed)
    rows = []
    for rnd in range(1, rounds + 1):
        for i in range(per_round):
            row = {"round": rnd, "driver_number": str(i + 1), "driver": f"D{i}",
                   "team": f"T{i % 3}"}
            for col in predict_grid.FEATURES:
                row[col] = np.nan
            row.update({
                "quali_pos": float((i + rnd) % per_round + 1),
                "quali_pos_last": (float((i + rnd - 1) % per_round + 1)
                                   if rnd > 1 else np.nan),
                "quali_form_3": rng.rand() * 5,
                "quali_form_season": rng.rand() * 5,
                "fp_best_delta": rng.rand(),
                "fp_race_pace_delta": rng.rand(),
                "fp_laps": float(rng.randint(10, 40)),
                "driver_points_before": float(rng.randint(0, 100)),
                "team_points_before": float(rng.randint(0, 200)),
                "team_id": float(i % 3),
            })
            rows.append(row)
    return pd.DataFrame(rows)


def test_build_features_shifts_quali_history():
    raw = pd.DataFrame({
        "round": [1, 1, 2, 2],
        "driver_number": ["1", "2", "1", "2"],
        "driver": ["A", "B", "A", "B"],
        "team": ["Team A", "Team A", "Team A", "Team A"],
        "quali_pos": [1.0, 2.0, 3.0, 4.0],
        "quali_delta": [0.0, 0.1, 0.2, 0.3],
        "points": [25.0, 18.0, 25.0, 18.0],
        "fp_best_delta": [0.1, 0.2, 0.3, 0.4],
        "fp_race_pace_delta": [0.5, 0.6, 0.7, 0.8],
        "fp3_best_delta": [np.nan] * 4,
        "fp3_race_pace_delta": [np.nan] * 4,
    })
    features = predict_grid.build_features(raw)
    first = features[features["round"] == 1]
    second = features[features["round"] == 2]
    assert first["quali_pos_last"].isna().all()  # round 1 has no previous quali
    assert list(second["quali_pos_last"]) == [1.0, 2.0]
    assert list(second["driver_points_before"]) == [25.0, 18.0]
    assert features.groupby("team")["team_id"].nunique().eq(1).all()


def test_anchor_value_falls_back_in_order():
    rows = pd.DataFrame({
        "quali_pos_last": [3.0, np.nan, np.nan],
        "quali_form_season": [9.0, 7.0, np.nan],
    })
    anchor = predict_grid.anchor_value(rows)
    assert list(anchor) == [3.0, 7.0, predict_grid.NEUTRAL_POSITION]


def test_usable_features_drops_all_nan_columns():
    used = predict_grid.usable_features(_features_frame(rounds=3))
    assert "fp3_best_delta" in predict_grid.FEATURES  # the synthetic season has it
    assert "fp3_best_delta" not in used  # ... but entirely NaN, so dropped
    assert "quali_pos_last" in used
    assert set(used) <= set(predict_grid.FEATURES)


def test_train_and_predict_anchor_mode():
    features = _features_frame(rounds=4)
    model, train, used, tuned = predict_grid.train_model(features, 4, mode="anchor")
    assert tuned is None
    # the anchor model additionally needs each driver's previous quali result,
    # which round 1 rows never have
    assert sorted(train["round"].unique()) == [2, 3]
    assert train["quali_pos_last"].notna().all()
    assert "sprint_quali_pos" not in used  # all-NaN training column dropped

    predicted = predict_grid.predict_round(model, features, 4, mode="anchor",
                                           use_features=used)
    assert len(predicted) == 10
    assert sorted(predicted["pred_pos"]) == list(range(1, 11))
    assert np.allclose(predicted["pred_score"],
                       predicted["anchor"] + predicted["model_output"])


def test_train_and_predict_direct_mode():
    features = _features_frame(rounds=4)
    model, train, used, _ = predict_grid.train_model(features, 4, mode="direct")
    # the direct model trains on round 1 too — it does not need the anchor
    assert sorted(train["round"].unique()) == [1, 2, 3]
    predicted = predict_grid.predict_round(model, features, 4, mode="direct",
                                           use_features=used)
    assert np.allclose(predicted["pred_score"], predicted["model_output"])
    assert sorted(predicted["pred_pos"]) == list(range(1, 11))


def test_train_model_requires_training_data():
    features = _features_frame(rounds=4)
    with pytest.raises(ValueError):
        predict_grid.train_model(features, target_round=1)
