"""Offline unit tests for pipelines/predict_fantasy.py: fantasy scoring calculation,
team knapsack optimizer, feature engineering (the one-round shift), usable
features guard, end-to-end train/predict cycles, strategic archetypes, and
the fitted-model cache thread-through.
"""

import numpy as np
import pandas as pd
import pytest
from sklearn.ensemble import HistGradientBoostingRegressor

import f1_common
import predict_fantasy


def _sample_race_df():
    """Synthetic completed round with 4 drivers across 2 teams."""
    return pd.DataFrame([
        {
            "round": 1,
            "event": "Test GP",
            "driver_number": "1",
            "driver": "VER",
            "team": "Red Bull",
            "grid": 1.0,
            "quali_pos": 1.0,
            "finish": 1.0,
            "points": 25.0,
            "status": "Finished",
            "sprint_finish": np.nan,
            "sprint_gain": np.nan,
        },
        {
            "round": 1,
            "event": "Test GP",
            "driver_number": "11",
            "driver": "PER",
            "team": "Red Bull",
            "grid": 4.0,
            "quali_pos": 4.0,
            "finish": 2.0,
            "points": 18.0,
            "status": "Finished",
            "sprint_finish": np.nan,
            "sprint_gain": np.nan,
        },
        {
            "round": 1,
            "event": "Test GP",
            "driver_number": "4",
            "driver": "NOR",
            "team": "McLaren",
            "grid": 2.0,
            "quali_pos": 2.0,
            "finish": 3.0,
            "points": 15.0,
            "status": "Finished",
            "sprint_finish": np.nan,
            "sprint_gain": np.nan,
        },
        {
            "round": 1,
            "event": "Test GP",
            "driver_number": "81",
            "driver": "PIA",
            "team": "McLaren",
            "grid": 3.0,
            "quali_pos": 3.0,
            "finish": np.nan,
            "points": 0.0,
            "status": "Retired",
            "sprint_finish": np.nan,
            "sprint_gain": np.nan,
        },
    ])


def test_calculate_driver_fantasy_points():
    df = _sample_race_df()
    pts = predict_fantasy.calculate_driver_fantasy_points(df)

    # VER: Quali P1 (10) + beat teammate in Q (2) + Race P1 (25) + gain 0 + classified (1) + beat teammate in R (3) = 41
    assert pts.iloc[0] == 41.0

    # PER: Quali P4 (7) + Race P2 (18) + gain (4 - 2 = +2) + classified (1) = 28
    assert pts.iloc[1] == 28.0

    # NOR: Quali P2 (9) + beat teammate in Q (2) + Race P3 (15) + gain (2 - 3 = -1) + classified (1) + beat teammate (3) = 29
    assert pts.iloc[2] == 29.0

    # PIA: Quali P3 (8) + Retired (-20) = -12
    assert pts.iloc[3] == -12.0


def test_calculate_constructor_fantasy_points():
    df = _sample_race_df()
    df["fantasy_points"] = predict_fantasy.calculate_driver_fantasy_points(df)
    c_df = predict_fantasy.calculate_constructor_fantasy_points(df)

    # Red Bull: drivers sum (41 + 28 = 69) + Both in Q3 (10) + Pit bonus (5) = 84
    rb = c_df[c_df["team"] == "Red Bull"].iloc[0]
    assert rb["constructor_points"] == 84.0

    # McLaren: drivers sum (29 + -12 = 17) + Both in Q3 (10) + Pit bonus (5) = 32
    mcl = c_df[c_df["team"] == "McLaren"].iloc[0]
    assert mcl["constructor_points"] == 32.0


def test_get_prices_and_fallbacks():
    drivers = ["VER", "NOR", "UNKNOWN_DRIVER"]
    constructors = ["Red Bull", "McLaren", "UNKNOWN_TEAM"]
    d_p, c_p = predict_fantasy.get_prices(2024, drivers, constructors)

    assert d_p["VER"] == 30.0
    assert d_p["NOR"] == 28.0
    assert 4.5 <= d_p["UNKNOWN_DRIVER"] <= 30.0

    assert c_p["Red Bull"] == 28.0
    assert c_p["McLaren"] == 29.0
    assert 5.0 <= c_p["UNKNOWN_TEAM"] <= 30.0


def test_optimize_team_budget_and_drs():
    # 6 drivers and 3 constructors
    d_df = pd.DataFrame([
        {"driver": "D1", "team": "T1", "pred_points": 30.0, "actual_points": 32.0},
        {"driver": "D2", "team": "T1", "pred_points": 25.0, "actual_points": 24.0},
        {"driver": "D3", "team": "T2", "pred_points": 20.0, "actual_points": 20.0},
        {"driver": "D4", "team": "T2", "pred_points": 15.0, "actual_points": 14.0},
        {"driver": "D5", "team": "T3", "pred_points": 10.0, "actual_points": 10.0},
        {"driver": "D6", "team": "T3", "pred_points": 5.0, "actual_points": 5.0},
    ])
    c_df = pd.DataFrame([
        {"team": "T1", "pred_points": 50.0, "actual_points": 50.0},
        {"team": "T2", "pred_points": 35.0, "actual_points": 35.0},
        {"team": "T3", "pred_points": 15.0, "actual_points": 15.0},
    ])

    d_prices = {"D1": 25.0, "D2": 20.0, "D3": 15.0, "D4": 10.0, "D5": 8.0, "D6": 5.0}
    c_prices = {"T1": 25.0, "T2": 15.0, "T3": 10.0}

    budget = 100.0
    res = predict_fantasy.optimize_team(d_df, c_df, d_prices, c_prices, budget=budget, top_k=3)

    opt = res["optimal"]
    assert len(opt["drivers"]) == 5
    assert len(opt["constructors"]) == 2
    assert opt["cost"] <= budget
    assert opt["remaining"] >= 0.0

    # DRS boost must be the highest predicted driver in the chosen 5
    chosen_names = [d["name"] for d in opt["drivers"]]
    expected_drs = max(opt["drivers"], key=lambda d: d["points"])["name"]
    assert opt["drs"] == expected_drs

    # Score calculation check: sum of 5 drivers + 2 constructors + DRS boost driver
    expected_score = sum(d["points"] for d in opt["drivers"]) + sum(c["points"] for c in opt["constructors"]) + max(d["points"] for d in opt["drivers"])
    assert opt["points"] == pytest.approx(expected_score)

    # Strategic archetypes present
    assert "balanced" in res["archetypes"]
    assert "mega_constructors" in res["archetypes"]
    assert "mega_drivers" in res["archetypes"]
    assert "max_ppm" in res["archetypes"]


def test_optimizer_never_returns_over_budget():
    drivers = pd.DataFrame([
        {"driver": f"D{i}", "team": f"T{i % 5}", "pred_points": 1.0}
        for i in range(5)
    ])
    constructors = pd.DataFrame([
        {"team": "T1", "pred_points": 1.0},
        {"team": "T2", "pred_points": 1.0},
    ])
    with pytest.raises(ValueError, match="no valid fantasy roster"):
        predict_fantasy.optimize_team(
            drivers, constructors,
            {f"D{i}": 10.0 for i in range(5)}, {"T1": 10.0, "T2": 10.0},
            budget=69.0,
        )


def _synthetic_season(rounds=4, per_round=10):
    rows = []
    rng = np.random.RandomState(42)
    for rnd in range(1, rounds + 1):
        for i in range(per_round):
            rows.append({
                "round": rnd,
                "event": f"Round {rnd}",
                "driver_number": str(i + 1),
                "driver": f"D{i+1}",
                "team": f"Team {i % 5 + 1}",
                "grid": float(i + 1),
                "quali_pos": float(i + 1),
                "quali_delta": float(i * 0.1),
                "finish": float((i + rnd) % per_round + 1),
                "points": float(max(0, 25 - i * 2)),
                "status": "Finished" if (i + rnd) % 6 != 0 else "Retired",
                "fp_race_pace_delta": rng.rand(),
                "fp_best_delta": rng.rand(),
                "fp_laps": float(rng.randint(15, 30)),
                "sprint_quali_pos": np.nan,
                "sprint_finish": np.nan,
                "sprint_gain": np.nan,
            })
    return pd.DataFrame(rows)


def test_build_features_shift_and_leakage():
    raw = _synthetic_season(rounds=3)
    features = predict_fantasy.build_features(raw, year=2024)

    r1 = features[features["round"] == 1]
    r2 = features[features["round"] == 2]

    # In round 1, shifted fantasy form features must be NaN
    assert r1["driver_fantasy_last"].isna().all()
    assert r1["driver_fantasy_form_3"].isna().all()
    assert r1["driver_fantasy_form_season"].isna().all()

    # In round 2, form must match round 1's ground truth fantasy points
    r1_idx = r1.set_index("driver_number")["fantasy_points"]
    r2_idx = r2.set_index("driver_number")["driver_fantasy_last"]
    for num in r1_idx.index:
        assert r2_idx.loc[num] == r1_idx.loc[num]

    # Both drivers of a team must see the same prior team history.  In
    # particular, the second row in a round must not see that round's team
    # aggregate through a raw-row shift.
    r2_team = features[features["round"] == 2].groupby("team")
    for _team, group in r2_team:
        assert group["team_fantasy_form_season"].nunique(dropna=False) == 1


def test_train_and_predict_cycle():
    raw = _synthetic_season(rounds=4)
    features = predict_fantasy.build_features(raw, year=2024)

    # Train delta model
    m_delta, tr_delta, used_delta, _ = predict_fantasy.train_model(features, target_round=4, mode="delta")
    assert isinstance(m_delta, HistGradientBoostingRegressor)
    d_delta = predict_fantasy.predict_drivers(m_delta, features, target_round=4, mode="delta", use_features=used_delta)
    assert len(d_delta) == 10
    assert "pred_points" in d_delta.columns

    # Train direct model
    m_direct, tr_direct, used_direct, _ = predict_fantasy.train_model(features, target_round=4, mode="direct")
    assert isinstance(m_direct, HistGradientBoostingRegressor)
    d_direct = predict_fantasy.predict_drivers(m_direct, features, target_round=4, mode="direct", use_features=used_direct)
    assert len(d_direct) == 10
    assert "pred_points" in d_direct.columns

    # Constructor prediction
    c_pred = predict_fantasy.predict_constructors_from_drivers(d_delta)
    assert len(c_pred) == 5
    assert "pred_points" in c_pred.columns


def test_backtest_records_and_final_predictions():
    raw = _synthetic_season(rounds=4)
    features = predict_fantasy.build_features(raw, year=2024)

    records = predict_fantasy.backtest_records(features, min_train_rounds=1, year=2024, budget=100.0)
    assert len(records) == 3  # rounds 2, 3, 4
    for r in records:
        assert "model_pts" in r
        assert "base_pts" in r
        assert "dream_pts" in r
        assert "capture_pct" in r
        assert r["budget_used"] <= 100.0

    pred = predict_fantasy.final_predictions(features, target_round=4, year=2024, budget=100.0)
    assert "team_optimal" in pred
    assert "alternatives" in pred
    assert "archetypes" in pred
    assert "d_board" in pred
    assert "c_board" in pred
    assert pred["team_optimal"]["cost"] <= 100.0

    # Importance frame
    imp = predict_fantasy.importance_frame(pred, mode="direct")
    assert not imp.empty
    assert "feature" in imp.columns
    assert "importance" in imp.columns


def test_model_selection_trains_only_requested_lane():
    raw = _synthetic_season(rounds=4)
    features = predict_fantasy.build_features(raw, year=2024)
    pred = predict_fantasy.final_predictions(
        features, target_round=4, year=2024, budget=100.0, models="direct")
    assert pred["direct_model"] is not None
    assert pred["delta_model"] is None
    assert pred["main_mode"] == "direct"


def test_fitted_model_cache_thread_through(monkeypatch, tmp_path):
    monkeypatch.setattr(f1_common, "MODEL_CACHE_DIR", tmp_path / "models")
    raw = _synthetic_season(rounds=3)
    features = predict_fantasy.build_features(raw, year=2024)

    m1, _, used, _ = predict_fantasy.train_model(features, target_round=3, mode="direct")
    train = features[(features["round"] < 3) & features["fantasy_points"].notna()]
    X = train[used].to_numpy(dtype=float)

    fits = []
    original = HistGradientBoostingRegressor.fit
    monkeypatch.setattr(HistGradientBoostingRegressor, "fit",
                        lambda self, X, y: (fits.append(1), original(self, X, y))[1])

    m2, _, cached_used, _ = predict_fantasy.train_model(features, target_round=3, mode="direct")

    # Second call must return the cached fitted model without calling fit()
    assert fits == []
    assert cached_used == used
    assert np.allclose(m1.predict(X), m2.predict(X))
