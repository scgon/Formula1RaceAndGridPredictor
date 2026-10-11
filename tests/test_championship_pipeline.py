"""Offline unit tests for pipelines/predict_championship.py: the
race+sprint points derivation, the snapshot feature build (inclusive of the
round itself), the final-points targets of complete seasons, the
leave-one-season-out training rule, and an end-to-end train/predict cycle
on synthetic data."""

import numpy as np
import pandas as pd
import pytest

import predict_championship as pc


def _raw_season(rounds=3, drivers=("A", "B", "C", "D"), sprint_rounds=()):
    """Race-CSV-shaped rows: one season, two teams, with sprint results on
    the given rounds (sprint points follow the 2022+ 8-...-1 scheme)."""
    rows = []
    for rnd in range(1, rounds + 1):
        for i, driver in enumerate(drivers):
            finish = float((rnd + i) % len(drivers) + 1)
            rows.append({
                "round": rnd,
                "event": f"Round {rnd}",
                "driver_number": str(i + 1),
                "driver": driver,
                "team": ["Team A", "Team B"][i % 2],
                "grid": float(i + 1),
                "finish": finish,
                "points": float(max(0, 13 - 3 * i)) if i < 4 else 0.0,
                "quali_delta": 0.1 * i,
                "status": "Finished",
                "sprint_finish": float(i + 1) if rnd in sprint_rounds else np.nan,
            })
    return pd.DataFrame(rows)


def _seasons(*specs):
    """[{"year", "data", "rounds_total", "complete"}] helper."""
    return [{"year": year, "data": _raw_season(rounds=rounds, sprint_rounds=sprints),
             "rounds_total": rounds, "complete": complete}
            for year, rounds, sprints, complete in specs]


def test_sprint_points_map_per_era():
    assert pc.sprint_points_map(2021) == {1: 3.0, 2: 2.0, 3: 1.0}
    scheme = pc.sprint_points_map(2024)
    assert scheme[1] == 8.0 and scheme[8] == 1.0 and 9 not in scheme
    assert pc.sprint_points_map(2018) == pc.sprint_points_map(2026)  # same scheme


def test_round_points_include_derived_sprint_points():
    data = _raw_season(rounds=1, sprint_rounds=(1,))
    snap = pc.build_snapshots(data, 2024, 1, "drivers")
    by_driver = snap.set_index("driver")
    # round-1: race points 13/10/7/4 + sprint points 8/7/6/5
    assert by_driver.loc["A", "points_so_far"] == pytest.approx(21.0)
    assert by_driver.loc["B", "points_so_far"] == pytest.approx(17.0)
    assert by_driver.loc["D", "points_so_far"] == pytest.approx(9.0)


def test_snapshots_are_inclusive_of_their_round():
    features = pc.build_features(_seasons((2024, 3, (), True)), "drivers")
    first = features[features["round"] == 1].set_index("driver")
    second = features[features["round"] == 2].set_index("driver")
    # a snapshot at round r already includes that round's points (unlike the
    # round pipelines' shift(1) "before this round" features)
    assert first.loc["A", "points_so_far"] == pytest.approx(first.loc["A", "round_points"])
    assert second.loc["A", "points_so_far"] == pytest.approx(
        first.loc["A", "points_so_far"] + second.loc["A", "round_points"])
    assert second.loc["A", "points_per_round"] == pytest.approx(
        second.loc["A", "points_so_far"] / 2)
    assert (features["rounds_remaining"] == features["year"].map(
        {2024: 3}) - features["round"]).all()


def test_final_points_only_for_complete_seasons():
    features = pc.build_features(
        _seasons((2023, 2, (), True), (2024, 2, (), False)), "drivers")
    assert features.loc[features["year"] == 2023, "final"].notna().all()
    assert features.loc[features["year"] == 2024, "final"].isna().all()
    # the 2023 finals are the last snapshot's points-so-far, broadcast to all rows
    finals = features[features["year"] == 2023].groupby("driver")["final"].first()
    last = features[(features["year"] == 2023) & (features["round"] == 2)].set_index("driver")
    for driver, final in finals.items():
        assert final == last.loc[driver, "points_so_far"]


def test_standings_and_gap_features():
    features = pc.build_features(_seasons((2024, 2, (), True)), "drivers")
    snap = features[features["round"] == 2]
    top = snap.loc[snap["points_so_far"].idxmax()]
    assert top["standings_pos"] == 1
    assert top["gap_to_leader"] == pytest.approx(0.0)
    assert top["leader_points"] == pytest.approx(top["points_so_far"])
    others = snap[snap["driver"] != top["driver"]]
    assert (others["gap_to_leader"] > 0).all()
    assert (others["standings_pos"] > 1).all()


def test_usable_features_drops_all_nan_columns():
    features = pc.build_features(_seasons((2024, 3, (), True)), "drivers")
    features["points_last3_rate"] = np.nan  # simulate an entirely empty column
    used = pc.usable_features(features, pc.CHAMPIONSHIPS["drivers"]["features"])
    assert "points_last3_rate" not in used
    assert "points_so_far" in used


def test_train_uses_only_earlier_seasons_up_to_the_snapshot_round():
    features = pc.build_features(
        _seasons((2022, 3, (), True), (2023, 3, (), True), (2024, 3, (), True)),
        "drivers")
    model, train, used, tuned = pc.train_model(features, 2024, 2, "remaining")
    assert tuned is None  # fast profile: fixed hyperparameters
    assert sorted(train["year"].unique()) == [2022, 2023]
    assert train["round"].max() == 2  # no snapshot later than the forecast round
    assert train["year"].max() < 2024  # the forecast season never trains its own model


def test_train_and_predict_cycle_remaining_mode():
    features = pc.build_features(
        _seasons((2022, 3, (), True), (2023, 3, (), True), (2024, 3, (), True)),
        "drivers")
    model, _, used, _ = pc.train_model(features, 2024, 2, "remaining")
    predicted = pc.predict_snapshot(model, features, 2024, 2, "remaining", used)
    assert len(predicted) == 4
    assert sorted(predicted["pred_pos"]) == [1, 2, 3, 4]
    # remaining mode anchors the projection at the current points and can
    # never project a total below what is already scored
    assert np.allclose(predicted["pred_final"],
                       predicted["points_so_far"] + np.clip(model.predict(
                           predicted[used].to_numpy(dtype=float)), 0.0, None))
    assert (predicted["pred_final"] >= predicted["points_so_far"]).all()


def test_direct_mode_clamps_at_current_points():
    features = pc.build_features(
        _seasons((2022, 3, (), True), (2023, 3, (), True), (2024, 3, (), True)),
        "drivers")
    model, _, used, _ = pc.train_model(features, 2024, 2, "direct")
    predicted = pc.predict_snapshot(model, features, 2024, 2, "direct", used)
    assert (predicted["pred_final"] >= predicted["points_so_far"]).all()


def test_baseline_projection_is_the_run_rate():
    features = pc.build_features(_seasons((2024, 3, (), True)), "drivers")
    base = pc.baseline_projection(features, 2024, 2, "drivers")
    expected = base["points_so_far"] + base["points_per_round"] * base["rounds_remaining"]
    assert np.allclose(base["pred_final"], expected)
    assert sorted(base["pred_pos"]) == [1, 2, 3, 4]


def test_backtest_records_score_against_final_standings():
    features = pc.build_features(
        _seasons((2022, 3, (), True), (2023, 3, (), True), (2024, 3, (), True)),
        "drivers")
    records = pc.backtest_records(features, min_train_rounds=1, models="both")
    # min_train_rounds=1 -> first forecast snapshot is round 2; the season's
    # final round (3) is skipped
    assert {rec["round"] for rec in records} == {2}
    assert {rec["season"] for rec in records} == {2023, 2024}
    rec = records[-1]
    assert rec["train_seasons"] == 2  # 2024 forecast trains on 2022+2023
    for field in ("remaining", "direct", "baseline"):
        assert rec[f"{field}_mae"] >= 0
        assert rec[f"{field}_champion"] in (0, 1)
        assert 0 <= rec[f"{field}_podium"] <= 3
        assert rec[f"{field}_top1"] in set(features["driver"])
    assert rec["actual_top1"] == features.loc[
        (features["year"] == 2024) & (features["round"] == 3) &
        (features["final"] == features.loc[(features["year"] == 2024) &
                                           (features["round"] == 3), "final"].max()),
        "driver"].iloc[0]
    # the earliest season cannot be backtested (nothing earlier to train on)
    assert all(rec["season"] != 2022 for rec in records)


def test_final_predictions_bundle_and_error():
    features = pc.build_features(
        _seasons((2022, 3, (), True), (2023, 3, (), True), (2024, 3, (), True)),
        "drivers")
    pred = pc.final_predictions(features, 2024, 2)
    assert "remaining" in pred and "direct" in pred and "baseline" in pred
    assert pred["train_seasons"] == 2
    assert "direct_pos" in pred["remaining"].columns
    assert "direct_final" in pred["remaining"].columns
    assert pc.importance_frame(pred, "remaining").columns.tolist() == [
        "feature", "importance", "std"]
    # forecasting the earliest bundled season has no training data
    oldest = pc.final_predictions(features, 2022, 2)
    assert "error" in oldest
    assert pc.importance_frame(oldest).empty


def test_constructors_championship_end_to_end():
    features = pc.build_features(
        _seasons((2022, 3, (), True), (2023, 3, (), True), (2024, 3, (), True)),
        "constructors")
    assert set(pc.CHAMPIONSHIPS["constructors"]["features"]) <= set(features.columns)
    # two teams -> four driver rows collapse to two team snapshots per round
    assert len(features) == 3 * 3 * 2
    pred = pc.final_predictions(features, 2024, 2, championship="constructors")
    assert "remaining" in pred
    assert len(pred["remaining"]) == 2
    # the entrant column is the team itself
    assert set(pred["remaining"]["team"]) == {"Team A", "Team B"}


def test_resolve_target_gate_and_modes(monkeypatch):
    data6 = _raw_season(rounds=6)
    data8 = _raw_season(rounds=8)
    schedule = pd.DataFrame({"RoundNumber": range(1, 9)})

    class FakeCompleted:
        def __init__(self, rounds):
            self.rounds = rounds

        def __call__(self, sched):
            return [(rn, f"Round {rn}") for rn in self.rounds]

    # 6 of the calendar's 8 rounds completed -> live mode, auto snapshot = last
    monkeypatch.setattr(pc.common, "completed_rounds", FakeCompleted(range(1, 7)))
    args = type("Args", (), {})()
    snapshot, mode, event = pc.resolve_target(args, 2026, schedule, data6)
    assert (snapshot, mode, event) == (6, "live", "Round 6")

    # auto snapshot needs min_train completed rounds before it
    args.min_train_rounds = 6
    with pytest.raises(SystemExit):
        pc.resolve_target(args, 2026, schedule, data6)
    args.min_train_rounds = 5
    snapshot, mode, _ = pc.resolve_target(args, 2026, schedule, data6)
    assert snapshot == 6

    # an explicit round missing from the data (not completed) is refused
    args.predict_round = 7
    with pytest.raises(SystemExit):
        pc.resolve_target(args, 2026, schedule, data6)

    # a season whose whole calendar ran is post mode
    monkeypatch.setattr(pc.common, "completed_rounds", FakeCompleted(range(1, 9)))
    args.predict_round = None
    _, mode, _ = pc.resolve_target(args, 2026, schedule, data8)
    assert mode == "post"


def test_season_is_complete_requires_final_round_in_data(monkeypatch):
    schedule = pd.DataFrame({"RoundNumber": [1, 2, 3]})
    monkeypatch.setattr(pc.common, "completed_rounds",
                        lambda sched: [(1, "a"), (2, "b"), (3, "c")])
    assert pc.season_is_complete(2024, schedule, _raw_season(rounds=3))
    # final round completed but not yet in the CSV: not complete
    assert not pc.season_is_complete(2024, schedule, _raw_season(rounds=2))
    # round 3 not completed yet: not complete
    monkeypatch.setattr(pc.common, "completed_rounds",
                        lambda sched: [(1, "a"), (2, "b")])
    assert not pc.season_is_complete(2024, schedule, _raw_season(rounds=3))
