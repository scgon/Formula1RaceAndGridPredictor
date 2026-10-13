"""Offline unit tests for pipelines/f1_common.py — verbosity gating,
UTC/schedule helpers, quali-lap extraction, the model factories (regressor
and classifier), the hyperparameter tuners, the fitted-model cache
(train_or_load), permutation importance and the season CSV cache.
Nothing here touches the network: collect_season runs against synthetic
schedules and stub loaders with DATA_DIR pointed at a tmp_path."""

import argparse
from datetime import timedelta

import joblib
import numpy as np
import pandas as pd
import pytest
from fastf1.exceptions import RateLimitExceededError
from sklearn.ensemble import HistGradientBoostingClassifier, HistGradientBoostingRegressor

import f1_common


# --- verbosity ---------------------------------------------------------------

def test_vprint_gated_on_verbosity(monkeypatch, capsys):
    monkeypatch.setattr(f1_common, "VERBOSITY", 1)
    f1_common.vprint(1, "at level")
    f1_common.vprint(2, "beyond")
    out = capsys.readouterr().out
    assert "at level" in out
    assert "beyond" not in out

    monkeypatch.setattr(f1_common, "VERBOSITY", 0)
    f1_common.vprint(1, "never")
    assert "never" not in capsys.readouterr().out

    monkeypatch.setattr(f1_common, "VERBOSITY", 2)
    f1_common.vprint(2, "verbose level")
    assert "verbose level" in capsys.readouterr().out


def test_verbosity_flags_map_to_levels(monkeypatch):
    monkeypatch.setattr(f1_common, "VERBOSITY", 9)

    def level(argv):
        parser = argparse.ArgumentParser()
        f1_common.add_verbosity_args(parser)
        f1_common.apply_verbosity(parser.parse_args(argv))
        return f1_common.VERBOSITY

    assert level([]) == 1
    assert level(["-q"]) == 0
    assert level(["--quiet"]) == 0
    assert level(["-v"]) == 2
    assert level(["--verbose"]) == 2


def test_verbosity_flags_are_mutually_exclusive():
    parser = argparse.ArgumentParser()
    f1_common.add_verbosity_args(parser)
    with pytest.raises(SystemExit):
        parser.parse_args(["-q", "-v"])


# --- small helpers -----------------------------------------------------------

def test_format_params():
    assert f1_common.format_params({}) == ""
    assert f1_common.format_params({"learning_rate": 0.08}) == "learning_rate=0.08"
    assert f1_common.format_params({"a": 1, "b": 2}) == "a=1, b=2"


def test_rank_corr():
    assert f1_common.rank_corr([1, 2, 3], [10, 20, 30]) == pytest.approx(1.0)
    assert f1_common.rank_corr([3, 2, 1], [10, 20, 30]) == pytest.approx(-1.0)
    assert np.isnan(f1_common.rank_corr([5, 5, 5], [1, 2, 3]))


def test_make_model_profiles():
    model = f1_common.make_model(categorical_features=[3])
    assert isinstance(model, HistGradientBoostingRegressor)
    assert model.random_state == 42
    assert model.categorical_features == [3]
    for key, value in f1_common.FAST_PARAMS.items():
        assert getattr(model, key) == value

    tuned = f1_common.make_model([3], profile="optimized",
                                 tuned_params={"learning_rate": 0.2, "max_iter": 400})
    assert tuned.learning_rate == 0.2
    assert tuned.max_iter == 400

    with pytest.raises(ValueError):
        f1_common.make_model([], profile="turbo")


# --- UTC/session helpers -------------------------------------------------------

def test_utc_now_is_naive_utc():
    now = f1_common.utc_now()
    assert now.tzinfo is None


def test_session_utc_finds_the_named_session():
    row = pd.Series({
        "Session1": "Practice 1", "Session1DateUtc": "2026-03-06 11:30:00",
        "Session2": "Qualifying", "Session2DateUtc": "2026-03-07 15:00:00",
        "Session3": "Race", "Session3DateUtc": "2026-03-08 14:00:00",
    })
    assert f1_common.session_utc(row, "Race") == pd.Timestamp("2026-03-08 14:00:00")
    assert f1_common.session_utc(row, "Qualifying") == pd.Timestamp("2026-03-07 15:00:00")
    assert f1_common.session_utc(row, "Practice 3") is None


def test_session_utc_handles_missing_and_nan_dates():
    assert f1_common.session_utc(pd.Series(dtype=object), "Race") is None
    row = pd.Series({"Session1": "Race", "Session1DateUtc": pd.NaT})
    assert f1_common.session_utc(row, "Race") is None


def _schedule_row(rn, name, race_utc):
    return {"RoundNumber": rn, "EventName": name,
            "Session1": "Race", "Session1DateUtc": race_utc}


def test_completed_rounds_respects_the_three_hour_buffer():
    now = f1_common.utc_now()
    schedule = pd.DataFrame([
        _schedule_row(1, "Long ago", now - timedelta(days=7)),
        _schedule_row(2, "Just started", now - timedelta(hours=1)),
        _schedule_row(3, "Still ahead", now + timedelta(days=7)),
    ])
    assert f1_common.completed_rounds(schedule) == [(1, "Long ago")]


# --- qualifying-lap extraction -------------------------------------------------

def _quali_results():
    return pd.DataFrame({
        "Q1": [pd.Timedelta("90.5s"), pd.Timedelta("91s"),
               pd.Timedelta("88s"), pd.NaT],
        "Q2": [pd.Timedelta("89.5s"), pd.NaT, pd.Timedelta("87.5s"), pd.NaT],
        "Q3": [pd.Timedelta("89s"), pd.NaT, pd.NaT, pd.NaT],
    }, index=["1", "44", "16", "99"])


def test_best_quali_seconds():
    qres = _quali_results()
    assert f1_common.best_quali_seconds(qres, "1") == pytest.approx(89.0)
    assert f1_common.best_quali_seconds(qres, "44") == pytest.approx(91.0)  # Q1 only
    assert np.isnan(f1_common.best_quali_seconds(qres, "10"))  # not on the grid
    assert np.isnan(f1_common.best_quali_seconds(qres, "99"))  # all sectors missing


def test_pole_seconds():
    assert f1_common.pole_seconds(_quali_results()) == pytest.approx(87.5)


# --- tuning and importance -------------------------------------------------------

def _synthetic_training_set(rounds=6, per_round=30, seed=0):
    rng = np.random.RandomState(seed)
    rounds_col = np.repeat(np.arange(1, rounds + 1), per_round)
    X = rng.rand(len(rounds_col), 4)
    y = 2 * X[:, 0] - X[:, 1] + rng.rand(len(rounds_col)) * 0.1
    return X, y, rounds_col


def test_tune_hyperparameters_needs_two_rounds():
    rng = np.random.RandomState(0)
    assert f1_common.tune_hyperparameters(rng.rand(20, 4), rng.rand(20),
                                           np.ones(20)) is None


def test_tune_hyperparameters_is_deterministic():
    X, y, rounds = _synthetic_training_set()
    first = f1_common.tune_hyperparameters(X, y, rounds, n_iter=3)
    second = f1_common.tune_hyperparameters(X, y, rounds, n_iter=3)
    assert first == second
    assert first is None or set(first) == set(f1_common.TUNED_SEARCH_SPACE)


def test_tune_hyperparameters_returns_params_without_improvement_margin(monkeypatch):
    # with no required margin, the search always adopts its best candidate
    # (the fast values themselves at worst — they are always a candidate)
    monkeypatch.setattr(f1_common, "TUNE_MIN_IMPROVEMENT", 0.0)
    X, y, rounds = _synthetic_training_set(rounds=4, seed=3)
    picked = f1_common.tune_hyperparameters(X, y, rounds, n_iter=3)
    assert isinstance(picked, dict)
    assert set(picked) == set(f1_common.TUNED_SEARCH_SPACE)


def test_importance_scores_dataframe():
    rng = np.random.RandomState(7)
    X = rng.rand(120, 3)
    y = 3 * X[:, 0] + rng.rand(120) * 0.1
    model = HistGradientBoostingRegressor(random_state=0).fit(X, y)
    frame = f1_common.importance_scores(model, ["alpha", "beta", "gamma"], X, y,
                                        n_repeats=2)
    assert list(frame.columns) == ["feature", "importance", "std"]
    assert len(frame) == 3
    assert list(frame["feature"]).count("alpha") == 1
    # ordered descending, with the informative feature on top
    assert frame["importance"].iloc[0] == frame["importance"].max()
    assert frame["feature"].iloc[0] == "alpha"


# --- fitted-model cache ---------------------------------------------------------

def test_model_cache_flags_map_to_enabled_state(monkeypatch):
    monkeypatch.setattr(f1_common, "MODEL_CACHE", False)

    def state(argv):
        parser = argparse.ArgumentParser()
        f1_common.add_model_cache_args(parser)
        f1_common.apply_model_cache(parser.parse_args(argv))
        return f1_common.MODEL_CACHE

    assert state([]) is True  # reusing fitted models is the default
    assert state(["--no-model-cache"]) is False


def test_train_or_load_fits_and_caches_under_fast_profile(monkeypatch, tmp_path):
    monkeypatch.setattr(f1_common, "MODEL_CACHE_DIR", tmp_path / "models")
    X, y, rounds = _synthetic_training_set()
    model, tuned = f1_common.train_or_load("race:gain", X, y, rounds)
    assert isinstance(model, HistGradientBoostingRegressor)
    assert tuned is None  # fast profile: fixed hyperparameters
    assert model.n_iter_ > 0  # fitted
    assert len(list((tmp_path / "models").glob("*.joblib"))) == 1


def test_train_or_load_reuses_the_cached_model_without_refitting(monkeypatch, tmp_path):
    monkeypatch.setattr(f1_common, "MODEL_CACHE_DIR", tmp_path / "models")
    X, y, rounds = _synthetic_training_set()
    fits = []
    original = HistGradientBoostingRegressor.fit

    def _counting_fit(self, X, y):
        fits.append(1)
        return original(self, X, y)

    monkeypatch.setattr(HistGradientBoostingRegressor, "fit", _counting_fit)
    first, _ = f1_common.train_or_load("race:gain", X, y, rounds)
    assert len(fits) == 1
    second, _ = f1_common.train_or_load("race:gain", X, y, rounds)
    assert len(fits) == 1  # identical re-run: served from the cache
    assert np.allclose(first.predict(X), second.predict(X))


def test_train_or_load_with_cache_disabled_retrains(monkeypatch, tmp_path):
    monkeypatch.setattr(f1_common, "MODEL_CACHE_DIR", tmp_path / "models")
    monkeypatch.setattr(f1_common, "MODEL_CACHE", False)
    X, y, rounds = _synthetic_training_set()
    fits = []
    original = HistGradientBoostingRegressor.fit

    def _counting_fit(self, X, y):
        fits.append(1)
        return original(self, X, y)

    monkeypatch.setattr(HistGradientBoostingRegressor, "fit", _counting_fit)
    f1_common.train_or_load("race:gain", X, y, rounds)
    f1_common.train_or_load("race:gain", X, y, rounds)
    assert len(fits) == 2
    assert not (tmp_path / "models").exists()  # nothing stored while disabled


def test_train_or_load_cache_keys_separate_lanes_and_data(monkeypatch, tmp_path):
    monkeypatch.setattr(f1_common, "MODEL_CACHE_DIR", tmp_path / "models")
    X, y, rounds = _synthetic_training_set()
    f1_common.train_or_load("race:gain", X, y, rounds)
    f1_common.train_or_load("race:direct", X, y, rounds)  # different lane
    f1_common.train_or_load("race:gain", X, y + 1.0, rounds)  # different data
    assert len(list((tmp_path / "models").glob("*.joblib"))) == 3


def test_train_or_load_classifier_reuses_the_cached_model(monkeypatch, tmp_path):
    monkeypatch.setattr(f1_common, "MODEL_CACHE_DIR", tmp_path / "models")
    X, _y, rounds = _synthetic_training_set()
    labels = (X[:, 0] > np.median(X[:, 0])).astype(int)
    fits = []
    original = HistGradientBoostingClassifier.fit

    def _counting_fit(self, X, y):
        fits.append(1)
        return original(self, X, y)

    monkeypatch.setattr(HistGradientBoostingClassifier, "fit", _counting_fit)
    first, tuned = f1_common.train_or_load("extras:pole", X, labels, rounds,
                                          classifier=True)
    assert isinstance(first, HistGradientBoostingClassifier)
    assert tuned is None
    assert list(first.classes_) == [0, 1]
    second, _ = f1_common.train_or_load("extras:pole", X, labels, rounds,
                                        classifier=True)
    assert len(fits) == 1  # cache hit, no refit
    assert np.allclose(first.predict_proba(X)[:, 1], second.predict_proba(X)[:, 1])


def test_train_or_load_optimized_profile_skips_the_tuner_on_a_hit(monkeypatch, tmp_path):
    monkeypatch.setattr(f1_common, "MODEL_CACHE_DIR", tmp_path / "models")
    X, y, rounds = _synthetic_training_set()
    calls = []
    real_tuner = f1_common.tune_hyperparameters

    def _spying_tuner(X, y, rounds, **kwargs):
        calls.append(1)
        return real_tuner(X, y, rounds, **kwargs)

    monkeypatch.setattr(f1_common, "tune_hyperparameters", _spying_tuner)
    first_model, first_tuned = f1_common.train_or_load(
        "race:gain", X, y, rounds, profile="optimized")
    assert len(calls) == 1
    assert first_tuned is None or set(first_tuned) == set(f1_common.TUNED_SEARCH_SPACE)
    second_model, second_tuned = f1_common.train_or_load(
        "race:gain", X, y, rounds, profile="optimized")
    assert len(calls) == 1  # the hyperparameter search is skipped on a hit
    assert second_tuned == first_tuned
    assert np.allclose(first_model.predict(X), second_model.predict(X))


def test_train_or_load_survives_a_corrupt_cache_entry(monkeypatch, tmp_path):
    monkeypatch.setattr(f1_common, "MODEL_CACHE_DIR", tmp_path / "models")
    monkeypatch.setattr(f1_common, "_model_cache_key", lambda *args: "poisoned")
    cache_dir = tmp_path / "models"
    cache_dir.mkdir()
    (cache_dir / "poisoned.joblib").write_bytes(b"not a joblib file")
    X, y, rounds = _synthetic_training_set()
    model, _tuned = f1_common.train_or_load("race:gain", X, y, rounds)
    assert model.n_iter_ > 0  # corrupt entry: retrained, not crashed
    # a loadable but wrongly-shaped entry is a miss too
    joblib.dump("nonsense", cache_dir / "poisoned.joblib")
    f1_common.train_or_load("race:gain", X, y, rounds)
    entry = joblib.load(cache_dir / "poisoned.joblib")
    assert set(entry) == {"model", "tuned"}  # replaced by the real entry


def test_train_or_load_ignores_an_unwritable_cache_dir(monkeypatch, tmp_path):
    blocker = tmp_path / "blocker"
    blocker.write_text("a regular file")
    monkeypatch.setattr(f1_common, "MODEL_CACHE_DIR", blocker / "models")
    X, y, rounds = _synthetic_training_set()
    model, _tuned = f1_common.train_or_load("race:gain", X, y, rounds)
    assert model.n_iter_ > 0  # the cache must never fail a run


# --- the season CSV cache ---------------------------------------------------------

REQUIRED = ["round", "driver_number", "driver", "finish"]


def _loader_frame(rn):
    return pd.DataFrame({
        "round": [rn, rn], "driver_number": ["1", "2"], "driver": ["A", "B"],
        "finish": [1.0, 2.0],
    })


def test_collect_season_reuses_cached_rounds_and_appends_new_ones(monkeypatch, tmp_path):
    monkeypatch.setattr(f1_common, "DATA_DIR", tmp_path)
    now = f1_common.utc_now()
    schedule = pd.DataFrame([
        _schedule_row(1, "One", now - timedelta(days=14)),
        _schedule_row(2, "Two", now - timedelta(days=7)),
    ])
    pd.DataFrame({
        "round": [1, 1], "driver_number": ["1", "2"], "driver": ["A", "B"],
        "finish": [2.0, 1.0], "stale_extra": ["x", "y"],
    }).to_csv(tmp_path / "season_2020.csv", index=False)

    calls = []

    def _load(year, rn, name):
        calls.append((year, rn, name))
        return _loader_frame(rn)

    data = f1_common.collect_season(
        2020, schedule, filename="season_{year}.csv",
        required_columns=REQUIRED, result_column="finish", load_round=_load)

    assert calls == [(2020, 2, "Two")]  # round 1 came from the cached CSV
    assert sorted(data["round"].unique()) == [1, 2]
    assert len(data) == 4  # both rounds, no duplicates
    saved = pd.read_csv(tmp_path / "season_2020.csv")
    assert sorted(saved["round"].unique()) == [1, 2]


def test_collect_season_skips_rounds_without_classified_results(monkeypatch, tmp_path):
    monkeypatch.setattr(f1_common, "DATA_DIR", tmp_path)
    now = f1_common.utc_now()
    schedule = pd.DataFrame([_schedule_row(1, "One", now - timedelta(days=7))])

    def _empty(year, rn, name):
        frame = _loader_frame(rn)
        frame["finish"] = np.nan  # race still in progress, nothing classified
        return frame

    data = f1_common.collect_season(
        2020, schedule, filename="season_{year}.csv",
        required_columns=REQUIRED, result_column="finish", load_round=_empty)
    assert data.empty
    assert not (tmp_path / "season_2020.csv").exists()  # nothing worth saving


def test_collect_season_with_nothing_completed(monkeypatch, tmp_path):
    monkeypatch.setattr(f1_common, "DATA_DIR", tmp_path)
    future = pd.DataFrame([_schedule_row(1, "Later",
                                         f1_common.utc_now() + timedelta(days=7))])

    def _boom(*args):
        raise AssertionError("load_round must not be called for future rounds")

    data = f1_common.collect_season(
        2020, future, filename="season_{year}.csv",
        required_columns=REQUIRED, result_column="finish", load_round=_boom)
    assert data.empty
    assert list(data.columns) == REQUIRED


def test_collect_season_stops_soft_on_the_rate_limit(monkeypatch, tmp_path, capsys):
    monkeypatch.setattr(f1_common, "DATA_DIR", tmp_path)
    now = f1_common.utc_now()
    schedule = pd.DataFrame([
        _schedule_row(1, "One", now - timedelta(days=14)),
        _schedule_row(2, "Two", now - timedelta(days=7)),
    ])
    calls = []

    def _limited(year, rn, name):
        calls.append(rn)
        raise RateLimitExceededError("rate limit reached")

    data = f1_common.collect_season(
        2020, schedule, filename="season_{year}.csv",
        required_columns=REQUIRED, result_column="finish", load_round=_limited)
    assert calls == [1]  # the first hit stops the crawl instead of retrying
    assert data.empty
    assert "rate limit" in capsys.readouterr().out.lower()
