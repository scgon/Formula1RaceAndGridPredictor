"""Shared helpers for the F1 prediction pipelines.

predict_race.py, predict_grid.py and predict_extras.py remain independent
entry points with their own features, targets and models; this module holds
the machinery they share: fastf1 cache setup, UTC handling, qualifying-lap
extraction, practice-lap features, the season CSV cache, the model factory,
permutation importance and the CLI verbosity gating (-q/--quiet, -v/--verbose;
the web app runs at the default level). The Streamlit app (app.py) builds on
the same functions.
"""

import re
import warnings
import time
from datetime import datetime, timedelta, timezone
from itertools import product
from pathlib import Path

import numpy as np
import pandas as pd
import fastf1
from fastf1.exceptions import RateLimitExceededError
from sklearn.ensemble import HistGradientBoostingRegressor
from sklearn.inspection import permutation_importance

warnings.filterwarnings("ignore")

# Repo root (this file lives in pipelines/); cache/ and data/ stay at the root
# so pipelines, web app and notebooks all share them.
BASE_DIR = Path(__file__).resolve().parent.parent
CACHE_DIR = BASE_DIR / "cache"
DATA_DIR = BASE_DIR / "data"

# Bundled team colors (year, team, color) kept current by the refresh
# workflow — what the web app renders from so a cold container never needs
# a live fastf1 session for colors (see season_team_colors).
TEAM_COLORS_CSV = DATA_DIR / "team_colors.csv"

MIN_TRAIN_ROUNDS = 5
STINT_WINDOW = 5
STINT_MIN_LAPS = 3
COMPLETION_BUFFER = timedelta(hours=3)

# fastf1 hard-stops at 500 uncached API requests per hour (500 calls/hour
# against Ergast/Jolpica). collect_season treats that limit as a soft stop:
# with rate_limit_wait=None it saves what it fetched and leaves the rest for
# the next run; with a wait interval it sleeps and retries instead. The wait
# cap bounds the patience of a single call (8 * 10 min ≈ one full rate-limit
# window) before it falls back to the save-and-stop behavior.
RATE_LIMIT_WAIT_SECS = 600
RATE_LIMIT_MAX_WAITS = 8

# --- verbosity ---------------------------------------------------------------

#: CLI output level, selected with the shared -v/--verbose and -q/--quiet
#: flags (add_verbosity_args / apply_verbosity). The web app never touches
#: it, so it always runs at the default:
#:   0 quiet   backtest summary + final prediction only (no progress lines,
#:             per-round backtest table or feature importance)
#:   1 default today's full output
#:   2 verbose  default plus live per-round backtest metrics,
#:             hyperparameter-tuner decisions and importance spread
VERBOSITY = 1


def set_verbosity(level):
    global VERBOSITY
    VERBOSITY = int(level)


def vprint(level, *args, **kwargs):
    """print() gated on the current VERBOSITY (prints when VERBOSITY >= level)."""
    if VERBOSITY >= level:
        print(*args, **kwargs)


def add_verbosity_args(parser):
    """The -v/--verbose and -q/--quiet flags shared by the pipeline CLIs;
    call apply_verbosity(args) right after parse_args."""
    group = parser.add_mutually_exclusive_group()
    group.add_argument("-q", "--quiet", action="store_true",
                       help="print less: only the backtest summary and the final "
                            "prediction (no progress lines, per-round backtest table "
                            "or feature importance)")
    group.add_argument("-v", "--verbose", action="store_true",
                       help="print more: live per-round backtest metrics, "
                            "hyperparameter-tuner decisions and importance spread")


def apply_verbosity(args):
    set_verbosity(2 if getattr(args, "verbose", False)
                  else 0 if getattr(args, "quiet", False) else 1)

# --- model profiles ---------------------------------------------------------

#: Selectable via the CLI `--model` flag and the web-app sidebar.
#:   fast      — fixed FAST_PARAMS hyperparameters; quickest to run.
#:   optimized — per-fit hyperparameter search (tune_hyperparameters) that
#:               minimizes cross-validated MAE over the training rounds;
#:               slower to run, usually lower MAE.
MODEL_PROFILES = ("fast", "optimized")

FAST_PARAMS = {
    "loss": "absolute_error",
    "learning_rate": 0.08,
    "max_iter": 150,
    "max_leaf_nodes": 15,
    "min_samples_leaf": 20,
    "l2_regularization": 1.0,
}

# Hyperparameter candidates sampled by the optimized profile. loss
# (absolute_error) and the categorical features are not searched.
TUNED_SEARCH_SPACE = {
    "learning_rate": [0.03, 0.06, 0.08, 0.12, 0.2],
    "max_iter": [100, 200, 400],
    "max_leaf_nodes": [3, 7, 15, 31],
    "min_samples_leaf": [5, 10, 20, 40],
    "l2_regularization": [0.0, 0.25, 1.0, 4.0],
}
TUNE_N_ITER = 20
TUNE_MAX_SPLITS = None
TUNE_MIN_IMPROVEMENT = 0.05
TUNE_STRIDE = 2
TUNE_RANDOM_STATE = 42


def setup():
    CACHE_DIR.mkdir(exist_ok=True)
    DATA_DIR.mkdir(exist_ok=True)
    fastf1.Cache.enable_cache(str(CACHE_DIR))
    try:
        fastf1.logger.set_log_level("WARNING")
    except Exception:
        pass


def utc_now():
    return pd.Timestamp(datetime.now(timezone.utc)).replace(tzinfo=None)


def session_utc(row, name):
    for i in range(1, 6):
        if row.get(f"Session{i}") == name:
            value = row.get(f"Session{i}DateUtc")
            if pd.isna(value):
                return None
            return pd.Timestamp(value)
    return None


def completed_rounds(schedule):
    """(round_number, event_name) pairs whose race started >COMPLETION_BUFFER ago."""
    now = utc_now()
    rounds = []
    for _, ev in schedule.iterrows():
        race_utc = session_utc(ev, "Race")
        if race_utc is not None and race_utc < now - COMPLETION_BUFFER:
            rounds.append((int(ev["RoundNumber"]), str(ev["EventName"])))
    return rounds


def season_team_colors(year, schedule=None):
    """{team name -> hex color} for one season, taken straight from the
    latest completed round's race results (the same source the notebooks
    use). Needs fastf1 session data: with a cold cache this downloads a
    race session — which is exactly what a fresh app container cannot
    afford once the API rate limit is exhausted — so the web app renders
    from the bundled data/team_colors.csv (written from this function by
    scripts/refresh_data.py) and calls this only for years the bundle does
    not cover.

    Only well-formed hex values are accepted: fastf1 session results can
    carry glitched TeamColor cells (e.g. the literal string 'nan' for Haas
    in Abu Dhabi 2021), which once bundled as '#nan' and rendered that
    team gray everywhere."""
    if schedule is None:
        schedule = fastf1.get_event_schedule(year, include_testing=False)
    for rn, _name in reversed(completed_rounds(schedule)):
        try:
            session = fastf1.get_session(year, rn, "R")
            session.load(laps=False, telemetry=False, weather=False, messages=False)
            colors = {t: "#" + c for t, c in
                      zip(session.results["TeamName"], session.results["TeamColor"])
                      if re.fullmatch(r"[0-9a-fA-F]{6}", str(c))}
            if colors:
                return colors
        except Exception:
            continue
    return {}


def best_quali_seconds(qres, num):
    if num not in qres.index:
        return np.nan
    times = []
    for col in ("Q1", "Q2", "Q3"):
        if col in qres.columns:
            value = qres.at[num, col]
            if pd.notna(value):
                times.append(pd.Timedelta(value).total_seconds())
    return min(times) if times else np.nan


def pole_seconds(qres):
    times = [best_quali_seconds(qres, num) for num in qres.index]
    times = [t for t in times if not np.isnan(t)]
    return min(times) if times else np.nan


def clean_practice_laps(session):
    laps = session.laps
    if laps is None or laps.empty:
        return pd.DataFrame()
    clean = laps[
        laps["LapTime"].notna()
        & laps["IsAccurate"].fillna(False)
        & laps["PitInTime"].isna()
        & laps["PitOutTime"].isna()
        & (laps["TrackStatus"].astype(str) == "1")
    ][["DriverNumber", "Stint", "LapNumber", "LapTime"]].copy()
    if clean.empty:
        return clean
    clean["DriverNumber"] = clean["DriverNumber"].astype(str)
    clean["seconds"] = clean["LapTime"].dt.total_seconds()
    return clean


def practice_session_frame(session):
    clean = clean_practice_laps(session)
    if clean.empty:
        return None
    best_lap = clean.groupby("DriverNumber")["seconds"].min()
    stint_best = {}
    for (num, _), group in clean.groupby(["DriverNumber", "Stint"]):
        if len(group) < STINT_MIN_LAPS:
            continue
        group = group.sort_values("LapNumber")
        pace = group["seconds"].rolling(STINT_WINDOW, min_periods=STINT_MIN_LAPS).mean().min()
        stint_best[num] = min(stint_best.get(num, np.inf), pace)
    if len(stint_best) < 3:
        return None
    stint = pd.Series(stint_best)
    return pd.DataFrame({
        "fp_best_delta": best_lap - best_lap.min(),
        "fp_laps": clean.groupby("DriverNumber").size(),
        "fp_race_pace_delta": stint - stint.min(),
    })


def practice_features(year, round_number, sessions):
    """Aggregate practice pace features over the given session identifiers:
    ("FP1", "FP2", "FP3") for the race pipeline, ("FP1", "FP2") for the grid
    pipeline (day-before-quali constraint)."""
    frames = []
    for identifier in sessions:
        try:
            session = fastf1.get_session(year, round_number, identifier)
            session.load(laps=True, telemetry=False, weather=False, messages=False)
            frame = practice_session_frame(session)
        except Exception:
            frame = None
        if frame is not None:
            frames.append(frame)
    if not frames:
        return pd.DataFrame()
    combined = pd.concat(frames)
    return combined.groupby(level=0).agg(
        fp_race_pace_delta=("fp_race_pace_delta", "min"),
        fp_best_delta=("fp_best_delta", "min"),
        fp_laps=("fp_laps", "sum"),
    )


def collect_season(year, schedule, *, filename, required_columns, result_column,
                   load_round, refresh=False, rate_limit_wait=None):
    """Fetch completed rounds into the season CSV, reusing whatever is cached.

    filename         CSV name pattern, e.g. "season_{year}.csv"
    required_columns columns the cached CSV must contain to be reused
    result_column    column holding classified results ("finish" / "quali_pos")
    load_round       callable (year, round_number, event_name) -> DataFrame
    rate_limit_wait  seconds to sleep before retrying a round when the F1 API
                     rate limit (500 uncached calls/hour) is hit; None stops
                     after the limit instead, leaving the remaining rounds
                     for the next run (already fetched rounds stay in the CSV)
    """
    path = DATA_DIR / filename.format(year=year)
    completed = completed_rounds(schedule)

    cached = None
    if path.exists() and not refresh:
        candidate = pd.read_csv(path, dtype={"driver_number": str})
        if set(required_columns).issubset(candidate.columns):
            cached = candidate
        else:
            print("cached season file has an outdated schema, re-downloading")
    have = set(cached["round"].unique()) if cached is not None else set()

    todo = [(rn, name) for rn, name in completed if rn not in have]
    print(f"Season {year}: {len(completed)} completed rounds ({len(have)} cached, {len(todo)} to fetch)")

    frames = []
    rate_limited = False
    waits_left = RATE_LIMIT_MAX_WAITS
    for rn, name in todo:
        while True:
            vprint(1, f"  fetching round {rn:>2}  {name} ...")
            try:
                frame = load_round(year, rn, name)
                break
            except RateLimitExceededError:
                if rate_limit_wait is None or waits_left <= 0:
                    print(f"  round {rn:>2} ({name}) skipped: F1 API rate limit "
                          "reached (500 calls/hour)")
                    rate_limited = True
                    break
                waits_left -= 1
                print(f"  API rate limit reached (500 calls/hour) — waiting "
                      f"{rate_limit_wait // 60} min, then retrying round {rn:>2} ...")
                time.sleep(rate_limit_wait)
            except Exception as exc:
                print(f"  round {rn:>2} ({name}) skipped: {type(exc).__name__}: {exc}")
                frame = None
                break
        if rate_limited:
            break
        if frame is None:
            continue
        if frame[result_column].notna().sum() == 0:
            print(f"  round {rn:>2} ({name}) skipped: no classified results yet")
            continue
        vprint(1, f"  fetched round {rn:>2}  {name}")
        frames.append(frame)
    if rate_limited:
        print("  Rounds fetched so far are saved; the rest will download on the next run.")

    if cached is not None:
        frames.insert(0, cached)
    if not frames:
        return pd.DataFrame(columns=required_columns)

    data = pd.concat(frames, ignore_index=True)
    data = data.drop_duplicates(subset=["round", "driver_number"], keep="last")
    data = data.sort_values(["round", "driver_number"]).reset_index(drop=True)
    data.to_csv(path, index=False)
    return data


def make_model(categorical_features, profile="fast", tuned_params=None):
    """HistGradientBoostingRegressor for the given profile.

    fast      — FAST_PARAMS, the project's fixed hyperparameters (quick).
    optimized — the hyperparameters returned by tune_hyperparameters for this
                training set; falls back to FAST_PARAMS when tuned_params is
                None (too little data to tune on).
    """
    if profile not in MODEL_PROFILES:
        raise ValueError(f"unknown model profile: {profile!r}")
    params = dict(FAST_PARAMS)
    if profile == "optimized" and tuned_params:
        params.update(tuned_params)
    return HistGradientBoostingRegressor(
        categorical_features=categorical_features,
        random_state=42,
        **params,
    )


def tune_hyperparameters(X, y, rounds, categorical_features=None,
                         n_iter=TUNE_N_ITER, random_state=TUNE_RANDOM_STATE):
    """Search TUNED_SEARCH_SPACE for the combination with the lowest
    cross-validated MAE on one model's training set.

    The CV mimics the rolling backtest: validation rounds are held out one
    at a time, each predicted by a model fit on all earlier rounds only;
    errors are pooled across folds. Shipped defaults validate on every
    TUNE_STRIDE-th training round (strided leave-one-round-out) so the
    score reflects the whole training window rather than the regime of a
    few recent rounds; setting TUNE_MAX_SPLITS to an int restricts
    validation to that many recent rounds instead. The FAST_PARAMS
    combination is always evaluated first and ties keep it. Because folds
    are small, a candidate must beat the fast hyperparameters by at least
    TUNE_MIN_IMPROVEMENT (relative MAE) before it is adopted — without
    that margin the search would chase fold noise and could end up worse
    out-of-sample than the fast profile.

    Returns the winning parameter dict, or None when the training set
    covers fewer than two rounds (nothing to cross-validate) or no
    candidate clearly beat the fast hyperparameters — in both cases
    callers fall back to the fast profile.
    """
    X = np.asarray(X, dtype=float)
    y = np.asarray(y, dtype=float)
    rounds = np.asarray(rounds)
    unique = np.unique(rounds)
    if TUNE_MAX_SPLITS is None:
        val_rounds = unique[1::TUNE_STRIDE]
    else:
        val_rounds = unique[-min(TUNE_MAX_SPLITS, len(unique) - 1):]
    if len(val_rounds) == 0:
        return None
    # A fold whose training share is all-NaN in some column (e.g. round 1
    # has no form or sprint data) would crash HistGradientBoosting — zero
    # those columns for the fit; a constant column cannot be split on, so
    # the model simply ignores that feature in that fold.
    fold_sets = []
    for r in val_rounds:
        X_train = X[rounds < r]
        all_nan = np.isnan(X_train).all(axis=0)
        if all_nan.any():
            X_train = X_train.copy()
            X_train[:, all_nan] = 0.0
        fold_sets.append((X_train, y[rounds < r], X[rounds == r], y[rounds == r]))

    keys = list(TUNED_SEARCH_SPACE)
    combos = list(product(*(TUNED_SEARCH_SPACE[key] for key in keys)))
    fast = tuple(FAST_PARAMS[key] for key in keys)
    rng = np.random.RandomState(random_state)
    sampled = [combo for combo in
               (combos[i] for i in rng.choice(len(combos), size=n_iter, replace=False))
               if combo != fast][:n_iter]
    candidates = [fast] + sampled

    def pooled_mae(combo):
        errors = []
        for X_train, y_train, X_val, y_val in fold_sets:
            model = HistGradientBoostingRegressor(
                categorical_features=categorical_features,
                random_state=random_state,
                loss="absolute_error",
                **dict(zip(keys, combo)),
            )
            model.fit(X_train, y_train)
            errors.append(np.abs(model.predict(X_val) - y_val))
        return float(np.concatenate(errors).mean())

    fast_mae = pooled_mae(fast)
    best_params = None
    best_mae = np.inf
    for combo in candidates:
        mae = fast_mae if combo == fast else pooled_mae(combo)
        if mae < best_mae:
            best_mae = mae
            best_params = dict(zip(keys, combo))
    keep_fast = best_mae > fast_mae * (1 - TUNE_MIN_IMPROVEMENT)
    if VERBOSITY >= 2:
        if best_params is None or keep_fast:
            vprint(2, f"  tuner: keeping fast hyperparameters (pooled MAE "
                      f"{fast_mae:.3f}; best found {best_mae:.3f})")
        else:
            vprint(2, f"  tuner: adopting {format_params(best_params)} (pooled MAE "
                      f"{best_mae:.3f} vs fast {fast_mae:.3f})")
    if keep_fast:
        return None
    return best_params


def format_params(params):
    """Compact one-line rendering of a hyperparameter dict for reports."""
    if not params:
        return ""
    return ", ".join(f"{key}={value}" for key, value in params.items())


def rank_corr(a, b):
    ra = pd.Series(a).rank().to_numpy()
    rb = pd.Series(b).rank().to_numpy()
    if np.std(ra) == 0 or np.std(rb) == 0:
        return np.nan
    return float(np.corrcoef(ra, rb)[0, 1])


def importance_scores(model, feature_names, X, target, n_repeats=5):
    """Permutation importance (MAE increase when shuffled) as a DataFrame,
    ordered like the CLI report (descending importance)."""
    result = permutation_importance(
        model, X, target,
        scoring="neg_mean_absolute_error",
        n_repeats=n_repeats,
        random_state=42,
    )
    order = np.argsort(result.importances_mean)[::-1]
    return pd.DataFrame({
        "feature": [feature_names[i] for i in order],
        "importance": result.importances_mean[order],
        "std": result.importances_std[order],
    })


def importance_report(model, feature_names, X, target, label, width=24):
    print(f"\nFeature importance ({label} model, permutation, increase in MAE when shuffled):")
    for row in importance_scores(model, feature_names, X, target).itertuples():
        spread = f" ±{row.std:.3f}" if VERBOSITY >= 2 else ""
        print(f"  {row.feature:<{width}} {row.importance:+.3f}{spread}")
