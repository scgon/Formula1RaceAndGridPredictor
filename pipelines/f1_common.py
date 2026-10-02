"""Shared helpers for the F1 race and grid prediction pipelines.

predict_race.py and predict_grid.py remain independent entry points with their
own features, targets and models; this module holds the machinery they share:
fastf1 cache setup, UTC handling, qualifying-lap extraction, practice-lap
features, the season CSV cache, the model factory and permutation importance.
The Streamlit app (app.py) builds on the same functions.
"""

import warnings
from datetime import datetime, timedelta, timezone
from pathlib import Path

import numpy as np
import pandas as pd
import fastf1
from sklearn.ensemble import HistGradientBoostingRegressor
from sklearn.inspection import permutation_importance

warnings.filterwarnings("ignore")

# Repo root (this file lives in pipelines/); cache/ and data/ stay at the root
# so pipelines, web app and notebooks all share them.
BASE_DIR = Path(__file__).resolve().parent.parent
CACHE_DIR = BASE_DIR / "cache"
DATA_DIR = BASE_DIR / "data"

MIN_TRAIN_ROUNDS = 5
STINT_WINDOW = 5
STINT_MIN_LAPS = 3
COMPLETION_BUFFER = timedelta(hours=3)


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
                   load_round, refresh=False):
    """Fetch completed rounds into the season CSV, reusing whatever is cached.

    filename         CSV name pattern, e.g. "season_{year}.csv"
    required_columns columns the cached CSV must contain to be reused
    result_column    column holding classified results ("finish" / "quali_pos")
    load_round       callable (year, round_number, event_name) -> DataFrame
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
    for rn, name in todo:
        try:
            frame = load_round(year, rn, name)
        except Exception as exc:
            print(f"  round {rn:>2} ({name}) skipped: {type(exc).__name__}: {exc}")
            continue
        if frame[result_column].notna().sum() == 0:
            print(f"  round {rn:>2} ({name}) skipped: no classified results yet")
            continue
        print(f"  fetched round {rn:>2}  {name}")
        frames.append(frame)

    if cached is not None:
        frames.insert(0, cached)
    if not frames:
        return pd.DataFrame(columns=required_columns)

    data = pd.concat(frames, ignore_index=True)
    data = data.drop_duplicates(subset=["round", "driver_number"], keep="last")
    data = data.sort_values(["round", "driver_number"]).reset_index(drop=True)
    data.to_csv(path, index=False)
    return data


def make_model(categorical_features):
    return HistGradientBoostingRegressor(
        loss="absolute_error",
        learning_rate=0.08,
        max_iter=150,
        max_leaf_nodes=15,
        min_samples_leaf=20,
        l2_regularization=1.0,
        categorical_features=categorical_features,
        random_state=42,
    )


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
        print(f"  {row.feature:<{width}} {row.importance:+.3f}")
