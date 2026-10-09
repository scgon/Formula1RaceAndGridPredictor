"""Milestone prediction: pole sitter, race winner, first retirement or
fastest-lap driver — ONE small binary classifier per run, selected with the
CLI --milestone flag or the web-app Milestone selectbox.

The selected model scores every driver of the target round with P(driver
achieves the milestone); the top probability is the predicted driver. The
pole model uses day-before-quali information only (FP1/FP2, sprint
qualifying, past qualifying form — the same constraint the grid pipeline
originally used), so it never needs the target round's qualifying and
runs in three modes: "post" reviews a completed round, "pre" predicts
after qualifying (the pole flag is known, so the call is scored) and
"prequali" predicts before qualifying — entrants come from practice (or
the last completed round) and the quali columns stay unknown. The other
three milestones predict from the pre-race information set (grid,
FP1-FP3, sprint results, form and reliability — the same as the race
pipeline) and therefore require qualifying to have happened. The
first-retirement and fastest-lap targets are derived from race lap data,
which neither of the other two pipelines collects, so this pipeline keeps
its own season CSV (extras_season_{year}.csv).

These are classifiers, not regressors, so the model factory, the
hyperparameter tuner (pooled cross-validated log loss instead of MAE) and
the permutation-importance helpers are local; everything else comes from
f1_common.py.
"""

import argparse
from datetime import datetime, timedelta
from itertools import product

import numpy as np
import pandas as pd
import fastf1
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.inspection import permutation_importance

import f1_common as common

MIN_TRAIN_ROUNDS = common.MIN_TRAIN_ROUNDS
PRACTICE_BUFFER = timedelta(hours=2)

PRACTICE_SESSIONS_QUALI = ("FP1", "FP2")   # pole model: day-before-quali only
PRACTICE_SESSIONS_RACE = ("FP1", "FP2", "FP3")  # race milestones: pre-race

FP12_COLS = ["fp12_race_pace_delta", "fp12_best_delta", "fp12_laps"]
FP_COLS = ["fp_race_pace_delta", "fp_best_delta", "fp_laps"]
SPRINT_QUALI_COLS = ["sprint_quali_pos", "sprint_quali_delta"]
SPRINT_RACE_COLS = ["sprint_finish", "sprint_gain"]

RAW_COLUMNS = [
    "round",
    "event",
    "driver_number",
    "driver",
    "team",
    "grid",
    "quali_pos",
    "quali_time",
    "quali_delta",
    "finish",
    "points",
    "status",
    "pole",
    "winner",
    "dnf",
    "first_dnf",
    "fastest_lap",
] + FP12_COLS + FP_COLS + SPRINT_QUALI_COLS + SPRINT_RACE_COLS

# Classifiers need smaller leaves than the regressor defaults: the positive
# class is one driver per round (~1/20 of the rows), so FAST_PARAMS'
# min_samples_leaf=20 would starve the trees. Everything else mirrors
# f1_common.FAST_PARAMS.
CLASSIFIER_FAST_PARAMS = {
    "learning_rate": 0.08,
    "max_iter": 150,
    "max_leaf_nodes": 15,
    "min_samples_leaf": 10,
    "l2_regularization": 1.0,
}

POLE_FEATURES = [
    "fp12_best_delta",
    "fp12_race_pace_delta",
    "fp12_laps",
    "team_fp12_best_delta",
    "team_fp12_race_pace_delta",
    "sprint_quali_pos",
    "sprint_quali_delta",
    "quali_pos_last",
    "quali_form_3",
    "quali_form_season",
    "team_quali_form_3",
    "team_quali_form_season",
    "driver_points_before",
    "team_points_before",
    "team_id",
]

WINNER_FEATURES = [
    "grid",
    "quali_delta",
    "team_quali_delta",
    "fp_best_delta",
    "fp_race_pace_delta",
    "fp_laps",
    "sprint_quali_pos",
    "sprint_finish",
    "sprint_gain",
    "driver_form_3",
    "driver_form_season",
    "driver_last_finish",
    "team_form_3",
    "team_form_season",
    "driver_points_before",
    "team_points_before",
    "driver_wins_before",
    "team_id",
]

FIRST_DNF_FEATURES = [
    "grid",
    "quali_delta",
    "team_quali_delta",
    "fp_best_delta",
    "fp_race_pace_delta",
    "fp_laps",
    "sprint_quali_pos",
    "sprint_finish",
    "sprint_gain",
    "driver_form_3",
    "driver_form_season",
    "driver_dnf_last",
    "driver_dnf_rate_season",
    "team_dnf_rate_season",
    "driver_points_before",
    "team_id",
]

FASTEST_LAP_FEATURES = [
    "grid",
    "quali_delta",
    "team_quali_delta",
    "fp_best_delta",
    "fp_race_pace_delta",
    "fp_laps",
    "sprint_quali_pos",
    "sprint_finish",
    "sprint_gain",
    "driver_form_3",
    "driver_form_season",
    "driver_fl_before",
    "driver_points_before",
    "team_id",
]

TARGETS = {
    "pole": {
        "label": "Pole position",
        "column": "pole",
        "rows_column": "quali_pos",
        "train_columns": ("quali_pos",),
        # the pole feature set never includes the target round's
        # qualifying, so this model can also predict a round before its
        # qualifying has run (the race milestones need the grid and cannot)
        "prequali": True,
        "features": POLE_FEATURES,
        "baseline_sort": (("quali_form_season", True), ("driver_number", True)),
        "baseline_note": "best average qualifying position this season",
    },
    "winner": {
        "label": "Race winner",
        "column": "winner",
        "rows_column": "grid",
        "train_columns": ("finish", "grid"),
        "features": WINNER_FEATURES,
        "baseline_sort": (("grid", True), ("driver_number", True)),
        "baseline_note": "the driver starting from pole (grid P1)",
    },
    "first_dnf": {
        "label": "First retirement",
        "column": "first_dnf",
        "rows_column": "grid",
        "train_columns": ("finish", "grid"),
        "features": FIRST_DNF_FEATURES,
        "baseline_sort": (("driver_dnf_count_before", False), ("driver_form_season", False), ("driver_number", True)),
        "baseline_note": "the driver with the most retirements this season",
    },
    "fastest_lap": {
        "label": "Fastest lap",
        "column": "fastest_lap",
        "rows_column": "grid",
        "train_columns": ("finish", "grid"),
        "features": FASTEST_LAP_FEATURES,
        "baseline_sort": (("driver_fl_before", False), ("quali_delta", True), ("driver_number", True)),
        "baseline_note": "most fastest laps this season, then the fastest qualifier",
    },
}

setup = common.setup
utc_now = common.utc_now
session_utc = common.session_utc


def _finished_like(status):
    """Classified as running the full distance — not a retirement: "Finished",
    the old "+N Lap(s)" classifications, the 2026 "Lapped", and disqualifications
    (a DSQ completed the race; the stewards removed the result afterwards)."""
    text = str(status)
    return (text == "Finished" or text == "Lapped" or text == "Disqualified"
            or text.startswith("+"))


def sprint_quali_features(year, round_number):
    """Sprint qualifying order (best SQ lap times; SQ session results are
    empty in fastf1 3.8.x), falling back to the sprint session's grid."""
    try:
        sq = fastf1.get_session(year, round_number, "SQ")
        sq.load(laps=True, telemetry=False, weather=False, messages=False)
        laps = sq.laps
        if laps is not None and not laps.empty:
            clean = laps[laps["LapTime"].notna() & (laps["Deleted"] != True)]
            if not clean.empty:
                best = clean.groupby(clean["DriverNumber"].astype(str))["LapTime"].min()
                if not best.empty:
                    return pd.DataFrame({
                        "sprint_quali_pos": best.rank(method="min").astype(float),
                        "sprint_quali_delta": (best - best.min()).dt.total_seconds(),
                    })
    except Exception:
        pass
    try:
        sprint = fastf1.get_session(year, round_number, "S")
        sprint.load(laps=False, telemetry=False, weather=False, messages=False)
        results = sprint.results
        if not results.empty and results["GridPosition"].notna().any():
            pos = {}
            for num in results.index:
                grid = results.at[num, "GridPosition"]
                pos[str(num)] = float(grid) if pd.notna(grid) and grid > 0 else np.nan
            return pd.DataFrame({
                "sprint_quali_pos": pd.Series(pos),
                "sprint_quali_delta": np.nan,
            })
    except Exception:
        pass
    return pd.DataFrame()


def sprint_race_features(year, round_number):
    """Sprint race result (sprint quali order comes from
    sprint_quali_features; the sprint runs before the race on sprint
    weekends, so it is valid pre-race information)."""
    try:
        sprint = fastf1.get_session(year, round_number, "S")
        sprint.load(laps=False, telemetry=False, weather=False, messages=False)
        results = sprint.results
        if results.empty or results["Position"].isna().all():
            return pd.DataFrame()
    except Exception:
        return pd.DataFrame()
    rows = {}
    for num in results.index:
        row = results.loc[num]
        grid = row["GridPosition"]
        grid = float(grid) if pd.notna(grid) and grid > 0 else np.nan
        finish = row["Position"]
        finish = float(finish) if pd.notna(finish) else np.nan
        rows[str(num)] = {
            "sprint_finish": finish,
            "sprint_gain": grid - finish if pd.notna(grid) and pd.notna(finish) else np.nan,
        }
    return pd.DataFrame.from_dict(rows, orient="index")


def _merge_weekend_features(frame, year, round_number):
    fp12 = common.practice_features(year, round_number, PRACTICE_SESSIONS_QUALI)
    if not fp12.empty:
        for col in FP12_COLS:
            frame[col] = frame["driver_number"].map(fp12[col.replace("fp12_", "fp_", 1)])
    fp = common.practice_features(year, round_number, PRACTICE_SESSIONS_RACE)
    if not fp.empty:
        for col in FP_COLS:
            frame[col] = frame["driver_number"].map(fp[col])
    sprint_quali = sprint_quali_features(year, round_number)
    if not sprint_quali.empty:
        for col in SPRINT_QUALI_COLS:
            frame[col] = frame["driver_number"].map(sprint_quali[col])
    sprint = sprint_race_features(year, round_number)
    if not sprint.empty:
        for col in SPRINT_RACE_COLS:
            frame[col] = frame["driver_number"].map(sprint[col])
    return frame


def load_completed_round(year, round_number, event_name):
    quali = fastf1.get_session(year, round_number, "Q")
    quali.load(laps=False, telemetry=False, weather=False, messages=False)
    race = fastf1.get_session(year, round_number, "R")
    race.load(laps=True, telemetry=False, weather=False, messages=False)
    qres = quali.results
    rres = race.results
    pole_time = common.pole_seconds(qres)

    # lap-derived milestones: fastest lap and (for the first retirement)
    # how many laps each driver completed before stopping
    laps = race.laps
    clean = pd.DataFrame()
    if laps is not None and not laps.empty:
        clean = laps[laps["LapTime"].notna()]
        if "Deleted" in clean.columns:
            clean = clean[clean["Deleted"] != True]
    laps_done = {}
    fl_driver = None
    if not clean.empty:
        by_car = clean.groupby(clean["DriverNumber"].astype(str))
        laps_done = by_car["LapNumber"].max().to_dict()
        best = by_car["LapTime"].min()
        if best.notna().any():
            fl_driver = str(best.idxmin())

    dnf_cars = {}
    quali_pos = {}
    for num in qres.index:
        pos = qres.at[num, "Position"]
        if pd.notna(pos):
            quali_pos[str(num)] = float(pos)
    for num in rres.index:
        finish = rres.at[num, "Position"]
        retired = pd.notna(finish) and not _finished_like(rres.at[num, "Status"])
        if retired:
            dnf_cars[str(num)] = int(laps_done.get(str(num), 0))
    first_dnf_cars = set()
    if dnf_cars:
        earliest = min(dnf_cars.values())
        first_dnf_cars = {num for num, laps in dnf_cars.items() if laps == earliest}

    points_map = {}
    for num in rres.index:
        value = rres.at[num, "Points"]
        points_map[str(num)] = float(value) if pd.notna(value) else 0.0
    rows = []
    for num in rres.index:
        row = rres.loc[num]
        grid = row["GridPosition"]
        grid = float(grid) if pd.notna(grid) and grid > 0 else np.nan
        finish = row["Position"]
        finish = float(finish) if pd.notna(finish) else np.nan
        status = str(row["Status"])
        # DNS rows (finish NaN) never retired *during* the race
        dnf = bool(pd.notna(finish) and not _finished_like(status))
        rows.append({
            "round": round_number,
            "event": event_name,
            "driver_number": str(num),
            "driver": row["Abbreviation"],
            "team": row["TeamName"],
            "grid": grid,
            "quali_pos": quali_pos.get(str(num), np.nan),
            "quali_time": common.best_quali_seconds(qres, num),
            "quali_delta": np.nan,
            "finish": finish,
            "points": points_map.get(str(num), 0.0),
            "status": status,
            "pole": int(quali_pos.get(str(num), np.nan) == 1),
            "winner": int(finish == 1),
            "dnf": int(dnf),
            "first_dnf": int(str(num) in first_dnf_cars),
            "fastest_lap": int(fl_driver == str(num)),
        })
    frame = pd.DataFrame(rows, columns=RAW_COLUMNS)
    frame["quali_delta"] = frame["quali_time"] - pole_time
    return _merge_weekend_features(frame, year, round_number)


def load_upcoming_round(year, round_number, event_name):
    """Pre-race frame from the (already run) qualifying results; race
    outcomes stay unknown, but the pole flag is known so the pole model's
    call can be scored against it."""
    quali = fastf1.get_session(year, round_number, "Q")
    quali.load(laps=False, telemetry=False, weather=False, messages=False)
    qres = quali.results
    pole_time = common.pole_seconds(qres)
    rows = []
    for num in qres.index:
        row = qres.loc[num]
        pos = row["Position"]
        grid = float(pos) if pd.notna(pos) else np.nan
        rows.append({
            "round": round_number,
            "event": event_name,
            "driver_number": str(num),
            "driver": row["Abbreviation"],
            "team": row["TeamName"],
            "grid": grid,
            "quali_pos": grid,
            "quali_time": common.best_quali_seconds(qres, num),
            "quali_delta": np.nan,
            "finish": np.nan,
            "points": 0.0,
            "status": "",
            "pole": int(grid == 1),
            "winner": 0,
            "dnf": 0,
            "first_dnf": 0,
            "fastest_lap": 0,
        })
    frame = pd.DataFrame(rows, columns=RAW_COLUMNS)
    frame["quali_delta"] = frame["quali_time"] - pole_time
    return _merge_weekend_features(frame, year, round_number)


def practice_entrants(year, round_number, sessions=PRACTICE_SESSIONS_RACE):
    for identifier in sessions:
        try:
            session = fastf1.get_session(year, round_number, identifier)
            session.load(laps=False, telemetry=False, weather=False, messages=False)
            if not session.results.empty:
                return session.results
        except Exception:
            continue
    return None


def load_upcoming_round_prequali(year, round_number, event_name, fallback):
    """Pre-quali frame for the pole model: qualifying has not happened yet,
    so there are no quali results. Entrants come from the first practice
    session with results, or from the last completed round when no practice
    has run either. The quali columns stay NaN and the pole flag is unknown
    (0 for every driver), so the model's call is only scored once the round
    completes."""
    results = practice_entrants(year, round_number)
    if results is not None:
        entries = [(str(num), results.at[num, "Abbreviation"], results.at[num, "TeamName"])
                   for num in results.index]
    else:
        entries = [(str(r.driver_number), r.driver, r.team) for r in fallback.itertuples()]
    rows = []
    for num, driver, team in entries:
        rows.append({
            "round": round_number,
            "event": event_name,
            "driver_number": num,
            "driver": driver,
            "team": team,
            "grid": np.nan,
            "quali_pos": np.nan,
            "quali_time": np.nan,
            "quali_delta": np.nan,
            "finish": np.nan,
            "points": 0.0,
            "status": "",
            "pole": 0,
            "winner": 0,
            "dnf": 0,
            "first_dnf": 0,
            "fastest_lap": 0,
        })
    frame = pd.DataFrame(rows, columns=RAW_COLUMNS)
    return _merge_weekend_features(frame, year, round_number)


def collect_season(year, schedule, refresh=False, rate_limit_wait=None):
    return common.collect_season(
        year, schedule,
        filename="extras_season_{year}.csv",
        required_columns=RAW_COLUMNS,
        result_column="finish",
        load_round=load_completed_round,
        refresh=refresh,
        rate_limit_wait=rate_limit_wait,
    )


def build_features(data):
    df = data.copy()
    df["driver_number"] = df["driver_number"].astype(str)
    df = df.sort_values(["round", "driver_number"]).reset_index(drop=True)
    df["team_id"] = pd.factorize(df["team"])[0]

    # round-local team aggregates
    df["team_quali_delta"] = df.groupby(["round", "team"], sort=False)["quali_delta"].transform("mean")
    df["team_race_mean"] = df.groupby(["round", "team"], sort=False)["finish"].transform("mean")
    df["team_race_points"] = df.groupby(["round", "team"], sort=False)["points"].transform("sum")
    df["team_dnf_mean"] = df.groupby(["round", "team"], sort=False)["dnf"].transform("mean")
    df["team_quali_mean"] = df.groupby(["round", "team"], sort=False)["quali_pos"].transform("mean")
    df["team_fp12_best_delta"] = df.groupby(["round", "team"], sort=False)["fp12_best_delta"].transform("mean")
    df["team_fp12_race_pace_delta"] = df.groupby(["round", "team"], sort=False)["fp12_race_pace_delta"].transform("mean")

    by_driver = df.groupby("driver_number", sort=False)
    # race form (all shifted: information from before the round)
    df["driver_last_finish"] = by_driver["finish"].transform(lambda s: s.shift(1))
    df["driver_form_3"] = by_driver["finish"].transform(lambda s: s.shift(1).rolling(3, min_periods=1).mean())
    df["driver_form_season"] = by_driver["finish"].transform(lambda s: s.shift(1).expanding(min_periods=1).mean())
    df["driver_points_before"] = by_driver["points"].transform(lambda s: s.cumsum().shift(1)).fillna(0.0)
    df["team_form_3"] = by_driver["team_race_mean"].transform(lambda s: s.shift(1).rolling(3, min_periods=1).mean())
    df["team_form_season"] = by_driver["team_race_mean"].transform(lambda s: s.shift(1).expanding(min_periods=1).mean())
    df["team_points_before"] = by_driver["team_race_points"].transform(lambda s: s.cumsum().shift(1)).fillna(0.0)
    # milestone history
    df["driver_wins_before"] = by_driver["winner"].transform(lambda s: s.cumsum().shift(1)).fillna(0.0)
    df["driver_fl_before"] = by_driver["fastest_lap"].transform(lambda s: s.cumsum().shift(1)).fillna(0.0)
    df["driver_dnf_count_before"] = by_driver["dnf"].transform(lambda s: s.cumsum().shift(1)).fillna(0.0)
    df["driver_dnf_last"] = by_driver["dnf"].transform(lambda s: s.shift(1)).fillna(0.0)
    df["driver_dnf_rate_season"] = by_driver["dnf"].transform(lambda s: s.shift(1).expanding(min_periods=1).mean())
    df["team_dnf_rate_season"] = by_driver["team_dnf_mean"].transform(lambda s: s.shift(1).expanding(min_periods=1).mean())
    # quali form
    df["quali_pos_last"] = by_driver["quali_pos"].transform(lambda s: s.shift(1))
    df["quali_form_3"] = by_driver["quali_pos"].transform(lambda s: s.shift(1).rolling(3, min_periods=1).mean())
    df["quali_form_season"] = by_driver["quali_pos"].transform(lambda s: s.shift(1).expanding(min_periods=1).mean())
    df["team_quali_form_3"] = by_driver["team_quali_mean"].transform(lambda s: s.shift(1).rolling(3, min_periods=1).mean())
    df["team_quali_form_season"] = by_driver["team_quali_mean"].transform(lambda s: s.shift(1).expanding(min_periods=1).mean())
    return df


# --- models ------------------------------------------------------------------

def usable_features(frame, feature_names):
    """feature_names minus any column that is entirely NaN in `frame` —
    sklearn's Hist gradient boosting cannot fit a feature with no observed
    values (e.g. quali form with only round 1 trained on, sprint quali
    before the first sprint weekend)."""
    matrix = frame[feature_names].to_numpy(dtype=float)
    return [name for name, col in zip(feature_names, matrix.T) if not np.isnan(col).all()]


def make_classifier(feature_names, profile="fast", tuned_params=None):
    if profile not in common.MODEL_PROFILES:
        raise ValueError(f"unknown model profile: {profile!r}")
    params = dict(CLASSIFIER_FAST_PARAMS)
    if profile == "optimized" and tuned_params:
        params.update(tuned_params)
    categorical = [feature_names.index("team_id")] if "team_id" in feature_names else []
    return HistGradientBoostingClassifier(
        categorical_features=categorical,
        random_state=42,
        **params,
    )


def tune_classifier(X, y, rounds, categorical_features=None,
                    n_iter=common.TUNE_N_ITER, random_state=common.TUNE_RANDOM_STATE):
    """Classifier twin of f1_common.tune_hyperparameters: search
    common.TUNED_SEARCH_SPACE for the combination with the lowest pooled
    cross-validated log loss on one model's training set (same rolling
    folds, same seeded candidate sampling, same minimum-improvement rule
    and fast-values fallback). Folds whose training share contains a
    single class are skipped — a classifier cannot be fit on them; when no
    fold is usable every candidate ties at infinity and None (fast values)
    is returned.
    """
    X = np.asarray(X, dtype=float)
    y = np.asarray(y, dtype=float)
    rounds = np.asarray(rounds)
    unique = np.unique(rounds)
    if common.TUNE_MAX_SPLITS is None:
        val_rounds = unique[1::common.TUNE_STRIDE]
    else:
        val_rounds = unique[-min(common.TUNE_MAX_SPLITS, len(unique) - 1):]
    if len(val_rounds) == 0:
        return None
    fold_sets = []
    for r in val_rounds:
        X_train = X[rounds < r]
        all_nan = np.isnan(X_train).all(axis=0)
        if all_nan.any():
            X_train = X_train.copy()
            X_train[:, all_nan] = 0.0
        fold_sets.append((X_train, y[rounds < r], X[rounds == r], y[rounds == r]))

    keys = list(common.TUNED_SEARCH_SPACE)
    combos = list(product(*(common.TUNED_SEARCH_SPACE[key] for key in keys)))
    fast = tuple(CLASSIFIER_FAST_PARAMS[key] for key in keys)
    rng = np.random.RandomState(random_state)
    sampled = [combo for combo in
               (combos[i] for i in rng.choice(len(combos), size=n_iter, replace=False))
               if combo != fast][:n_iter]
    candidates = [fast] + sampled

    def pooled_log_loss(combo):
        losses = []
        for X_train, y_train, X_val, y_val in fold_sets:
            if np.unique(y_train).size < 2:
                continue
            model = HistGradientBoostingClassifier(
                categorical_features=categorical_features,
                random_state=random_state,
                **dict(zip(keys, combo)),
            )
            proba = model.fit(X_train, y_train).predict_proba(X_val)[:, 1]
            proba = np.clip(proba, 1e-6, 1 - 1e-6)
            losses.append(-(y_val * np.log(proba) + (1 - y_val) * np.log(1 - proba)))
        if not losses:
            return np.inf
        return float(np.concatenate(losses).mean())

    fast_loss = pooled_log_loss(fast)
    best_params = None
    best_loss = np.inf
    for combo in candidates:
        loss = fast_loss if combo == fast else pooled_log_loss(combo)
        if loss < best_loss:
            best_loss = loss
            best_params = dict(zip(keys, combo))
    keep_fast = best_loss > fast_loss * (1 - common.TUNE_MIN_IMPROVEMENT)
    if common.VERBOSITY >= 2 and np.isfinite(fast_loss) and np.isfinite(best_loss):
        if best_params is None or keep_fast:
            common.vprint(2, f"  tuner: keeping fast hyperparameters (pooled log loss "
                          f"{fast_loss:.3f}; best found {best_loss:.3f})")
        else:
            common.vprint(2, f"  tuner: adopting {common.format_params(best_params)} "
                          f"(pooled log loss {best_loss:.3f} vs fast {fast_loss:.3f})")
    if keep_fast:
        return None
    return best_params


def target_rows(features, round_number, name):
    """Rows of one round a model predicts over: quali participants for the
    pole model, drivers with a grid slot for the race milestones. A
    pre-quali pole target has no quali positions yet, so the pole model
    scores every entrant of the round instead."""
    spec = TARGETS[name]
    rows = features[features["round"] == round_number]
    eligible = rows[rows[spec["rows_column"]].notna()]
    if eligible.empty and spec.get("prequali"):
        return rows
    return eligible


def baseline_pick(rows, name):
    """The naive pick for one milestone (the sort in TARGETS; NaN values
    sort last, driver_number breaks ties deterministically)."""
    cols = [c for c, _ in TARGETS[name]["baseline_sort"]]
    ascending = [a for _, a in TARGETS[name]["baseline_sort"]]
    ordered = rows.sort_values(cols, ascending=ascending, na_position="last")
    return str(ordered.iloc[0]["driver"])


def actual_drivers(features, round_number, name):
    """Set of drivers who actually achieved the milestone in the round
    (empty when the round's outcome is unknown — e.g. a race not yet run,
    or a round where nobody retired). First retirements can be tied (a
    lap-one pile-up), so this is genuinely a set."""
    spec = TARGETS[name]
    rows = features[(features["round"] == round_number) & (features[spec["column"]] == 1)]
    return set(rows["driver"])


def train_model(features, target_round, name, profile="fast"):
    """Fit one milestone classifier on all rounds before `target_round`.

    Returns (model, train frame, used feature list, tuned hyperparameters);
    raises ValueError when there is no training data or no positive
    examples (a classifier cannot be fit on a single class).
    """
    spec = TARGETS[name]
    mask = features["round"] < target_round
    for col in spec["train_columns"]:
        mask &= features[col].notna()
    train = features[mask]
    if train.empty:
        raise ValueError(f"no training data before round {target_round}")
    target = train[spec["column"]].to_numpy(dtype=int)
    if np.unique(target).size < 2:
        raise ValueError(f"no {spec['label'].lower()} examples in the training rounds "
                         f"before round {target_round}")
    used = usable_features(train, spec["features"])
    X = train[used].to_numpy(dtype=float)
    categorical = [used.index("team_id")] if "team_id" in used else None
    tuned = None
    if profile == "optimized":
        tuned = tune_classifier(X, target, train["round"].to_numpy(),
                                categorical_features=categorical)
    model = make_classifier(used, profile, tuned)
    model.fit(X, target)
    return model, train, used, tuned


def predict_round(model, features, target_round, name, use_features=None):
    """Score every eligible driver of the round; rows sorted by predicted
    probability (ties broken by the driver_number pre-sort)."""
    spec = TARGETS[name]
    use_features = spec["features"] if use_features is None else use_features
    rows = target_rows(features, target_round, name).copy()
    rows = rows.sort_values("driver_number")
    rows["probability"] = model.predict_proba(rows[use_features].to_numpy(dtype=float))[:, 1]
    rows["pred_rank"] = rows["probability"].rank(method="first", ascending=False).astype(int)
    return rows.sort_values("pred_rank")


def backtest_records(features, min_train_rounds, profile="fast", milestone="pole"):
    """Rolling backtest of ONE milestone model (selected with the
    `milestone` key of TARGETS / the CLI --milestone flag / the web-app
    Milestone selectbox); one metrics dict per predicted round.

    Each record carries the hit flags for the model top pick, the model
    top-3 and the naive baseline, the predicted/baseline/actual driver
    names, and the probability the model assigned to its own pick and to
    the actual driver. Rounds without a first retirement (nobody retired)
    score NaN when the first-retirement model is selected, so hit rates
    average only over rounds where the milestone exists. A pre-quali
    target round (pole model) likewise rides along as the last row with
    NaN hits — its outcome is not known yet, exactly like the
    race-milestone rows of a pre-mode round. Prints one live progress
    line per trained model (streamed into the web app's log).
    """
    spec = TARGETS[milestone]
    rounds = sorted(features["round"].unique())
    tune_note = " (tuning hyperparameters)" if profile == "optimized" else ""
    records = []
    for r in rounds:
        if sum(1 for x in rounds if x < r) < min_train_rounds:
            continue
        event = str(features.loc[features["round"] == r, "event"].iloc[0])
        rec = {"round": r, "event": event,
               "dnf_any": int(((features["round"] == r) & (features["dnf"] == 1)).any())}
        common.vprint(1, f"backtest round {r:>2}  {event}: training {spec['label'].lower()} model{tune_note}...")
        try:
            model, train, used, _ = train_model(features, r, milestone, profile)
        except ValueError:
            rec.update({"hit": np.nan, "top3": np.nan, "base": np.nan,
                        "top1": None, "actual": None, "baseline": None,
                        "pick_proba": np.nan, "actual_proba": np.nan})
            records.append(rec)
            continue
        rec["train_rounds"] = int(train["round"].nunique())
        ranked = predict_round(model, features, r, milestone, used)
        actual = actual_drivers(features, r, milestone)
        top1 = str(ranked.iloc[0]["driver"])
        top3 = [str(d) for d in ranked.head(3)["driver"]]
        base = baseline_pick(target_rows(features, r, milestone), milestone)
        pick_proba = float(ranked.iloc[0]["probability"])
        actual_proba = float(ranked.loc[ranked["driver"].isin(actual), "probability"].max()) \
            if actual else np.nan
        if not actual:
            hit = top3_hit = base_hit = np.nan
        else:
            hit = float(top1 in actual)
            top3_hit = float(any(d in actual for d in top3))
            base_hit = float(base in actual)
        rec.update({
            "hit": hit,
            "top3": top3_hit,
            "base": base_hit,
            "top1": top1,
            "actual": " / ".join(sorted(actual)) if actual else None,
            "baseline": base,
            "pick_proba": pick_proba,
            "actual_proba": actual_proba,
        })
        records.append(rec)
        if common.VERBOSITY >= 2 and pd.notna(rec.get("hit")):
            outcome = "hit" if rec["hit"] == 1 else "miss"
            common.vprint(2, f"  round {r:>2} result: pick {rec['top1']} "
                          f"({100 * rec['pick_proba']:.0f}%) | baseline {rec['baseline']} "
                          f"| actual {rec['actual']} | {outcome}")
    return records


def _hit_rate(records, field):
    values = [rec[field] for rec in records if pd.notna(rec.get(field))]
    return float(np.mean(values)) if values else np.nan


def backtest_report(records, milestone="pole"):
    spec = TARGETS[milestone]
    label = spec["label"].lower()
    print(f"\n=== Rolling backtest: {label} (model top pick vs naive baselines) ===")
    if common.VERBOSITY >= 1:
        print(f"{'rd':>3}  {'event':<26} {'pick':>6} {'baseline':>9} {'actual':>10} {'hit':>6}")
        for rec in records:
            hit = rec.get("hit")
            mark = "hit" if hit == 1 else ("-" if pd.notna(hit) else "?")
            actual = rec.get("actual") or "—"
            if milestone == "first_dnf" and not rec.get("dnf_any"):
                actual = "no retirement"
            print(f"{rec['round']:>3}  {rec['event']:<26} {str(rec.get('top1') or '—'):>6} "
                  f"{str(rec.get('baseline') or '—'):>9} {actual:>10} {mark:>6}")
    if not records:
        print("not enough completed rounds for a backtest yet")
        return
    n = len(records)
    n_scored = sum(1 for rec in records if pd.notna(rec.get("hit")))
    denom = f" ({n_scored} rounds with a retirement)" if milestone == "first_dnf" else ""
    print(f"\nMean over {n} predicted rounds{denom}:")
    for title, field in (("model top-1 hit", "hit"), ("model top-3 hit", "top3"),
                         ("baseline hit", "base")):
        rate = _hit_rate(records, field)
        print(f"{title:<22}" + (f"{100 * rate:>9.0f}%" if pd.notna(rate) else "        –"))


def final_predictions(features, target, profile="fast", milestone="pole"):
    """Train ONE final milestone model (the `milestone` key of TARGETS) on
    rounds before `target` and score the target round's drivers.

    Returns a single entry holding the milestone name, the model, its
    training frame, the feature list it actually used, the hyperparameters
    the profile picked, the ranked prediction table (with probabilities),
    the top pick, the naive baseline pick, and the actual milestone
    drivers when already known (pole always — qualifying precedes the
    race; the race milestones only in post mode). When the model cannot
    train (e.g. no positive examples in the training rounds) the entry
    carries an "error" message instead. Prints one live progress line
    (streamed into the web app's log).
    """
    spec = TARGETS[milestone]
    tune_note = " (tuning hyperparameters)" if profile == "optimized" else ""
    common.vprint(1, f"training final {spec['label'].lower()} model{tune_note}...")
    try:
        model, train, used, tuned = train_model(features, target, milestone, profile)
    except ValueError as exc:
        return {"name": milestone, "error": str(exc)}
    table = predict_round(model, features, target, milestone, used)
    actual = sorted(actual_drivers(features, target, milestone))
    top1 = str(table.iloc[0]["driver"])
    return {
        "name": milestone,
        "model": model,
        "train": train,
        "features": used,
        "params": tuned,
        "table": table,
        "top1": top1,
        "top3": [str(d) for d in table.head(3)["driver"]],
        "baseline": baseline_pick(target_rows(features, target, milestone), milestone),
        "actual": actual,
        "hit": (top1 in actual) if actual else None,
        "train_rounds": int(train["round"].nunique()),
    }


def importance_scores(model, feature_names, X, y, n_repeats=5):
    """Permutation importance (log-loss increase when shuffled) as a
    DataFrame, ordered like the CLI report (descending importance)."""
    result = permutation_importance(
        model, X, y,
        scoring="neg_log_loss",
        n_repeats=n_repeats,
        random_state=42,
    )
    order = np.argsort(result.importances_mean)[::-1]
    return pd.DataFrame({
        "feature": [feature_names[i] for i in order],
        "importance": result.importances_mean[order],
        "std": result.importances_std[order],
    })


def importance_frame(pred):
    """Permutation importance for the final model, as a DataFrame."""
    if "error" in pred:
        return pd.DataFrame(columns=["feature", "importance", "std"])
    used = pred["features"]
    return importance_scores(
        pred["model"], used, pred["train"][used].to_numpy(dtype=float),
        pred["train"][TARGETS[pred["name"]]["column"]].to_numpy(dtype=int))


def importance_report(pred, width=24):
    if "error" in pred:
        return
    scores = importance_frame(pred)
    print(f"\nFeature importance ({TARGETS[pred['name']]['label']} model, permutation, "
          "increase in log loss when shuffled):")
    for row in scores.itertuples():
        spread = f" ±{row.std:.3f}" if common.VERBOSITY >= 2 else ""
        print(f"  {row.feature:<{width}} {row.importance:+.3f}{spread}")


def final_report(features, year, target, event_name, mode, profile="fast", milestone="pole"):
    pred = final_predictions(features, target, profile, milestone)
    spec = TARGETS[milestone]
    print(f"\n=== Prediction: {year} {event_name} (round {target}) — {spec['label']} ===")
    if mode == "pre" and milestone == "pole":
        print("pre-race mode: qualifying has already decided pole, so this day-before-quali "
              "model's call is scored below")
    if mode == "prequali":
        print("pre-quali mode: qualifying has not happened yet, so this model predicts from")
        print("practice, sprint qualifying and season form — run again after qualifying to score it")
    if "error" in pred:
        print(f"\nNo model — {pred['error']}")
        return pred
    print(f"\n{spec['label']} — predicted: {pred['top1']}")
    print(f"baseline pick ({spec['baseline_note']}): {pred['baseline']}")
    table = pred["table"] if common.VERBOSITY >= 2 else pred["table"].head(5)
    for i, (_, row) in enumerate(table.iterrows(), 1):
        print(f"  {i:>2}  {row['driver']:<4} {row['team']:<20} {100 * row['probability']:>5.1f}%")
    if pred["actual"]:
        verdict = "hit" if pred["top1"] in pred["actual"] else "miss"
        print(f"  actual: {' / '.join(pred['actual'])}  ({verdict})")
    elif mode == "post" or milestone == "pole":
        print("  actual: not known yet")
    if profile == "optimized":
        untuned = "fast defaults (no clearly better combination found)"
        print(f"\nTuned hyperparameters ({spec['label']} model): "
              f"{common.format_params(pred['params']) or untuned}")
    if common.VERBOSITY >= 1:
        importance_report(pred)
    return pred


def practice_done_by(row, now):
    for name in ("Practice 3", "Practice 2", "Practice 1"):
        utc = session_utc(row, name)
        if utc is not None and utc < now - PRACTICE_BUFFER:
            return True
    return False


def _prequali_note(row, now):
    if not practice_done_by(row, now):
        print(f"\nNo practice sessions have finished yet, so the pole model relies on")
        print("season form only - run again after practice (or qualifying) for sharper predictions")


def resolve_target(args, year, schedule, data):
    completed = sorted(data["round"].unique())
    if not completed:
        raise SystemExit(f"no completed rounds found for the {year} season")
    now = utc_now()
    upcoming = []
    for _, row in schedule.iterrows():
        race_utc = session_utc(row, "Race")
        if race_utc is not None and race_utc > now:
            upcoming.append(row)

    # only the pole model can run before a round's qualifying (its feature
    # set never includes the target round's qualifying); the race
    # milestones need the grid and keep requiring quali to be done
    milestone = getattr(args, "milestone", None) or "pole"
    prequali_ok = bool(TARGETS[milestone].get("prequali"))

    if args.next:
        if not upcoming:
            raise SystemExit("no upcoming race left on this season's calendar")
        row = upcoming[0]
        quali_utc = session_utc(row, "Qualifying")
        if quali_utc is None or quali_utc > now:
            if prequali_ok:
                _prequali_note(row, now)
                return int(row["RoundNumber"]), "prequali", str(row["EventName"])
            raise SystemExit(f"qualifying for the {row['EventName']} has not happened yet, retry after quali")
        return int(row["RoundNumber"]), "pre", str(row["EventName"])

    if args.predict_round is not None:
        rn = args.predict_round
        match = schedule[schedule["RoundNumber"] == rn]
        if match.empty:
            raise SystemExit(f"round {rn} is not on the {year} calendar")
        row = match.iloc[0]
        race_utc = session_utc(row, "Race")
        quali_utc = session_utc(row, "Qualifying")
        if race_utc is not None and race_utc < now:
            if rn in completed:
                return rn, "post", str(row["EventName"])
            raise SystemExit(f"round {rn} happened recently but is not in the cache yet, retry in a few hours")
        if quali_utc is not None and quali_utc < now:
            return rn, "pre", str(row["EventName"])
        if prequali_ok:
            _prequali_note(row, now)
            return rn, "prequali", str(row["EventName"])
        raise SystemExit(f"cannot predict round {rn}: qualifying has not happened yet")

    if upcoming:
        row = upcoming[0]
        quali_utc = session_utc(row, "Qualifying")
        if quali_utc is not None and quali_utc < now:
            print(f"\nNext round ({row['EventName']}) has qualifying done -> predicting its milestones")
            return int(row["RoundNumber"]), "pre", str(row["EventName"])
        if prequali_ok:
            print(f"\nQualifying for the next round ({row['EventName']}) has not happened yet,")
            print(f"so the pole model predicts it before qualifying runs")
            _prequali_note(row, now)
            return int(row["RoundNumber"]), "prequali", str(row["EventName"])
        last_event = str(data.loc[data["round"] == completed[-1], "event"].iloc[0])
        print(f"\nQualifying for the next round ({row['EventName']}) has not happened yet,")
        print(f"so the models demonstrate on the last completed round instead")
        return completed[-1], "post", last_event

    last_event = str(data.loc[data["round"] == completed[-1], "event"].iloc[0])
    print(f"\nThe {year} season is finished -> demonstrating on the final round")
    return completed[-1], "post", last_event


def main():
    parser = argparse.ArgumentParser(
        description="F1 milestone prediction (pole, winner, first retirement, fastest lap) "
                    "using fastf1 and gradient boosting")
    parser.add_argument("--season", type=int, default=None,
                        help="season year (default: current year, falls back to previous year)")
    parser.add_argument("--predict-round", type=int, default=None,
                        help="round number to predict (default: auto-select)")
    parser.add_argument("--next", action="store_true",
                        help="predict the next race weekend's milestones (the race milestones "
                             "require its qualifying to be done; the pole model also predicts "
                             "before qualifying)")
    parser.add_argument("--min-train-rounds", type=int, default=MIN_TRAIN_ROUNDS,
                        help="minimum completed rounds needed before first prediction (default: 5)")
    parser.add_argument("--milestone", choices=list(TARGETS), default="pole",
                        help="which milestone model to run: pole, winner, first_dnf or "
                             "fastest_lap — only this one model is trained and backtested "
                             "(default: pole)")
    parser.add_argument("--model", choices=common.MODEL_PROFILES, default="fast",
                        help="model profile: 'fast' uses fixed hyperparameters, "
                             "'optimized' tunes them per model by minimizing "
                             "cross-validated log loss (better calibrated picks, slower run)")
    parser.add_argument("--refresh", action="store_true",
                        help="re-download season data, ignoring the local cache")
    common.add_verbosity_args(parser)
    args = parser.parse_args()
    common.apply_verbosity(args)

    setup()
    year = args.season or datetime.now().year
    schedule = fastf1.get_event_schedule(year, include_testing=False)
    data = collect_season(year, schedule, refresh=args.refresh)
    if data.empty:
        year -= 1
        print(f"\nNo completed rounds this season yet, falling back to {year}")
        schedule = fastf1.get_event_schedule(year, include_testing=False)
        data = collect_season(year, schedule, refresh=args.refresh)
        if data.empty:
            raise SystemExit("no race data available")

    target, mode, event_name = resolve_target(args, year, schedule, data)
    if mode == "prequali":
        completed_rounds = sorted(data["round"].unique())
        fallback = data[data["round"] == completed_rounds[-1]]
        upcoming = load_upcoming_round_prequali(year, target, event_name, fallback)
        data = pd.concat([data[data["round"] != target], upcoming], ignore_index=True)
    elif mode == "pre":
        upcoming = load_upcoming_round(year, target, event_name)
        data = pd.concat([data[data["round"] != target], upcoming], ignore_index=True)

    spec = TARGETS[args.milestone]
    features = build_features(data)
    print(f"\nMilestone: {spec['label'].lower()}")
    print(f"Dataset: {features['round'].nunique()} rounds, {len(features)} driver-round records")
    print(f"Features: {', '.join(spec['features'])}")

    backtest_report(backtest_records(features, args.min_train_rounds, args.model,
                                     args.milestone), args.milestone)
    final_report(features, year, target, event_name, mode, args.model, args.milestone)


if __name__ == "__main__":
    main()
