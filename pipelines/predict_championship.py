"""Final championship points prediction: remaining-points model (final −
points so far) vs direct model (absolute final points), baselined against a
naive run-rate projection (current points + season-average per round ×
remaining rounds). Runs for the drivers' or the constructors' championship
(--championship), in two modes: "live" forecasts the final standings of an
in-progress season as of its last completed round (or any earlier round via
--predict-round), "post" reviews a completed season the same way.

Unlike the round-based pipelines the models train ACROSS seasons: a
mid-season snapshot cannot know its own season's final points, so the
target of every training row is a driver's or team's final championship
points of a COMPLETED earlier season. Training snapshots come from every
complete season bundled in data/ (the race pipeline's season_*.csv — no new
data format), restricted to rounds ≤ the forecast round so the training
distribution mirrors the live task; the forecast season itself never
contributes training rows. The backtest is therefore leave-one-season-out:
every complete bundled season is forecast after each of its rounds, using
only earlier complete seasons, and the season's final round is skipped
(there is nothing left to forecast).

Championship points are the race results' points plus sprint points
derived from the stored sprint finishing positions (2021 awarded 3-2-1,
2022+ award 8-7-6-5-4-3-2-1; the derivation reproduces the official
standings of every bundled season). The naive run-rate baseline that
everything is compared against extrapolates current scoring pace over the
remaining rounds.
"""

import argparse
from datetime import datetime

import numpy as np
import pandas as pd
import fastf1

import f1_common as common
import predict_race as race

MIN_TRAIN_ROUNDS = common.MIN_TRAIN_ROUNDS

# Sprint points by finishing position, per points scheme. fastf1 race
# results award race-session points only (fastest-lap points included, since
# they are scored in the race), so sprint points are added from the stored
# sprint positions. 2021: 3-2-1; since 2022: 8-7-6-5-4-3-2-1; before 2021
# there were no sprints (the column stays empty). Positions outside the
# scheme score nothing.
SPRINT_POINTS_2021 = {1: 3.0, 2: 2.0, 3: 1.0}
SPRINT_POINTS = {pos: float(9 - pos) for pos in range(1, 9)}

setup = common.setup
utc_now = common.utc_now
session_utc = common.session_utc


def sprint_points_map(year):
    return SPRINT_POINTS_2021 if year == 2021 else SPRINT_POINTS


def _finished_like(status):
    """Classified as running the full distance — not a retirement: "Finished",
    the old "+N Lap(s)" classifications, the 2026 "Lapped", and disqualifications
    (a DSQ completed the race; the stewards removed the result afterwards)."""
    text = str(status)
    return (text == "Finished" or text == "Lapped" or text == "Disqualified"
            or text.startswith("+"))


def collect_season(year, schedule, refresh=False, rate_limit_wait=None):
    """The championship pipeline reads the race pipeline's season files —
    same filename, same loader, same refresh workflow — so this is a thin
    pass-through to keep data collection in one place."""
    return race.collect_season(year, schedule, refresh=refresh,
                               rate_limit_wait=rate_limit_wait)


DRIVER_FEATURES = [
    "rounds_done",
    "rounds_remaining",
    "points_so_far",
    "points_per_round",
    "points_rate_projection",
    "points_last3_rate",
    "standings_pos",
    "gap_to_leader",
    "leader_points",
    "finish_avg",
    "finish_form3",
    "best_finish",
    "quali_delta_avg",
    "quali_delta_form3",
    "wins",
    "podiums",
    "dnf_count",
    "dnf_rate",
    "team_points_so_far",
    "team_points_per_round",
    "team_points_last3_rate",
    "team_finish_avg",
    "team_dnf_rate",
    "team_id",
    "driver_id",
]

TEAM_FEATURES = [
    "rounds_done",
    "rounds_remaining",
    "points_so_far",
    "points_per_round",
    "points_rate_projection",
    "points_last3_rate",
    "standings_pos",
    "gap_to_leader",
    "leader_points",
    "finish_avg",
    "best_finish",
    "quali_delta_avg",
    "podiums",
    "dnf_count",
    "dnf_rate",
    "team_id",
]

CHAMPIONSHIPS = {
    "drivers": {
        "label": "Drivers' championship",
        "entrant": "driver",
        "tiebreak": "driver_number",
        "features": DRIVER_FEATURES,
    },
    "constructors": {
        "label": "Constructors' championship",
        "entrant": "team",
        "tiebreak": "team",
        "features": TEAM_FEATURES,
    },
}


# --- season data -------------------------------------------------------------

def bundled_season_years():
    """Years with a bundled race-season CSV (the training data source)."""
    years = []
    for path in sorted(common.DATA_DIR.glob("season_*.csv")):
        try:
            years.append(int(path.stem.split("_")[1]))
        except (IndexError, ValueError):
            continue
    return years


def read_season_csv(year):
    return pd.read_csv(common.DATA_DIR / f"season_{year}.csv",
                       dtype={"driver_number": str})


def season_is_complete(year, schedule, data):
    """Whether the season's whole calendar has run AND is covered by the
    data (the final round only counts COMPLETION_BUFFER after its race, so
    a race in progress never produces a complete season)."""
    rounds_total = int(schedule["RoundNumber"].max())
    if rounds_total not in {rn for rn, _ in common.completed_rounds(schedule)}:
        return False
    return bool(data["round"].max() >= rounds_total)


def load_training_seasons(schedule_loader=None):
    """{year: (data, rounds_total)} for every complete season bundled in
    data/ — the pipeline's training data. Incomplete seasons (a season in
    progress has no known final standings and cannot provide targets) are
    skipped with a note. `schedule_loader` lets the web app share its
    TTL-cached schedule lookup; the CLI passes nothing and uses fastf1."""
    if schedule_loader is None:
        def schedule_loader(y):
            return fastf1.get_event_schedule(y, include_testing=False)
    seasons = {}
    for year in bundled_season_years():
        try:
            data = read_season_csv(year)
        except (OSError, ValueError) as exc:
            common.vprint(1, f"season {year}: unreadable ({exc}), skipped")
            continue
        schedule = schedule_loader(year)
        if season_is_complete(year, schedule, data):
            seasons[year] = (data, int(schedule["RoundNumber"].max()))
        else:
            common.vprint(1, f"season {year}: still in progress, not usable as training data")
    return seasons


def complete_season_years(schedule_loader=None):
    """Sorted years of every complete bundled season — the candidate seasons
    of the leave-one-season-out backtest and its first-season setting. The
    same completeness rule load_training_seasons applies, without keeping
    the frames (cheap disk reads; the web app re-runs it per sidebar render).
    `schedule_loader` lets the web app share its TTL-cached schedule lookup;
    the CLI passes nothing and uses fastf1."""
    if schedule_loader is None:
        def schedule_loader(y):
            return fastf1.get_event_schedule(y, include_testing=False)
    years = []
    for year in bundled_season_years():
        try:
            data = read_season_csv(year)
        except (OSError, ValueError):
            continue
        if season_is_complete(year, schedule_loader(year), data):
            years.append(year)
    return sorted(years)


# --- snapshots ----------------------------------------------------------------

def _season_points(data, year):
    """One season's rows with round_points (race + derived sprint points)
    and retirement flags added."""
    df = data.copy()
    df["driver_number"] = df["driver_number"].astype(str)
    df = df.sort_values(["round", "driver_number"]).reset_index(drop=True)
    df["round_points"] = (df["points"].fillna(0.0)
                          + df["sprint_finish"].map(sprint_points_map(year)).fillna(0.0))
    df["dnf"] = [int(pd.notna(f) and not _finished_like(s))
                 for f, s in zip(df["finish"], df["status"])]
    df["win_flag"] = (df["finish"] == 1).astype(int)
    df["podium_flag"] = (df["finish"] <= 3).astype(int)
    return df


def _driver_snapshots(df, year, rounds_total):
    """One row per (driver, round): the state of the championship AFTER that
    round completed — points so far, scoring rates, standings position, form
    and reliability stats, all inclusive of the round itself (a snapshot at
    the last completed round is exactly the live forecasting task)."""
    d = df.sort_values(["driver_number", "round"]).copy()
    by_driver = d.groupby("driver_number", sort=False)
    d["rounds_done"] = by_driver.cumcount() + 1
    d["points_so_far"] = by_driver["round_points"].cumsum()
    d["points_last3_rate"] = by_driver["round_points"].transform(
        lambda s: s.rolling(3, min_periods=1).mean())
    d["finish_avg"] = by_driver["finish"].transform(
        lambda s: s.expanding(min_periods=1).mean())
    d["finish_form3"] = by_driver["finish"].transform(
        lambda s: s.rolling(3, min_periods=1).mean())
    d["best_finish"] = by_driver["finish"].transform(
        lambda s: s.expanding(min_periods=1).min())
    d["quali_delta_avg"] = by_driver["quali_delta"].transform(
        lambda s: s.expanding(min_periods=1).mean())
    d["quali_delta_form3"] = by_driver["quali_delta"].transform(
        lambda s: s.rolling(3, min_periods=1).mean())
    d["wins"] = d.groupby("driver_number", sort=False)["win_flag"].cumsum()
    d["podiums"] = d.groupby("driver_number", sort=False)["podium_flag"].cumsum()
    d["dnf_count"] = d.groupby("driver_number", sort=False)["dnf"].cumsum()
    d["points_per_round"] = d["points_so_far"] / d["rounds_done"]
    d["dnf_rate"] = d["dnf_count"] / d["rounds_done"]
    d["rounds_remaining"] = rounds_total - d["round"]
    # the naive run-rate projection as an explicit feature: trees cannot
    # multiply, so handing them the product lets the models learn corrections
    # to the baseline instead of re-deriving it from the components (and
    # losing to it late in the season, where the run-rate is near-exact)
    d["points_rate_projection"] = d["points_per_round"] * d["rounds_remaining"]

    team_rounds = _team_rounds(df)
    # only the team-level features ride along on driver rows (renamed with a
    # team_ prefix — the team-round frame's own columns use the same generic
    # names the driver columns already use)
    d = d.merge(
        team_rounds[["round", "team", "points_so_far", "points_per_round",
                     "points_last3_rate", "finish_avg", "dnf_rate"]].rename(columns={
            "points_so_far": "team_points_so_far",
            "points_per_round": "team_points_per_round",
            "points_last3_rate": "team_points_last3_rate",
            "finish_avg": "team_finish_avg",
            "dnf_rate": "team_dnf_rate",
        }),
        on=["round", "team"], how="left")

    d["standings_pos"] = d.groupby("round")["points_so_far"].rank(
        method="min", ascending=False)
    d["leader_points"] = d.groupby("round")["points_so_far"].transform("max")
    d["gap_to_leader"] = d["leader_points"] - d["points_so_far"]
    return d


def _team_rounds(df):
    """Per (round, team) aggregates — the constructors' snapshot base, and
    the source of the driver snapshot's team features (one row per team per
    round, so expanding stats count every round exactly once)."""
    tr = df.groupby(["round", "team"], sort=False).agg(
        event=("event", "first"),
        round_points=("round_points", "sum"),
        team_round_finish=("finish", "mean"),
        team_round_best=("finish", "min"),
        team_round_qdelta=("quali_delta", "mean"),
        team_round_dnf=("dnf", "sum"),
        team_round_podiums=("podium_flag", "sum"),
    ).reset_index().sort_values(["team", "round"])
    by_team = tr.groupby("team", sort=False)
    tr["rounds_done"] = by_team.cumcount() + 1
    tr["points_so_far"] = by_team["round_points"].cumsum()
    tr["points_per_round"] = tr["points_so_far"] / tr["rounds_done"]
    tr["points_last3_rate"] = by_team["round_points"].transform(
        lambda s: s.rolling(3, min_periods=1).mean())
    tr["finish_avg"] = by_team["team_round_finish"].transform(
        lambda s: s.expanding(min_periods=1).mean())
    tr["best_finish"] = by_team["team_round_best"].transform(
        lambda s: s.expanding(min_periods=1).min())
    tr["quali_delta_avg"] = by_team["team_round_qdelta"].transform(
        lambda s: s.expanding(min_periods=1).mean())
    tr["podiums"] = by_team["team_round_podiums"].cumsum()
    tr["dnf_count"] = by_team["team_round_dnf"].cumsum()
    tr["dnf_rate"] = tr["dnf_count"] / tr["rounds_done"]
    return tr


def _team_snapshots(df, year, rounds_total):
    """One row per (team, round) for the constructors' championship — the
    same expanding-state shape as the driver snapshots."""
    tr = _team_rounds(df).copy()
    tr["rounds_remaining"] = rounds_total - tr["round"]
    tr["points_rate_projection"] = tr["points_per_round"] * tr["rounds_remaining"]
    tr["standings_pos"] = tr.groupby("round")["points_so_far"].rank(
        method="min", ascending=False)
    tr["leader_points"] = tr.groupby("round")["points_so_far"].transform("max")
    tr["gap_to_leader"] = tr["leader_points"] - tr["points_so_far"]
    return tr


DRIVER_SNAPSHOT_COLUMNS = [
    "round", "event", "driver_number", "driver", "team",
    "rounds_done", "rounds_remaining", "round_points", "points_so_far",
    "points_per_round", "points_rate_projection", "points_last3_rate",
    "standings_pos", "gap_to_leader", "leader_points",
    "finish_avg", "finish_form3", "best_finish",
    "quali_delta_avg", "quali_delta_form3",
    "wins", "podiums", "dnf_count", "dnf_rate",
    "team_points_so_far", "team_points_per_round", "team_points_last3_rate",
    "team_finish_avg", "team_dnf_rate",
]

TEAM_SNAPSHOT_COLUMNS = [
    "round", "event", "team",
    "rounds_done", "rounds_remaining", "round_points", "points_so_far",
    "points_per_round", "points_rate_projection", "points_last3_rate",
    "standings_pos", "gap_to_leader", "leader_points",
    "finish_avg", "best_finish", "quali_delta_avg",
    "podiums", "dnf_count", "dnf_rate",
]


def build_snapshots(data, year, rounds_total, championship="drivers"):
    """One season's snapshot frame for the selected championship."""
    df = _season_points(data, year)
    if championship == "constructors":
        snap = _team_snapshots(df, year, rounds_total)
        snap = snap[TEAM_SNAPSHOT_COLUMNS]
    else:
        snap = _driver_snapshots(df, year, rounds_total)
        snap = snap[DRIVER_SNAPSHOT_COLUMNS]
    snap.insert(0, "year", year)
    return snap.reset_index(drop=True)


def build_features(seasons, championship="drivers"):
    """Snapshot frames of several seasons concatenated into one, with stable
    team/driver identity ids and the final-points target attached to every
    snapshot of a completed season (NaN for a season in progress, which can
    never provide its own targets).

    seasons: iterable of dicts {year, data, rounds_total, complete}.
    """
    spec = CHAMPIONSHIPS[championship]
    frames = [build_snapshots(s["data"], s["year"], s["rounds_total"], championship)
              for s in seasons]
    features = pd.concat(frames, ignore_index=True)
    features = features.sort_values(
        ["year", "round", spec["tiebreak"]]).reset_index(drop=True)
    features["team_id"] = pd.factorize(features["team"])[0]
    if championship == "drivers":
        # identity = the stable three-letter code (permanent car numbers are
        # not: the reigning champion runs #1)
        features["driver_id"] = pd.factorize(features["driver"])[0]
    features["final"] = np.nan
    for season in seasons:
        if not season["complete"]:
            continue
        mask = features["year"] == season["year"]
        rows = features[mask]
        # final points = the entrant's points at their last completed round
        # (standings freeze for a driver who misses the season finale)
        last_round = rows.groupby(spec["entrant"])["round"].transform("max")
        finals = rows.loc[rows["round"] == last_round].set_index(spec["entrant"])["points_so_far"]
        features.loc[mask, "final"] = rows[spec["entrant"]].map(finals)
    return features


def load_feature_seasons(year, data, schedule, schedule_loader=None,
                        championship="drivers", complete=None):
    """Complete bundled seasons plus the target season, turned into one
    snapshot feature frame. The target season rides along even while in
    progress (its rows get no target); complete seasons provide the training
    data whatever the target is."""
    if complete is None:
        complete = season_is_complete(year, schedule, data)
    seasons = []
    training = load_training_seasons(schedule_loader)
    for y in sorted(training):
        if y == year:
            continue
        tdata, rounds_total = training[y]
        seasons.append({"year": y, "data": tdata, "rounds_total": rounds_total,
                        "complete": True})
    seasons.append({"year": year, "data": data,
                    "rounds_total": int(schedule["RoundNumber"].max()),
                    "complete": complete})
    if len(training) > 1:
        span = f"{min(training)}-{max(training)}"
    elif training:
        span = str(min(training))
    else:
        span = "none"
    print(f"Training data: {len(training)} complete bundled season(s) ({span})")
    return build_features(seasons, championship)


# --- models -------------------------------------------------------------------

def usable_features(frame, feature_names):
    """feature_names minus any column that is entirely NaN in `frame` —
    sklearn's Hist gradient boosting cannot fit a feature with no observed
    values (the driver-id column can be all-unknown early, and quali/race
    form is empty for a debut-only entrant set)."""
    matrix = frame[feature_names].to_numpy(dtype=float)
    return [name for name, col in zip(feature_names, matrix.T) if not np.isnan(col).all()]


def train_model(features, target_year, snapshot_round, mode="remaining",
                profile="fast", championship="drivers"):
    """Fit one championship model on snapshots of every COMPLETE season
    before `target_year`, at rounds up to `snapshot_round` (the training
    distribution mirrors the live task: the forecast season itself never
    contributes training rows, and later snapshots of past seasons are
    excluded so early- and late-season forecasts train alike).

    mode "remaining" learns the points still to be scored (final − points
    so far) — the race pipeline's gain-model analogue, anchored at the
    current standings; mode "direct" learns the absolute final points.
    Returns (model, train frame, used feature list, tuned hyperparameters).
    """
    spec = CHAMPIONSHIPS[championship]
    train = features[
        (features["year"] < target_year)
        & (features["round"] <= snapshot_round)
        & features["final"].notna()
    ]
    if train.empty:
        raise ValueError(f"no complete earlier season to train on for a {target_year} forecast")
    if mode == "remaining":
        target = (train["final"] - train["points_so_far"]).to_numpy(dtype=float)
    else:
        target = train["final"].to_numpy(dtype=float)
    used = usable_features(train, spec["features"])
    X = train[used].to_numpy(dtype=float)
    categorical = [used.index(c) for c in ("team_id", "driver_id") if c in used]
    tuned = None
    if profile == "optimized":
        # leave-one-season-out folds: the tuner's round array is the season year
        tuned = common.tune_hyperparameters(X, target, train["year"].to_numpy(),
                                             categorical_features=categorical)
    model = common.make_model(categorical, profile, tuned)
    model.fit(X, target)
    return model, train, used, tuned


def _rank_frame(rows, value, championship):
    """Deterministic standings order for one projected column: sort by the
    projection (desc), then current points (desc), then the stable
    entrant key."""
    spec = CHAMPIONSHIPS[championship]
    ordered = rows.sort_values([value, "points_so_far", spec["tiebreak"]],
                               ascending=[False, False, True])
    ordered = ordered.copy()
    ordered["pred_pos"] = np.arange(1, len(ordered) + 1)
    return ordered.sort_values("pred_pos")


def predict_snapshot(model, features, target_year, snapshot_round, mode,
                     use_features, championship="drivers"):
    """Projected final standings for one season as of one round. Remaining
    model output is clamped at zero (points cannot be un-scored), direct
    model output at the current points (a final total below what is already
    scored is impossible); the rank ties break on current points."""
    rows = features[(features["year"] == target_year)
                    & (features["round"] == snapshot_round)].copy()
    pred = model.predict(rows[use_features].to_numpy(dtype=float))
    if mode == "remaining":
        rows["pred_final"] = rows["points_so_far"] + np.clip(pred, 0.0, None)
    else:
        rows["pred_final"] = np.maximum(pred, rows["points_so_far"])
    return _rank_frame(rows, "pred_final", championship)


def baseline_projection(features, target_year, snapshot_round, championship="drivers"):
    """The naive forecast every model is compared against: current points +
    season-average points per round × remaining rounds (run-rate
    extrapolation)."""
    rows = features[(features["year"] == target_year)
                    & (features["round"] == snapshot_round)].copy()
    rows["pred_final"] = rows["points_so_far"] + rows["points_per_round"] * rows["rounds_remaining"]
    return _rank_frame(rows, "pred_final", championship)


# --- backtest -------------------------------------------------------------------

def backtest_records(features, min_train_rounds, profile="fast", models="both",
                     championship="drivers", first_season=None):
    """Leave-one-season-out backtest: every complete bundled season forecast
    after each of its rounds (from the first round with `min_train_rounds`
    completed rounds before it, up to but excluding the season's final
    round — as of the final round there is nothing left to forecast), each
    forecast trained only on complete seasons before it, at snapshot rounds
    up to the forecast round.

    first_season scopes the EVALUATION to that season and later (None — the
    default, auto — forecasts every complete bundled season that has at
    least one earlier complete season to train on; the earliest bundled
    season is never forecast, whatever this is set to). The final forecast
    is unaffected: it always trains on every complete earlier season.

    One record per (season, round): the final-points MAE, champion hit,
    top-3 overlap and rank correlation of the selected models and of the
    run-rate baseline, the top picks and the actual champion, and the
    number of training seasons. Prints one live progress line per forecast
    (streamed into the web app's log).
    """
    spec = CHAMPIONSHIPS[championship]
    entrant = spec["entrant"]
    records = []
    tune_note = " (tuning hyperparameters)" if profile == "optimized" else ""
    model_names = [m for m in ("remaining", "direct") if models in ("both", m)]
    if first_season is not None:
        common.vprint(1, f"backtest scope: seasons from {first_season} onwards "
                         "(first backtest season setting)")
    for year in sorted(int(y) for y in features["year"].unique()):
        if first_season is not None and year < first_season:
            continue
        rows_season = features[features["year"] == year]
        if rows_season["final"].isna().all():
            continue  # a season in progress has no final standings to score
        n_train = int(features.loc[(features["year"] < year)
                                   & features["final"].notna(), "year"].nunique())
        if n_train == 0:
            common.vprint(1, f"backtest {year}: no complete earlier season to train on, skipped")
            continue
        rounds_total = int(rows_season["round"].max())
        for rnd in range(min_train_rounds + 1, rounds_total):
            rows = features[(features["year"] == year) & (features["round"] == rnd)]
            event = str(rows["event"].iloc[0])
            common.vprint(1, f"backtest {year} round {rnd:>2}  {event}: training "
                          f"{' + '.join(model_names)} model{'s' if len(model_names) > 1 else ''}{tune_note}...")
            actual = rows.sort_values(["final", spec["tiebreak"]],
                                      ascending=[False, True])
            actual_top1 = str(actual.iloc[0][entrant])
            actual_top3 = set(actual.head(3)[entrant])
            rec = {"season": year, "round": rnd, "event": event,
                   "train_seasons": n_train}
            try:
                fitted = {}
                for mode in model_names:
                    model, _train, used, _tuned = train_model(
                        features, year, rnd, mode, profile, championship)
                    fitted[mode] = (model, used)
            except ValueError:
                continue
            for mode, (model, used) in fitted.items():
                pred = predict_snapshot(model, features, year, rnd, mode, used, championship)
                top1 = str(pred.iloc[0][entrant])
                rec[f"{mode}_mae"] = float((pred["pred_final"] - pred["final"]).abs().mean())
                rec[f"{mode}_champion"] = int(top1 == actual_top1)
                rec[f"{mode}_podium"] = len(set(pred.head(3)[entrant]) & actual_top3)
                rec[f"{mode}_corr"] = common.rank_corr(pred["pred_pos"], -pred["final"])
                rec[f"{mode}_top1"] = top1
            base = baseline_projection(features, year, rnd, championship)
            base_top1 = str(base.iloc[0][entrant])
            rec["baseline_mae"] = float((base["pred_final"] - base["final"]).abs().mean())
            rec["baseline_champion"] = int(base_top1 == actual_top1)
            rec["baseline_podium"] = len(set(base.head(3)[entrant]) & actual_top3)
            rec["baseline_corr"] = common.rank_corr(base["pred_pos"], -base["final"])
            rec["baseline_top1"] = base_top1
            rec["actual_top1"] = actual_top1
            records.append(rec)
            detail = "/".join(f"{rec[f'{m}_mae']:.0f}" for m in model_names)
            common.vprint(2, f"  {year} round {rnd:>2} result: MAE {detail}/base {rec['baseline_mae']:.0f} | "
                          f"champion {'/'.join(rec[f'{m}_top1'] for m in model_names)}/"
                          f"{base_top1} vs {actual_top1}")
    return records


def _first_correct(records, season, field):
    for rec in records:
        if rec["season"] == season and rec.get(field) == 1:
            return int(rec["round"])
    return None


def _rate(records, field, season=None):
    values = [rec[field] for rec in records
              if field in rec and (season is None or rec["season"] == season)
              and pd.notna(rec.get(field))]
    return float(np.mean(values)) if values else np.nan


def backtest_report(records, models="both", championship="drivers"):
    show_remaining = models in ("both", "remaining")
    show_direct = models in ("both", "direct")
    print(f"\n=== Rolling backtest: remaining-points model vs direct model vs run-rate baseline ===")
    print("leave-one-season-out: every complete bundled season forecast after each round,")
    print("each forecast trained only on earlier complete seasons (the final round is skipped —")
    print("nothing is left to forecast as of the season's last round)")
    if not records:
        print("not enough completed seasons for a backtest yet")
        return
    seasons = sorted({rec["season"] for rec in records})
    if common.VERBOSITY >= 1:
        print(f"\n{'season':>6} {'fcsts':>5}  {'first correct round':>19}  "
              f"{'champion hit rate':>25}  {'final-points MAE':>25}")
        for season in seasons:
            recs = [rec for rec in records if rec["season"] == season]
            firsts = "/".join(
                (str(_first_correct(records, season, f"{m}_champion")) if show else "")
                for m, show in (("remaining", show_remaining), ("direct", show_direct),
                                ("baseline", True)))
            hits = "/".join(
                (f"{100 * _rate(recs, f'{m}_champion'):.0f}%" if show else "")
                for m, show in (("remaining", show_remaining), ("direct", show_direct),
                                ("baseline", True)))
            maes = "/".join(
                (f"{_rate(recs, f'{m}_mae'):.1f}" if show else "")
                for m, show in (("remaining", show_remaining), ("direct", show_direct),
                                ("baseline", True)))
            print(f"{season:>6} {len(recs):>5}  {firsts:>19}  {hits:>25}  {maes:>25}")
    n = len(records)
    print(f"\nMean over {n} forecasts ({len(seasons)} seasons):")
    header = f"{'':>26}"
    if show_remaining:
        header += f"{'remaining':>11}"
    if show_direct:
        header += f"{'direct':>9}"
    header += f"{'baseline':>9}"
    print(header)
    line = f"{'champion hit rate':<26}"
    for field in ("remaining_champion", "direct_champion", "baseline_champion"):
        if field.startswith("remaining") and not show_remaining:
            continue
        if field.startswith("direct") and not show_direct:
            continue
        line += f"{100 * _rate(records, field):>10.0f}%"
    print(line)
    line = f"{'top-3 standings overlap':<26}"
    for field in ("remaining_podium", "direct_podium", "baseline_podium"):
        if field.startswith("remaining") and not show_remaining:
            continue
        if field.startswith("direct") and not show_direct:
            continue
        line += f"{_rate(records, field):>10.2f}"
    print(line)
    line = f"{'final-points MAE':<26}"
    for field in ("remaining_mae", "direct_mae", "baseline_mae"):
        if field.startswith("remaining") and not show_remaining:
            continue
        if field.startswith("direct") and not show_direct:
            continue
        line += f"{_rate(records, field):>10.2f}"
    print(line)
    line = f"{'rank correlation':<26}"
    for field in ("remaining_corr", "direct_corr", "baseline_corr"):
        if field.startswith("remaining") and not show_remaining:
            continue
        if field.startswith("direct") and not show_direct:
            continue
        rate = _rate(records, field)
        line += f"{rate:>10.2f}" if pd.notna(rate) else f"{'-':>10}"
    print(line)


# --- final forecast -------------------------------------------------------------

def final_predictions(features, target_year, snapshot_round, profile="fast",
                      models="both", championship="drivers"):
    """Train the final models on every complete season before `target_year`
    and project the final standings as of `snapshot_round`.

    Returns a bundle with the ranked projection frames of the selected
    models ("remaining" / "direct"), the run-rate baseline, the models
    themselves, the training frame, the feature lists actually used, the
    hyperparameters the profile picked and the number of training seasons;
    with both models selected the remaining frame carries direct_pos and
    direct_final columns for side-by-side display. When there is no
    training data (forecasting the earliest bundled season) the bundle
    carries an "error" message instead. Prints one live progress line per
    trained model (streamed into the web app's log).
    """
    spec = CHAMPIONSHIPS[championship]
    entrant = spec["entrant"]
    tune_note = " (tuning hyperparameters)" if profile == "optimized" else ""
    bundle = {"championship": championship}
    try:
        if models in ("both", "remaining"):
            common.vprint(1, f"training final remaining-points model{tune_note}...")
            remaining_model, train, remaining_used, remaining_params = train_model(
                features, target_year, snapshot_round, "remaining", profile, championship)
            remaining = predict_snapshot(remaining_model, features, target_year,
                                          snapshot_round, "remaining", remaining_used,
                                          championship)
            bundle.update({"remaining_model": remaining_model, "remaining": remaining,
                           "remaining_features": remaining_used,
                           "remaining_params": remaining_params, "train": train})
        if models in ("both", "direct"):
            common.vprint(1, f"training final direct model{tune_note}...")
            direct_model, direct_train, direct_used, direct_params = train_model(
                features, target_year, snapshot_round, "direct", profile, championship)
            direct = predict_snapshot(direct_model, features, target_year,
                                      snapshot_round, "direct", direct_used,
                                      championship)
            bundle.update({"direct_model": direct_model, "direct": direct,
                           "direct_features": direct_used,
                           "direct_params": direct_params})
            if "remaining" in bundle:
                key = direct.set_index(entrant)
                remaining["direct_pos"] = remaining[entrant].map(key["pred_pos"]).astype(int)
                remaining["direct_final"] = remaining[entrant].map(key["pred_final"])
    except ValueError as exc:
        return {"championship": championship, "error": str(exc)}
    bundle["baseline"] = baseline_projection(features, target_year,
                                             snapshot_round, championship)
    bundle["train_seasons"] = int(bundle["train"]["year"].nunique())
    return bundle


def importance_frame(pred, mode="remaining"):
    """Permutation importance for one of the final models, as a DataFrame."""
    if "error" in pred:
        return pd.DataFrame(columns=["feature", "importance", "std"])
    train = pred["train"]
    used = pred[f"{mode}_features"]
    if mode == "remaining":
        target = (train["final"] - train["points_so_far"]).to_numpy(dtype=float)
    else:
        target = train["final"].to_numpy(dtype=float)
    return common.importance_scores(pred[f"{mode}_model"], used,
                                    train[used].to_numpy(dtype=float), target)


def final_report(features, year, target, event_name, mode, profile="fast",
                 models="both", championship="drivers", rounds_total=None):
    spec = CHAMPIONSHIPS[championship]
    entrant = spec["entrant"]
    pred = final_predictions(features, year, target, profile, models, championship)
    label = spec["label"].lower()
    print(f"\n=== Prediction: final {year} {label} standings (as of round {target} — {event_name}) ===")
    if rounds_total:
        if mode == "live":
            print(f"season in progress: {target} of {rounds_total} rounds completed, "
                  f"{rounds_total - target} to go")
        else:
            print(f"season complete ({rounds_total} rounds)")
            if target == rounds_total:
                print("as of the final round there is nothing left to forecast — this is the")
                print("converged state; use --predict-round for an earlier retro-forecast")
    if "error" in pred:
        print(f"\nNo model — {pred['error']}")
        return pred
    with_remaining = "remaining" in pred
    with_direct = "direct" in pred
    main = pred["remaining"] if with_remaining else pred["direct"]
    points_fmt = lambda v: f"{v:g}"
    baseline_finals = pred["baseline"].set_index(entrant)["pred_final"]
    # constructors: the entrant IS the team — a separate team column would repeat it
    name_cols = (f"{entrant:<4} {'team':<20} " if championship == "drivers"
                 else f"{entrant:<20} ")
    header = f"{'rem':>4} "
    if with_direct:
        header += f"{'direct':>6}  "
    header += f"{name_cols}{'now':>5} "
    if with_remaining and with_direct:
        header += f"{'rem pts':>8} {'dir pts':>8} "
    else:
        header += f"{'proj pts':>8} "
    header += f"{'base pts':>8}"
    if mode == "post":
        header += f" {'final':>6}"
        if with_remaining:
            header += f" {'err rem':>7}"
        if with_direct:
            header += f" {'err dir':>7}"
    print(header)
    for _, row in main.iterrows():
        line = f"{'P' + str(row['pred_pos']):>4} "
        if with_direct:
            line += f"{'P' + str(row['direct_pos']):>6}  "
        line += (f"{row[entrant]:<4} {row['team']:<20} " if championship == "drivers"
                 else f"{row[entrant]:<20} ")
        line += f"{points_fmt(row['points_so_far']):>5} "
        if with_remaining and with_direct:
            line += f"{points_fmt(row['pred_final']):>8} {points_fmt(row['direct_final']):>8} "
        else:
            line += f"{points_fmt(row['pred_final']):>8} "
        line += f"{points_fmt(baseline_finals.get(row[entrant], float('nan'))):>8}"
        if mode == "post":
            line += f" {points_fmt(row['final']):>6}"
            if with_remaining:
                line += f" {row['pred_final'] - row['final']:>+7g}"
            if with_direct:
                err = (row["direct_final"] if with_remaining else row["pred_final"]) - row["final"]
                line += f" {err:>+7g}"
        print(line)
    top = main.iloc[0]
    second = main.iloc[1] if len(main) > 1 else None
    top_base = pred["baseline"].iloc[0]
    if with_remaining:
        margin = f", margin {top['pred_final'] - second['pred_final']:+g} over {second[entrant]}" \
            if second is not None else ""
        print(f"\nPredicted champion — remaining model: {top[entrant]} "
              f"({points_fmt(top['pred_final'])} pts{margin})")
    if with_direct:
        top_direct = (main.sort_values("direct_pos").iloc[0] if with_direct and with_remaining
                      else pred["direct"].iloc[0])
        print(f"Predicted champion — direct model:    {top_direct[entrant]} "
              f"({points_fmt(top_direct['direct_final'] if with_remaining else top_direct['pred_final'])} pts)")
    print(f"Baseline projection champion:          {top_base[entrant]} "
          f"({points_fmt(top_base['pred_final'])} pts)")
    if mode == "post":
        actual_rows = features[(features["year"] == year) & (features["round"] == target)]
        actual = actual_rows.sort_values(["final", spec["tiebreak"]],
                                        ascending=[False, True])
        actual_champion = actual.iloc[0]
        verdicts = []
        if with_remaining:
            verdicts.append(f"remaining model {'hit' if top[entrant] == actual_champion[entrant] else 'miss'}")
        if with_direct:
            top_direct = (main.sort_values("direct_pos").iloc[0] if with_remaining
                          else pred["direct"].iloc[0])
            verdicts.append(f"direct model {'hit' if top_direct[entrant] == actual_champion[entrant] else 'miss'}")
        verdicts.append(f"baseline {'hit' if top_base[entrant] == actual_champion[entrant] else 'miss'}")
        print(f"Actual champion:                       {actual_champion[entrant]} "
              f"({points_fmt(actual_champion['final'])} pts) — {'; '.join(verdicts)}")
    if profile == "optimized":
        untuned = "fast defaults (no clearly better combination found)"
        if with_remaining:
            print(f"\nTuned hyperparameters (remaining model): "
                  f"{common.format_params(pred['remaining_params']) or untuned}")
        if with_direct:
            print(f"Tuned hyperparameters (direct model):   "
                  f"{common.format_params(pred['direct_params']) or untuned}")
    if common.VERBOSITY >= 1:
        train = pred["train"]
        if with_remaining:
            common.importance_report(pred["remaining_model"], pred["remaining_features"],
                                     train[pred["remaining_features"]].to_numpy(dtype=float),
                                     (train["final"] - train["points_so_far"]).to_numpy(dtype=float),
                                     "remaining", width=22)
        if with_direct:
            common.importance_report(pred["direct_model"], pred["direct_features"],
                                     train[pred["direct_features"]].to_numpy(dtype=float),
                                     train["final"].to_numpy(dtype=float),
                                     "direct", width=22)
    return pred


# --- target resolution -----------------------------------------------------------

def resolve_target(args, year, schedule, data):
    """(snapshot_round, mode, event_name) for the forecast: the round the
    standings are projected AS OF. Auto = the last completed round; an
    explicit round must already be completed (a future round carries no
    information). A snapshot needs at least `min_train_rounds` completed
    rounds before it — the same gate the backtest applies to its first
    forecast. mode is "live" for a season in progress, "post" once the
    season's whole calendar has completed."""
    completed = sorted(int(r) for r in data["round"].unique())
    if not completed:
        raise SystemExit(f"no completed rounds found for the {year} season")
    snapshot = getattr(args, "predict_round", None)
    if snapshot is None:
        snapshot = completed[-1]
    else:
        snapshot = int(snapshot)
        if snapshot not in completed:
            raise SystemExit(f"round {snapshot} has not been completed yet "
                             f"(last completed round: {completed[-1]})")
    min_train = getattr(args, "min_train_rounds", MIN_TRAIN_ROUNDS)
    rounds_before = sum(1 for r in completed if r < snapshot)
    if rounds_before < min_train:
        raise SystemExit(f"a forecast as of round {snapshot} has only {rounds_before} completed "
                         f"rounds before it — a championship forecast needs at least {min_train}; "
                         f"pick a later round or lower the minimum")
    event = str(data.loc[data["round"] == snapshot, "event"].iloc[0])
    mode = "post" if season_is_complete(year, schedule, data) else "live"
    return snapshot, mode, event


def main():
    parser = argparse.ArgumentParser(
        description="F1 final championship points prediction using fastf1 and gradient boosting")
    parser.add_argument("--season", type=int, default=None,
                        help="season year whose final standings are forecast "
                             "(default: current year, falls back to previous year)")
    parser.add_argument("--championship", choices=list(CHAMPIONSHIPS), default="drivers",
                        help="which championship to forecast: drivers (default) or constructors")
    parser.add_argument("--models", choices=("both", "remaining", "direct"), default="both",
                        help="which models to train, backtest and forecast: both (default), "
                             "only the remaining-points model, or only the direct model")
    parser.add_argument("--predict-round", "--snapshot-round", type=int, default=None,
                        dest="predict_round",
                        help="project the standings as of this completed round "
                             "(default: the season's last completed round)")
    parser.add_argument("--min-train-rounds", type=int, default=MIN_TRAIN_ROUNDS,
                        help="minimum completed rounds before the first forecast "
                             "round (default: 5)")
    parser.add_argument("--first-backtest-season", type=int, default=None,
                        help="first season the leave-one-season-out backtest "
                             "forecasts (default: every complete bundled season "
                             "that has an earlier complete season to train on)")
    parser.add_argument("--model", choices=common.MODEL_PROFILES, default="fast",
                        help="model profile: 'fast' uses fixed hyperparameters, "
                             "'optimized' tunes them per model by minimizing "
                             "cross-validated MAE (better error, slower run)")
    parser.add_argument("--refresh", action="store_true",
                        help="re-download the target season's data, ignoring the local cache")
    common.add_verbosity_args(parser)
    args = parser.parse_args()
    common.apply_verbosity(args)

    setup()
    year = args.season or datetime.now().year
    schedule = fastf1.get_event_schedule(year, include_testing=False)
    data = collect_season(year, schedule, refresh=args.refresh)
    complete = season_is_complete(year, schedule, data)

    def forecastable(frame):
        # auto snapshot = the last completed round; it needs min_train rounds
        # before it (a complete season always qualifies)
        if frame.empty:
            return False
        return complete or int(frame["round"].max()) > args.min_train_rounds

    if not forecastable(data):
        year -= 1
        print(f"\nNo usable completed rounds in {year + 1} yet, falling back to {year}")
        schedule = fastf1.get_event_schedule(year, include_testing=False)
        data = collect_season(year, schedule, refresh=args.refresh)
        complete = season_is_complete(year, schedule, data)
        if data.empty:
            raise SystemExit("no race data available")

    target, mode, event_name = resolve_target(args, year, schedule, data)
    spec = CHAMPIONSHIPS[args.championship]
    features = load_feature_seasons(year, data, schedule, None, args.championship, complete)
    print(f"\nChampionship: {spec['label']}")
    print(f"Forecast: {year} as of round {target} — {event_name} ({mode} mode)")
    print(f"Dataset: {features['year'].nunique()} seasons, "
          f"{len(features)} {spec['entrant']}-round snapshots")
    print(f"Features: {', '.join(spec['features'])}")

    backtest_report(backtest_records(features, args.min_train_rounds, args.model,
                                     args.models, args.championship,
                                     first_season=args.first_backtest_season),
                    args.models, args.championship)
    final_report(features, year, target, event_name, mode, args.model, args.models,
                 args.championship, rounds_total=int(schedule["RoundNumber"].max()))


if __name__ == "__main__":
    main()
