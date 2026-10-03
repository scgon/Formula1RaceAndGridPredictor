"""Qualifying/grid prediction: anchor model (change vs last quali) vs direct
model (absolute position), baselined against last-quali persistence. Only
FP1/FP2 and sprint qualifying feed the features (day-before-quali constraint).
Shared machinery lives in f1_common.py.
"""

import argparse
from datetime import datetime, timedelta

import numpy as np
import pandas as pd
import fastf1

import f1_common as common

MIN_TRAIN_ROUNDS = common.MIN_TRAIN_ROUNDS
PRACTICE_BUFFER = timedelta(hours=2)
PRACTICE_SESSIONS = ("FP1", "FP2")
NEUTRAL_POSITION = 11.5

PRACTICE_COLS = ["fp_race_pace_delta", "fp_best_delta", "fp_laps"]
SPRINT_QUALI_COLS = ["sprint_quali_pos", "sprint_quali_delta"]

FEATURES = [
    "fp_best_delta",
    "fp_race_pace_delta",
    "fp_laps",
    "team_fp_best_delta",
    "team_fp_race_pace_delta",
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

RAW_COLUMNS = [
    "round",
    "event",
    "driver_number",
    "driver",
    "team",
    "quali_pos",
    "quali_time",
    "quali_delta",
    "points",
] + PRACTICE_COLS + SPRINT_QUALI_COLS

setup = common.setup
utc_now = common.utc_now
session_utc = common.session_utc


def practice_done_by(row, now):
    for name in ("Practice 2", "Practice 1"):
        utc = session_utc(row, name)
        if utc is not None and utc < now - PRACTICE_BUFFER:
            return True
    return False


def practice_entrants(year, round_number, sessions=PRACTICE_SESSIONS):
    for identifier in sessions:
        try:
            session = fastf1.get_session(year, round_number, identifier)
            session.load(laps=False, telemetry=False, weather=False, messages=False)
            if not session.results.empty:
                return session.results
        except Exception:
            continue
    return None


def sprint_quali_features(year, round_number):
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


def _merge_weekend_features(frame, year, round_number):
    practice = common.practice_features(year, round_number, PRACTICE_SESSIONS)
    if not practice.empty:
        for col in PRACTICE_COLS:
            frame[col] = frame["driver_number"].map(practice[col])
    sprint_quali = sprint_quali_features(year, round_number)
    if not sprint_quali.empty:
        for col in SPRINT_QUALI_COLS:
            frame[col] = frame["driver_number"].map(sprint_quali[col])
    return frame


def load_completed_round(year, round_number, event_name):
    quali = fastf1.get_session(year, round_number, "Q")
    quali.load(laps=False, telemetry=False, weather=False, messages=False)
    race = fastf1.get_session(year, round_number, "R")
    race.load(laps=False, telemetry=False, weather=False, messages=False)
    qres = quali.results
    rres = race.results
    pole = common.pole_seconds(qres)
    points_map = {}
    for num in rres.index:
        value = rres.at[num, "Points"]
        points_map[str(num)] = float(value) if pd.notna(value) else 0.0
    rows = []
    for num in qres.index:
        row = qres.loc[num]
        pos = row["Position"]
        rows.append({
            "round": round_number,
            "event": event_name,
            "driver_number": str(num),
            "driver": row["Abbreviation"],
            "team": row["TeamName"],
            "quali_pos": float(pos) if pd.notna(pos) else np.nan,
            "quali_time": common.best_quali_seconds(qres, num),
            "quali_delta": np.nan,
            "points": points_map.get(str(num), 0.0),
        })
    frame = pd.DataFrame(rows, columns=RAW_COLUMNS)
    frame["quali_delta"] = frame["quali_time"] - pole
    return _merge_weekend_features(frame, year, round_number)


def load_upcoming_round(year, round_number, event_name, fallback):
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
            "quali_pos": np.nan,
            "quali_time": np.nan,
            "quali_delta": np.nan,
            "points": 0.0,
        })
    frame = pd.DataFrame(rows, columns=RAW_COLUMNS)
    return _merge_weekend_features(frame, year, round_number)


def collect_season(year, schedule, refresh=False):
    return common.collect_season(
        year, schedule,
        filename="quali_season_{year}.csv",
        required_columns=RAW_COLUMNS,
        result_column="quali_pos",
        load_round=load_completed_round,
        refresh=refresh,
    )


def build_features(data):
    df = data.copy()
    df["driver_number"] = df["driver_number"].astype(str)
    df = df.sort_values(["round", "driver_number"]).reset_index(drop=True)
    df["team_id"] = pd.factorize(df["team"])[0]
    df["team_fp_best_delta"] = df.groupby(["round", "team"], sort=False)["fp_best_delta"].transform("mean")
    df["team_fp_race_pace_delta"] = df.groupby(["round", "team"], sort=False)["fp_race_pace_delta"].transform("mean")
    df["team_quali_mean"] = df.groupby(["round", "team"], sort=False)["quali_pos"].transform("mean")
    df["team_points"] = df.groupby(["round", "team"], sort=False)["points"].transform("sum")
    by_driver = df.groupby("driver_number", sort=False)
    df["quali_pos_last"] = by_driver["quali_pos"].transform(lambda s: s.shift(1))
    df["quali_form_3"] = by_driver["quali_pos"].transform(lambda s: s.shift(1).rolling(3, min_periods=1).mean())
    df["quali_form_season"] = by_driver["quali_pos"].transform(lambda s: s.shift(1).expanding(min_periods=1).mean())
    df["driver_points_before"] = by_driver["points"].transform(lambda s: s.cumsum().shift(1)).fillna(0.0)
    df["team_quali_form_3"] = by_driver["team_quali_mean"].transform(lambda s: s.shift(1).rolling(3, min_periods=1).mean())
    df["team_quali_form_season"] = by_driver["team_quali_mean"].transform(lambda s: s.shift(1).expanding(min_periods=1).mean())
    df["team_points_before"] = by_driver["team_points"].transform(lambda s: s.cumsum().shift(1)).fillna(0.0)
    return df


def anchor_value(rows):
    return rows["quali_pos_last"].fillna(rows["quali_form_season"]).fillna(NEUTRAL_POSITION)


def usable_features(frame):
    """FEATURES minus any column that is entirely NaN in `frame` — sklearn's
    HistGradientBoosting cannot fit a feature with no observed values (e.g.
    the quali form features when only round 1 exists, or sprint quali
    features before the first sprint weekend of the season)."""
    matrix = frame[FEATURES].to_numpy(dtype=float)
    return [name for name, col in zip(FEATURES, matrix.T) if not np.isnan(col).all()]


def make_model(feature_names, profile="fast", tuned_params=None):
    categorical = [feature_names.index("team_id")] if "team_id" in feature_names else []
    return common.make_model(categorical, profile, tuned_params)


def train_model(features, target_round, mode="anchor", profile="fast"):
    """Fit one model on all rounds before `target_round`.

    Returns (model, train frame, used feature list, tuned hyperparameters).
    With profile="optimized" the hyperparameters come from a CV-MAE search
    over the training set (None when nothing clearly beat the fast defaults,
    in which case the model falls back to the fast profile's fixed values).
    """
    if mode == "anchor":
        train = features[
            (features["round"] < target_round)
            & features["quali_pos"].notna()
            & features["quali_pos_last"].notna()
        ]
        target = (train["quali_pos"] - train["quali_pos_last"]).to_numpy(dtype=float)
    else:
        train = features[
            (features["round"] < target_round)
            & features["quali_pos"].notna()
        ]
        target = train["quali_pos"].to_numpy(dtype=float)
    if train.empty:
        raise ValueError(f"no training data before round {target_round}")
    used = usable_features(train)
    X = train[used].to_numpy(dtype=float)
    categorical = [used.index("team_id")] if "team_id" in used else None
    tuned = None
    if profile == "optimized":
        tuned = common.tune_hyperparameters(
            X, target, train["round"].to_numpy(),
            categorical_features=categorical,
        )
    model = make_model(used, profile, tuned)
    model.fit(X, target)
    return model, train, used, tuned


def predict_round(model, features, target_round, mode="anchor", use_features=None):
    use_features = FEATURES if use_features is None else use_features
    rows = features[features["round"] == target_round].copy()
    rows = rows.sort_values(["driver_number"])
    rows["model_output"] = model.predict(rows[use_features].to_numpy(dtype=float))
    if mode == "anchor":
        anchor = anchor_value(rows)
        rows["anchor"] = anchor
        rows["pred_score"] = anchor + rows["model_output"]
    else:
        rows["pred_score"] = rows["model_output"]
    rows["pred_pos"] = rows["pred_score"].rank(method="first").astype(int)
    return rows.sort_values("pred_pos")


def backtest_records(features, min_train_rounds, profile="fast"):
    """Rolling backtest of anchor vs direct model; one metrics dict per predicted round.

    Alongside the CLI-reported metrics, each record carries the pole points
    (+15 for a correctly predicted pole), the predicted/actual pole names
    and the number of rounds each model trained on (used by the web app).
    Prints one live progress line per trained model (streamed into the web
    app's download log).
    """
    rounds = sorted(features["round"].unique())
    tune_note = " (tuning hyperparameters)" if profile == "optimized" else ""
    records = []
    for r in rounds:
        if sum(1 for x in rounds if x < r) < min_train_rounds:
            continue
        event = str(features.loc[features["round"] == r, "event"].iloc[0])
        try:
            print(f"backtest round {r:>2}  {event}: training anchor model{tune_note}...")
            anchor_model, anchor_train, anchor_used, _ = train_model(features, r, "anchor", profile)
            print(f"backtest round {r:>2}  {event}: training direct model{tune_note}...")
            direct_model, direct_train, direct_used, _ = train_model(features, r, "direct", profile)
        except ValueError:
            continue
        anchor = predict_round(anchor_model, features, r, "anchor", anchor_used)
        direct = predict_round(direct_model, features, r, "direct", direct_used)
        anchor_scored = anchor[anchor["quali_pos"].notna()]
        direct_scored = direct[direct["quali_pos"].notna()]
        pole_row = features.loc[(features["round"] == r) & (features["quali_pos"] == 1), "driver"]
        if anchor_scored.empty or pole_row.empty:
            continue
        pole = pole_row.iloc[0]
        actual_top3 = set(features.loc[(features["round"] == r) & (features["quali_pos"] <= 3), "driver"])
        persistence = anchor_value(anchor_scored).rank(method="first")
        anchor_top1 = anchor.iloc[0]["driver"]
        direct_top1 = direct.iloc[0]["driver"]
        records.append({
            "round": r,
            "event": event,
            "anchor_mae": float((anchor_scored["pred_pos"] - anchor_scored["quali_pos"]).abs().mean()),
            "direct_mae": float((direct_scored["pred_pos"] - direct_scored["quali_pos"]).abs().mean()),
            "persistence_mae": float((persistence - anchor_scored["quali_pos"]).abs().mean()),
            "anchor_podium": len(actual_top3 & set(anchor.head(3)["driver"])),
            "direct_podium": len(actual_top3 & set(direct.head(3)["driver"])),
            "anchor_pole": 1 if anchor_top1 == pole else 0,
            "direct_pole": 1 if direct_top1 == pole else 0,
            "anchor_corr": common.rank_corr(anchor_scored["pred_pos"], anchor_scored["quali_pos"]),
            "direct_corr": common.rank_corr(direct_scored["pred_pos"], direct_scored["quali_pos"]),
            "anchor_points": 15 if anchor_top1 == pole else 0,
            "direct_points": 15 if direct_top1 == pole else 0,
            "anchor_top1": anchor_top1,
            "direct_top1": direct_top1,
            "actual_top1": pole,
            "anchor_train_rounds": int(anchor_train["round"].nunique()),
            "direct_train_rounds": int(direct_train["round"].nunique()),
        })
    return records


def backtest_report(records):
    print("\n=== Rolling backtest: anchor model (change vs last quali) vs direct model (absolute quali position) ===")
    print(f"{'rd':>3}  {'event':<26} {'anchor':>6} {'direct':>6} {'lastQ':>6} {'podium':>8} {'pole':>8}")
    for rec in records:
        print(f"{rec['round']:>3}  {rec['event']:<26} {rec['anchor_mae']:>6.1f} {rec['direct_mae']:>6.1f} "
              f"{rec['persistence_mae']:>6.1f} {f'{rec['anchor_podium']}/{rec['direct_podium']}':>8} "
              f"{('hit' if rec['anchor_pole'] else '-') + '/' + ('hit' if rec['direct_pole'] else '-'):>8}")
    if not records:
        print("not enough completed rounds for a backtest yet")
        return
    n = len(records)
    print(f"\nMean over {n} predicted qualifying sessions:")
    print(f"{'':<28} {'anchor':>7} {'direct':>8} {'last-quali':>10}")
    print(f"{'position MAE':<28}"
          f"{np.mean([x['anchor_mae'] for x in records]):>7.2f}"
          f"{np.mean([x['direct_mae'] for x in records]):>8.2f}"
          f"{np.mean([x['persistence_mae'] for x in records]):>10.2f}")
    print(f"{'podium hit rate':<28}"
          f"{100 * sum(x['anchor_podium'] for x in records) / (3 * n):>6.0f}%"
          f"{100 * sum(x['direct_podium'] for x in records) / (3 * n):>7.0f}%")
    print(f"{'pole hit rate':<28}"
          f"{100 * sum(x['anchor_pole'] for x in records) / n:>6.0f}%"
          f"{100 * sum(x['direct_pole'] for x in records) / n:>7.0f}%")
    print(f"{'rank correlation':<28}"
          f"{np.nanmean([x['anchor_corr'] for x in records]):>7.2f}"
          f"{np.nanmean([x['direct_corr'] for x in records]):>8.2f}")


def final_predictions(features, target, profile="fast"):
    """Train both models on rounds before `target` and predict `target`.

    Returns models, the anchor training set and both prediction frames; the
    anchor frame carries a direct_pos column for side-by-side display. The
    feature lists actually used by each model (all-NaN training columns
    dropped) are returned as anchor_features / direct_features, and the
    hyperparameters the profile selected as anchor_params / direct_params
    (None = fixed fast values, either because profile="fast" or because
    there was too little data to tune on). Prints one live progress line
    per trained model (streamed into the web app's download log).
    """
    tune_note = " (tuning hyperparameters)" if profile == "optimized" else ""
    print(f"training final anchor model{tune_note}...")
    anchor_model, train_set, anchor_used, anchor_params = train_model(features, target, "anchor", profile)
    print(f"training final direct model{tune_note}...")
    direct_model, _, direct_used, direct_params = train_model(features, target, "direct", profile)
    anchor = predict_round(anchor_model, features, target, "anchor", anchor_used)
    direct = predict_round(direct_model, features, target, "direct", direct_used)
    anchor["direct_pos"] = anchor["driver"].map(direct.set_index("driver")["pred_pos"]).astype(int)
    return {"anchor_model": anchor_model, "direct_model": direct_model,
            "train": train_set, "anchor": anchor, "direct": direct,
            "anchor_features": anchor_used, "direct_features": direct_used,
            "anchor_params": anchor_params, "direct_params": direct_params}


def importance_frame(pred, mode="anchor"):
    """Permutation importance for one of the final models, as a DataFrame."""
    train = pred["train"]
    if mode == "anchor":
        used = pred["anchor_features"]
        target = (train["quali_pos"] - train["quali_pos_last"]).to_numpy(dtype=float)
        model = pred["anchor_model"]
    else:
        used = pred["direct_features"]
        target = train["quali_pos"].to_numpy(dtype=float)
        model = pred["direct_model"]
    return common.importance_scores(model, used, train[used].to_numpy(dtype=float), target)


def final_report(features, year, target, event_name, mode, profile="fast"):
    pred = final_predictions(features, target, profile)
    anchor = pred["anchor"]
    print(f"\n=== Qualifying prediction: {year} {event_name} (round {target}) ===")
    print("predicting the qualifying classification; starting-grid penalties are not applied")
    if mode == "pre":
        print("day-before-quali mode: features come from FP1/FP2, sprint qualifying on sprint weekends, and earlier weekends\n")
    header = f"{'anchor':>6} {'direct':>6}  {'driver':<4} {'team':<20}"
    if mode == "post":
        header += f" {'actual':>7} {'delta':>6}"
    print(header)
    for _, row in anchor.iterrows():
        line = (f"{'P' + str(row['pred_pos']):>6} {'P' + str(row['direct_pos']):>6}  "
                f"{row['driver']:<4} {row['team']:<20}")
        if mode == "post":
            delta = int(row["pred_pos"] - row["quali_pos"]) if pd.notna(row["quali_pos"]) else ""
            actual = f"P{int(row['quali_pos'])}" if pd.notna(row["quali_pos"]) else "-"
            line += f" {actual:>7} {str(delta):>6}"
        print(line)
    print(f"\nPredicted pole (anchor model)   : {anchor.iloc[0]['driver']}")
    print(f"Predicted pole (direct model)   : {anchor.sort_values('direct_pos').iloc[0]['driver']}")
    if mode == "post":
        pole_rows = features.loc[(features["round"] == target) & (features["quali_pos"] == 1), "driver"]
        if not pole_rows.empty:
            print(f"Actual pole                     : {pole_rows.iloc[0]}")
    train = pred["train"]
    if profile == "optimized":
        untuned = "fast defaults (no clearly better combination found)"
        print(f"\nTuned hyperparameters (anchor model) : "
              f"{common.format_params(pred['anchor_params']) or untuned}")
        print(f"Tuned hyperparameters (direct model) : "
              f"{common.format_params(pred['direct_params']) or untuned}")
    common.importance_report(pred["anchor_model"], pred["anchor_features"],
                             train[pred["anchor_features"]].to_numpy(dtype=float),
                             (train["quali_pos"] - train["quali_pos_last"]).to_numpy(dtype=float),
                             "anchor", width=24)
    common.importance_report(pred["direct_model"], pred["direct_features"],
                             train[pred["direct_features"]].to_numpy(dtype=float),
                             train["quali_pos"].to_numpy(dtype=float),
                             "direct", width=24)


def resolve_target(args, year, schedule, data):
    completed = sorted(data["round"].unique())
    if not completed:
        raise SystemExit(f"no completed weekends found for the {year} season")
    now = utc_now()
    upcoming = []
    for _, row in schedule.iterrows():
        race_utc = session_utc(row, "Race")
        if race_utc is not None and race_utc > now:
            upcoming.append(row)

    if args.next:
        if not upcoming:
            raise SystemExit("no upcoming race left on this season's calendar")
        row = upcoming[0]
        quali_utc = session_utc(row, "Qualifying")
        if quali_utc is not None and quali_utc < now:
            rn = int(row["RoundNumber"])
            raise SystemExit(f"qualifying for the {row['EventName']} already happened, "
                             f"use --predict-round {rn} to review it")
        return int(row["RoundNumber"]), "pre", str(row["EventName"])

    if args.predict_round is not None:
        rn = args.predict_round
        match = schedule[schedule["RoundNumber"] == rn]
        if match.empty:
            raise SystemExit(f"round {rn} is not on the {year} calendar")
        row = match.iloc[0]
        quali_utc = session_utc(row, "Qualifying")
        race_utc = session_utc(row, "Race")
        if quali_utc is not None and quali_utc < now:
            if rn in completed:
                return rn, "post", str(row["EventName"])
            raise SystemExit(f"round {rn} happened recently but is not in the cache yet, retry in a few hours")
        return rn, "pre", str(row["EventName"])

    if upcoming:
        row = upcoming[0]
        quali_utc = session_utc(row, "Qualifying")
        if quali_utc is None or quali_utc > now:
            if not practice_done_by(row, now):
                print(f"\nPractice for the next round ({row['EventName']}) has not finished yet,")
                print("predictions will rely on historical form only - run again after FP2 for best results")
            else:
                print(f"\nNext up: {row['EventName']} - practice done, predicting its qualifying")
            return int(row["RoundNumber"]), "pre", str(row["EventName"])
        print(f"\nQualifying for the next round ({row['EventName']}) already happened,")
        print("so the model demonstrates on the last completed qualifying instead")
    else:
        print(f"\nThe {year} season is finished -> demonstrating on the final round")
    last_event = str(data.loc[data["round"] == completed[-1], "event"].iloc[0])
    return completed[-1], "post", last_event


def main():
    parser = argparse.ArgumentParser(
        description="F1 qualifying/grid prediction from Friday practice and sprint qualifying, "
                    "using fastf1 and gradient boosting")
    parser.add_argument("--season", type=int, default=None,
                        help="season year (default: current year, falls back to previous year)")
    parser.add_argument("--predict-round", type=int, default=None,
                        help="round number to predict (default: auto-select)")
    parser.add_argument("--next", action="store_true",
                        help="predict the next round's qualifying (fails if it already happened)")
    parser.add_argument("--min-train-rounds", type=int, default=MIN_TRAIN_ROUNDS,
                        help="minimum completed weekends before first backtest prediction (default: 5)")
    parser.add_argument("--model", choices=common.MODEL_PROFILES, default="fast",
                        help="model profile: 'fast' uses fixed hyperparameters, "
                             "'optimized' tunes them per model by minimizing "
                             "cross-validated MAE (better error, slower run)")
    parser.add_argument("--refresh", action="store_true",
                        help="re-download season data, ignoring the local cache")
    args = parser.parse_args()

    setup()
    year = args.season or datetime.now().year
    schedule = fastf1.get_event_schedule(year, include_testing=False)
    data = collect_season(year, schedule, refresh=args.refresh)
    if data.empty:
        year -= 1
        print(f"\nNo completed weekends this season yet, falling back to {year}")
        schedule = fastf1.get_event_schedule(year, include_testing=False)
        data = collect_season(year, schedule, refresh=args.refresh)
        if data.empty:
            raise SystemExit("no data available")

    target, mode, event_name = resolve_target(args, year, schedule, data)
    if mode == "pre":
        completed_rounds = sorted(data["round"].unique())
        fallback = data[data["round"] == completed_rounds[-1]]
        upcoming = load_upcoming_round(year, target, event_name, fallback)
        data = pd.concat([data[data["round"] != target], upcoming], ignore_index=True)

    features = build_features(data)
    print(f"\nDataset: {features['round'].nunique()} rounds, {len(features)} driver-quali records")
    print(f"Features: {', '.join(FEATURES)}")

    backtest_report(backtest_records(features, args.min_train_rounds, args.model))
    final_report(features, year, target, event_name, mode, args.model)


if __name__ == "__main__":
    main()
