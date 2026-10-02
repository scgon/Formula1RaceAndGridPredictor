"""Race finish prediction: gain model (finish - grid) vs direct model (absolute
finish), baselined against grid order. Shared machinery lives in f1_common.py.
"""

import argparse
from datetime import datetime

import numpy as np
import pandas as pd
import fastf1

import f1_common as common

MIN_TRAIN_ROUNDS = common.MIN_TRAIN_ROUNDS

PRACTICE_SESSIONS = ("FP1", "FP2", "FP3")

PRACTICE_COLS = ["fp_race_pace_delta", "fp_best_delta", "fp_laps"]
SPRINT_COLS = ["sprint_quali_pos", "sprint_finish", "sprint_gain"]

FEATURES = [
    "grid",
    "quali_delta",
    "team_quali_delta",
    "fp_race_pace_delta",
    "fp_best_delta",
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
    "team_id",
]

RAW_COLUMNS = [
    "round",
    "event",
    "driver_number",
    "driver",
    "team",
    "grid",
    "quali_time",
    "finish",
    "points",
    "status",
] + PRACTICE_COLS + SPRINT_COLS

setup = common.setup
utc_now = common.utc_now
session_utc = common.session_utc


def _row_from_result(res, num, round_number, event_name):
    row = res.loc[num]
    grid = row["GridPosition"]
    grid = float(grid) if pd.notna(grid) and grid > 0 else np.nan
    finish = row["Position"]
    finish = float(finish) if pd.notna(finish) else np.nan
    points = row["Points"]
    points = float(points) if pd.notna(points) else 0.0
    return {
        "round": round_number,
        "event": event_name,
        "driver_number": str(num),
        "driver": row["Abbreviation"],
        "team": row["TeamName"],
        "grid": grid,
        "quali_time": np.nan,
        "finish": finish,
        "points": points,
        "status": str(row["Status"]),
    }


def sprint_features(year, round_number):
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
            "sprint_quali_pos": grid,
            "sprint_finish": finish,
            "sprint_gain": grid - finish if pd.notna(grid) and pd.notna(finish) else np.nan,
        }
    return pd.DataFrame.from_dict(rows, orient="index")


def _merge_weekend_features(frame, year, round_number):
    practice = common.practice_features(year, round_number, PRACTICE_SESSIONS)
    if not practice.empty:
        for col in PRACTICE_COLS:
            frame[col] = frame["driver_number"].map(practice[col])
    sprint = sprint_features(year, round_number)
    if not sprint.empty:
        for col in SPRINT_COLS:
            frame[col] = frame["driver_number"].map(sprint[col])
    return frame


def load_completed_round(year, round_number, event_name):
    quali = fastf1.get_session(year, round_number, "Q")
    quali.load(laps=False, telemetry=False, weather=False, messages=False)
    race = fastf1.get_session(year, round_number, "R")
    race.load(laps=False, telemetry=False, weather=False, messages=False)
    qres = quali.results
    rres = race.results
    pole = common.pole_seconds(qres)
    rows = []
    for num in rres.index:
        entry = _row_from_result(rres, num, round_number, event_name)
        entry["quali_time"] = common.best_quali_seconds(qres, num)
        rows.append(entry)
    frame = pd.DataFrame(rows, columns=RAW_COLUMNS)
    frame["quali_delta"] = frame["quali_time"] - pole
    return _merge_weekend_features(frame, year, round_number)


def load_upcoming_round(year, round_number, event_name):
    quali = fastf1.get_session(year, round_number, "Q")
    quali.load(laps=False, telemetry=False, weather=False, messages=False)
    qres = quali.results
    pole = common.pole_seconds(qres)
    rows = []
    for num in qres.index:
        row = qres.loc[num]
        grid = row["Position"]
        entry = {
            "round": round_number,
            "event": event_name,
            "driver_number": str(num),
            "driver": row["Abbreviation"],
            "team": row["TeamName"],
            "grid": float(grid) if pd.notna(grid) else np.nan,
            "quali_time": common.best_quali_seconds(qres, num),
            "finish": np.nan,
            "points": 0.0,
            "status": "",
        }
        rows.append(entry)
    frame = pd.DataFrame(rows, columns=RAW_COLUMNS)
    frame["quali_delta"] = frame["quali_time"] - pole
    return _merge_weekend_features(frame, year, round_number)


def collect_season(year, schedule, refresh=False):
    return common.collect_season(
        year, schedule,
        filename="season_{year}.csv",
        required_columns=RAW_COLUMNS + ["quali_delta"],
        result_column="finish",
        load_round=load_completed_round,
        refresh=refresh,
    )


def build_features(data):
    df = data.copy()
    df["driver_number"] = df["driver_number"].astype(str)
    df = df.sort_values(["round", "driver_number"]).reset_index(drop=True)
    df["team_id"] = pd.factorize(df["team"])[0]
    df["team_race_mean"] = df.groupby(["round", "team"], sort=False)["finish"].transform("mean")
    df["team_race_points"] = df.groupby(["round", "team"], sort=False)["points"].transform("sum")
    df["team_quali_delta"] = df.groupby(["round", "team"], sort=False)["quali_delta"].transform("mean")
    by_driver = df.groupby("driver_number", sort=False)
    df["driver_last_finish"] = by_driver["finish"].transform(lambda s: s.shift(1))
    df["driver_form_3"] = by_driver["finish"].transform(lambda s: s.shift(1).rolling(3, min_periods=1).mean())
    df["driver_form_season"] = by_driver["finish"].transform(lambda s: s.shift(1).expanding(min_periods=1).mean())
    df["driver_points_before"] = by_driver["points"].transform(lambda s: s.cumsum().shift(1)).fillna(0.0)
    df["team_form_3"] = by_driver["team_race_mean"].transform(lambda s: s.shift(1).rolling(3, min_periods=1).mean())
    df["team_form_season"] = by_driver["team_race_mean"].transform(lambda s: s.shift(1).expanding(min_periods=1).mean())
    df["team_points_before"] = by_driver["team_race_points"].transform(lambda s: s.cumsum().shift(1)).fillna(0.0)
    return df


def make_model():
    return common.make_model([FEATURES.index("team_id")])


def train_model(features, target_round, mode="gain"):
    train = features[
        (features["round"] < target_round)
        & features["finish"].notna()
        & features["grid"].notna()
    ]
    if train.empty:
        raise ValueError(f"no training data before round {target_round}")
    if mode == "gain":
        target = (train["finish"] - train["grid"]).to_numpy(dtype=float)
    else:
        target = train["finish"].to_numpy(dtype=float)
    model = make_model()
    model.fit(train[FEATURES].to_numpy(dtype=float), target)
    return model, train


def predict_round(model, features, target_round, mode="gain"):
    rows = features[(features["round"] == target_round) & features["grid"].notna()].copy()
    rows = rows.sort_values(["grid", "driver_number"])
    rows["model_output"] = model.predict(rows[FEATURES].to_numpy(dtype=float))
    if mode == "gain":
        rows["pred_finish"] = rows["grid"] + rows["model_output"]
    else:
        rows["pred_finish"] = rows["model_output"]
    rows["pred_pos"] = rows["pred_finish"].rank(method="first").astype(int)
    return rows.sort_values("pred_pos")


def podium_points(pred_top3, actual_top3):
    """Podium scoring for the web app: +15 per exact position match,
    +5 per predicted podium driver in the wrong slot, +100 perfect-podium bonus."""
    points = sum(15 if p == a else (5 if p in actual_top3 else 0)
                 for p, a in zip(pred_top3, actual_top3))
    if list(pred_top3) == list(actual_top3):
        points += 100
    return points


def backtest_records(features, min_train_rounds):
    """Rolling backtest of gain vs direct model; one metrics dict per predicted round.

    Alongside the CLI-reported metrics, each record carries the podium
    points and the predicted/actual winner names used by the web app.
    """
    rounds = sorted(features["round"].unique())
    records = []
    for r in rounds:
        if sum(1 for x in rounds if x < r) < min_train_rounds:
            continue
        try:
            gain_model, _ = train_model(features, r, "gain")
            direct_model, _ = train_model(features, r, "direct")
        except ValueError:
            continue
        gain = predict_round(gain_model, features, r, "gain")
        direct = predict_round(direct_model, features, r, "direct")
        gain_scored = gain[gain["finish"].notna()]
        direct_scored = direct[direct["finish"].notna()]
        winner_row = features.loc[(features["round"] == r) & (features["finish"] == 1), "driver"]
        if gain_scored.empty or winner_row.empty:
            continue
        winner = winner_row.iloc[0]
        actual_rows = features.loc[(features["round"] == r) & (features["finish"] <= 3)]
        actual_top3_order = list(actual_rows.sort_values("finish")["driver"])
        actual_top3 = set(actual_rows["driver"])
        gain_top3 = list(gain.head(3)["driver"])
        direct_top3 = list(direct.head(3)["driver"])
        records.append({
            "round": r,
            "event": str(features.loc[features["round"] == r, "event"].iloc[0]),
            "gain_mae": float((gain_scored["pred_pos"] - gain_scored["finish"]).abs().mean()),
            "direct_mae": float((direct_scored["pred_pos"] - direct_scored["finish"]).abs().mean()),
            "grid_mae": float((gain_scored["grid"].rank(method="first") - gain_scored["finish"]).abs().mean()),
            "gain_podium": len(actual_top3 & set(gain_top3)),
            "direct_podium": len(actual_top3 & set(direct_top3)),
            "grid_podium": len(actual_top3 & set(gain_scored.nsmallest(3, "grid")["driver"])),
            "gain_winner": 1 if gain_top3[0] == winner else 0,
            "direct_winner": 1 if direct_top3[0] == winner else 0,
            "grid_winner": 1 if gain_scored.nsmallest(1, "grid")["driver"].iloc[0] == winner else 0,
            "gain_corr": common.rank_corr(gain_scored["pred_pos"], gain_scored["finish"]),
            "direct_corr": common.rank_corr(direct_scored["pred_pos"], direct_scored["finish"]),
            "gain_points": podium_points(gain_top3, actual_top3_order),
            "direct_points": podium_points(direct_top3, actual_top3_order),
            "gain_top1": gain_top3[0],
            "direct_top1": direct_top3[0],
            "actual_top1": winner,
        })
    return records


def backtest_report(records):
    print("\n=== Rolling backtest: gain model (finish - grid) vs direct model (absolute finish) ===")
    print(f"{'rd':>3}  {'event':<26} {'gain':>5} {'direct':>6} {'grid':>5} {'podium':>8} {'winner':>8}")
    for rec in records:
        print(f"{rec['round']:>3}  {rec['event']:<26} {rec['gain_mae']:>5.1f} {rec['direct_mae']:>6.1f} "
              f"{rec['grid_mae']:>5.1f} {f'{rec['gain_podium']}/{rec['direct_podium']}':>8} "
              f"{('hit' if rec['gain_winner'] else '-') + '/' + ('hit' if rec['direct_winner'] else '-'):>8}")
    if not records:
        print("not enough completed rounds for a backtest yet")
        return
    n = len(records)
    print(f"\nMean over {n} predicted rounds:")
    print(f"{'':<28} {'gain':>7} {'direct':>8} {'grid-only':>9}")
    print(f"{'position MAE':<28}"
          f"{np.mean([x['gain_mae'] for x in records]):>7.2f}"
          f"{np.mean([x['direct_mae'] for x in records]):>8.2f}"
          f"{np.mean([x['grid_mae'] for x in records]):>9.2f}")
    print(f"{'podium hit rate':<28}"
          f"{100 * sum(x['gain_podium'] for x in records) / (3 * n):>6.0f}%"
          f"{100 * sum(x['direct_podium'] for x in records) / (3 * n):>7.0f}%"
          f"{100 * sum(x['grid_podium'] for x in records) / (3 * n):>8.0f}%")
    print(f"{'winner hit rate':<28}"
          f"{100 * sum(x['gain_winner'] for x in records) / n:>6.0f}%"
          f"{100 * sum(x['direct_winner'] for x in records) / n:>7.0f}%"
          f"{100 * sum(x['grid_winner'] for x in records) / n:>8.0f}%")
    print(f"{'rank correlation':<28}"
          f"{np.nanmean([x['gain_corr'] for x in records]):>7.2f}"
          f"{np.nanmean([x['direct_corr'] for x in records]):>8.2f}")


def final_predictions(features, target):
    """Train both models on rounds before `target` and predict `target`.

    Returns models, the training set and both prediction frames; the gain
    frame carries a direct_pos column for side-by-side display.
    """
    gain_model, train_set = train_model(features, target, "gain")
    direct_model, _ = train_model(features, target, "direct")
    gain = predict_round(gain_model, features, target, "gain")
    direct = predict_round(direct_model, features, target, "direct")
    gain["direct_pos"] = gain["driver"].map(direct.set_index("driver")["pred_pos"]).astype(int)
    return {"gain_model": gain_model, "direct_model": direct_model,
            "train": train_set, "gain": gain, "direct": direct}


def importance_frame(pred, mode="gain"):
    """Permutation importance for one of the final models, as a DataFrame."""
    train = pred["train"]
    X = train[FEATURES].to_numpy(dtype=float)
    if mode == "gain":
        target = (train["finish"] - train["grid"]).to_numpy(dtype=float)
        model = pred["gain_model"]
    else:
        target = train["finish"].to_numpy(dtype=float)
        model = pred["direct_model"]
    return common.importance_scores(model, FEATURES, X, target)


def final_report(features, year, target, event_name, mode):
    pred = final_predictions(features, target)
    gain = pred["gain"]
    print(f"\n=== Prediction: {year} {event_name} (round {target}) ===")
    if mode == "pre":
        print("grid estimated from qualifying classification (grid penalties not applied)\n")
    header = f"{'gain':>4} {'direct':>6}  {'driver':<4} {'team':<20} {'grid':>4}"
    if mode == "post":
        header += f" {'actual':>7} {'delta':>6}"
    print(header)
    for _, row in gain.iterrows():
        line = (f"{'P' + str(row['pred_pos']):>4} {'P' + str(row['direct_pos']):>6}  "
                f"{row['driver']:<4} {row['team']:<20} {int(row['grid']):>4}")
        if mode == "post":
            delta = int(row["pred_pos"] - row["finish"]) if pd.notna(row["finish"]) else ""
            actual = f"P{int(row['finish'])}" if pd.notna(row["finish"]) else "DNF"
            line += f" {actual:>7} {str(delta):>6}"
        print(line)
    print(f"\nPredicted podium (gain model)   : {' '.join(gain.head(3)['driver'])}")
    print(f"Predicted podium (direct model) : {' '.join(gain.sort_values('direct_pos').head(3)['driver'])}")
    if mode == "post":
        actual_rows = features.loc[(features["round"] == target) & (features["finish"] <= 3)]
        actual_top3 = list(actual_rows.sort_values("finish")["driver"])
        print(f"Actual podium                   : {' '.join(actual_top3)}")
    else:
        print(f"Predicted winner (gain model)   : {gain.iloc[0]['driver']}")
        print(f"Predicted winner (direct model) : {gain.sort_values('direct_pos').iloc[0]['driver']}")
    train = pred["train"]
    X = train[FEATURES].to_numpy(dtype=float)
    common.importance_report(pred["gain_model"], FEATURES, X,
                             (train["finish"] - train["grid"]).to_numpy(dtype=float),
                             "gain", width=22)
    common.importance_report(pred["direct_model"], FEATURES, X,
                             train["finish"].to_numpy(dtype=float),
                             "direct", width=22)


def resolve_target(args, year, schedule, data):
    completed = sorted(data["round"].unique())
    if not completed:
        raise SystemExit(f"no completed races found for the {year} season")
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
        if quali_utc is None or quali_utc > now:
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
        raise SystemExit(f"cannot predict round {rn}: qualifying has not happened yet")

    if upcoming:
        row = upcoming[0]
        quali_utc = session_utc(row, "Qualifying")
        if quali_utc is not None and quali_utc < now:
            print(f"\nNext round ({row['EventName']}) has qualifying done -> predicting that race")
            return int(row["RoundNumber"]), "pre", str(row["EventName"])
        last_event = str(data.loc[data["round"] == completed[-1], "event"].iloc[0])
        print(f"\nQualifying for the next round ({row['EventName']}) has not happened yet,")
        print(f"so the model demonstrates on the last completed race instead")
        return completed[-1], "post", last_event

    last_event = str(data.loc[data["round"] == completed[-1], "event"].iloc[0])
    print(f"\nThe {year} season is finished -> demonstrating on the final round")
    return completed[-1], "post", last_event


def main():
    parser = argparse.ArgumentParser(
        description="F1 race result prediction using fastf1 and gradient boosting")
    parser.add_argument("--season", type=int, default=None,
                        help="season year (default: current year, falls back to previous year)")
    parser.add_argument("--predict-round", type=int, default=None,
                        help="round number to predict (default: auto-select)")
    parser.add_argument("--next", action="store_true",
                        help="predict the next race on the calendar (requires qualifying to be done)")
    parser.add_argument("--min-train-rounds", type=int, default=MIN_TRAIN_ROUNDS,
                        help="minimum completed rounds needed before first prediction (default: 5)")
    parser.add_argument("--refresh", action="store_true",
                        help="re-download season data, ignoring the local cache")
    args = parser.parse_args()

    setup()
    year = args.season or datetime.now().year
    schedule = fastf1.get_event_schedule(year, include_testing=False)
    data = collect_season(year, schedule, refresh=args.refresh)
    if data.empty:
        year -= 1
        print(f"\nNo completed races this season yet, falling back to {year}")
        schedule = fastf1.get_event_schedule(year, include_testing=False)
        data = collect_season(year, schedule, refresh=args.refresh)
        if data.empty:
            raise SystemExit("no race data available")

    target, mode, event_name = resolve_target(args, year, schedule, data)
    if mode == "pre":
        upcoming = load_upcoming_round(year, target, event_name)
        data = pd.concat([data[data["round"] != target], upcoming], ignore_index=True)

    features = build_features(data)
    n_rounds = features["round"].nunique()
    print(f"\nDataset: {n_rounds} rounds, {len(features)} driver-race records")
    print(f"Features: {', '.join(FEATURES)}")

    backtest_report(backtest_records(features, args.min_train_rounds))
    final_report(features, year, target, event_name, mode)


if __name__ == "__main__":
    main()
