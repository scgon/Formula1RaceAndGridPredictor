import argparse
import warnings
from datetime import datetime, timedelta, timezone
from pathlib import Path

import numpy as np
import pandas as pd
import fastf1
from sklearn.ensemble import HistGradientBoostingRegressor
from sklearn.inspection import permutation_importance

warnings.filterwarnings("ignore")

BASE_DIR = Path(__file__).resolve().parent
CACHE_DIR = BASE_DIR / "cache"
DATA_DIR = BASE_DIR / "data"

MIN_TRAIN_ROUNDS = 5
STINT_WINDOW = 5
STINT_MIN_LAPS = 3
COMPLETION_BUFFER = timedelta(hours=3)

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


def _clean_practice_laps(session):
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


def _practice_session_frame(session):
    clean = _clean_practice_laps(session)
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


def practice_features(year, round_number):
    frames = []
    for identifier in ("FP1", "FP2", "FP3"):
        try:
            session = fastf1.get_session(year, round_number, identifier)
            session.load(laps=True, telemetry=False, weather=False, messages=False)
            frame = _practice_session_frame(session)
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
    practice = practice_features(year, round_number)
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
    pole = pole_seconds(qres)
    rows = []
    for num in rres.index:
        entry = _row_from_result(rres, num, round_number, event_name)
        entry["quali_time"] = best_quali_seconds(qres, num)
        rows.append(entry)
    frame = pd.DataFrame(rows, columns=RAW_COLUMNS)
    frame["quali_delta"] = frame["quali_time"] - pole
    return _merge_weekend_features(frame, year, round_number)


def load_upcoming_round(year, round_number, event_name):
    quali = fastf1.get_session(year, round_number, "Q")
    quali.load(laps=False, telemetry=False, weather=False, messages=False)
    qres = quali.results
    pole = pole_seconds(qres)
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
            "quali_time": best_quali_seconds(qres, num),
            "finish": np.nan,
            "points": 0.0,
            "status": "",
        }
        rows.append(entry)
    frame = pd.DataFrame(rows, columns=RAW_COLUMNS)
    frame["quali_delta"] = frame["quali_time"] - pole
    return _merge_weekend_features(frame, year, round_number)


def collect_season(year, schedule, refresh=False):
    path = DATA_DIR / f"season_{year}.csv"
    now = utc_now()
    completed = []
    for _, ev in schedule.iterrows():
        race_utc = session_utc(ev, "Race")
        if race_utc is not None and race_utc < now - COMPLETION_BUFFER:
            completed.append((int(ev["RoundNumber"]), str(ev["EventName"])))

    cached = None
    if path.exists() and not refresh:
        candidate = pd.read_csv(path, dtype={"driver_number": str})
        if set(RAW_COLUMNS + ["quali_delta"]).issubset(candidate.columns):
            cached = candidate
        else:
            print("cached season file lacks practice/sprint features, re-downloading")
    have = set(cached["round"].unique()) if cached is not None else set()

    todo = [(rn, name) for rn, name in completed if rn not in have]
    print(f"Season {year}: {len(completed)} completed races ({len(have)} cached, {len(todo)} to fetch)")

    frames = []
    for rn, name in todo:
        try:
            frame = load_completed_round(year, rn, name)
        except Exception as exc:
            print(f"  round {rn:>2} ({name}) skipped: {type(exc).__name__}: {exc}")
            continue
        if frame["finish"].notna().sum() == 0:
            print(f"  round {rn:>2} ({name}) skipped: no classified results yet")
            continue
        print(f"  fetched round {rn:>2}  {name}")
        frames.append(frame)

    if cached is not None:
        frames.insert(0, cached)
    if not frames:
        return pd.DataFrame(columns=RAW_COLUMNS + ["quali_delta"])

    data = pd.concat(frames, ignore_index=True)
    data = data.drop_duplicates(subset=["round", "driver_number"], keep="last")
    data = data.sort_values(["round", "driver_number"]).reset_index(drop=True)
    data.to_csv(path, index=False)
    return data


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
    return HistGradientBoostingRegressor(
        loss="absolute_error",
        learning_rate=0.08,
        max_iter=150,
        max_leaf_nodes=15,
        min_samples_leaf=20,
        l2_regularization=1.0,
        categorical_features=[FEATURES.index("team_id")],
        random_state=42,
    )


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


def rank_corr(a, b):
    ra = pd.Series(a).rank().to_numpy()
    rb = pd.Series(b).rank().to_numpy()
    if np.std(ra) == 0 or np.std(rb) == 0:
        return np.nan
    return float(np.corrcoef(ra, rb)[0, 1])


def backtest_report(features, min_train_rounds):
    rounds = sorted(features["round"].unique())
    records = []
    print("\n=== Rolling backtest: gain model (finish - grid) vs direct model (absolute finish) ===")
    print(f"{'rd':>3}  {'event':<26} {'gain':>5} {'direct':>6} {'grid':>5} {'podium':>8} {'winner':>8}")
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
        actual_top3 = set(features.loc[(features["round"] == r) & (features["finish"] <= 3), "driver"])
        rec = {
            "round": r,
            "event": str(features.loc[features["round"] == r, "event"].iloc[0]),
            "gain_mae": float((gain_scored["pred_pos"] - gain_scored["finish"]).abs().mean()),
            "direct_mae": float((direct_scored["pred_pos"] - direct_scored["finish"]).abs().mean()),
            "grid_mae": float((gain_scored["grid"].rank(method="first") - gain_scored["finish"]).abs().mean()),
            "gain_podium": len(actual_top3 & set(gain.head(3)["driver"])),
            "direct_podium": len(actual_top3 & set(direct.head(3)["driver"])),
            "grid_podium": len(actual_top3 & set(gain_scored.nsmallest(3, "grid")["driver"])),
            "gain_winner": 1 if gain.iloc[0]["driver"] == winner else 0,
            "direct_winner": 1 if direct.iloc[0]["driver"] == winner else 0,
            "grid_winner": 1 if gain_scored.nsmallest(1, "grid")["driver"].iloc[0] == winner else 0,
            "gain_corr": rank_corr(gain_scored["pred_pos"], gain_scored["finish"]),
            "direct_corr": rank_corr(direct_scored["pred_pos"], direct_scored["finish"]),
        }
        records.append(rec)
        print(f"{r:>3}  {rec['event']:<26} {rec['gain_mae']:>5.1f} {rec['direct_mae']:>6.1f} "
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


def importance_report(model, train, mode="gain"):
    if mode == "gain":
        target = (train["finish"] - train["grid"]).to_numpy(dtype=float)
    else:
        target = train["finish"].to_numpy(dtype=float)
    result = permutation_importance(
        model,
        train[FEATURES].to_numpy(dtype=float),
        target,
        scoring="neg_mean_absolute_error",
        n_repeats=5,
        random_state=42,
    )
    order = np.argsort(result.importances_mean)[::-1]
    print(f"\nFeature importance ({mode} model, permutation, increase in MAE when shuffled):")
    for idx in order:
        print(f"  {FEATURES[idx]:<22} {result.importances_mean[idx]:+.3f}")


def final_report(features, year, target, event_name, mode):
    gain_model, train_set = train_model(features, target, "gain")
    direct_model, _ = train_model(features, target, "direct")
    gain = predict_round(gain_model, features, target, "gain")
    direct = predict_round(direct_model, features, target, "direct")
    gain["direct_pos"] = gain["driver"].map(direct.set_index("driver")["pred_pos"]).astype(int)
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
    importance_report(gain_model, train_set, "gain")
    importance_report(direct_model, train_set, "direct")


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

    backtest_report(features, args.min_train_rounds)
    final_report(features, year, target, event_name, mode)


if __name__ == "__main__":
    main()
