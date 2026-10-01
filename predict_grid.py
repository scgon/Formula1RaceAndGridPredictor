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


def practice_done_by(row, now):
    for name in ("Practice 2", "Practice 1"):
        utc = session_utc(row, name)
        if utc is not None and utc < now - PRACTICE_BUFFER:
            return True
    return False


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


def practice_features(year, round_number, sessions=PRACTICE_SESSIONS):
    frames = []
    for identifier in sessions:
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
    practice = practice_features(year, round_number)
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
    pole = pole_seconds(qres)
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
            "quali_time": best_quali_seconds(qres, num),
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
    path = DATA_DIR / f"quali_season_{year}.csv"
    now = utc_now()
    completed = []
    for _, ev in schedule.iterrows():
        race_utc = session_utc(ev, "Race")
        if race_utc is not None and race_utc < now - COMPLETION_BUFFER:
            completed.append((int(ev["RoundNumber"]), str(ev["EventName"])))

    cached = None
    if path.exists() and not refresh:
        candidate = pd.read_csv(path, dtype={"driver_number": str})
        if set(RAW_COLUMNS).issubset(candidate.columns):
            cached = candidate
        else:
            print("cached season file has an outdated schema, re-downloading")
    have = set(cached["round"].unique()) if cached is not None else set()

    todo = [(rn, name) for rn, name in completed if rn not in have]
    print(f"Season {year}: {len(completed)} completed weekends ({len(have)} cached, {len(todo)} to fetch)")

    frames = []
    for rn, name in todo:
        try:
            frame = load_completed_round(year, rn, name)
        except Exception as exc:
            print(f"  round {rn:>2} ({name}) skipped: {type(exc).__name__}: {exc}")
            continue
        if frame["quali_pos"].notna().sum() == 0:
            print(f"  round {rn:>2} ({name}) skipped: no classified quali results yet")
            continue
        print(f"  fetched round {rn:>2}  {name}")
        frames.append(frame)

    if cached is not None:
        frames.insert(0, cached)
    if not frames:
        return pd.DataFrame(columns=RAW_COLUMNS)

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


def train_model(features, target_round, mode="anchor"):
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
    model = make_model()
    model.fit(train[FEATURES].to_numpy(dtype=float), target)
    return model, train


def predict_round(model, features, target_round, mode="anchor"):
    rows = features[features["round"] == target_round].copy()
    rows = rows.sort_values(["driver_number"])
    rows["model_output"] = model.predict(rows[FEATURES].to_numpy(dtype=float))
    if mode == "anchor":
        anchor = anchor_value(rows)
        rows["anchor"] = anchor
        rows["pred_score"] = anchor + rows["model_output"]
    else:
        rows["pred_score"] = rows["model_output"]
    rows["pred_pos"] = rows["pred_score"].rank(method="first").astype(int)
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
    print("\n=== Rolling backtest: anchor model (change vs last quali) vs direct model (absolute quali position) ===")
    print(f"{'rd':>3}  {'event':<26} {'anchor':>6} {'direct':>6} {'lastQ':>6} {'podium':>8} {'pole':>8}")
    for r in rounds:
        if sum(1 for x in rounds if x < r) < min_train_rounds:
            continue
        try:
            anchor_model, _ = train_model(features, r, "anchor")
            direct_model, _ = train_model(features, r, "direct")
        except ValueError:
            continue
        anchor = predict_round(anchor_model, features, r, "anchor")
        direct = predict_round(direct_model, features, r, "direct")
        anchor_scored = anchor[anchor["quali_pos"].notna()]
        direct_scored = direct[direct["quali_pos"].notna()]
        pole_row = features.loc[(features["round"] == r) & (features["quali_pos"] == 1), "driver"]
        if anchor_scored.empty or pole_row.empty:
            continue
        pole = pole_row.iloc[0]
        actual_top3 = set(features.loc[(features["round"] == r) & (features["quali_pos"] <= 3), "driver"])
        persistence = anchor_value(anchor_scored).rank(method="first")
        rec = {
            "round": r,
            "event": str(features.loc[features["round"] == r, "event"].iloc[0]),
            "anchor_mae": float((anchor_scored["pred_pos"] - anchor_scored["quali_pos"]).abs().mean()),
            "direct_mae": float((direct_scored["pred_pos"] - direct_scored["quali_pos"]).abs().mean()),
            "persistence_mae": float((persistence - anchor_scored["quali_pos"]).abs().mean()),
            "anchor_podium": len(actual_top3 & set(anchor.head(3)["driver"])),
            "direct_podium": len(actual_top3 & set(direct.head(3)["driver"])),
            "anchor_pole": 1 if anchor.iloc[0]["driver"] == pole else 0,
            "direct_pole": 1 if direct.iloc[0]["driver"] == pole else 0,
            "anchor_corr": rank_corr(anchor_scored["pred_pos"], anchor_scored["quali_pos"]),
            "direct_corr": rank_corr(direct_scored["pred_pos"], direct_scored["quali_pos"]),
        }
        records.append(rec)
        print(f"{r:>3}  {rec['event']:<26} {rec['anchor_mae']:>6.1f} {rec['direct_mae']:>6.1f} "
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


def importance_report(model, train, mode="anchor"):
    if mode == "anchor":
        target = (train["quali_pos"] - train["quali_pos_last"]).to_numpy(dtype=float)
    else:
        target = train["quali_pos"].to_numpy(dtype=float)
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
        print(f"  {FEATURES[idx]:<24} {result.importances_mean[idx]:+.3f}")


def final_report(features, year, target, event_name, mode):
    anchor_model, train_set = train_model(features, target, "anchor")
    direct_model, _ = train_model(features, target, "direct")
    anchor = predict_round(anchor_model, features, target, "anchor")
    direct = predict_round(direct_model, features, target, "direct")
    anchor["direct_pos"] = anchor["driver"].map(direct.set_index("driver")["pred_pos"]).astype(int)
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
    importance_report(anchor_model, train_set, "anchor")
    importance_report(direct_model, train_set, "direct")


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

    backtest_report(features, args.min_train_rounds)
    final_report(features, year, target, event_name, mode)


if __name__ == "__main__":
    main()
