"""Race finish prediction: gain model (finish - grid) vs direct model (absolute
finish), baselined against grid order. Runs in three modes: "post" reviews a
completed round, "pre" predicts after qualifying (grid known, both models),
and "prequali" predicts before qualifying — no grid exists yet, so the gain
model has nothing to anchor to and the direct model predicts alone from
practice, sprint and season-form features. Shared machinery lives in
f1_common.py.
"""

import argparse
from datetime import datetime, timedelta

import numpy as np
import pandas as pd
import fastf1

import f1_common as common

MIN_TRAIN_ROUNDS = common.MIN_TRAIN_ROUNDS
PRACTICE_BUFFER = timedelta(hours=2)

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


def load_upcoming_round_prequali(year, round_number, event_name, fallback):
    """Pre-quali frame: qualifying has not happened, so there are no quali
    results yet. Entrants come from the first practice session with results,
    or from the last completed round when no practice has run either. Grid
    and quali features stay NaN — the direct model predicts from practice,
    sprint and season-form features (the gain model needs a grid and is
    skipped in this mode)."""
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
            "quali_time": np.nan,
            "finish": np.nan,
            "points": 0.0,
            "status": "",
        })
    frame = pd.DataFrame(rows, columns=RAW_COLUMNS)
    frame["quali_delta"] = np.nan
    return _merge_weekend_features(frame, year, round_number)


def collect_season(year, schedule, refresh=False, rate_limit_wait=None):
    return common.collect_season(
        year, schedule,
        filename="season_{year}.csv",
        required_columns=RAW_COLUMNS + ["quali_delta"],
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


def usable_features(frame):
    """FEATURES minus any column that is entirely NaN in `frame` — sklearn's
    HistGradientBoosting cannot fit a feature with no observed values (e.g.
    the form features when only round 1 exists, or sprint features before
    the first sprint weekend of the season)."""
    matrix = frame[FEATURES].to_numpy(dtype=float)
    return [name for name, col in zip(FEATURES, matrix.T) if not np.isnan(col).all()]


def make_model(feature_names, profile="fast", tuned_params=None):
    categorical = [feature_names.index("team_id")] if "team_id" in feature_names else []
    return common.make_model(categorical, profile, tuned_params)


def train_model(features, target_round, mode="gain", profile="fast"):
    """Fit one model on all rounds before `target_round`.

    Returns (model, train frame, used feature list, tuned hyperparameters).
    With profile="optimized" the hyperparameters come from a CV-MAE search
    over the training set (None when nothing clearly beat the fast defaults,
    in which case the model falls back to the fast profile's fixed values).
    """
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


def predict_round(model, features, target_round, mode="gain", use_features=None):
    use_features = FEATURES if use_features is None else use_features
    rows = features[features["round"] == target_round].copy()
    # the gain model needs a grid to anchor to; the direct model predicts
    # grid-less (pre-quali) rounds over all entrants, completed rounds
    # (which always have a grid) exactly as before
    if mode == "gain" or rows["grid"].notna().any():
        rows = rows[rows["grid"].notna()]
    rows = rows.sort_values(["grid", "driver_number"])
    rows["model_output"] = model.predict(rows[use_features].to_numpy(dtype=float))
    if mode == "gain":
        rows["pred_finish"] = rows["grid"] + rows["model_output"]
    else:
        rows["pred_finish"] = rows["model_output"]
    rows["pred_pos"] = rows["pred_finish"].rank(method="first").astype(int)
    return rows.sort_values("pred_pos")


def podium_points(pred_top3, actual_top3):
    """Podium scoring for the web app: +15 per exact position match,
    +5 per predicted podium driver in the wrong slot."""
    return sum(15 if p == a else (5 if p in actual_top3 else 0)
               for p, a in zip(pred_top3, actual_top3))


def backtest_records(features, min_train_rounds, profile="fast", models="both"):
    """Rolling backtest of the selected models; one metrics dict per predicted
    round. models picks what trains: "both" (default), "gain" or "direct" —
    records then carry only the trained model's fields alongside the shared
    baseline/actual fields.

    Alongside the CLI-reported metrics, each record carries the podium
    points, the predicted/actual winner names and the number of rounds each
    model trained on (used by the web app). Prints one live progress line
    per trained model (streamed into the web app's download log).
    """
    rounds = sorted(features["round"].unique())
    tune_note = " (tuning hyperparameters)" if profile == "optimized" else ""
    records = []
    for r in rounds:
        if sum(1 for x in rounds if x < r) < min_train_rounds:
            continue
        # a pre-quali target round has no grid and no results: the gain
        # model has nothing to anchor to and there is nothing to score
        if features.loc[features["round"] == r, "grid"].notna().sum() == 0:
            continue
        event = str(features.loc[features["round"] == r, "event"].iloc[0])
        try:
            if models in ("both", "gain"):
                common.vprint(1, f"backtest round {r:>2}  {event}: training gain model{tune_note}...")
                gain_model, gain_train, gain_used, _ = train_model(features, r, "gain", profile)
            if models in ("both", "direct"):
                common.vprint(1, f"backtest round {r:>2}  {event}: training direct model{tune_note}...")
                direct_model, direct_train, direct_used, _ = train_model(features, r, "direct", profile)
        except ValueError:
            continue
        gain = direct = None
        if models in ("both", "gain"):
            gain = predict_round(gain_model, features, r, "gain", gain_used)
        if models in ("both", "direct"):
            direct = predict_round(direct_model, features, r, "direct", direct_used)
        scored = (gain if gain is not None else direct)
        scored = scored[scored["finish"].notna()]
        winner_row = features.loc[(features["round"] == r) & (features["finish"] == 1), "driver"]
        if scored.empty or winner_row.empty:
            continue
        winner = winner_row.iloc[0]
        actual_rows = features.loc[(features["round"] == r) & (features["finish"] <= 3)]
        actual_top3_order = list(actual_rows.sort_values("finish")["driver"])
        actual_top3 = set(actual_rows["driver"])
        rec = {"round": r, "event": event}
        if gain is not None:
            gain_top3 = list(gain.head(3)["driver"])
            rec.update({
                "gain_mae": float((scored["pred_pos"] - scored["finish"]).abs().mean()),
                "gain_podium": len(actual_top3 & set(gain_top3)),
                "gain_winner": 1 if gain_top3[0] == winner else 0,
                "gain_corr": common.rank_corr(scored["pred_pos"], scored["finish"]),
                "gain_points": podium_points(gain_top3, actual_top3_order),
                "gain_top1": gain_top3[0],
                "gain_train_rounds": int(gain_train["round"].nunique()),
            })
        if direct is not None:
            direct_top3 = list(direct.head(3)["driver"])
            rec.update({
                "direct_mae": float((direct[direct["finish"].notna()]["pred_pos"]
                                     - direct[direct["finish"].notna()]["finish"]).abs().mean()),
                "direct_podium": len(actual_top3 & set(direct_top3)),
                "direct_winner": 1 if direct_top3[0] == winner else 0,
                "direct_corr": common.rank_corr(direct[direct["finish"].notna()]["pred_pos"],
                                                direct[direct["finish"].notna()]["finish"]),
                "direct_points": podium_points(direct_top3, actual_top3_order),
                "direct_top1": direct_top3[0],
                "direct_train_rounds": int(direct_train["round"].nunique()),
            })
        rec.update({
            "grid_mae": float((scored["grid"].rank(method="first") - scored["finish"]).abs().mean()),
            "grid_podium": len(actual_top3 & set(scored.nsmallest(3, "grid")["driver"])),
            "grid_winner": 1 if scored.nsmallest(1, "grid")["driver"].iloc[0] == winner else 0,
            "actual_top1": winner,
        })
        records.append(rec)
        detail = ""
        if gain is not None:
            detail += f"gain {rec['gain_mae']:.1f}"
        if gain is not None and direct is not None:
            detail += " / "
        if direct is not None:
            detail += f"direct {rec['direct_mae']:.1f}"
        podium = "/".join(str(rec[k]) for k in
                         ("gain_podium", "direct_podium") if k in rec)
        winner_cells = "/".join("hit" if rec[k] else "-" for k in
                                ("gain_winner", "direct_winner") if k in rec)
        common.vprint(2, f"  round {r:>2} result: {detail} / grid {rec['grid_mae']:.1f} "
                      f"| podium {podium} | winner {winner_cells}")
    return records


def backtest_report(records, models="both"):
    show_gain = models in ("both", "gain")
    show_direct = models in ("both", "direct")
    if show_gain and show_direct:
        print("\n=== Rolling backtest: gain model (finish - grid) vs direct model (absolute finish) ===")
    else:
        print(f"\n=== Rolling backtest: {models} model vs grid-order baseline ===")
    if common.VERBOSITY >= 1:
        header = f"{'rd':>3}  {'event':<26}"
        if show_gain:
            header += f" {'gain':>5}"
        if show_direct:
            header += f" {'direct':>6}"
        header += f" {'grid':>5} {'podium':>8} {'winner':>8}"
        print(header)
        for rec in records:
            line = f"{rec['round']:>3}  {rec['event']:<26}"
            if show_gain:
                line += f" {rec['gain_mae']:>5.1f}"
            if show_direct:
                line += f" {rec['direct_mae']:>6.1f}"
            podium = "/".join(str(rec[k]) for k in ("gain_podium", "direct_podium") if k in rec)
            winner = "/".join("hit" if rec[k] else "-" for k in
                              ("gain_winner", "direct_winner") if k in rec)
            line += f" {rec['grid_mae']:>5.1f} {podium:>8} {winner:>8}"
            print(line)
    if not records:
        print("not enough completed rounds for a backtest yet")
        return
    n = len(records)
    print(f"\nMean over {n} predicted rounds:")
    header = f"{'':<28}"
    if show_gain:
        header += f" {'gain':>7}"
    if show_direct:
        header += f" {'direct':>8}"
    header += f" {'grid-only':>9}"
    print(header)
    line = f"{'position MAE':<28}"
    if show_gain:
        line += f"{np.mean([x['gain_mae'] for x in records]):>7.2f}"
    if show_direct:
        line += f"{np.mean([x['direct_mae'] for x in records]):>8.2f}"
    line += f"{np.mean([x['grid_mae'] for x in records]):>9.2f}"
    print(line)
    line = f"{'podium hit rate':<28}"
    if show_gain:
        line += f"{100 * sum(x['gain_podium'] for x in records) / (3 * n):>6.0f}%"
    if show_direct:
        line += f"{100 * sum(x['direct_podium'] for x in records) / (3 * n):>7.0f}%"
    line += f"{100 * sum(x['grid_podium'] for x in records) / (3 * n):>8.0f}%"
    print(line)
    line = f"{'winner hit rate':<28}"
    if show_gain:
        line += f"{100 * sum(x['gain_winner'] for x in records) / n:>6.0f}%"
    if show_direct:
        line += f"{100 * sum(x['direct_winner'] for x in records) / n:>7.0f}%"
    line += f"{100 * sum(x['grid_winner'] for x in records) / n:>8.0f}%"
    print(line)
    line = f"{'rank correlation':<28}"
    if show_gain:
        line += f"{np.nanmean([x['gain_corr'] for x in records]):>7.2f}"
    if show_direct:
        line += f"{np.nanmean([x['direct_corr'] for x in records]):>8.2f}"
    print(line)


def final_predictions(features, target, profile="fast", models="both"):
    """Train the final models on rounds before `target` and predict `target`.

    Returns models, the training set and the prediction frames of the
    selected models; with both models selected the gain frame carries a
    direct_pos column for side-by-side display. The feature lists actually
    used by each model (all-NaN training columns dropped) are returned as
    gain_features / direct_features, and the hyperparameters the profile
    selected as gain_params / direct_params (None = fixed fast values,
    either because profile="fast" or because there was too little data to
    tune on). models selects what trains: "both" (default), "gain" or
    "direct" — the bundle contains only the selected models' keys (a
    pre-quali target has no grid to anchor the gain model to, so callers
    pass models="direct" there). Prints one live progress line per trained
    model (streamed into the web app's download log).
    """
    tune_note = " (tuning hyperparameters)" if profile == "optimized" else ""
    bundle = {}
    if models in ("both", "gain"):
        common.vprint(1, f"training final gain model{tune_note}...")
        gain_model, train_set, gain_used, gain_params = train_model(features, target, "gain", profile)
        gain = predict_round(gain_model, features, target, "gain", gain_used)
        bundle.update({"gain_model": gain_model, "gain": gain,
                       "gain_features": gain_used, "gain_params": gain_params,
                       "train": train_set})
    if models in ("both", "direct"):
        common.vprint(1, f"training final direct model{tune_note}...")
        direct_model, direct_train, direct_used, direct_params = train_model(features, target, "direct", profile)
        direct = predict_round(direct_model, features, target, "direct", direct_used)
        bundle.update({
            "direct_model": direct_model,
            "train": direct_train,
            "direct": direct,
            "direct_features": direct_used,
            "direct_params": direct_params,
        })
        if models == "both":
            gain["direct_pos"] = gain["driver"].map(direct.set_index("driver")["pred_pos"]).astype(int)
    return bundle


def importance_frame(pred, mode="gain"):
    """Permutation importance for one of the final models, as a DataFrame."""
    train = pred["train"]
    if mode == "gain":
        used = pred["gain_features"]
        target = (train["finish"] - train["grid"]).to_numpy(dtype=float)
        model = pred["gain_model"]
    else:
        used = pred["direct_features"]
        target = train["finish"].to_numpy(dtype=float)
        model = pred["direct_model"]
    return common.importance_scores(model, used, train[used].to_numpy(dtype=float), target)


def final_report(features, year, target, event_name, mode, profile="fast", models="both"):
    if mode == "prequali" and models == "gain":
        print("no grid yet, so the gain model cannot anchor to it - running the direct model only\n")
    if mode == "prequali":
        models = "direct"
    pred = final_predictions(features, target, profile, models)
    with_gain = "gain" in pred
    with_direct = "direct" in pred
    rows = pred["gain"] if with_gain else pred["direct"]
    print(f"\n=== Prediction: {year} {event_name} (round {target}) ===")
    if mode == "prequali":
        print("pre-quali mode: qualifying has not happened yet, so there is no grid to\n"
              "anchor the gain model to - the direct model predicts the race alone,\n"
              "from practice, sprint and season-form features\n")
    elif mode == "pre":
        print("grid estimated from qualifying classification (grid penalties not applied)\n")
    if with_gain and with_direct:
        header = f"{'gain':>4} {'direct':>6}  {'driver':<4} {'team':<20} {'grid':>4}"
    elif with_gain:
        header = f"{'gain':>4}  {'driver':<4} {'team':<20} {'grid':>4}"
    else:
        header = f"{'direct':>6}  {'driver':<4} {'team':<20}"
    if mode == "post":
        header += f" {'actual':>7} {'delta':>6}"
    print(header)
    for _, row in rows.iterrows():
        if with_gain and with_direct:
            line = (f"{'P' + str(row['pred_pos']):>4} {'P' + str(row['direct_pos']):>6}  "
                    f"{row['driver']:<4} {row['team']:<20} {int(row['grid']):>4}")
        elif with_gain:
            line = (f"{'P' + str(row['pred_pos']):>4}  "
                    f"{row['driver']:<4} {row['team']:<20} {int(row['grid']):>4}")
        else:
            line = (f"{'P' + str(row['pred_pos']):>6}  "
                    f"{row['driver']:<4} {row['team']:<20}")
        if mode == "post":
            delta = int(row["pred_pos"] - row["finish"]) if pd.notna(row["finish"]) else ""
            actual = f"P{int(row['finish'])}" if pd.notna(row["finish"]) else "DNF"
            line += f" {actual:>7} {str(delta):>6}"
        print(line)
    if with_gain:
        print(f"\nPredicted podium (gain model)   : {' '.join(pred['gain'].head(3)['driver'])}")
    if with_direct:
        direct_podium = (pred['gain'].sort_values('direct_pos') if with_gain
                         else pred['direct'])
        lead = "" if with_gain else "\n"
        print(f"{lead}Predicted podium (direct model) : "
              f"{' '.join(direct_podium.head(3)['driver'])}")
    if mode == "post":
        actual_rows = features.loc[(features["round"] == target) & (features["finish"] <= 3)]
        actual_top3 = list(actual_rows.sort_values("finish")["driver"])
        print(f"Actual podium                   : {' '.join(actual_top3)}")
    else:
        if with_gain:
            print(f"Predicted winner (gain model)   : {pred['gain'].iloc[0]['driver']}")
        if with_direct:
            direct_order = (pred['gain'].sort_values('direct_pos') if with_gain
                            else pred['direct'])
            print(f"Predicted winner (direct model) : {direct_order.iloc[0]['driver']}")
    train = pred["train"]
    if profile == "optimized":
        untuned = "fast defaults (no clearly better combination found)"
        if with_gain:
            print(f"\nTuned hyperparameters (gain model)   : "
                  f"{common.format_params(pred['gain_params']) or untuned}")
        if with_direct:
            print(f"Tuned hyperparameters (direct model) : "
                  f"{common.format_params(pred['direct_params']) or untuned}")
    if with_gain and common.VERBOSITY >= 1:
        common.importance_report(pred["gain_model"], pred["gain_features"],
                                 train[pred["gain_features"]].to_numpy(dtype=float),
                                 (train["finish"] - train["grid"]).to_numpy(dtype=float),
                                 "gain", width=22)
    if with_direct and common.VERBOSITY >= 1:
        common.importance_report(pred["direct_model"], pred["direct_features"],
                                 train[pred["direct_features"]].to_numpy(dtype=float),
                                 train["finish"].to_numpy(dtype=float),
                                 "direct", width=22)


def practice_done_by(row, now):
    for name in ("Practice 3", "Practice 2", "Practice 1"):
        utc = session_utc(row, name)
        if utc is not None and utc < now - PRACTICE_BUFFER:
            return True
    return False


def _prequali_note(row, now):
    if not practice_done_by(row, now):
        print(f"\nNo practice sessions have finished yet, so the direct model relies on")
        print("season form only - run again after practice (or qualifying) for sharper predictions")


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
            _prequali_note(row, now)
            return int(row["RoundNumber"]), "prequali", str(row["EventName"])
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
        _prequali_note(row, now)
        return rn, "prequali", str(row["EventName"])

    if upcoming:
        row = upcoming[0]
        quali_utc = session_utc(row, "Qualifying")
        if quali_utc is not None and quali_utc < now:
            print(f"\nNext round ({row['EventName']}) has qualifying done -> predicting that race")
            return int(row["RoundNumber"]), "pre", str(row["EventName"])
        print(f"\nQualifying for the next round ({row['EventName']}) has not happened yet,")
        print(f"so the direct model predicts it without a grid (the gain model needs one)")
        _prequali_note(row, now)
        return int(row["RoundNumber"]), "prequali", str(row["EventName"])

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
                        help="predict the next race on the calendar (after its qualifying "
                             "with both models, before it with the direct model only)")
    parser.add_argument("--min-train-rounds", type=int, default=MIN_TRAIN_ROUNDS,
                        help="minimum completed rounds needed before first prediction (default: 5)")
    parser.add_argument("--model", choices=common.MODEL_PROFILES, default="fast",
                        help="model profile: 'fast' uses fixed hyperparameters, "
                             "'optimized' tunes them per model by minimizing "
                             "cross-validated MAE (better error, slower run)")
    parser.add_argument("--models", choices=("both", "gain", "direct"), default="both",
                        help="which models to train, backtest and predict: both "
                             "(default), only the gain model, or only the direct model "
                             "(a pre-quali target always falls back to direct-only — "
                             "there is no grid to anchor the gain model to)")
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
        print(f"\nNo completed races this season yet, falling back to {year}")
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

    features = build_features(data)
    n_rounds = features["round"].nunique()
    print(f"\nDataset: {n_rounds} rounds, {len(features)} driver-race records")
    print(f"Features: {', '.join(FEATURES)}")

    backtest_report(backtest_records(features, args.min_train_rounds, args.model, args.models),
                    args.models)
    final_report(features, year, target, event_name, mode, args.model, args.models)


if __name__ == "__main__":
    main()
