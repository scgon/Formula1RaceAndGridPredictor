"""Custom race prediction: a user-supplied qualifying order -> predicted finish.

The other order pipelines predict a calendar round from the data the providers
published for it. This one predicts a *hypothetical* race: the user designs the
starting grid and the model says how that race would have ended.

The rules that keep the prediction honest:

* the custom race is a real round of a reference season — only the grid is
  replaced. Everything else (the form before the round, the circuit history)
  is what the race pipeline would have used for that same round;
* the model trains on the races *before* that round only (every earlier
  bundled season plus the reference season's earlier rounds), so the custom
  grid is scored out of sample, exactly like the rolling backtest;
* only information that exists before the race starts is used: the grid, the
  driver's and the team's form, the points so far and the circuit history. A
  hypothetical race has no qualifying classification, no practice pace and no
  sprint result, so those features of the race pipeline are deliberately not
  part of the feature set;
* the round's real rows are *replaced* by the custom ones (not added to), so
  neither the form features nor the circuit feature can see the race the
  custom grid is standing in for.

Two models are trained, mirroring the race pipeline: the **gain** model
predicts the positions gained relative to the grid (the custom grid is the
anchor) and the **direct** model predicts the absolute finishing position.
The grid order itself is the baseline.

Nothing is downloaded: the pipeline works on the bundled season CSVs.
"""

import argparse
import unicodedata

import numpy as np
import pandas as pd

import f1_common as common
import predict_race as race

MIN_TRAIN_ROUNDS = common.MIN_TRAIN_ROUNDS

# The hyperparameter search is bounded like the fantasy pipeline's: the
# training pool spans every earlier bundled season, so an unbounded strided
# search would run one validation fold per pooled round (minutes per fit).
TUNE_MAX_SPLITS = 4
TUNE_MAX_TRAIN_ROUNDS = 40

# The features are deliberately limited to what a race prediction can know
# before the race starts. The race pipeline additionally uses the qualifying
# gap, the practice pace and the sprint results — none of which exist for a
# hypothetical race, where the user supplies the grid and nothing else.
FEATURES = [
    "grid",
    "driver_form_3",
    "driver_form_season",
    "driver_last_finish",
    "team_form_3",
    "team_form_season",
    "driver_points_before",
    "team_points_before",
    "circuit_form",
    "circuit_team_form",
    "team_id",
]

# The season CSVs spell the same circuit differently across eras ("Mexican
# Grand Prix" until 2020, "Mexico City Grand Prix" from 2021; "Brazilian
# Grand Prix" before the 2021 rename; the 2020 Bahrain race was billed as
# Sakhir, the 2020 British race as the 70th Anniversary race, the Styrian
# and Tuscan races as their own). The circuit features are keyed on the
# normalized name with those aliases folded in, so a driver's history at a
# circuit is not split across the eras it was renamed in.
EVENT_ALIASES = {
    "brazilian": "sao paulo",
    "mexican": "mexico city",
    "sakhir": "bahrain",
    "70th anniversary": "british",
    "styrian": "austrian",
    "tuscan": "italian",
}


# ---------------------------------------------------------------------------
# data
# ---------------------------------------------------------------------------

def bundled_years():
    """Years whose race CSV is bundled in the repo — the only data this
    pipeline ever reads (it never downloads a session)."""
    years = []
    for path in common.DATA_DIR.glob("season_*.csv"):
        try:
            years.append(int(path.stem.split("_")[1]))
        except (IndexError, ValueError):
            continue
    return sorted(set(years))


def load_season(season):
    """One bundled season's rows (an empty frame when it is not bundled)."""
    path = common.DATA_DIR / f"season_{season}.csv"
    try:
        frame = pd.read_csv(path, dtype={"driver_number": str})
    except (OSError, ValueError):
        return pd.DataFrame()
    frame["season"] = int(season)
    return frame


def load_data(up_to_year=None):
    """Every bundled season (optionally only up to `up_to_year`) in one frame."""
    frames = []
    for year in bundled_years():
        if up_to_year is not None and year > up_to_year:
            continue
        frame = load_season(year)
        if not frame.empty:
            frames.append(frame)
    if not frames:
        raise ValueError("no bundled season data is available")
    return pd.concat(frames, ignore_index=True)


def normalise_event(value):
    """Circuit key: lower case, accent-free, without the 'grand prix' tail."""
    text = unicodedata.normalize("NFKD", str(value))
    text = text.encode("ascii", "ignore").decode()
    text = text.lower().replace("grand prix", " ")
    text = " ".join(text.split())
    return EVENT_ALIASES.get(text, text)


def round_key(season, round_number):
    """Sortable key of a round across seasons (season * 1000 + round)."""
    return int(season) * 1000 + int(round_number)


# ---------------------------------------------------------------------------
# features
# ---------------------------------------------------------------------------

def build_features(data):
    """Feature frame for every race row.

    The features are exactly what a race prediction can know before the race
    starts: the grid, the driver's and the team's form, the points so far and
    the circuit history. Nothing that only exists after the race (the result
    itself, lap times) is used, so the same frame serves the backtest and the
    custom race.
    """
    df = data.copy()
    df["driver_number"] = df["driver_number"].astype(str)
    df["team"] = df["team"].map(common.canonical_team)
    df["event_key"] = df["event"].map(normalise_event)
    df = df.sort_values(["season", "round", "driver_number"]).reset_index(drop=True)
    df["team_id"] = pd.factorize(df["team"])[0]
    df["team_race_mean"] = df.groupby(["season", "round", "team"], sort=False)["finish"].transform("mean")
    df["team_race_points"] = df.groupby(["season", "round", "team"], sort=False)["points"].transform("sum")
    # Driver identity: the three-letter code, scoped to its season. The code
    # alone is not an identity across seasons (SAI was Sainz, then
    # Sargeant) and a car number is not either (#3 was Ricciardo before
    # Verstappen) — inside one season the code identifies the driver.
    df["driver_key"] = df["season"].astype(str) + "|" + df["driver"].astype(str)
    by_driver = df.groupby("driver_key", sort=False)
    df["driver_last_finish"] = by_driver["finish"].transform(lambda s: s.shift(1))
    df["driver_form_3"] = by_driver["finish"].transform(lambda s: s.shift(1).rolling(3, min_periods=1).mean())
    df["driver_form_season"] = by_driver["finish"].transform(lambda s: s.shift(1).expanding(min_periods=1).mean())
    df["driver_points_before"] = by_driver["points"].transform(lambda s: s.cumsum().shift(1)).fillna(0.0)
    df["team_form_3"] = by_driver["team_race_mean"].transform(lambda s: s.shift(1).rolling(3, min_periods=1).mean())
    df["team_form_season"] = by_driver["team_race_mean"].transform(lambda s: s.shift(1).expanding(min_periods=1).mean())
    df["team_points_before"] = by_driver["team_race_points"].transform(lambda s: s.cumsum().shift(1)).fillna(0.0)
    # Circuit form is the one feature that looks across seasons: how the
    # driver (and his team) finished at this circuit before. It is keyed on
    # the three-letter code, the same convention the championship pipeline
    # uses for driver identity across seasons.
    circuit_driver = df.groupby(["driver", "event_key"], sort=False)["finish"]
    df["circuit_form"] = circuit_driver.transform(lambda s: s.shift(1).expanding(min_periods=1).mean())
    circuit_team = df.groupby(["team", "event_key"], sort=False)["finish"]
    df["circuit_team_form"] = circuit_team.transform(lambda s: s.shift(1).expanding(min_periods=1).mean())
    df["round_key"] = df["season"].astype(int) * 1000 + df["round"].astype(int)
    return df


def usable_features(frame):
    """FEATURES minus every column that is entirely NaN in `frame` —
    sklearn's HistGradientBoosting cannot fit a feature with no observed
    value (the circuit features stay empty for a circuit nobody raced twice)."""
    matrix = frame[FEATURES].to_numpy(dtype=float)
    return [name for name, col in zip(FEATURES, matrix.T) if not np.isnan(col).all()]


# ---------------------------------------------------------------------------
# models
# ---------------------------------------------------------------------------

def train_model(features, season, target_round, mode="gain", profile="fast"):
    """Fit one model on every race before `target_round` of `season`.

    "Before" means every earlier bundled season plus the reference season's
    earlier rounds — the same rule the rolling backtest uses, so the custom
    race is scored out of sample exactly like a real round.

    Returns (model, train frame, used feature list, tuned hyperparameters).
    Every fit goes through f1_common.train_or_load, which reuses an identical
    previously fitted model instead of retraining.
    """
    key = round_key(season, target_round)
    train = features[(features["round_key"] < key)
                     & features["finish"].notna()
                     & features["grid"].notna()]
    if train.empty:
        raise ValueError(f"no training data before round {target_round}")
    if mode == "gain":
        target = (train["finish"] - train["grid"]).to_numpy(dtype=float)
    else:
        target = train["finish"].to_numpy(dtype=float)
    used = usable_features(train)
    X = train[used].to_numpy(dtype=float)
    categorical = [used.index("team_id")] if "team_id" in used else None
    model, tuned = common.train_or_load(
        f"custom-race:{mode}", X, target, train["round_key"].to_numpy(),
        profile=profile, categorical_features=categorical,
        tune_max_splits=TUNE_MAX_SPLITS, tune_max_train_rounds=TUNE_MAX_TRAIN_ROUNDS)
    return model, train, used, tuned


def predict_rows(model, features, season, target_round, mode="gain", use_features=None):
    """Predicted finishing order of one round of one season.

    The gain model predicts the positions gained relative to the grid, so the
    finishing position is the custom grid plus the predicted gain; the direct
    model predicts the absolute finishing position.
    """
    use_features = FEATURES if use_features is None else use_features
    rows = features[(features["season"] == season)
                    & (features["round"] == target_round)].copy()
    if rows.empty:
        raise ValueError(f"round {target_round} has no rows to predict")
    rows = rows.sort_values(["grid", "driver_number"])
    output = model.predict(rows[use_features].to_numpy(dtype=float))
    rows["model_output"] = output
    rows["pred_finish"] = (rows["grid"] + output) if mode == "gain" else output
    rows["pred_pos"] = rows["pred_finish"].rank(method="first").astype(int)
    return rows.sort_values("pred_pos").reset_index(drop=True)


def custom_features(features, season, target_round, qualifying_order):
    """The feature frame as it stands before `target_round`, with that round's
    own rows replaced by the user's grid.

    Replacing the round's rows (rather than adding to them) is what keeps the
    prediction honest: the round's real result is no longer in the frame, so
    the circuit feature cannot see it and the form features cannot see the
    race the custom grid stands in for. The features are recomputed on the
    resulting frame, which gives the custom rows the same "state before the
    race" the real rows had.
    """
    key = round_key(season, target_round)
    history = features[features["round_key"] < key]
    entrants = features[(features["season"] == season)
                        & (features["round"] == target_round)]
    if entrants.empty:
        raise ValueError(f"round {target_round} is not in the {season} data")
    order = [str(driver) for driver in qualifying_order]
    if len(set(order)) != len(order):
        raise ValueError("the qualifying order lists a driver more than once")
    missing = [driver for driver in entrants["driver"].unique() if driver not in order]
    if missing:
        raise ValueError("the qualifying order is missing: "
                         + ", ".join(sorted(str(m) for m in missing)))
    rows = []
    for position, driver in enumerate(order, 1):
        match = entrants[entrants["driver"] == driver]
        if match.empty:
            raise ValueError(f"{driver!r} did not enter round {target_round}")
        row = match.iloc[0].copy()
        row["grid"] = float(position)
        row["finish"] = np.nan
        row["points"] = 0.0
        row["status"] = ""
        rows.append(row)
    custom = pd.DataFrame(rows)
    return build_features(pd.concat([history, custom], ignore_index=True))


def custom_predictions(features, season, target_round, qualifying_order,
                       profile="fast", models="both"):
    """Predict the hypothetical race: the round's real grid replaced by
    `qualifying_order` (driver codes in P1 order).

    Returns the same bundle shape as the race pipeline's final_predictions:
    the predicted frames (gain / direct), the models, the training set, the
    feature lists actually used and the tuned hyperparameters. Prints one
    live progress line per trained model (streamed into the web app's log).
    """
    tune_note = " (tuning hyperparameters)" if profile == "optimized" else ""
    frame = custom_features(features, season, target_round, qualifying_order)
    bundle = {}
    if models in ("both", "gain"):
        common.vprint(1, f"training final gain model{tune_note}...")
        gain_model, train, gain_used, gain_params = train_model(
            frame, season, target_round, "gain", profile)
        gain = predict_rows(gain_model, frame, season, target_round, "gain", gain_used)
        bundle.update({"gain_model": gain_model, "gain": gain,
                       "gain_features": gain_used, "gain_params": gain_params,
                       "train": train})
    if models in ("both", "direct"):
        common.vprint(1, f"training final direct model{tune_note}...")
        direct_model, direct_train, direct_used, direct_params = train_model(
            frame, season, target_round, "direct", profile)
        direct = predict_rows(direct_model, frame, season, target_round, "direct", direct_used)
        bundle.update({"direct_model": direct_model, "direct": direct,
                       "train": direct_train, "direct_features": direct_used,
                       "direct_params": direct_params})
        if models == "both":
            gain["direct_pos"] = gain["driver"].map(direct.set_index("driver")["pred_pos"]).astype(int)
    return bundle


def importance_frame(bundle, mode="gain"):
    """Permutation importance for one of the final models, as a DataFrame."""
    train = bundle["train"]
    if mode == "gain":
        used = bundle["gain_features"]
        target = (train["finish"] - train["grid"]).to_numpy(dtype=float)
        model = bundle["gain_model"]
    else:
        used = bundle["direct_features"]
        target = train["finish"].to_numpy(dtype=float)
        model = bundle["direct_model"]
    return common.importance_scores(model, used, train[used].to_numpy(dtype=float), target)


# ---------------------------------------------------------------------------
# backtest
# ---------------------------------------------------------------------------

def backtest_records(features, season, min_train_rounds=MIN_TRAIN_ROUNDS,
                     profile="fast", models="both"):
    """Rolling backtest of the reference season: every completed round
    predicted with its real grid by a model trained only on earlier races.

    Alongside the reported metrics, each record carries the podium points, the
    predicted/actual winner names and the number of rounds each model trained
    on (used by the web app). Prints one live progress line per trained model
    (streamed into the web app's log).
    """
    season_rows = features[features["season"] == season]
    rounds = sorted(season_rows["round"].unique())
    tune_note = " (tuning hyperparameters)" if profile == "optimized" else ""
    records = []
    for r in rounds:
        if sum(1 for x in rounds if x < r) < min_train_rounds:
            continue
        event = str(season_rows.loc[season_rows["round"] == r, "event"].iloc[0])
        try:
            if models in ("both", "gain"):
                common.vprint(1, f"backtest round {r:>2}  {event}: training gain model{tune_note}...")
                gain_model, gain_train, gain_used, _ = train_model(
                    features, season, r, "gain", profile)
            if models in ("both", "direct"):
                common.vprint(1, f"backtest round {r:>2}  {event}: training direct model{tune_note}...")
                direct_model, direct_train, direct_used, _ = train_model(
                    features, season, r, "direct", profile)
        except ValueError:
            continue
        gain = direct = None
        if models in ("both", "gain"):
            gain = predict_rows(gain_model, features, season, r, "gain", gain_used)
        if models in ("both", "direct"):
            direct = predict_rows(direct_model, features, season, r, "direct", direct_used)
        scored = gain if gain is not None else direct
        scored = scored[scored["finish"].notna()]
        winner_row = features.loc[(features["season"] == season)
                                  & (features["round"] == r)
                                  & (features["finish"] == 1), "driver"]
        if scored.empty or winner_row.empty:
            continue
        winner = winner_row.iloc[0]
        actual_rows = features.loc[(features["season"] == season)
                                   & (features["round"] == r)
                                   & (features["finish"] <= 3)]
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
                "gain_points": race.podium_points(gain_top3, actual_top3_order),
                "gain_top1": gain_top3[0],
                "gain_train_rounds": int(gain_train[["season", "round"]].drop_duplicates().shape[0]),
            })
        if direct is not None:
            direct_top3 = list(direct.head(3)["driver"])
            scored_direct = direct[direct["finish"].notna()]
            rec.update({
                "direct_mae": float((scored_direct["pred_pos"] - scored_direct["finish"]).abs().mean()),
                "direct_podium": len(actual_top3 & set(direct_top3)),
                "direct_winner": 1 if direct_top3[0] == winner else 0,
                "direct_corr": common.rank_corr(scored_direct["pred_pos"], scored_direct["finish"]),
                "direct_points": race.podium_points(direct_top3, actual_top3_order),
                "direct_top1": direct_top3[0],
                "direct_train_rounds": int(direct_train[["season", "round"]].drop_duplicates().shape[0]),
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
    """Print the rolling backtest the way the race pipeline does."""
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


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def _season_rounds(features, season):
    season_rows = features[features["season"] == season]
    return sorted(season_rows["round"].unique()), season_rows


def main(argv=None):
    parser = argparse.ArgumentParser(
        description="Predict a hypothetical race from a user-supplied qualifying order")
    parser.add_argument("--season", type=int, default=None,
                        help="reference season (default: the latest bundled season)")
    parser.add_argument("--round", type=int, default=None,
                        help="round to predict (default: the season's last completed round)")
    parser.add_argument("--circuit", default=None,
                        help="circuit name (default: the round's event name)")
    parser.add_argument("--grid", default=None,
                        help="comma-separated driver codes in qualifying order "
                             "(default: the round's real grid)")
    parser.add_argument("--min-train-rounds", type=int, default=MIN_TRAIN_ROUNDS,
                        help="minimum completed rounds needed before the first backtested "
                             "round (default: 5)")
    parser.add_argument("--model", choices=common.MODEL_PROFILES, default="fast",
                        help="model profile: 'fast' uses fixed hyperparameters, "
                             "'optimized' tunes them per model by minimizing "
                             "cross-validated MAE (better error, slower run)")
    common.add_verbosity_args(parser)
    common.add_model_cache_args(parser)
    args = parser.parse_args(argv)
    common.apply_verbosity(args)
    common.apply_model_cache(args)
    common.setup()

    years = bundled_years()
    if not years:
        raise SystemExit("no bundled season data is available")
    season = args.season or years[-1]
    data = load_data(season)
    features = build_features(data)
    rounds, season_rows = _season_rounds(features, season)
    if not rounds:
        raise SystemExit(f"no bundled data for the {season} season")
    target = args.round if args.round is not None else rounds[-1]
    if target not in rounds:
        raise SystemExit(f"round {target} is not in the bundled {season} data")
    event = str(season_rows.loc[season_rows["round"] == target, "event"].iloc[0])
    circuit = args.circuit or event
    entrants = season_rows[season_rows["round"] == target]
    if args.grid:
        order = [x.strip() for x in args.grid.split(",") if x.strip()]
    else:
        order = list(entrants.sort_values("grid")["driver"])

    print(f"\nDataset: {len(features)} driver-race records over "
          f"{features['season'].nunique()} bundled seasons")
    print(f"Features: {', '.join(FEATURES)}")
    backtest_report(backtest_records(features, season, args.min_train_rounds, args.model))

    bundle = custom_predictions(features, season, target, order, args.model)
    gain = bundle["gain"]
    direct = bundle.get("direct")
    print(f"\n=== Custom race: {season} {circuit} (round {target}) ===")
    header = f"{'grid':>4} {'gain':>5}"
    if direct is not None:
        header += f" {'direct':>6}"
    header += f"  {'driver':<4} {'team':<20}"
    print(header)
    for _, row in gain.iterrows():
        line = f"{int(row['grid']):>4} {'P' + str(row['pred_pos']):>5}"
        if direct is not None:
            line += f" {'P' + str(row['direct_pos']):>6}"
        line += f"  {row['driver']:<4} {row['team']:<20}"
        print(line)
    if direct is not None:
        print(f"\nPredicted podium (gain model)   : {' '.join(gain.head(3)['driver'])}")
        print(f"Predicted podium (direct model) : {' '.join(direct.head(3)['driver'])}")
    else:
        print(f"\nPredicted podium (gain model)   : {' '.join(gain.head(3)['driver'])}")
    if args.model == "optimized":
        untuned = "fast defaults (no clearly better combination found)"
        print(f"\nTuned hyperparameters (gain model)   : "
              f"{common.format_params(bundle['gain_params']) or untuned}")
        if direct is not None:
            print(f"Tuned hyperparameters (direct model) : "
                  f"{common.format_params(bundle['direct_params']) or untuned}")
    if common.VERBOSITY >= 1:
        common.importance_report(bundle["gain_model"], bundle["gain_features"],
                                 bundle["train"][bundle["gain_features"]].to_numpy(dtype=float),
                                 (bundle["train"]["finish"] - bundle["train"]["grid"]).to_numpy(dtype=float),
                                 "gain", width=22)
        if direct is not None:
            common.importance_report(bundle["direct_model"], bundle["direct_features"],
                                     bundle["train"][bundle["direct_features"]].to_numpy(dtype=float),
                                     bundle["train"]["finish"].to_numpy(dtype=float),
                                     "direct", width=22)


if __name__ == "__main__":
    main()
