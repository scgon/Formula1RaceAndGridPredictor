"""Formula 1 Fantasy team optimizer: predicts expected fantasy points for drivers
and constructors, selects the optimal team (5 drivers + 2 constructors + 1 DRS
boost driver) staying strictly under the budget cap (default $100.0M), and
provides strategic archetypes and alternative lineups.

Runs in three modes:
- "post": reviews a completed round (compares recommended team vs actual dream
  team and baseline, computing actual fantasy points scored and capture rate).
- "pre": predicts after qualifying (full grid is known, maximizing positional
  gain/loss and qualifying accuracy).
- "prequali": predicts before qualifying (uses practice pace FP1/FP2, sprint
  sessions, and season form — ideal for locking lineups before the deadline).

Shared machinery lives in f1_common.py.
"""

import argparse
import heapq
import itertools
from datetime import datetime, timedelta
from typing import Any, Dict, List, Optional, Sequence, Tuple

import numpy as np
import pandas as pd
import fastf1

import f1_common as common
import predict_race as race

MIN_TRAIN_ROUNDS = common.MIN_TRAIN_ROUNDS
DEFAULT_BUDGET = 100.0

PRACTICE_BUFFER = timedelta(hours=2)
PRACTICE_SESSIONS = ("FP1", "FP2", "FP3")

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
    "driver_fantasy_form_3",
    "driver_fantasy_form_season",
    "driver_fantasy_last",
    "driver_dnf_rate",
    "team_fantasy_form_3",
    "team_fantasy_form_season",
    "driver_points_before",
    "team_points_before",
    "team_id",
]

PRICES_FILE = common.DATA_DIR / "fantasy_prices.csv"

setup = common.setup
utc_now = common.utc_now
session_utc = common.session_utc
vprint = common.vprint


def is_finished_like(status: Any) -> bool:
    """Classified as finishing the race (not a retirement / DNF)."""
    text = str(status)
    return (text in ("Finished", "Lapped", "Disqualified")
            or text.startswith("+"))


# ---------------------------------------------------------------------------
# Fantasy scoring engine (official F1 Fantasy rules)
# ---------------------------------------------------------------------------

def calculate_driver_fantasy_points(df: pd.DataFrame) -> pd.Series:
    """Calculate official F1 Fantasy points scored by each driver row.

    For upcoming / uncompleted rounds (finish is NaN), returns NaN.
    """
    scores = pd.Series(np.nan, index=df.index, dtype=float)

    finish_pts_map = {1: 25.0, 2: 18.0, 3: 15.0, 4: 12.0, 5: 10.0,
                      6: 8.0, 7: 6.0, 8: 4.0, 9: 2.0, 10: 1.0}

    # Process each completed round separately
    for rnd, grp in df.groupby("round", sort=False):
        has_status = grp["status"].fillna("").astype(str).str.strip().ne("").any()
        if not has_status and grp["finish"].isna().all():
            continue

        for team_name, tm_grp in grp.groupby("team", sort=False):
            for idx, row in tm_grp.iterrows():
                f = row.get("finish")
                status = str(row.get("status", "")).strip()
                if not status and pd.isna(f):
                    continue

                pts = 0.0
                g = row.get("grid")

                # Qualifying points (top 10: 10 down to 1)
                q = row.get("quali_pos") if "quali_pos" in row and pd.notna(row.get("quali_pos")) else g
                if pd.notna(q):
                    q_int = int(q)
                    if 1 <= q_int <= 10:
                        pts += float(11 - q_int)

                # Teammate beat in qualifying (+2)
                teammates = tm_grp[tm_grp.index != idx]
                if not teammates.empty and pd.notna(q):
                    tm_q = teammates.iloc[0].get("quali_pos") if "quali_pos" in teammates.columns else teammates.iloc[0].get("grid")
                    if pd.notna(tm_q) and q < tm_q:
                        pts += 2.0

                # Race finish
                status = str(row.get("status", ""))
                finished = is_finished_like(status)

                if finished:
                    pts += finish_pts_map.get(int(f), 0.0)
                    # Positions gained/lost (+1 per gain, -1 per loss)
                    if pd.notna(g):
                        pts += float(g - f)
                    # Classified finish (+1)
                    pts += 1.0

                    # Beat teammate in race (+3)
                    if not teammates.empty:
                        tm_f = teammates.iloc[0].get("finish")
                        tm_status = str(teammates.iloc[0].get("status", ""))
                        tm_finished = is_finished_like(tm_status)
                        if not tm_finished or (pd.notna(tm_f) and f < tm_f):
                            pts += 3.0
                else:
                    # DNF penalty (-20)
                    pts -= 20.0

                # Sprint points (if applicable)
                sf = row.get("sprint_finish")
                if pd.notna(sf):
                    sf_int = int(sf)
                    if 1 <= sf_int <= 8:
                        pts += float(9 - sf_int)
                    sg = row.get("sprint_gain")
                    if pd.notna(sg):
                        pts += float(sg)

                scores.at[idx] = pts

    return scores


def calculate_constructor_fantasy_points(df: pd.DataFrame,
                                         driver_pts_col: str = "fantasy_points") -> pd.DataFrame:
    """Compute constructor fantasy points for each completed round and team.

    Constructor score = sum of both drivers' points + qualifying progression
    bonus + pit stop efficiency bonus (+5 baseline).
    """
    rows = []
    for (rnd, team), grp in df.groupby(["round", "team"], sort=False):
        if grp[driver_pts_col].isna().all():
            continue

        d_sum = grp[driver_pts_col].sum()

        # Qualifying progression bonus based on grid / quali_pos
        grids = []
        for _, r in grp.iterrows():
            g = r.get("quali_pos") if "quali_pos" in r and pd.notna(r.get("quali_pos")) else r.get("grid")
            grids.append(g if pd.notna(g) else 20.0)

        q3_count = sum(1 for g in grids if g <= 10.0)
        q2_count = sum(1 for g in grids if g <= 15.0)

        bonus = 0.0
        if q3_count == 2:
            bonus += 10.0
        elif q3_count == 1:
            bonus += 5.0
        elif q2_count == 2:
            bonus += 3.0
        elif q2_count == 1:
            bonus += 1.0
        else:
            bonus -= 1.0

        # Pit stop baseline bonus
        bonus += 5.0

        rows.append({
            "round": rnd,
            "team": team,
            "constructor_points": d_sum + bonus,
            "constructor_driver_sum": d_sum,
            "constructor_bonus": bonus,
        })

    return pd.DataFrame(rows)


# ---------------------------------------------------------------------------
# Pricing & Budget Engine
# ---------------------------------------------------------------------------

def load_price_table() -> pd.DataFrame:
    """Read the bundled fantasy prices CSV if present."""
    if PRICES_FILE.exists():
        try:
            return pd.read_csv(PRICES_FILE)
        except Exception:
            pass
    return pd.DataFrame(columns=["year", "type", "name", "price"])


def get_prices(year: int, drivers: Sequence[str], constructors: Sequence[str],
               season_points: Optional[Dict[str, float]] = None) -> Tuple[Dict[str, float], Dict[str, float]]:
    """Return ({driver -> price}, {constructor -> price}) in millions.

    Uses data/fantasy_prices.csv first; falls back to performance-based
    scaling if an asset is missing.
    """
    tbl = load_price_table()
    y_tbl = tbl[tbl["year"] == year] if not tbl.empty else pd.DataFrame()

    d_prices = {}
    c_prices = {}

    for _, row in y_tbl.iterrows():
        p_type = str(row["type"]).lower()
        name = str(row["name"])
        price = float(row["price"])
        if p_type == "driver":
            d_prices[name] = price
        elif p_type == "constructor":
            c_prices[name] = price

    # Fallback for any missing drivers
    for d in drivers:
        if d not in d_prices:
            pts = (season_points.get(d, 0.0) if season_points else 0.0)
            d_prices[d] = float(np.clip(5.0 + pts * 0.15, 4.5, 30.0))

    # Fallback for any missing constructors
    for c in constructors:
        if c not in c_prices:
            # Check team lineage aliases
            alias_match = None
            for lineage in (
                {"Red Bull", "Red Bull Racing"},
                {"Renault", "Alpine", "Alpine F1 Team"},
                {"Sauber", "Alfa Romeo", "Alfa Romeo Racing", "Kick Sauber", "Audi"},
                {"Toro Rosso", "AlphaTauri", "RB", "RB F1 Team", "Racing Bulls"},
                {"Force India", "Racing Point", "Aston Martin"},
                {"Cadillac", "Cadillac F1 Team"},
            ):
                if c in lineage:
                    for known in lineage:
                        if known in c_prices:
                            alias_match = c_prices[known]
                            break
                if alias_match is not None:
                    break
            if alias_match is not None:
                c_prices[c] = alias_match
            else:
                pts = (season_points.get(c, 0.0) if season_points else 0.0)
                c_prices[c] = float(np.clip(6.0 + pts * 0.20, 5.0, 30.0))

    return d_prices, c_prices


# ---------------------------------------------------------------------------
# Team Optimizer (Fast 0-1 Knapsack with DRS Boost)
# ---------------------------------------------------------------------------

def optimize_team(driver_df: pd.DataFrame,
                  constructor_df: pd.DataFrame,
                  d_prices: Dict[str, float],
                  c_prices: Dict[str, float],
                  budget: float = DEFAULT_BUDGET,
                  top_k: int = 5,
                  score_col: str = "pred_points") -> Dict[str, Any]:
    """Find the optimal F1 Fantasy team (5 drivers + 2 constructors + 1 DRS
    booster) under the budget cap.

    Returns the global optimal team, top K alternatives, and strategic
    archetypes.
    """
    if budget < 0:
        raise ValueError("budget must be non-negative")
    # Prepare driver items
    d_items = []
    for _, row in driver_df.iterrows():
        name = str(row["driver"])
        pts = float(row[score_col])
        actual_pts = float(row.get("actual_points", np.nan))
        price = d_prices.get(name, 10.0)
        team = str(row["team"])
        d_items.append({
            "name": name,
            "team": team,
            "price": price,
            "points": pts,
            "actual_points": actual_pts,
            "ppm": pts / price if price > 0 else 0.0,
        })

    # Prepare constructor items
    c_items = []
    for _, row in constructor_df.iterrows():
        name = str(row["team"])
        pts = float(row[score_col])
        actual_pts = float(row.get("actual_points", np.nan))
        price = c_prices.get(name, 12.0)
        c_items.append({
            "name": name,
            "team": name,
            "price": price,
            "points": pts,
            "actual_points": actual_pts,
            "ppm": pts / price if price > 0 else 0.0,
        })

    # Precompute constructor pairs
    c_pairs = []
    for c1, c2 in itertools.combinations(c_items, 2):
        cost = c1["price"] + c2["price"]
        pts = c1["points"] + c2["points"]
        act = c1["actual_points"] + c2["actual_points"]
        c_pairs.append((cost, pts, act, (c1, c2)))

    # Sort constructor pairs by cost ascending
    c_pairs.sort(key=lambda x: x[0])
    min_c_cost = c_pairs[0][0] if c_pairs else 0.0

    # Precompute driver 5-tuples
    d_tuples = []
    for combo in itertools.combinations(d_items, 5):
        cost = sum(d["price"] for d in combo)
        if cost + min_c_cost > budget:
            continue
        base_pts = sum(d["points"] for d in combo)
        base_act = sum(d["actual_points"] for d in combo)
        # DRS booster: doubles highest predicted driver in chosen 5
        drs_d = max(combo, key=lambda d: d["points"])
        total_pts = base_pts + drs_d["points"]
        total_act = base_act + (drs_d["actual_points"] if pd.notna(drs_d["actual_points"]) else 0.0)
        d_tuples.append((cost, total_pts, total_act, combo, drs_d))

    top_c_names = set(c["name"] for c in sorted(c_items, key=lambda c: c["points"], reverse=True)[:3])
    top_d_names = set(d["name"] for d in sorted(d_items, key=lambda d: d["points"], reverse=True)[:3])

    heap = []
    counter = 0

    best_mega_c = None
    best_mega_d = None
    best_ppm = None

    for d_cost, d_pts, d_act, d_combo, drs_d in d_tuples:
        rem_budget = budget - d_cost
        has_top_d = sum(1 for d in d_combo if d["name"] in top_d_names) >= 2

        for c_cost, c_pts, c_act, c_combo in c_pairs:
            if c_cost > rem_budget:
                break
            tot_cost = d_cost + c_cost
            tot_pts = d_pts + c_pts
            tot_act = d_act + c_act

            counter += 1
            entry = (tot_pts, -tot_cost, counter, d_combo, c_combo, drs_d["name"], tot_cost, tot_act)

            if len(heap) < top_k:
                heapq.heappush(heap, entry)
            elif entry > heap[0]:
                heapq.heapreplace(heap, entry)

            # Strategy archetypes
            if any(c["name"] in top_c_names for c in c_combo):
                if best_mega_c is None or tot_pts > best_mega_c[0]:
                    best_mega_c = (tot_pts, tot_cost, d_combo, c_combo, drs_d["name"], tot_act)

            if has_top_d:
                if best_mega_d is None or tot_pts > best_mega_d[0]:
                    best_mega_d = (tot_pts, tot_cost, d_combo, c_combo, drs_d["name"], tot_act)

            ppm = tot_pts / tot_cost if tot_cost > 0 else 0.0
            if best_ppm is None or ppm > best_ppm[0]:
                best_ppm = (ppm, tot_pts, tot_cost, d_combo, c_combo, drs_d["name"], tot_act)

    def _format_item(tot_pts, tot_cost, d_combo, c_combo, drs_name, tot_act):
        return {
            "drivers": tuple(d_combo),
            "constructors": tuple(c_combo),
            "drs": drs_name,
            "cost": round(tot_cost, 2),
            "remaining": round(budget - tot_cost, 2),
            "points": round(tot_pts, 2),
            "actual_points": round(tot_act, 2) if pd.notna(tot_act) else np.nan,
        }

    if not heap:
        # Never return a roster over the cap: callers must be able to trust
        # that every recommendation satisfies the platform's hard budget.
        cheapest_d = sorted(d_items, key=lambda d: d["price"])[:5]
        cheapest_c = sorted(c_items, key=lambda c: c["price"])[:2]
        minimum_cost = sum(d["price"] for d in cheapest_d) + sum(c["price"] for c in cheapest_c)
        raise ValueError(
            f"no valid fantasy roster fits the ${budget:.1f}M budget "
            f"(minimum roster cost is ${minimum_cost:.1f}M)"
        )

    # Extract top lineups sorted by points descending
    sorted_heap = sorted(heap, key=lambda x: (x[0], -x[6]), reverse=True)
    optimal = _format_item(sorted_heap[0][0], sorted_heap[0][6], sorted_heap[0][3],
                           sorted_heap[0][4], sorted_heap[0][5], sorted_heap[0][7])

    alternatives = [
        _format_item(it[0], it[6], it[3], it[4], it[5], it[7])
        for it in sorted_heap[1:top_k]
    ]

    mega_c_fmt = (_format_item(best_mega_c[0], best_mega_c[1], best_mega_c[2],
                               best_mega_c[3], best_mega_c[4], best_mega_c[5])
                  if best_mega_c is not None else optimal)

    mega_d_fmt = (_format_item(best_mega_d[0], best_mega_d[1], best_mega_d[2],
                               best_mega_d[3], best_mega_d[4], best_mega_d[5])
                  if best_mega_d is not None else optimal)

    ppm_fmt = (_format_item(best_ppm[1], best_ppm[2], best_ppm[3],
                            best_ppm[4], best_ppm[5], best_ppm[6])
               if best_ppm is not None else optimal)

    archetypes = {
        "balanced": optimal,
        "mega_constructors": mega_c_fmt,
        "mega_drivers": mega_d_fmt,
        "max_ppm": ppm_fmt,
    }

    return {
        "optimal": optimal,
        "alternatives": alternatives,
        "archetypes": archetypes,
        "budget": budget,
        "valid_count": counter,
    }


# ---------------------------------------------------------------------------
# Feature Engineering
# ---------------------------------------------------------------------------

def collect_season(year: int, schedule: pd.DataFrame, refresh: bool = False,
                   rate_limit_wait: Optional[int] = None) -> pd.DataFrame:
    """Thin wrapper around race.collect_season: uses the bundled season CSVs."""
    return race.collect_season(year, schedule, refresh=refresh, rate_limit_wait=rate_limit_wait)


def load_upcoming_round(year: int, round_number: int, event_name: str) -> pd.DataFrame:
    """Upcoming frame after qualifying (grid known)."""
    return race.load_upcoming_round(year, round_number, event_name)


def load_upcoming_round_prequali(year: int, round_number: int, event_name: str,
                                fallback: pd.DataFrame) -> pd.DataFrame:
    """Upcoming frame before qualifying (practice/form only)."""
    return race.load_upcoming_round_prequali(year, round_number, event_name, fallback)


def load_upcoming_round_checked(year: int, round_number: int, event_name: str,
                                fallback: pd.DataFrame) -> Tuple[pd.DataFrame, str]:
    """Upcoming frame with lag-checked mode."""
    return race.load_upcoming_round_checked(year, round_number, event_name, fallback)


def build_features(data: pd.DataFrame, year: Optional[int] = None) -> pd.DataFrame:
    """Build driver and team rolling fantasy features from season data.

    Shift(1) is applied so past performance never leaks into the predicted
    round.
    """
    df = data.copy()
    if year is not None:
        df["year"] = int(year)
    elif "year" not in df.columns:
        df["year"] = datetime.now().year

    df["driver_number"] = df["driver_number"].astype(str)
    df = df.sort_values(["round", "driver_number"]).reset_index(drop=True)

    # Ground truth fantasy points
    df["fantasy_points"] = calculate_driver_fantasy_points(df)

    # Team round points
    df["team_fantasy_round"] = df.groupby(["round", "team"], sort=False)["fantasy_points"].transform("sum")
    df["team_race_points"] = df.groupby(["round", "team"], sort=False)["points"].transform("sum")
    df["team_quali_delta"] = df.groupby(["round", "team"], sort=False)["quali_delta"].transform("mean")
    df["team_id"] = pd.factorize(df["team"])[0]

    # Driver rolling features (shifted by 1 round)
    by_driver = df.groupby("driver_number", sort=False)
    df["driver_fantasy_last"] = by_driver["fantasy_points"].transform(lambda s: s.shift(1))
    df["driver_fantasy_form_3"] = by_driver["fantasy_points"].transform(lambda s: s.shift(1).rolling(3, min_periods=1).mean())
    df["driver_fantasy_form_season"] = by_driver["fantasy_points"].transform(lambda s: s.shift(1).expanding(min_periods=1).mean())

    # Driver DNF rate
    df["driver_dnf_rate"] = by_driver["status"].transform(
        lambda s: (~s.map(is_finished_like)).shift(1).expanding(min_periods=1).mean()
    ).fillna(0.0)

    df["driver_points_before"] = by_driver["points"].transform(lambda s: s.cumsum().shift(1)).fillna(0.0)

    # Team features must shift *team-round* aggregates, not raw driver rows.
    # Shifting the raw rows would expose the current round's aggregate to the
    # second driver of each team (the rows are sorted by round and driver).
    team_round = (df.groupby(["team", "round"], sort=False)
                    .agg(team_fantasy_round=("fantasy_points", "sum"),
                         team_race_points=("points", "sum"))
                    .reset_index()
                    .sort_values(["team", "round"]))
    by_team_round = team_round.groupby("team", sort=False)
    team_round["team_fantasy_form_3"] = by_team_round["team_fantasy_round"].transform(
        lambda s: s.shift(1).rolling(3, min_periods=1).mean())
    team_round["team_fantasy_form_season"] = by_team_round["team_fantasy_round"].transform(
        lambda s: s.shift(1).expanding(min_periods=1).mean())
    team_round["team_points_before"] = by_team_round["team_race_points"].transform(
        lambda s: s.shift(1).cumsum()).fillna(0.0)
    team_history = team_round.set_index(["team", "round"])
    keys = pd.MultiIndex.from_frame(df[["team", "round"]])
    for col in ("team_fantasy_form_3", "team_fantasy_form_season", "team_points_before"):
        df[col] = team_history[col].reindex(keys).to_numpy()

    return df


def usable_features(frame: pd.DataFrame) -> List[str]:
    """FEATURES minus any column that is entirely NaN in `frame`."""
    matrix = frame[FEATURES].to_numpy(dtype=float)
    return [name for name, col in zip(FEATURES, matrix.T) if not np.isnan(col).all()]


# ---------------------------------------------------------------------------
# Model Training & Inference
# ---------------------------------------------------------------------------

def train_model(features: pd.DataFrame, target_round: int, mode: str = "delta",
                profile: str = "fast") -> Tuple[Any, pd.DataFrame, List[str], Optional[Dict[str, Any]]]:
    """Fit one fantasy model on all rounds before `target_round`.

    Returns (model, train_frame, used_features, tuned_hyperparameters).
    All fits go through f1_common.train_or_load.
    """
    train = features[
        (features["round"] < target_round)
        & features["fantasy_points"].notna()
    ]
    if train.empty:
        raise ValueError(f"no training data before round {target_round}")

    if mode == "delta":
        baseline = train["driver_fantasy_form_season"].fillna(train["fantasy_points"])
        target = (train["fantasy_points"] - baseline).to_numpy(dtype=float)
    else:
        target = train["fantasy_points"].to_numpy(dtype=float)

    used = usable_features(train)
    X = train[used].to_numpy(dtype=float)
    categorical = [used.index("team_id")] if "team_id" in used else None

    model, tuned = common.train_or_load(
        f"fantasy:{mode}", X, target, train["round"].to_numpy(),
        profile=profile, categorical_features=categorical
    )
    return model, train, used, tuned


def predict_drivers(model: Any, features: pd.DataFrame, target_round: int,
                    mode: str = "delta", use_features: Optional[List[str]] = None) -> pd.DataFrame:
    """Predict expected driver fantasy points for target_round."""
    use_features = FEATURES if use_features is None else use_features
    rows = features[features["round"] == target_round].copy()
    if rows.empty:
        raise ValueError(f"round {target_round} has no rows to predict")

    # In pre mode with a known grid, keep entrants with grid; in prequali predict all
    if rows["grid"].notna().any():
        rows = rows[rows["grid"].notna()].copy()

    X = rows[use_features].to_numpy(dtype=float)
    model_output = model.predict(X)

    if mode == "delta":
        baseline = rows["driver_fantasy_form_season"].fillna(0.0).to_numpy(dtype=float)
        rows["pred_points"] = np.round(baseline + model_output, 2)
    else:
        rows["pred_points"] = np.round(model_output, 2)

    rows["actual_points"] = rows["fantasy_points"]
    return rows.sort_values("pred_points", ascending=False).reset_index(drop=True)


def predict_constructors_from_drivers(driver_preds: pd.DataFrame) -> pd.DataFrame:
    """Predict constructor fantasy points from the drivers' predicted points."""
    rows = []
    for team, grp in driver_preds.groupby("team", sort=False):
        d_pred_sum = grp["pred_points"].sum()
        act_sum = grp["actual_points"].sum() if grp["actual_points"].notna().all() else np.nan

        # Estimated qualifying progression bonus
        grids = grp["grid"].dropna().tolist() if "grid" in grp else []
        if len(grids) == 2:
            q3_count = sum(1 for g in grids if g <= 10.0)
            q2_count = sum(1 for g in grids if g <= 15.0)
            bonus = 10.0 if q3_count == 2 else (5.0 if q3_count == 1 else (3.0 if q2_count == 2 else 0.0))
        else:
            bonus = 5.0  # general expectation

        # Pit stop bonus
        bonus += 5.0

        rows.append({
            "team": team,
            "pred_points": round(d_pred_sum + bonus, 2),
            "actual_points": round(act_sum + bonus, 2) if pd.notna(act_sum) else np.nan,
            "driver_points_sum": round(d_pred_sum, 2),
            "bonus": bonus,
        })

    return pd.DataFrame(rows).sort_values("pred_points", ascending=False).reset_index(drop=True)


# ---------------------------------------------------------------------------
# Backtest & Pipeline API
# ---------------------------------------------------------------------------

def backtest_records(features: pd.DataFrame, min_train_rounds: int = MIN_TRAIN_ROUNDS,
                     profile: str = "fast", models: str = "both",
                     budget: float = DEFAULT_BUDGET,
                     year: Optional[int] = None) -> List[Dict[str, Any]]:
    """Rolling leave-one-round-out backtest.

    For each round, trains models on previous rounds, predicts driver/constructor
    points, and picks the optimal team under the cap. Compares against naive
    baseline team and actual dream team.
    """
    completed_rounds = sorted(features[features["fantasy_points"].notna()]["round"].unique())
    eligible = [r for r in completed_rounds if r > min_train_rounds]
    if not eligible:
        return []

    if year is None:
        year = int(features["year"].iloc[0]) if "year" in features.columns else datetime.now().year

    records = []

    for target_round in eligible:
        event_name = str(features[features["round"] == target_round]["event"].iloc[0])

        # Get prices
        round_drivers = features[features["round"] == target_round]["driver"].unique()
        round_teams = features[features["round"] == target_round]["team"].unique()
        d_prices, c_prices = get_prices(year, round_drivers, round_teams)

        # Baseline predictions: driver season average so far
        baseline_df = features[features["round"] == target_round].copy()
        baseline_df["pred_points"] = baseline_df["driver_fantasy_form_season"].fillna(0.0)
        baseline_df["actual_points"] = baseline_df["fantasy_points"]
        c_baseline_df = predict_constructors_from_drivers(baseline_df)

        base_opt = optimize_team(baseline_df, c_baseline_df, d_prices, c_prices, budget=budget, top_k=1)
        base_act = base_opt["optimal"]["actual_points"]

        # Actual optimal dream team (with hindsight)
        actual_df = features[features["round"] == target_round].copy()
        actual_df["pred_points"] = actual_df["fantasy_points"]
        actual_df["actual_points"] = actual_df["fantasy_points"]
        c_act_df = predict_constructors_from_drivers(actual_df)

        dream_opt = optimize_team(actual_df, c_act_df, d_prices, c_prices, budget=budget, top_k=1)
        dream_act = dream_opt["optimal"]["actual_points"]

        # Train model(s)
        train_mode = "direct" if models == "direct" else "delta"
        model, train_df, used, _ = train_model(features, target_round, mode=train_mode, profile=profile)

        d_pred = predict_drivers(model, features, target_round, mode=train_mode, use_features=used)
        c_pred = predict_constructors_from_drivers(d_pred)

        model_opt = optimize_team(d_pred, c_pred, d_prices, c_prices, budget=budget, top_k=1)
        model_act = model_opt["optimal"]["actual_points"]

        # Prediction accuracy metrics on drivers
        y_true = d_pred["actual_points"].to_numpy(dtype=float)
        y_pred = d_pred["pred_points"].to_numpy(dtype=float)
        valid = ~np.isnan(y_true) & ~np.isnan(y_pred)
        mae = float(np.mean(np.abs(y_true[valid] - y_pred[valid]))) if np.any(valid) else np.nan
        corr = common.rank_corr(y_pred[valid], y_true[valid]) if np.any(valid) else np.nan

        capture_pct = (model_act / dream_act * 100.0) if (pd.notna(model_act) and pd.notna(dream_act) and dream_act > 0) else np.nan

        records.append({
            "round": target_round,
            "event": event_name,
            "model_team": [d["name"] for d in model_opt["optimal"]["drivers"]],
            "model_constructors": [c["name"] for c in model_opt["optimal"]["constructors"]],
            "drs": model_opt["optimal"]["drs"],
            "model_pts": model_act,
            "base_pts": base_act,
            "dream_pts": dream_act,
            "capture_pct": capture_pct,
            "budget_used": model_opt["optimal"]["cost"],
            "mae": mae,
            "rank_corr": corr,
            "train_rounds": int(train_df["round"].nunique()),
        })

    return records


def final_predictions(features: pd.DataFrame, target_round: int, profile: str = "fast",
                      models: str = "both", budget: float = DEFAULT_BUDGET,
                      year: Optional[int] = None) -> Dict[str, Any]:
    """Fit final models and compute recommended team and alternatives."""
    if models not in ("both", "delta", "direct"):
        raise ValueError(f"unknown fantasy model selection: {models!r}")
    if year is None:
        year = int(features["year"].iloc[0]) if "year" in features.columns else datetime.now().year

    round_drivers = features[features["round"] == target_round]["driver"].unique()
    round_teams = features[features["round"] == target_round]["team"].unique()
    d_prices, c_prices = get_prices(year, round_drivers, round_teams)

    delta_model = direct_model = None
    delta_features = direct_features = None
    delta_params = direct_params = None
    train_df = None
    d_delta = c_delta = d_direct = c_direct = None

    if models in ("both", "delta"):
        delta_model, train_df, delta_features, delta_params = train_model(
            features, target_round, mode="delta", profile=profile
        )
        d_delta = predict_drivers(delta_model, features, target_round,
                                  mode="delta", use_features=delta_features)
        c_delta = predict_constructors_from_drivers(d_delta)

    if models in ("both", "direct"):
        direct_model, direct_train, direct_features, direct_params = train_model(
            features, target_round, mode="direct", profile=profile
        )
        if train_df is None:
            train_df = direct_train
        d_direct = predict_drivers(direct_model, features, target_round,
                                   mode="direct", use_features=direct_features)
        c_direct = predict_constructors_from_drivers(d_direct)

    # Primary chosen model
    primary_d = d_direct if models == "direct" else d_delta
    primary_c = c_direct if models == "direct" else c_delta
    main_mode = "direct" if models == "direct" else "delta"

    team_opt = optimize_team(primary_d, primary_c, d_prices, c_prices, budget=budget, top_k=5)

    # In post mode, also optimize the actual dream team
    actual_dream = None
    if primary_d["actual_points"].notna().any():
        act_d = primary_d.copy()
        act_d["pred_points"] = act_d["actual_points"]
        act_c = predict_constructors_from_drivers(act_d)
        actual_dream = optimize_team(act_d, act_c, d_prices, c_prices, budget=budget, top_k=1)["optimal"]

    # Build full assets table (drivers + constructors with pricing and PPM)
    d_board = primary_d.copy()
    d_board["price"] = d_board["driver"].map(d_prices).fillna(10.0)
    d_board["ppm"] = np.round(d_board["pred_points"] / d_board["price"], 2)
    chosen_drivers = set(d["name"] for d in team_opt["optimal"]["drivers"])
    d_board["selected"] = d_board["driver"].isin(chosen_drivers)
    d_board["is_drs"] = d_board["driver"] == team_opt["optimal"]["drs"]

    c_board = primary_c.copy()
    c_board["price"] = c_board["team"].map(c_prices).fillna(12.0)
    c_board["ppm"] = np.round(c_board["pred_points"] / c_board["price"], 2)
    chosen_c = set(c["name"] for c in team_opt["optimal"]["constructors"])
    c_board["selected"] = c_board["team"].isin(chosen_c)

    return {
        "delta": d_delta,
        "direct": d_direct,
        "primary_drivers": primary_d,
        "primary_constructors": primary_c,
        "c_delta": c_delta,
        "c_direct": c_direct,
        "team_optimal": team_opt["optimal"],
        "alternatives": team_opt["alternatives"],
        "archetypes": team_opt["archetypes"],
        "actual_dream": actual_dream,
        "d_board": d_board,
        "c_board": c_board,
        "train": train_df,
        "train_rounds": int(train_df["round"].nunique()),
        "delta_model": delta_model,
        "direct_model": direct_model,
        "delta_features": delta_features,
        "direct_features": direct_features,
        "delta_params": delta_params,
        "direct_params": direct_params,
        "main_mode": main_mode,
        "prices": {"drivers": d_prices, "constructors": c_prices},
    }


def importance_frame(pred: Dict[str, Any], mode: str = "delta") -> pd.DataFrame:
    """Permutation importance of the specified final model."""
    model = pred[f"{mode}_model"]
    if model is None:
        raise ValueError(f"{mode} model was not trained")
    used = pred[f"{mode}_features"]
    train = pred["train"]
    if mode == "delta":
        baseline = train["driver_fantasy_form_season"].fillna(train["fantasy_points"])
        target = (train["fantasy_points"] - baseline).to_numpy(dtype=float)
    else:
        target = train["fantasy_points"].to_numpy(dtype=float)

    X = train[used].to_numpy(dtype=float)
    return common.importance_scores(model, used, X, target)


# ---------------------------------------------------------------------------
# CLI Reports & Target Resolution
# ---------------------------------------------------------------------------

def _prequali_note(row: pd.Series, now: datetime) -> None:
    fp1_utc = session_utc(row, "Practice 1")
    if fp1_utc is not None and now < fp1_utc:
        diff = fp1_utc - now
        hours = int(diff.total_seconds() // 3600)
        vprint(1, f"Note: Practice 1 has not started yet ({hours}h away) - features will be based on past form only.")


def resolve_target(args: argparse.Namespace, year: int, schedule: pd.DataFrame,
                   data: pd.DataFrame) -> Tuple[int, str, str]:
    """Target weekend resolution matching race pipeline."""
    completed = sorted(data["round"].unique())
    if not completed:
        raise SystemExit(f"no completed races found for the {year} season")
    now = utc_now()
    upcoming = []
    for _, row in schedule.iterrows():
        race_utc = session_utc(row, "Race")
        if race_utc is not None and race_utc > now:
            upcoming.append(row)

    if getattr(args, "next", False):
        if not upcoming:
            raise SystemExit("no upcoming race left on this season's calendar")
        row = upcoming[0]
        quali_utc = session_utc(row, "Qualifying")
        if quali_utc is None or quali_utc > now:
            _prequali_note(row, now)
            return int(row["RoundNumber"]), "prequali", str(row["EventName"])
        return int(row["RoundNumber"]), "pre", str(row["EventName"])

    if getattr(args, "predict_round", None) is not None:
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
        if quali_utc is None or quali_utc > now:
            _prequali_note(row, now)
            return int(row["RoundNumber"]), "prequali", str(row["EventName"])
        return int(row["RoundNumber"]), "pre", str(row["EventName"])

    last = completed[-1]
    event = str(schedule[schedule["RoundNumber"] == last]["EventName"].iloc[0])
    return last, "post", event


def backtest_report(records: List[Dict[str, Any]], models: str = "both") -> None:
    """Print the backtest metrics table to stdout."""
    if not records:
        print("No rounds available for backtesting.")
        return

    print("\n" + "=" * 78)
    print("FANTASY BACKTEST SUMMARY (Rolling Leave-One-Round-Out)")
    print("=" * 78)

    model_tot = 0.0
    base_tot = 0.0
    dream_tot = 0.0

    if common.VERBOSITY >= 1:
        header = f"{'Rnd':<4} {'Event':<24} {'Model Pts':>10} {'Base Pts':>10} {'Dream Pts':>10} {'Capture%':>10} {'Cost':>7}"
        print(header)
        print("-" * 78)

    for r in records:
        m_pts = r["model_pts"]
        b_pts = r["base_pts"]
        d_pts = r["dream_pts"]
        cap = f"{r['capture_pct']:.1f}%" if pd.notna(r["capture_pct"]) else "N/A"
        if common.VERBOSITY >= 1:
            print(f"{r['round']:<4} {r['event'][:24]:<24} {m_pts:>10.1f} {b_pts:>10.1f} {d_pts:>10.1f} {cap:>10} ${r['budget_used']:>5.1f}M")

        if pd.notna(m_pts): model_tot += m_pts
        if pd.notna(b_pts): base_tot += b_pts
        if pd.notna(d_pts): dream_tot += d_pts

    if common.VERBOSITY >= 1:
        print("-" * 78)
    tot_cap = (model_tot / dream_tot * 100.0) if dream_tot > 0 else 0.0
    print(f"{'TOTAL':<29} {model_tot:>10.1f} {base_tot:>10.1f} {dream_tot:>10.1f} {tot_cap:>9.1f}%")
    print(f"\nNet advantage over naive baseline: {model_tot - base_tot:+.1f} fantasy points\n")


def final_report(pred: Dict[str, Any], event_name: str, mode: str, target_round: int,
                 budget: float = DEFAULT_BUDGET) -> None:
    """Print the optimal fantasy team recommendation to stdout."""
    opt = pred["team_optimal"]
    print("=" * 78)
    print(f"OPTIMAL F1 FANTASY TEAM: Round {target_round} - {event_name}")
    print(f"Budget Cap: ${budget:.1f}M  |  Total Spent: ${opt['cost']:.1f}M  |  Remaining: ${opt['remaining']:.1f}M")
    print(f"Projected Team Score: {opt['points']:.1f} pts  (DRS Booster: {opt['drs']} 2x)")
    print("=" * 78)

    print("\nSELECTED DRIVERS (5):")
    for d in opt["drivers"]:
        drs_tag = " [DRS BOOST 2x]" if d["name"] == opt["drs"] else ""
        print(f"  • {d['name']:<4} ({d['team']:<18}) : ${d['price']:>4.1f}M  |  {d['points']:>5.1f} pts  |  {d['ppm']:>4.2f} PPM{drs_tag}")

    print("\nSELECTED CONSTRUCTORS (2):")
    for c in opt["constructors"]:
        print(f"  • {c['name']:<24} : ${c['price']:>4.1f}M  |  {c['points']:>5.1f} pts  |  {c['ppm']:>4.2f} PPM")

    if pred.get("actual_dream") is not None:
        act = pred["actual_dream"]
        print("\n" + "-" * 78)
        print(f"ACTUAL RESULTS (POST REVIEW):")
        print(f"  Recommended Team Actual Score: {opt.get('actual_points', 0):.1f} pts")
        print(f"  Dream Team Perfect Ceiling:    {act.get('actual_points', 0):.1f} pts (${act['cost']:.1f}M)")
        if act.get('actual_points', 0) > 0:
            cap = opt.get('actual_points', 0) / act.get('actual_points', 1) * 100.0
            print(f"  Capture Rate of Optimal:       {cap:.1f}%")
        print("-" * 78)

    print("\nTOP ALTERNATIVE LINEUPS:")
    for i, alt in enumerate(pred.get("alternatives", []), start=2):
        d_str = ", ".join(d["name"] for d in alt["drivers"])
        c_str = ", ".join(c["name"] for c in alt["constructors"])
        print(f"  Rank #{i}: {alt['points']:.1f} pts | ${alt['cost']:.1f}M | DRS: {alt['drs']} | Drivers: [{d_str}] | Teams: [{c_str}]")
    print()


# ---------------------------------------------------------------------------
# CLI Entry Point
# ---------------------------------------------------------------------------

def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--season", type=int, default=datetime.now().year, help="championship season year")
    parser.add_argument("--predict-round", type=int, default=None, help="round number to predict or review")
    parser.add_argument("--next", action="store_true", help="predict the next upcoming race on the calendar")
    parser.add_argument("--budget", type=float, default=DEFAULT_BUDGET, help="fantasy budget cap in millions (default 100.0)")
    parser.add_argument("--refresh", action="store_true", help="re-download season data")
    parser.add_argument("--min-train-rounds", type=int, default=MIN_TRAIN_ROUNDS, help="minimum training rounds before backtest")
    parser.add_argument("--model", choices=common.MODEL_PROFILES, default="fast", help="model profile (fast or optimized)")
    parser.add_argument("--models", choices=("both", "delta", "direct"), default="both", help="which model(s) to train")
    common.add_model_cache_args(parser)
    common.add_verbosity_args(parser)

    args = parser.parse_args()
    common.apply_model_cache(args)
    common.apply_verbosity(args)
    setup()

    schedule = fastf1.get_event_schedule(args.season, include_testing=False)
    data = collect_season(args.season, schedule, refresh=args.refresh)

    if data.empty and args.season > 2018:
        prev = args.season - 1
        print(f"No completed rounds in {args.season} yet, falling back to {prev}")
        schedule = fastf1.get_event_schedule(prev, include_testing=False)
        data = collect_season(prev, schedule, refresh=args.refresh)
        args.season = prev

    target_round, mode, event_name = resolve_target(args, args.season, schedule, data)
    print(f"\n{args.season} Round {target_round}: {event_name} ({mode} mode)")

    if mode == "prequali":
        completed = sorted(data["round"].unique())
        fallback = data[data["round"] == completed[-1]]
        upcoming = load_upcoming_round_prequali(args.season, target_round, event_name, fallback)
        data = pd.concat([data[data["round"] != target_round], upcoming], ignore_index=True)
    elif mode == "pre":
        completed = sorted(data["round"].unique())
        fallback = data[data["round"] == completed[-1]]
        upcoming, mode = load_upcoming_round_checked(args.season, target_round, event_name, fallback)
        data = pd.concat([data[data["round"] != target_round], upcoming], ignore_index=True)

    features = build_features(data, year=args.season)

    records = backtest_records(features, min_train_rounds=args.min_train_rounds,
                               profile=args.model, models=args.models, budget=args.budget,
                               year=args.season)
    backtest_report(records, models=args.models)

    pred = final_predictions(features, target_round, profile=args.model,
                             models=args.models, budget=args.budget,
                             year=args.season)
    final_report(pred, event_name, mode, target_round, budget=args.budget)


if __name__ == "__main__":
    main()
