"""Championship prediction page: final drivers' or constructors' championship
points — remaining-points model vs direct model vs the run-rate baseline."""

import sys
from pathlib import Path

_HERE = Path(__file__).resolve().parent
for _p in (_HERE, _HERE.parent / "pipelines"):
    if str(_p) not in sys.path:
        sys.path.insert(0, str(_p))

import _bootstrap
_bootstrap.fresh_modules()

import pandas as pd
import streamlit as st

import predict_championship
import webapp_common as wc

KIND = "championship"
VERSION_KEY = f"{KIND}_data_version"
RESULT_KEY = f"{KIND}_result"
ERROR_KEY = f"{KIND}_run_error"
NOTICE_KEY = f"{KIND}_reload_notice"

CHAMPIONSHIP_OPTIONS = {"Drivers' championship": "drivers",
                        "Constructors' championship": "constructors"}

st.title("Championship prediction")
st.caption("Projects the **final championship points table** at the end of the season, for "
           "drivers or constructors. **Remaining-points model**: how many points each "
           "entrant will still score from here, anchored at the current standings. "
           "**Direct model**: the final total outright. **Baseline**: a naive run-rate "
           "extrapolation (current points + season-average per round × remaining rounds). "
           "A mid-season snapshot cannot know its own season's final points, so models "
           "train across past completed seasons and are backtested leave-one-season-out "
           "over the bundled past seasons (the *First backtest season* setting scopes "
           "how far back) — the same data files the race pipeline collects, no new "
           "downloads.")

if VERSION_KEY not in st.session_state:
    st.session_state[VERSION_KEY] = 0

# --- sidebar ---------------------------------------------------------------
with st.sidebar:
    st.header("Settings")
    year, force_refresh, reload_pressed = wc.season_inputs(KIND)
    championship_label = st.selectbox(
        "Championship", list(CHAMPIONSHIP_OPTIONS), key=f"{KIND}_championship",
        help="Which championship to forecast — only this one is trained, "
             "backtested and projected.")
    championship = CHAMPIONSHIP_OPTIONS[championship_label]
    schedule = wc.get_schedule(year)
    completed = wc.completed_round_numbers(schedule)
    selection = wc.snapshot_round_selectbox(KIND, schedule, set(completed))
    min_train = wc.first_backtest_control(KIND, completed)
    first_season = wc.first_backtest_season_control(
        KIND, predict_championship.complete_season_years(wc.get_schedule))
    profile = wc.model_profile_input(KIND)
    run_pressed = st.button("Run prediction", type="primary", key=f"{KIND}_run",
                            help="Execute the pipeline with the current settings — "
                                 "changing settings never starts it automatically.")

# --- explicit actions (widgets alone never start anything) ------------------
if reload_pressed:
    st.session_state[VERSION_KEY] += 1
    log_box = st.empty()
    with wc.capture_stdout(log_box, refresh_secs=0.3):
        wc.load_season(KIND, year, force_refresh)
    log_box.empty()
    st.session_state[NOTICE_KEY] = True

if run_pressed:
    result = wc.run_championship_pipeline(year, st.session_state[VERSION_KEY],
                                           force_refresh, selection, min_train,
                                           profile, championship, first_season)
    if "error" in result:
        st.session_state[ERROR_KEY] = result["error"]
    else:
        st.session_state[RESULT_KEY] = result
        st.session_state[ERROR_KEY] = None

if st.session_state.pop(NOTICE_KEY, False):
    st.success("Season data reloaded — press **Run prediction** to refresh the results.")

if st.session_state.get(ERROR_KEY):
    st.error(st.session_state[ERROR_KEY])

result = st.session_state.get(RESULT_KEY)
if result is None:
    if not st.session_state.get(ERROR_KEY):
        st.info("Pick a championship and configure the season and snapshot round in the "
                "sidebar, then press **Run prediction**.")
    st.stop()

if result["settings"] != (year, st.session_state[VERSION_KEY], force_refresh,
                          selection, min_train, profile, championship, first_season):
    st.warning("Settings changed since the last run — press **Run prediction** to update the results.")

# --- unpack the run --------------------------------------------------------
used_year = result["year"]
if used_year != result["requested_year"]:
    st.warning(f"No completed rounds in {result['requested_year']} yet — showing the "
               f"{used_year} season instead.")

target = result["target"]          # the snapshot round the forecast is as of
mode = result["mode"]
event_name = result["event"]
rounds_total = result["rounds_total"]
features = result["features"]
records = result["records"]
pred = result["pred"]
imp_main = result["imp_main"]
imp_direct = result["imp_direct"]
championship = result["championship"]
spec = predict_championship.CHAMPIONSHIPS[championship]
entrant = spec["entrant"]
colors = wc.team_colors(used_year)

# (season, entrant) -> team, for coloring backtest picks across seasons
team_lookup = {(int(y), str(e)): str(t) for y, e, t in
               features[["year", entrant, "team"]].drop_duplicates().itertuples(index=False)}

# --- header ----------------------------------------------------------------
headline = f"{used_year} {spec['label'].lower()} — as of round {target} ({event_name})"
if mode == "live":
    st.subheader(f":material/rocket_launch: Forecast: {headline}")
    st.caption(f"Season in progress: {target} of {rounds_total} rounds completed, "
               f"{rounds_total - target} to go. Everything below uses only information "
               f"available as of the end of round {target}.")
else:
    st.subheader(f":material/history: Review: {headline}")
    st.caption("This season is complete — the models were trained only on earlier "
               "seasons, so this is a genuine out-of-sample review. Rows are sorted by "
               "the actual final standings.")
    if target == rounds_total:
        st.caption(":material/info: As of the final round there is nothing left to "
                   "forecast — this is the converged state. Pick an earlier *Forecast "
                   "as of* round in the sidebar for a retro-forecast mid-season.")

if result["resolve_log"].strip():
    st.caption(result["resolve_log"].strip())

if result["train_seasons"] < 2:
    st.caption(f":material/info: This forecast's models were trained on a single earlier "
               f"season — thin data, so treat the projection with caution.")

# --- final standings projection --------------------------------------------
st.subheader("Projected final standings")
if "error" in pred:
    st.warning(f"No model — {pred['error']}")
else:
    main = pred["remaining"]           # carries direct_pos + direct_final columns
    baseline_finals = pred["baseline"].set_index(entrant)["pred_final"]
    entrant_team = (dict(zip(main["driver"], main["team"])) if championship == "drivers"
                    else {t: t for t in main["team"]})

    if mode == "post":
        ordered = main.sort_values(["final", "pred_pos"], ascending=[False, True])
    else:
        ordered = main.sort_values("pred_pos")

    pos_fmt = lambda v: "" if pd.isna(v) else f"P{v:.0f}"
    points_fmt = lambda v: "" if pd.isna(v) else f"{v:.0f}"
    error_fmt = lambda v: "" if pd.isna(v) else f"{int(round(v)):+d}"
    rate_fmt = lambda v: "" if pd.isna(v) else f"{v:.1f}"

    def pace(final):
        """Points per round a projection implies for the rounds still to run
        (projected final − points now, spread over rounds remaining) — blank
        in the converged state, where no rounds are left to forecast."""
        return ((final - ordered["points_so_far"]) / ordered["rounds_remaining"]).where(
            ordered["rounds_remaining"] > 0)

    disp = pd.DataFrame({
        "Remaining model": ordered["pred_pos"].to_numpy(),
        "Direct model": ordered["direct_pos"].to_numpy(),
        "Now": ordered["standings_pos"].to_numpy(),
        "Points now": ordered["points_so_far"].to_numpy(),
        "Final points — remaining": ordered["pred_final"].to_numpy(),
        "Pts/round — remaining": pace(ordered["pred_final"]).to_numpy(),
        "Final points — direct": ordered["direct_final"].to_numpy(),
        "Pts/round — direct": pace(ordered["direct_final"]).to_numpy(),
        "Baseline points": ordered[entrant].map(baseline_finals).to_numpy(),
        "Avg pts/round": ordered["points_per_round"].to_numpy(),
    }).reset_index(drop=True)
    name_col = "Driver" if championship == "drivers" else "Team"
    # constructors: the entrant IS the team, so one team column is both
    disp.insert(3, name_col, ordered[entrant].to_numpy())
    if championship == "drivers":
        disp.insert(4, "Team", ordered["team"].to_numpy())
    fmt = {"Remaining model": pos_fmt, "Direct model": pos_fmt, "Now": pos_fmt,
           "Points now": points_fmt, "Final points — remaining": points_fmt,
           "Pts/round — remaining": rate_fmt, "Final points — direct": points_fmt,
           "Pts/round — direct": rate_fmt, "Baseline points": points_fmt,
           "Avg pts/round": rate_fmt}
    if mode == "post":
        disp["Actual final points"] = ordered["final"].to_numpy()
        disp["Error (remaining)"] = (ordered["pred_final"] - ordered["final"]).to_numpy()
        disp["Error (direct)"] = (ordered["direct_final"] - ordered["final"]).to_numpy()
        fmt["Actual final points"] = points_fmt
        fmt["Error (remaining)"] = error_fmt
        fmt["Error (direct)"] = error_fmt

    styled = disp.style.format(fmt, na_rep="")
    for col in ("Remaining model", "Direct model", "Now"):
        styled = styled.map(wc.position_css, subset=[col])
    if mode == "post":
        styled = styled.map(wc.zero_error_css, subset=["Error (remaining)", "Error (direct)"])
    if championship == "drivers":
        styled = wc.style_driver_team_columns(styled, ordered, colors)
    else:
        styled = styled.map(lambda team: wc.team_css(team, colors), subset=["Team"])
    st.dataframe(styled, width="stretch", hide_index=True)
    st.caption("*Pts/round* — the pace each projection implies for the rounds still to "
               "run: projected final points minus points now, divided by rounds "
               "remaining. *Avg pts/round* — the entrant's season average so far, the "
               "exact rate the run-rate baseline extrapolates.")

    # --- champion callouts ---------------------------------------------------
    top = main.iloc[0]
    second = main.iloc[1] if len(main) > 1 else None
    top_direct = main.sort_values("direct_pos").iloc[0]
    top_base = pred["baseline"].iloc[0]
    margin = (f", projected margin +{top['pred_final'] - second['pred_final']:.0f} "
              f"over {wc.driver_md(second[entrant], entrant_team, colors)}"
              if second is not None else "")
    c1, c2 = st.columns(2)
    c1.success(f"**Predicted champion — remaining model:** "
               f"{wc.driver_md(top[entrant], entrant_team, colors)} "
               f"({top['pred_final']:.0f} pts{margin})")
    c2.success(f"**Predicted champion — direct model:** "
               f"{wc.driver_md(top_direct[entrant], entrant_team, colors)} "
               f"({top_direct['direct_final']:.0f} pts)")
    st.info(f"**Baseline projection (run-rate):** "
            f"{wc.driver_md(top_base[entrant], entrant_team, colors)} "
            f"({top_base['pred_final']:.0f} pts)")
    if mode == "post":
        actual_rows = main.sort_values(["final", spec["tiebreak"]],
                                       ascending=[False, True])
        actual_champion = actual_rows.iloc[0][entrant]
        marks = []
        for label, name in (("remaining", top[entrant]),
                            ("direct", top_direct[entrant]),
                            ("baseline", top_base[entrant])):
            hit = "hit" if name == actual_champion else "miss"
            icon = ":material/check_circle:" if hit == "hit" else ":material/cancel:"
            marks.append(f"{label} model {hit} {icon}" if label != "baseline"
                         else f"baseline {hit} {icon}")
        st.info(f"**Actual champion: {wc.driver_md(actual_champion, entrant_team, colors)}** "
                f"({actual_rows.iloc[0]['final']:g} pts) — {'; '.join(marks)}",
                icon=":material/emoji_events:")

    # --- projected points chart ---------------------------------------------
    st.markdown(f"**Projected final points — remaining model**")
    st.altair_chart(wc.ranked_bars_chart(main, entrant, "pred_final", colors,
                                        "projected final points"),
                    theme="streamlit", width="stretch")
    st.caption("Every entrant's projected final total in their team's color; the "
              "current points are in the table above.")

    # --- projected standings order diagram (drivers only) ---------------------
    if championship == "drivers":
        current = main.assign(now_pos=main["points_so_far"].rank(method="first",
                                                                  ascending=False).astype(int))
        lanes = [("remaining model", main, "pred_pos"),
                 ("direct model", main, "direct_pos"),
                 ("current standings", current, "now_pos")]
        chart = wc.predicted_order_chart(
            lanes, {1.5: "champion", 3.5: "podium"}, colors, "final position")
        st.altair_chart(chart, theme="streamlit", width="stretch")
        st.caption("Driver codes sit at each model's projected final championship "
                   "position (bottom lane = remaining model, middle = direct model, "
                   "top lane = the current standings); dashed lines mark the champion "
                   "and final-podium cuts.")

# --- rolling backtest --------------------------------------------------------
st.divider()
st.subheader("Rolling backtest")
st.caption("Leave-one-season-out: each included season is forecast after each of its "
           "rounds from the first forecast round onwards, every forecast trained only "
           "on earlier complete seasons (a season's final round has nothing left to "
           "forecast and is skipped). The *First backtest season* setting scopes "
           "which seasons are included.")

if not records:
    st.info("Not enough completed seasons for a backtest yet.")
else:
    bt = pd.DataFrame(records)
    n = len(bt)

    if result["first_season"] is not None:
        st.caption(f":material/filter_alt: *First backtest season* setting: the metrics "
                   f"below cover seasons **{result['first_season']}** and later — earlier "
                   f"seasons are excluded from the evaluation.")
    else:
        st.caption(f":material/filter_alt: *First backtest season*: Auto — every complete "
                   f"bundled season with training data, from **{int(bt['season'].min())}** "
                   f"onwards. The earliest bundled season is never forecast: nothing "
                   f"earlier exists to train its forecasts on.")

    r1c1, r1c2, r1c3 = st.columns(3)
    r1c1.metric("Final-points MAE — remaining", f"{bt['remaining_mae'].mean():.1f}",
                delta=f"{bt['remaining_mae'].mean() - bt['baseline_mae'].mean():+.1f} vs baseline",
                delta_color="inverse")
    r1c2.metric("Final-points MAE — direct", f"{bt['direct_mae'].mean():.1f}",
                delta=f"{bt['direct_mae'].mean() - bt['baseline_mae'].mean():+.1f} vs baseline",
                delta_color="inverse")
    r1c3.metric("Final-points MAE — baseline", f"{bt['baseline_mae'].mean():.1f}")

    r2c1, r2c2, r2c3 = st.columns(3)
    r2c1.metric("Champion hit — remaining", f"{100 * bt['remaining_champion'].mean():.0f}%")
    r2c2.metric("Champion hit — direct", f"{100 * bt['direct_champion'].mean():.0f}%")
    r2c3.metric("Champion hit — baseline", f"{100 * bt['baseline_champion'].mean():.0f}%")

    st.caption(f"Top-3 standings overlap (of 3): remaining **{bt['remaining_podium'].mean():.2f}**, "
               f"direct **{bt['direct_podium'].mean():.2f}**, baseline "
               f"**{bt['baseline_podium'].mean():.2f}** — rank correlation: remaining "
               f"**{bt['remaining_corr'].mean():.2f}**, direct **{bt['direct_corr'].mean():.2f}**, "
               f"baseline **{bt['baseline_corr'].mean():.2f}**, over {n} forecasts of "
               f"{bt['season'].nunique()} seasons.")

    thin = sorted(int(s) for s in bt.loc[bt["train_seasons"] < 2, "season"].unique())
    if thin:
        st.caption(f":material/info: {', '.join(str(s) for s in thin)} were forecast by "
                   f"models trained on a single earlier season — thin training data, "
                   f"treat those seasons' metrics with caution.")

    st.markdown("**Champion hit rate by round** "
                "(share of backtested seasons whose champion was the top pick)")
    hit_by_round = bt.groupby("round")[["remaining_champion", "direct_champion",
                                        "baseline_champion"]].mean()
    hit_by_round.columns = ["Remaining model", "Direct model", "Baseline"]
    st.line_chart(hit_by_round, x_label="Round", y_label="Champion hit rate")

    st.markdown("**Final-points MAE by round** (mean across seasons)")
    mae_by_round = bt.groupby("round")[["remaining_mae", "direct_mae",
                                        "baseline_mae"]].mean()
    mae_by_round.columns = ["Remaining model", "Direct model", "Baseline"]
    st.line_chart(mae_by_round, x_label="Round", y_label="Points MAE")

    def first_correct(sub, field):
        hits = sub[sub[field] == 1]
        return int(hits["round"].min()) if len(hits) else pd.NA

    rows = []
    for season, sub in bt.groupby("season"):
        rows.append({
            "Season": int(season),
            "Actual champion": sub["actual_top1"].iloc[0],
            "First correct round — remaining": first_correct(sub, "remaining_champion"),
            "First correct round — direct": first_correct(sub, "direct_champion"),
            "First correct round — baseline": first_correct(sub, "baseline_champion"),
            "Champion hit rate — remaining": sub["remaining_champion"].mean(),
            "Champion hit rate — direct": sub["direct_champion"].mean(),
            "Champion hit rate — baseline": sub["baseline_champion"].mean(),
            "Points MAE — remaining": sub["remaining_mae"].mean(),
            "Points MAE — direct": sub["direct_mae"].mean(),
            "Points MAE — baseline": sub["baseline_mae"].mean(),
            "Forecasts": len(sub),
        })
    seasons_table = pd.DataFrame(rows)

    css_frame = pd.DataFrame("", index=seasons_table.index, columns=seasons_table.columns)
    for i, row in seasons_table.iterrows():
        team = team_lookup.get((int(row["Season"]), str(row["Actual champion"])), "")
        css_frame.at[i, "Actual champion"] = wc.team_css(team, colors, bold=True)
    rate_fmt = lambda v: "" if pd.isna(v) else f"{100 * v:.0f}%"
    styled = (seasons_table.style
              .format({col: "{:.0f}" for col in seasons_table.columns
                       if col.startswith("First correct") or col == "Season"}
                      | {col: rate_fmt for col in seasons_table.columns
                         if col.startswith("Champion hit")}
                      | {col: "{:.1f}" for col in seasons_table.columns
                         if col.startswith("Points MAE")}, na_rep="—")
              .apply(lambda _: css_frame, axis=None))
    st.markdown("**Per-season summary** — when was each champion first predicted?")
    st.dataframe(styled, width="stretch", hide_index=True)

    with st.expander("Per-round forecast table"):
        picks = bt[["season", "round", "event", "remaining_top1", "direct_top1",
                    "baseline_top1", "actual_top1", "remaining_champion",
                    "direct_champion", "baseline_champion"]].copy()
        picks.columns = ["Season", "Round", "Event", "Pick (remaining)",
                         "Pick (direct)", "Pick (baseline)", "Actual champion",
                         "Hit (rem)", "Hit (dir)", "Hit (base)"]
        for col in ("Hit (rem)", "Hit (dir)", "Hit (base)"):
            picks[col] = picks[col].map({1: "hit", 0: "–"})
        pick_cols = ["Pick (remaining)", "Pick (direct)", "Pick (baseline)",
                     "Actual champion"]
        css_frame = pd.DataFrame("", index=picks.index, columns=picks.columns)
        for i, row in picks.iterrows():
            for col in pick_cols[:-1]:
                hit = row["Actual champion"] == row[col] and row[
                    {"Pick (remaining)": "Hit (rem)",
                     "Pick (direct)": "Hit (dir)",
                     "Pick (baseline)": "Hit (base)"}[col]] == "hit"
                team = team_lookup.get((int(row["Season"]), str(row[col])), "")
                css_frame.at[i, col] = wc.team_css(team, colors, bold=hit)
            team = team_lookup.get((int(row["Season"]), str(row["Actual champion"])), "")
            css_frame.at[i, "Actual champion"] = wc.team_css(team, colors, bold=True)
        styled = picks.style.apply(lambda _: css_frame, axis=None)
        st.dataframe(styled, width="stretch", hide_index=True)
        st.caption("Entrant names are colored by team; picks that matched the actual "
                   "champion are in bold.")

# --- feature importance ------------------------------------------------------
st.divider()
st.subheader("Feature importance")
st.caption("Permutation importance on the final training pool — increase in MAE when a "
           "feature is shuffled.")
if not imp_main.empty:
    ic1, ic2 = st.columns(2)
    with ic1:
        st.markdown("**Remaining-points model**")
        st.bar_chart(imp_main.set_index("feature")["importance"], horizontal=True,
                     x_label="MAE increase", y_label="")
    with ic2:
        st.markdown("**Direct model**")
        st.bar_chart(imp_direct.set_index("feature")["importance"], horizontal=True,
                     x_label="MAE increase", y_label="")

# --- diagnostics ---------------------------------------------------------------
st.divider()
st.caption(f"Dataset: {result['train_seasons']} training season(s) + the "
           f"{used_year} target, {len(features)} {entrant}-round snapshots "
           f"({features['round'].nunique()} calendar rounds). "
           f"Features: {', '.join(spec['features'])}")
wc.tuned_params_expander(result, {"remaining": "Remaining-points model",
                                  "direct": "Direct model"})
wc.log_expander("Season data log", result["season_log"])
if result["train_log"].strip():
    wc.log_expander("Training-season log", result["train_log"])
