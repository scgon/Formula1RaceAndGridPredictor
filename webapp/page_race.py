"""Race prediction page: gain model vs direct model vs grid-order baseline."""

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

import predict_race
import webapp_common as wc

KIND = "race"
VERSION_KEY = f"{KIND}_data_version"
RESULT_KEY = f"{KIND}_result"
ERROR_KEY = f"{KIND}_run_error"
NOTICE_KEY = f"{KIND}_reload_notice"

st.title("Race Prediction")
st.caption("Predicts the race finishing order. **Gain model**: positions gained vs grid. "
           "**Direct model**: absolute finish. **Baseline**: grid order. "
           "Everything below uses only information available before the race starts. "
           "Before qualifying runs there is no grid to anchor the gain model to, so the "
           "direct model predicts alone in that window.")

if VERSION_KEY not in st.session_state:
    st.session_state[VERSION_KEY] = 0

# --- sidebar ---------------------------------------------------------------
with st.sidebar:
    st.header("Settings")
    year, force_refresh, reload_pressed = wc.season_inputs(KIND)
    schedule = wc.get_schedule(year)
    completed = wc.completed_round_numbers(schedule)
    selection = wc.target_selectbox(KIND, schedule, set(completed))
    min_train = wc.first_backtest_control(KIND, completed)
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
    result = wc.run_pipeline(KIND, year, st.session_state[VERSION_KEY],
                             force_refresh, selection, min_train, profile)
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
        st.info("Configure the season and target round in the sidebar, then press "
                "**Run prediction**.")
    st.stop()

if result["settings"] != (year, st.session_state[VERSION_KEY], force_refresh,
                         selection, min_train, profile):
    st.warning("Settings changed since the last run — press **Run prediction** to update the results.")

# --- unpack the run --------------------------------------------------------
used_year = result["year"]
if used_year != result["requested_year"]:
    st.warning(f"No completed races in {result['requested_year']} yet — showing the {used_year} season instead.")

target = result["target"]
mode = result["mode"]
event_name = result["event"]
features = result["features"]
records = result["records"]
gain = result["main"]
imp_gain = result["imp_main"]
imp_direct = result["imp_direct"]
colors = wc.team_colors(used_year)
driver_team = wc.team_by_driver(features)  # driver -> team, latest row wins

# --- header ----------------------------------------------------------------
headline = f"{used_year} {event_name} — round {target}"
if mode in ("pre", "prequali"):
    st.subheader(f":material/rocket_launch: Prediction: {headline}")
    if mode == "prequali":
        st.caption("Pre-qualifying mode. Qualifying has not happened, so there is no grid "
                   "yet: the gain model has nothing to anchor to and is skipped — the "
                   "direct model predicts the finish order from practice, sprint and "
                   "season-form features. The backtest below still scores both models on "
                   "completed rounds.")
    else:
        st.caption("Pre-race mode. Grid estimated from the qualifying classification "
                   "(starting-grid penalties are not applied).")
else:
    st.subheader(f":material/history: Review: {headline}")
    st.caption("This round is already completed — models were trained only on earlier rounds, "
               "so this is a genuine out-of-sample review. Rows are sorted by the actual result.")

if result["resolve_log"].strip():
    st.caption(result["resolve_log"].strip())

if result["train_rounds"] < 2:
    wc.degenerate_training_note("the starting-grid order")

# --- prediction table ------------------------------------------------------
if mode == "post":
    sort_key = gain["finish"].fillna(99)
    ordered = gain.assign(_key=sort_key).sort_values(["_key", "pred_pos"]).drop(columns="_key")
else:
    ordered = gain.sort_values("pred_pos")

error_fmt = lambda v: "" if pd.isna(v) else f"{int(v):+d}"
if mode == "prequali":
    disp = pd.DataFrame({
        "Direct model": ordered["pred_pos"].astype(int),
        "Driver": ordered["driver"],
        "Team": ordered["team"],
    }).reset_index(drop=True)
    fmt = {"Direct model": "P{:.0f}"}
else:
    disp = pd.DataFrame({
        "Gain model": ordered["pred_pos"].astype(int),
        "Direct model": ordered["direct_pos"].astype(int),
        "Driver": ordered["driver"],
        "Team": ordered["team"],
        "Grid": ordered["grid"],
    }).reset_index(drop=True)
    fmt = {
        "Gain model": "P{:.0f}",
        "Direct model": "P{:.0f}",
        "Grid": "P{:.0f}",
    }
if mode == "post":
    disp["Actual"] = ordered["finish"].to_numpy()
    disp["Error (gain)"] = (ordered["pred_pos"] - ordered["finish"]).to_numpy()
    disp["Error (direct)"] = (ordered["direct_pos"] - ordered["finish"]).to_numpy()
    fmt["Actual"] = lambda v: "DNF" if pd.isna(v) else f"P{v:.0f}"
    fmt["Error (gain)"] = error_fmt
    fmt["Error (direct)"] = error_fmt

styled = disp.style.format(fmt, na_rep="")
for col in ("Gain model", "Direct model", "Grid", "Actual"):
    if col in disp.columns:
        styled = styled.map(wc.position_css, subset=[col])
if mode == "post":
    styled = styled.map(wc.zero_error_css, subset=["Error (gain)", "Error (direct)"])
styled = wc.style_driver_team_columns(styled, ordered, colors)
st.dataframe(styled, width="stretch", hide_index=True)

if mode == "prequali":
    st.success(f"**Predicted podium (direct model):** "
               f"{wc.drivers_md(gain.head(3)['driver'], driver_team, colors)}")
else:
    p1, p2 = st.columns(2)
    p1.success(f"**Predicted podium (gain):** "
               f"{wc.drivers_md(gain.head(3)['driver'], driver_team, colors)}")
    p2.success(f"**Predicted podium (direct):** "
               f"{wc.drivers_md(gain.sort_values('direct_pos').head(3)['driver'], driver_team, colors)}")
if mode == "post":
    actual_rows = features.loc[(features["round"] == target) & (features["finish"] <= 3)]
    actual_top3 = list(actual_rows.sort_values("finish")["driver"])
    st.info(f"**Actual podium:** {wc.drivers_md(actual_top3, driver_team, colors)}",
            icon=":material/emoji_events:")

# --- final predicted order diagram -----------------------------------------
if mode == "prequali":
    lanes = [("direct model", gain, "pred_pos")]
    lane_caption = ("Driver codes sit at the direct model's predicted finishing slot; "
                    "dashed lines mark the podium and points cuts. There is no starting "
                    "grid yet — qualifying has not happened.")
else:
    lanes = [("gain model", gain, "pred_pos"), ("direct model", gain, "direct_pos")]
    lanes.append(("actual result" if mode == "post" else "starting grid",
                  gain, "finish" if mode == "post" else "grid"))
    lane_caption = ("Driver codes sit at each model's predicted finishing slot "
                    "(bottom lane = gain model, middle = direct model, "
                    "top lane = actual result or starting grid); "
                    "dashed lines mark the podium and points cuts.")
chart = wc.predicted_order_chart(
    lanes, {3.5: "podium cut", 10.5: "points cut"}, colors, "finishing position")
st.altair_chart(chart, theme="streamlit", width="stretch")
st.caption(lane_caption)

# --- backtest --------------------------------------------------------------
st.divider()
st.subheader("Rolling backtest")
st.caption("Each round predicted using only the rounds before it — the fair comparison of all three approaches.")

if not records:
    st.info("Not enough completed rounds for a backtest yet.")
else:
    bt = pd.DataFrame(records)
    n = len(bt)
    m = {
        "gain_mae": bt["gain_mae"].mean(),
        "direct_mae": bt["direct_mae"].mean(),
        "grid_mae": bt["grid_mae"].mean(),
        "gain_podium": 100 * bt["gain_podium"].sum() / (3 * n),
        "direct_podium": 100 * bt["direct_podium"].sum() / (3 * n),
        "gain_winner": 100 * bt["gain_winner"].mean(),
        "direct_winner": 100 * bt["direct_winner"].mean(),
        "gain_corr": bt["gain_corr"].mean(),
        "direct_corr": bt["direct_corr"].mean(),
    }

    r1c1, r1c2, r1c3 = st.columns(3)
    r1c1.metric("Position MAE — gain", f"{m['gain_mae']:.2f}",
                delta=f"{m['gain_mae'] - m['grid_mae']:+.2f} vs grid baseline", delta_color="inverse")
    r1c2.metric("Position MAE — direct", f"{m['direct_mae']:.2f}",
                delta=f"{m['direct_mae'] - m['grid_mae']:+.2f} vs grid baseline", delta_color="inverse")
    r1c3.metric("Position MAE — grid baseline", f"{m['grid_mae']:.2f}")

    r2c1, r2c2, r2c3, r2c4 = st.columns(4)
    r2c1.metric("Podium hit — gain", f"{m['gain_podium']:.0f}%")
    r2c2.metric("Podium hit — direct", f"{m['direct_podium']:.0f}%")
    r2c3.metric("Winner hit — gain", f"{m['gain_winner']:.0f}%")
    r2c4.metric("Winner hit — direct", f"{m['direct_winner']:.0f}%")

    st.caption(f"Rank correlation over {n} predicted rounds — gain: **{m['gain_corr']:.2f}**, "
               f"direct: **{m['direct_corr']:.2f}**")

    wc.degenerate_backtest_note(bt, ["gain_train_rounds", "direct_train_rounds"],
                                "the starting-grid order")

    st.markdown("**Prediction error by round**")
    mae_df = bt.set_index("round")[["gain_mae", "direct_mae", "grid_mae"]].rename(columns={
        "gain_mae": "Gain model", "direct_mae": "Direct model", "grid_mae": "Grid baseline"})
    st.bar_chart(mae_df, x_label="Round", y_label="MAE (positions)", stack=False)

    with st.expander("Per-round backtest table"):
        table = bt[["round", "event", "gain_mae", "direct_mae", "grid_mae",
                    "gain_podium", "direct_podium", "gain_winner", "direct_winner"]].copy()
        table.columns = ["Round", "Event", "Gain MAE", "Direct MAE", "Grid MAE",
                         "Gain podium hits", "Direct podium hits", "Gain winner", "Direct winner"]
        table["Gain winner"] = table["Gain winner"].map({1: "win", 0: "–"})
        table["Direct winner"] = table["Direct winner"].map({1: "win", 0: "–"})
        st.dataframe(table, width="stretch", hide_index=True)

    # --- podium points table ----------------------------------------------
    st.subheader("Podium points")
    st.caption("+15 for a correct winner/P2/P3 prediction, +5 when a driver is picked on the "
               "podium but in the wrong slot. Maximum 45 points per round.")

    pts = bt[["round", "event", "gain_points", "direct_points"]].copy()
    pts.columns = ["Round", "Event", "Gain model", "Direct model"]
    pts["Round"] = pts["Round"].astype("Int64")
    summary = pd.DataFrame([
        {"Round": pd.NA, "Event": "Season total",
         "Gain model": bt["gain_points"].sum(), "Direct model": bt["direct_points"].sum()},
        {"Round": pd.NA, "Event": "Average per round",
         "Gain model": bt["gain_points"].mean(), "Direct model": bt["direct_points"].mean()},
    ])
    pts = pd.concat([pts, summary], ignore_index=True)
    points_fmt = lambda v: "" if pd.isna(v) else f"{v:g}"
    styled = (pts.style
              .format({"Round": "{:.0f}", "Gain model": points_fmt, "Direct model": points_fmt}, na_rep="")
              .apply(wc.bold_row_style, subset=pd.IndexSlice[len(pts) - 2:, :]))
    st.dataframe(styled, width="stretch", hide_index=True)

    # --- predicted winner table ---------------------------------------------
    st.subheader("Predicted winner by round")
    drivers = bt[["round", "event", "gain_top1", "direct_top1", "actual_top1"]].copy()
    drivers.columns = ["Round", "Event", "Predicted winner (gain)",
                       "Predicted winner (direct)", "Actual winner"]

    css_frame = pd.DataFrame("", index=drivers.index, columns=drivers.columns)
    for i, row in drivers.iterrows():
        for col in ("Predicted winner (gain)", "Predicted winner (direct)"):
            hit = row[col] == row["Actual winner"]
            css_frame.at[i, col] = wc.team_css(driver_team.get(row[col], ""), colors, bold=hit)
        css_frame.at[i, "Actual winner"] = wc.team_css(driver_team.get(row["Actual winner"], ""), colors, bold=True)

    styled = drivers.style.apply(lambda _: css_frame, axis=None)
    st.dataframe(styled, width="stretch", hide_index=True)
    st.caption("Driver names are colored by team; predictions that matched the actual winner are in bold.")

# --- feature importance ----------------------------------------------------
st.divider()
st.subheader("Feature importance")
st.caption("Permutation importance on the final training set — increase in MAE when a feature is shuffled.")
if mode == "prequali":
    st.markdown("**Direct model**")
    st.bar_chart(imp_direct.set_index("feature")["importance"], horizontal=True,
                 x_label="MAE increase", y_label="")
else:
    ic1, ic2 = st.columns(2)
    with ic1:
        st.markdown("**Gain model**")
        st.bar_chart(imp_gain.set_index("feature")["importance"], horizontal=True,
                     x_label="MAE increase", y_label="")
    with ic2:
        st.markdown("**Direct model**")
        st.bar_chart(imp_direct.set_index("feature")["importance"], horizontal=True,
                     x_label="MAE increase", y_label="")

# --- diagnostics ------------------------------------------------------------
st.divider()
st.caption(f"Dataset: {features['round'].nunique()} rounds, {len(features)} driver-race records. "
           f"Features: {', '.join(predict_race.FEATURES)}")
wc.tuned_params_expander(result, {"direct": "Direct"} if mode == "prequali"
                         else {"gain": "Gain", "direct": "Direct"})
wc.log_expander("Season data log", result["season_log"])
if result["prep_log"].strip():
    wc.log_expander("Weekend download log", result["prep_log"])
