"""Race prediction page: gain model vs direct model vs grid-order baseline."""

import sys
from pathlib import Path

_HERE = Path(__file__).resolve().parent
for _p in (_HERE, _HERE.parent / "pipelines"):
    if str(_p) not in sys.path:
        sys.path.insert(0, str(_p))

import pandas as pd
import streamlit as st

import predict_race
import webapp_common as wc

KIND = "race"

st.title("Race Prediction")
st.caption("Predicts the race finishing order. **Gain model**: positions gained vs grid. "
           "**Direct model**: absolute finish. **Baseline**: grid order. "
           "Everything below uses only information available before the race starts.")

# --- sidebar controls ------------------------------------------------------
with st.sidebar:
    st.header("Settings")
    year, data_version, force_refresh = wc.season_controls(KIND)

# --- data ------------------------------------------------------------------
with st.spinner("Loading season data (newly completed rounds are downloaded — this can take minutes on first use)..."):
    season = wc.load_season(KIND, year, data_version, force_refresh)

if season["data"] is None:
    st.error("No race data available for this or the previous season.")
    st.stop()

used_year = season["year"]
data = season["data"]
schedule = season["schedule"]
if used_year != year:
    st.warning(f"No completed races in {year} yet — showing the {used_year} season instead.")

completed = sorted(int(r) for r in data["round"].unique())

with st.sidebar:
    selection = wc.target_selectbox(KIND, schedule, set(completed))
    min_train = wc.first_backtest_control(KIND, completed)

try:
    target, mode, event_name, resolve_log = wc.resolve_target(KIND, selection, used_year, schedule, data)
except wc.TargetUnavailable as exc:
    st.error(str(exc))
    wc.log_expander("Season data log", season["log"])
    st.stop()

common_args = (KIND, used_year, data_version, force_refresh, target, mode, event_name)

# --- pipeline --------------------------------------------------------------
try:
    with st.spinner("Preparing features (downloads upcoming-weekend sessions in pre-race mode)..."):
        prep = wc.prepare_features(*common_args)

    with st.spinner("Running rolling backtest (one model pair per round)..."):
        records = wc.backtest_records(*common_args, min_train)

    with st.spinner("Training final models and computing permutation importance..."):
        pred = wc.final_predictions(*common_args)
        imp_gain = wc.importance_frame(*common_args, "gain")
        imp_direct = wc.importance_frame(*common_args, "direct")
except Exception as exc:  # download failures etc.
    st.error(f"Pipeline failed: {type(exc).__name__}: {exc}")
    st.stop()

features = prep["features"]
gain = pred["gain"]
colors = wc.team_colors(used_year)

# --- header ----------------------------------------------------------------
headline = f"{used_year} {event_name} — round {target}"
if mode == "pre":
    st.subheader(f":material/rocket_launch: Prediction: {headline}")
    st.caption("Pre-race mode. Grid estimated from the qualifying classification "
               "(starting-grid penalties are not applied).")
else:
    st.subheader(f":material/history: Review: {headline}")
    st.caption("This round is already completed — models were trained only on earlier rounds, "
               "so this is a genuine out-of-sample review. Rows are sorted by the actual result.")

if resolve_log.strip():
    st.caption(resolve_log.strip())

# --- prediction table ------------------------------------------------------
if mode == "post":
    sort_key = gain["finish"].fillna(99)
    ordered = gain.assign(_key=sort_key).sort_values(["_key", "pred_pos"]).drop(columns="_key")
else:
    ordered = gain.sort_values("pred_pos")

disp = pd.DataFrame({
    "Gain model": ordered["pred_pos"].astype(int),
    "Direct model": ordered["direct_pos"].astype(int),
    "Driver": ordered["driver"],
    "Team": ordered["team"],
    "Grid": ordered["grid"],
}).reset_index(drop=True)

error_fmt = lambda v: "" if pd.isna(v) else f"{int(v):+d}"
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
styled = styled.map(lambda t: wc.team_css(t, colors, bold=True), subset=["Driver"])
styled = styled.map(lambda t: wc.team_css(t, colors), subset=["Team"])
st.dataframe(styled, width="stretch", hide_index=True)

p1, p2 = st.columns(2)
p1.success(f"**Predicted podium (gain):** {' · '.join(gain.head(3)['driver'])}")
p2.success(f"**Predicted podium (direct):** {' · '.join(gain.sort_values('direct_pos').head(3)['driver'])}")
if mode == "post":
    actual_rows = features.loc[(features["round"] == target) & (features["finish"] <= 3)]
    actual_top3 = list(actual_rows.sort_values("finish")["driver"])
    st.info(f"**Actual podium:** {' · '.join(actual_top3)}", icon=":material/emoji_events:")

# --- final predicted order diagram -----------------------------------------
lanes = [("gain model", gain, "pred_pos"), ("direct model", gain, "direct_pos")]
lanes.append(("actual result" if mode == "post" else "starting grid",
              gain, "finish" if mode == "post" else "grid"))
chart = wc.predicted_order_chart(
    lanes, {3.5: "podium cut", 10.5: "points cut"}, colors, "finishing position")
st.altair_chart(chart, theme="streamlit", width="stretch")
st.caption("Driver codes sit at each model's predicted finishing slot "
           "(top lane = gain model, bottom = actual result or starting grid); "
           "dashed lines mark the podium and points cuts.")

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

    c1, c2 = st.columns(2)
    mae_df = bt.set_index("round")[["gain_mae", "direct_mae", "grid_mae"]].rename(columns={
        "gain_mae": "Gain model", "direct_mae": "Direct model", "grid_mae": "Grid baseline"})
    corr_df = bt.set_index("round")[["gain_corr", "direct_corr"]].rename(columns={
        "gain_corr": "Gain model", "direct_corr": "Direct model"})
    with c1:
        st.markdown("**Prediction error by round**")
        st.bar_chart(mae_df, x_label="Round", y_label="MAE (positions)")
    with c2:
        st.markdown("**Rank correlation with the actual result**")
        st.bar_chart(corr_df, x_label="Round", y_label="Spearman rho")

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
               "podium but in the wrong slot, and a **+100 bonus for a perfect podium**. "
               "Maximum 145 points per round.")

    pts = bt[["round", "event", "gain_points", "direct_points"]].copy()
    pts.columns = ["Round", "Event", "Gain model", "Direct model"]
    pts["Round"] = pts["Round"].astype("Int64")
    totals = pd.DataFrame([{
        "Round": pd.NA, "Event": "Season total",
        "Gain model": pts["Gain model"].sum(),
        "Direct model": pts["Direct model"].sum(),
    }])
    pts = pd.concat([pts, totals], ignore_index=True)
    styled = pts.style.format({"Round": "{:.0f}"}, na_rep="").apply(wc.bold_row_style, subset=pd.IndexSlice[len(pts) - 1, :])
    st.dataframe(styled, width="stretch", hide_index=True)

    # --- predicted winner table ---------------------------------------------
    st.subheader("Predicted winner by round")
    drivers = bt[["round", "event", "gain_top1", "direct_top1", "actual_top1"]].copy()
    drivers.columns = ["Round", "Event", "Predicted winner (gain)",
                       "Predicted winner (direct)", "Actual winner"]
    driver_team = features.groupby("driver")["team"].last().to_dict()

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
wc.log_expander("Season data log", season["log"])
if prep["log"].strip():
    wc.log_expander("Weekend download log", prep["log"])
