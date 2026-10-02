"""Qualifying prediction page: anchor model vs direct model vs persistence baseline."""

import sys
from pathlib import Path

_HERE = Path(__file__).resolve().parent
for _p in (_HERE, _HERE.parent / "pipelines"):
    if str(_p) not in sys.path:
        sys.path.insert(0, str(_p))

import pandas as pd
import streamlit as st

import predict_grid
import webapp_common as wc

KIND = "grid"

st.title("Qualifying Prediction")
st.caption("Predicts the qualifying classification. **Anchor model**: change vs each driver's "
           "previous quali result. **Direct model**: absolute position. "
           "**Baseline**: repeating the last qualifying order. "
           "Day-before-quali constraint: only FP1/FP2 and sprint qualifying feed the features — "
           "never FP3 or the sprint race.")

# --- sidebar controls ------------------------------------------------------
with st.sidebar:
    st.header("Settings")
    year, data_version, force_refresh = wc.season_controls(KIND)

# --- data ------------------------------------------------------------------
with st.spinner("Loading season data (newly completed rounds are downloaded — this can take minutes on first use)..."):
    season = wc.load_season(KIND, year, data_version, force_refresh)

if season["data"] is None:
    st.error("No qualifying data available for this or the previous season.")
    st.stop()

used_year = season["year"]
data = season["data"]
schedule = season["schedule"]
if used_year != year:
    st.warning(f"No completed weekends in {year} yet — showing the {used_year} season instead.")

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
    with st.spinner("Preparing features (downloads FP1/FP2 and sprint-quali sessions in pre-quali mode)..."):
        prep = wc.prepare_features(*common_args)

    with st.spinner("Running rolling backtest (one model pair per round)..."):
        records = wc.backtest_records(*common_args, min_train)

    with st.spinner("Training final models and computing permutation importance..."):
        pred = wc.final_predictions(*common_args)
        imp_anchor = wc.importance_frame(*common_args, "anchor")
        imp_direct = wc.importance_frame(*common_args, "direct")
except Exception as exc:  # download failures etc.
    st.error(f"Pipeline failed: {type(exc).__name__}: {exc}")
    st.stop()

features = prep["features"]
anchor = pred["anchor"]
colors = wc.team_colors(used_year)

# --- header ----------------------------------------------------------------
headline = f"{used_year} {event_name} — round {target}"
if mode == "pre":
    st.subheader(f":material/rocket_launch: Prediction: {headline}")
    st.caption("Pre-qualifying mode. Features come from FP1/FP2, sprint qualifying on sprint "
               "weekends, and earlier weekends. Starting-grid penalties are not applied.")
else:
    st.subheader(f":material/history: Review: {headline}")
    st.caption("This qualifying is already completed — models were trained only on earlier rounds, "
               "so this is a genuine out-of-sample review. Rows are sorted by the actual result.")

if resolve_log.strip():
    st.caption(resolve_log.strip())

# --- prediction table ------------------------------------------------------
if mode == "post":
    sort_key = anchor["quali_pos"].fillna(99)
    ordered = anchor.assign(_key=sort_key).sort_values(["_key", "pred_pos"]).drop(columns="_key")
else:
    ordered = anchor.sort_values("pred_pos")

disp = pd.DataFrame({
    "Anchor model": ordered["pred_pos"].astype(int),
    "Direct model": ordered["direct_pos"].astype(int),
    "Driver": ordered["driver"],
    "Team": ordered["team"],
}).reset_index(drop=True)

error_fmt = lambda v: "" if pd.isna(v) else f"{int(v):+d}"
fmt = {
    "Anchor model": "P{:.0f}",
    "Direct model": "P{:.0f}",
}
if mode == "post":
    disp["Actual"] = ordered["quali_pos"].to_numpy()
    disp["Error (anchor)"] = (ordered["pred_pos"] - ordered["quali_pos"]).to_numpy()
    disp["Error (direct)"] = (ordered["direct_pos"] - ordered["quali_pos"]).to_numpy()
    fmt["Actual"] = lambda v: "-" if pd.isna(v) else f"P{v:.0f}"
    fmt["Error (anchor)"] = error_fmt
    fmt["Error (direct)"] = error_fmt

styled = disp.style.format(fmt, na_rep="")
for col in ("Anchor model", "Direct model", "Actual"):
    if col in disp.columns:
        styled = styled.map(wc.position_css, subset=[col])
styled = styled.map(lambda t: wc.team_css(t, colors, bold=True), subset=["Driver"])
styled = styled.map(lambda t: wc.team_css(t, colors), subset=["Team"])
st.dataframe(styled, width="stretch", hide_index=True)

p1, p2 = st.columns(2)
p1.success(f"**Predicted pole (anchor):** {anchor.iloc[0]['driver']}")
p2.success(f"**Predicted pole (direct):** {anchor.sort_values('direct_pos').iloc[0]['driver']}")
if mode == "post":
    pole_rows = features.loc[(features["round"] == target) & (features["quali_pos"] == 1), "driver"]
    if not pole_rows.empty:
        st.info(f"**Actual pole:** {pole_rows.iloc[0]}", icon=":material/emoji_events:")

# --- final predicted grids diagram -----------------------------------------
lanes = [("anchor model", anchor, "pred_pos"), ("direct model", anchor, "direct_pos")]
lanes.append(("actual grid" if mode == "post" else "last quali grid",
              anchor, "quali_pos" if mode == "post" else "quali_pos_last"))
chart = wc.predicted_order_chart(
    lanes, {10.5: "Q3 cut", 15.5: "Q2 cut"}, colors, "grid slot")
st.altair_chart(chart, theme="streamlit", width="stretch")
st.caption("Driver codes sit at each model's predicted grid slot "
           "(top lane = anchor model, bottom = actual grid or last quali grid); "
           "dashed lines mark the Q3 and Q2 cuts.")

# --- backtest --------------------------------------------------------------
st.divider()
st.subheader("Rolling backtest")
st.caption("Each qualifying predicted using only the rounds before it — the fair comparison of all three approaches.")

if not records:
    st.info("Not enough completed rounds for a backtest yet.")
else:
    bt = pd.DataFrame(records)
    n = len(bt)
    m = {
        "anchor_mae": bt["anchor_mae"].mean(),
        "direct_mae": bt["direct_mae"].mean(),
        "persistence_mae": bt["persistence_mae"].mean(),
        "anchor_podium": 100 * bt["anchor_podium"].sum() / (3 * n),
        "direct_podium": 100 * bt["direct_podium"].sum() / (3 * n),
        "anchor_pole": 100 * bt["anchor_pole"].mean(),
        "direct_pole": 100 * bt["direct_pole"].mean(),
        "anchor_corr": bt["anchor_corr"].mean(),
        "direct_corr": bt["direct_corr"].mean(),
    }

    r1c1, r1c2, r1c3 = st.columns(3)
    r1c1.metric("Position MAE — anchor", f"{m['anchor_mae']:.2f}",
                delta=f"{m['anchor_mae'] - m['persistence_mae']:+.2f} vs last-quali baseline",
                delta_color="inverse")
    r1c2.metric("Position MAE — direct", f"{m['direct_mae']:.2f}",
                delta=f"{m['direct_mae'] - m['persistence_mae']:+.2f} vs last-quali baseline",
                delta_color="inverse")
    r1c3.metric("Position MAE — persistence baseline", f"{m['persistence_mae']:.2f}")

    r2c1, r2c2, r2c3, r2c4 = st.columns(4)
    r2c1.metric("Podium hit — anchor", f"{m['anchor_podium']:.0f}%")
    r2c2.metric("Podium hit — direct", f"{m['direct_podium']:.0f}%")
    r2c3.metric("Pole hit — anchor", f"{m['anchor_pole']:.0f}%")
    r2c4.metric("Pole hit — direct", f"{m['direct_pole']:.0f}%")

    st.caption(f"Rank correlation over {n} predicted qualifying sessions — anchor: **{m['anchor_corr']:.2f}**, "
               f"direct: **{m['direct_corr']:.2f}**")

    c1, c2 = st.columns(2)
    mae_df = bt.set_index("round")[["anchor_mae", "direct_mae", "persistence_mae"]].rename(columns={
        "anchor_mae": "Anchor model", "direct_mae": "Direct model",
        "persistence_mae": "Last-quali baseline"})
    corr_df = bt.set_index("round")[["anchor_corr", "direct_corr"]].rename(columns={
        "anchor_corr": "Anchor model", "direct_corr": "Direct model"})
    with c1:
        st.markdown("**Qualifying prediction error by round**")
        st.bar_chart(mae_df, x_label="Round", y_label="MAE (positions)")
    with c2:
        st.markdown("**Rank correlation with the actual qualifying order**")
        st.bar_chart(corr_df, x_label="Round", y_label="Spearman rho")

    with st.expander("Per-round backtest table"):
        table = bt[["round", "event", "anchor_mae", "direct_mae", "persistence_mae",
                    "anchor_podium", "direct_podium", "anchor_pole", "direct_pole"]].copy()
        table.columns = ["Round", "Event", "Anchor MAE", "Direct MAE", "Persistence MAE",
                         "Anchor podium hits", "Direct podium hits", "Anchor pole", "Direct pole"]
        table["Anchor pole"] = table["Anchor pole"].map({1: "pole", 0: "–"})
        table["Direct pole"] = table["Direct pole"].map({1: "pole", 0: "–"})
        st.dataframe(table, width="stretch", hide_index=True)

    # --- pole points table ---------------------------------------------------
    st.subheader("Pole points")
    st.caption("+15 for a correctly predicted pole position, 0 otherwise. "
               "Maximum 15 points per round.")

    pts = bt[["round", "event", "anchor_points", "direct_points"]].copy()
    pts.columns = ["Round", "Event", "Anchor model", "Direct model"]
    pts["Round"] = pts["Round"].astype("Int64")
    totals = pd.DataFrame([{
        "Round": pd.NA, "Event": "Season total",
        "Anchor model": pts["Anchor model"].sum(),
        "Direct model": pts["Direct model"].sum(),
    }])
    pts = pd.concat([pts, totals], ignore_index=True)
    styled = pts.style.format({"Round": "{:.0f}"}, na_rep="").apply(wc.bold_row_style, subset=pd.IndexSlice[len(pts) - 1, :])
    st.dataframe(styled, width="stretch", hide_index=True)

    # --- predicted pole table -------------------------------------------------
    st.subheader("Predicted pole by round")
    poles = bt[["round", "event", "anchor_top1", "direct_top1", "actual_top1"]].copy()
    poles.columns = ["Round", "Event", "Predicted pole (anchor)",
                     "Predicted pole (direct)", "Actual pole"]
    driver_team = features.groupby("driver")["team"].last().to_dict()

    css_frame = pd.DataFrame("", index=poles.index, columns=poles.columns)
    for i, row in poles.iterrows():
        for col in ("Predicted pole (anchor)", "Predicted pole (direct)"):
            hit = row[col] == row["Actual pole"]
            css_frame.at[i, col] = wc.team_css(driver_team.get(row[col], ""), colors, bold=hit)
        css_frame.at[i, "Actual pole"] = wc.team_css(driver_team.get(row["Actual pole"], ""), colors, bold=True)

    styled = poles.style.apply(lambda _: css_frame, axis=None)
    st.dataframe(styled, width="stretch", hide_index=True)
    st.caption("Driver names are colored by team; predictions that matched the actual pole are in bold.")

# --- feature importance ----------------------------------------------------
st.divider()
st.subheader("Feature importance")
st.caption("Permutation importance on the final training set — increase in MAE when a feature is shuffled.")
ic1, ic2 = st.columns(2)
with ic1:
    st.markdown("**Anchor model**")
    st.bar_chart(imp_anchor.set_index("feature")["importance"], horizontal=True,
                 x_label="MAE increase", y_label="")
with ic2:
    st.markdown("**Direct model**")
    st.bar_chart(imp_direct.set_index("feature")["importance"], horizontal=True,
                 x_label="MAE increase", y_label="")

# --- diagnostics ------------------------------------------------------------
st.divider()
st.caption(f"Dataset: {features['round'].nunique()} rounds, {len(features)} driver-quali records. "
           f"Features: {', '.join(predict_grid.FEATURES)}")
wc.log_expander("Season data log", season["log"])
if prep["log"].strip():
    wc.log_expander("Weekend download log", prep["log"])
