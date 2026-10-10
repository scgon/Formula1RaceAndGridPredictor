"""Extras prediction page: one selected milestone model (pole, race winner,
first retirement or fastest lap), presented like the race/quali pages."""

import sys
from pathlib import Path

_HERE = Path(__file__).resolve().parent
for _p in (_HERE, _HERE.parent / "pipelines"):
    if str(_p) not in sys.path:
        sys.path.insert(0, str(_p))

import _bootstrap
_bootstrap.fresh_modules()

import numpy as np
import pandas as pd
import streamlit as st

import predict_extras
import webapp_common as wc

KIND = "extras"
VERSION_KEY = f"{KIND}_data_version"
RESULT_KEY = f"{KIND}_result"
ERROR_KEY = f"{KIND}_run_error"
NOTICE_KEY = f"{KIND}_reload_notice"

MILESTONE_OPTIONS = {spec["label"]: name for name, spec in predict_extras.TARGETS.items()}

st.title("Milestones & Extras: one model, one pick")
st.caption("Six small models, each predicting a single weekend milestone instead of a "
           "full order — pick one in the sidebar and only it runs. **Pole position** uses "
           "day-before-quali information only (FP1/FP2, sprint qualifying, past quali "
           "form), so it can predict before qualifying runs. **Race winner**, "
           "**First retirement** and **Fastest lap** use the pre-race information set "
           "(grid, FP1–FP3, sprint results, form and reliability history). "
           "**Sprint pole** and **Sprint winner** predict the sprint weekend's "
           "qualifying and race from Friday practice and past sprint / quali form — "
           "they exist on sprint weekends only. Each model scores every driver with a "
           "probability; the top pick is the prediction, always compared against a "
           "naive baseline.")

if VERSION_KEY not in st.session_state:
    st.session_state[VERSION_KEY] = 0

# --- sidebar ---------------------------------------------------------------
with st.sidebar:
    st.header("Settings")
    year, force_refresh, reload_pressed = wc.season_inputs(KIND)
    milestone_label = st.selectbox(
        "Milestone", list(MILESTONE_OPTIONS), key=f"{KIND}_milestone",
        help="Which of the six milestone models to run — only this one is "
             "trained and backtested.")
    milestone = MILESTONE_OPTIONS[milestone_label]
    spec = predict_extras.TARGETS[milestone]
    schedule = wc.get_schedule(year)
    completed = wc.completed_round_numbers(schedule)
    selection = wc.target_selectbox(KIND, schedule, set(completed), milestone)
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
    result = wc.run_extras_pipeline(year, st.session_state[VERSION_KEY],
                                    force_refresh, selection, min_train, profile,
                                    milestone)
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
        st.info("Pick a milestone and configure the season and target round in the "
                "sidebar, then press **Run prediction**.")
    st.stop()

if result["settings"] != (year, st.session_state[VERSION_KEY], force_refresh,
                          selection, min_train, profile, milestone):
    st.warning("Settings changed since the last run — press **Run prediction** to update the results.")

# --- unpack the run --------------------------------------------------------
used_year = result["year"]
if used_year != result["requested_year"]:
    st.warning(f"No completed rounds in {result['requested_year']} yet — showing the {used_year} season instead.")

target = result["target"]
mode = result["mode"]
event_name = result["event"]
features = result["features"]
records = result["records"]
pred = result["pred"]
imp = result["imp"]
spec = predict_extras.TARGETS[result["milestone"]]
colors = wc.team_colors(used_year)
driver_team = wc.team_by_driver(features)  # driver -> team, latest row wins

# --- header ----------------------------------------------------------------
headline = f"{used_year} {event_name} — round {target}"
if mode == "prequali":
    st.subheader(f":material/rocket_launch: Prediction: {headline} — {spec['label']}")
    st.caption("Pre-quali mode. Qualifying has not happened yet, so the pole model "
               "predicts from practice, sprint qualifying and season form — its call "
               "is scored once the round completes.")
elif mode == "pre":
    st.subheader(f":material/rocket_launch: Prediction: {headline} — {spec['label']}")
    if spec.get("pre_scored"):
        st.caption(f"Pre-race mode. The {spec['label'].lower()} happens before the race, "
                   "so the model's call is already scored below.")
    else:
        st.caption("Pre-race mode. The grid is known from qualifying; this race "
                   "milestone stays unknown until the race runs.")
else:
    st.subheader(f":material/history: Review: {headline} — {spec['label']}")
    st.caption("This round is already completed — the model was trained only on earlier "
               "rounds, so this is a genuine out-of-sample review.")

if result["resolve_log"].strip():
    st.caption(result["resolve_log"].strip())

if result["train_rounds"] < 2:
    st.caption(":material/info: This round's model was trained on a single round of data "
               "(~20 rows) — far too little to learn anything real, so treat its pick "
               "with caution.")

# --- prediction table ------------------------------------------------------
st.subheader(f"{spec['label']} — driver probabilities")
st.caption("Every eligible driver scored by the selected model; the top probability is "
           "the prediction. *Baseline pick* = the naive rule it is compared against.")

if "error" in pred:
    st.warning(f"No model — {pred['error']}")
else:
    table = pred["table"]
    achieved_css = "background-color: #b7f0c8; color: #0b5c2a; font-weight: 600"

    disp = pd.DataFrame({
        "Rank": table["pred_rank"].astype(int),
        "Driver": table["driver"],
        "Team": table["team"],
        "Probability": table["probability"].astype(float),
    }).reset_index(drop=True)
    fmt = {"Probability": "{:.1%}"}
    if pred["actual"]:
        # positional assignment (like the race/quali pages): `table` still
        # carries the season frame's row labels while `disp` was reset to
        # 0..n-1, and a Series assignment aligns by label — without .to_numpy()
        # every cell lands NaN and the column renders blank.
        achieved = table["driver"].isin(pred["actual"]).map({True: "yes", False: ""})
        disp["Achieved"] = achieved.to_numpy()
        fmt["Achieved"] = lambda v: "" if pd.isna(v) or v == "" else str(v)

    styled = disp.style.format(fmt, na_rep="")
    styled = styled.map(wc.position_css, subset=["Rank"])
    styled = styled.map(wc.probability_css, subset=["Probability"])
    if "Achieved" in disp.columns:
        styled = styled.map(lambda v: achieved_css if v == "yes" else "", subset=["Achieved"])
    styled = wc.style_driver_team_columns(styled, table, colors)
    st.dataframe(styled, width="stretch", hide_index=True)

    p1, p2 = st.columns(2)
    proba = float(table.iloc[0]["probability"])
    p1.success(f"**Predicted {spec['label'].lower()}: "
               f"{wc.driver_md(pred['top1'], driver_team, colors)}** ({proba:.0%})")
    p2.info(f"**Baseline pick** ({spec['baseline_note']}): "
            f"{wc.driver_md(pred['baseline'], driver_team, colors)}")
    if pred["actual"]:
        verdict = "hit" if pred["top1"] in pred["actual"] else "miss"
        icon = ":material/check_circle:" if verdict == "hit" else ":material/cancel:"
        st.info(f"**Actual: {wc.drivers_md(pred['actual'], driver_team, colors, sep=' / ')}** "
                f"— the model's call was a **{verdict}** {icon}")
    elif mode == "post" or result["milestone"] == "pole":
        st.caption("Actual: not known yet.")

    st.markdown("**Top 10 driver probabilities**")
    st.altair_chart(wc.probability_chart(table, colors),
                    theme="streamlit", width="stretch")
    st.caption("One bar per driver, colored by their team.")

# --- rolling backtest ------------------------------------------------------
st.divider()
st.subheader("Rolling backtest")
st.caption("Each round predicted using only the rounds before it — the selected model's "
            "top pick against its naive baseline.")

if not records:
    st.info("Not enough completed rounds for a backtest yet.")
else:
    bt = pd.DataFrame(records)

    def rate(field):
        values = bt[field].dropna()
        return float(values.mean()) if len(values) else np.nan

    model_rate, base_rate, top3_rate = rate("hit"), rate("base"), rate("top3")
    scored = int(bt["hit"].notna().sum())

    metric_cols = st.columns(4, border=True)
    metric_cols[0].metric(
        "Model top-1 hit rate",
        f"{100 * model_rate:.0f}%" if pd.notna(model_rate) else "–",
        (f"{100 * (model_rate - base_rate):+.0f} pts vs baseline"
         if pd.notna(model_rate) and pd.notna(base_rate) else None))
    metric_cols[1].metric("Baseline hit rate",
                          f"{100 * base_rate:.0f}%" if pd.notna(base_rate) else "–")
    metric_cols[2].metric("Model top-3 hit rate",
                          f"{100 * top3_rate:.0f}%" if pd.notna(top3_rate) else "–")
    metric_cols[3].metric("Rounds scored", scored,
                          help="Rounds the milestone exists in (a race with no retirement "
                               "has no first retiree; the sprint milestones only exist on "
                               "sprint weekends).")

    if result["milestone"] == "first_dnf":
        st.caption("First-retirement rates average only over rounds with at least one "
                   "retirement (a race with none has no first retiree); tied first "
                   "retirements (lap-one pile-ups) count as a hit when any of them was "
                   "picked.")

    flagged = sorted(int(r) for r in bt.loc[
        bt.get("train_rounds", pd.Series(index=bt.index, dtype=float)).fillna(99) < 2, "round"])
    if flagged:
        rounds_txt = ", ".join(f"round {r}" for r in flagged)
        were = "was" if len(flagged) == 1 else "were"
        st.caption(f":material/info: {rounds_txt} {were} predicted by a model trained on a "
                   "single round of data (~20 rows) — too little to learn from, so treat "
                   "those rounds' hit flags with caution.")

    st.markdown("**Cumulative hits by round — model vs baseline**")
    cumulative = bt.set_index("round")[["hit", "base"]].fillna(0).cumsum()
    cumulative.columns = ["Model", "Baseline"]
    st.line_chart(cumulative, x_label="Round", y_label="Cumulative hits")

    st.markdown("**Model confidence by round**")
    confidence = bt.set_index("round")[["pick_proba", "actual_proba"]]
    confidence.columns = ["probability of the model's pick",
                          "probability assigned to the actual driver"]
    st.line_chart(confidence, x_label="Round", y_label="Model probability")

    # --- per-round picks table ----------------------------------------------
    st.subheader("Picks by round")
    picks = bt[["round", "event", "top1", "baseline", "actual", "hit"]].copy()
    picks.columns = ["Round", "Event", "Model pick", "Baseline pick", "Actual", "Hit"]
    picks["Round"] = picks["Round"].astype("Int64")
    picks["Actual"] = picks["Actual"].fillna(
        "no retirement" if result["milestone"] == "first_dnf" else "—")
    picks["Hit"] = picks["Hit"].map({1.0: "hit", 0.0: "–"}).fillna("–")

    css_frame = pd.DataFrame("", index=picks.index, columns=picks.columns)
    for i, row in picks.iterrows():
        hit = row["Hit"] == "hit"
        css_frame.at[i, "Model pick"] = wc.team_css(
            driver_team.get(row["Model pick"], ""), colors, bold=hit)
        css_frame.at[i, "Baseline pick"] = wc.team_css(
            driver_team.get(row["Baseline pick"], ""), colors)
        if row["Actual"] not in ("—", "no retirement"):
            css_frame.at[i, "Actual"] = wc.team_css(
                driver_team.get(row["Actual"], ""), colors, bold=True)
    styled = picks.style.apply(lambda _: css_frame, axis=None)
    st.dataframe(styled, width="stretch", hide_index=True)
    st.caption("Driver names are colored by team; model picks that matched the actual "
               "milestone are in bold.")

    # --- milestone points table ----------------------------------------------
    st.subheader("Milestone points")
    st.caption("+15 when the model's top pick achieved the milestone, 0 otherwise. "
               "Maximum 15 points per round.")

    pts = bt[["round", "event", "hit"]].copy()
    pts["points"] = pts["hit"] * 15
    pts = pts[["round", "event", "points"]].rename(
        columns={"round": "Round", "event": "Event", "points": "Model"})
    pts["Round"] = pts["Round"].astype("Int64")
    summary = pd.DataFrame([
        {"Round": pd.NA, "Event": "Season total", "Model": pts["Model"].sum()},
        {"Round": pd.NA, "Event": "Average per round", "Model": pts["Model"].mean()},
    ])
    pts = pd.concat([pts, summary], ignore_index=True)
    points_fmt = lambda v: "" if pd.isna(v) else f"{v:g}"
    styled = (pts.style
              .format({"Round": "{:.0f}", "Model": points_fmt}, na_rep="")
              .apply(wc.bold_row_style, subset=pd.IndexSlice[len(pts) - 2:, :]))
    st.dataframe(styled, width="stretch", hide_index=True)

# --- feature importance -----------------------------------------------------
st.divider()
st.subheader("Feature importance")
st.caption("Permutation importance on the final training set — increase in log loss "
           "when a feature is shuffled.")
if not imp.empty:
    st.bar_chart(imp.set_index("feature")["importance"], horizontal=True,
                 x_label="log-loss increase", y_label="")

# --- diagnostics ------------------------------------------------------------
st.divider()
st.caption(f"Dataset: {features['round'].nunique()} rounds, {len(features)} driver-round "
           f"records. {spec['label']} model features: {', '.join(spec['features'])}")
wc.tuned_params_expander(result, {result["milestone"]: spec["label"]})
wc.log_expander("Season data log", result["season_log"])
if result["prep_log"].strip():
    wc.log_expander("Weekend download log", result["prep_log"])
