"""Formula 1 Fantasy prediction & lineup optimization page: predicts expected fantasy
points for all drivers and constructors, and selects the optimal team (5 drivers +
2 constructors + 1 DRS booster) while staying strictly under the budget cap.
"""

import sys
from pathlib import Path

_HERE = Path(__file__).resolve().parent
for _p in (_HERE, _HERE.parent / "pipelines"):
    if str(_p) not in sys.path:
        sys.path.insert(0, str(_p))

import _bootstrap
_bootstrap.fresh_modules()

import altair as alt
import numpy as np
import pandas as pd
import streamlit as st

import predict_fantasy
import webapp_common as wc

KIND = "fantasy"
VERSION_KEY = f"{KIND}_data_version"
RESULT_KEY = f"{KIND}_result"
ERROR_KEY = f"{KIND}_run_error"
NOTICE_KEY = f"{KIND}_reload_notice"

st.title("F1 Fantasy: Team Optimizer & Budget Cap")
st.caption("Predicts expected Formula 1 Fantasy points and uses integer knapsack optimization "
           "to select the **optimal team (5 drivers + 2 constructors + 1 DRS 2x booster)** "
           "under your chosen budget cap (official standard is **$100.0M**). "
           "**Direct model**: predicts absolute fantasy points (drives the recommendation). "
           "**Delta model**: predicts points above or below season form (reported in parallel). "
           "Both train on every bundled season before the target, so early-season rounds "
           "inherit the level knowledge of completed seasons. "
           "**Pre-quali mode**: predictions are made from practice pace (FP1/FP2) and form "
           "before qualifying runs — ideal for locking lineups before the deadline.")

if VERSION_KEY not in st.session_state:
    st.session_state[VERSION_KEY] = 0

# --- sidebar ---------------------------------------------------------------
with st.sidebar:
    st.header("Settings")
    year, force_refresh, reload_pressed = wc.season_inputs(KIND)
    budget = st.slider("Budget cap ($M)", min_value=80.0, max_value=130.0,
                       value=100.0, step=0.5,
                       help="Team salary cap in millions of dollars (official F1 Fantasy default is $100.0M).")
    models = st.selectbox(
        "Model formulation", ["both", "delta", "direct"],
        format_func=lambda x: "Delta model (vs season form)" if x == "delta"
        else ("Direct points model" if x == "direct" else "Both (Direct + Delta)"),
        help="Direct model (absolute points) drives the recommendation and backtest "
             "under Both/Direct; the Delta model anchors to the driver's season "
             "scoring average and is reported in parallel under Both.")
    schedule = wc.get_schedule(year)
    completed = wc.completed_round_numbers(schedule)
    selection = wc.target_selectbox(KIND, schedule, set(completed))
    min_train = wc.first_backtest_control(KIND, completed)
    profile = wc.model_profile_input(KIND)
    run_pressed = st.button("Run prediction", type="primary", key=f"{KIND}_run",
                            help="Execute the pipeline with current settings — "
                                 "changing settings never runs automatically.")

# --- actions ---------------------------------------------------------------
if reload_pressed:
    st.session_state[VERSION_KEY] += 1
    log_box = st.empty()
    with wc.capture_stdout(log_box, refresh_secs=0.3):
        wc.load_season(KIND, year, force_refresh)
    log_box.empty()
    st.session_state[NOTICE_KEY] = True

if run_pressed:
    result = wc.run_fantasy_pipeline(
        year, st.session_state[VERSION_KEY], force_refresh, selection, min_train,
        profile=profile, budget=budget, models=models)
    if "error" in result:
        st.session_state[ERROR_KEY] = result["error"]
    else:
        st.session_state[RESULT_KEY] = result
        st.session_state[ERROR_KEY] = None

if st.session_state.pop(NOTICE_KEY, False):
    st.success("Season data reloaded — press **Run prediction** to refresh results.")

if st.session_state.get(ERROR_KEY):
    st.error(st.session_state[ERROR_KEY])

result = st.session_state.get(RESULT_KEY)
if result is None:
    if not st.session_state.get(ERROR_KEY):
        st.info("Configure the season, budget cap, and target round in the sidebar, "
                "then press **Run prediction**.")
    st.stop()

if result["settings"] != (year, st.session_state[VERSION_KEY], force_refresh,
                          selection, min_train, profile, budget, models):
    st.warning("Settings changed since the last run — press **Run prediction** to update the results.")

# --- unpack the run --------------------------------------------------------
used_year = result["year"]
if used_year != result["requested_year"]:
    st.warning(f"No completed races in {result['requested_year']} yet — showing {used_year} instead.")

target = result["target"]
mode = result["mode"]
event_name = result["event"]
features = result["features"]
optimal = result["optimal"]
alternatives = result["alternatives"]
archetypes = result["archetypes"]
actual_dream = result.get("actual_dream")
d_board = result["d_board"]
c_board = result["c_board"]
records = result["records"]
imp_delta = result.get("imp_delta")
imp_direct = result.get("imp_direct")
colors = wc.team_colors(used_year)

# Header
st.header(f"{used_year} Round {target}: {event_name}")
mode_labels = {
    "post": "Review mode (round completed — showing actual fantasy scores)",
    "pre": "Prediction mode (qualifying completed — starting grid known)",
    "prequali": "Pre-qualifying mode (practice & form only — optimal for locking before quali deadline)",
}
st.caption(mode_labels.get(mode, mode))

if mode == "prequali":
    st.info("Qualifying has not taken place yet. Predictions are powered by FP1/FP2 pace, "
            "sprint sessions, and rolling season form so you can finalize and lock your fantasy "
            "lineup before the weekend deadline.")

# Top KPIs
k1, k2, k3, k4 = st.columns(4)
with k1:
    st.metric("Projected team score", f"{optimal['points']:.1f} pts")
with k2:
    st.metric("Budget cap used", f"${optimal['cost']:.1f}M / ${budget:.1f}M",
              delta=f"${optimal['remaining']:.1f}M remaining", delta_color="normal")
with k3:
    st.metric("DRS booster (2x)", f"{optimal['drs']}")
with k4:
    if mode == "post" and pd.notna(optimal.get("actual_points")):
        act_pts = optimal["actual_points"]
        ceiling = actual_dream["actual_points"] if actual_dream else np.nan
        cap_str = f"{(act_pts / ceiling * 100):.1f}% ceiling" if pd.notna(ceiling) and ceiling > 0 else ""
        st.metric("Actual team score", f"{act_pts:.1f} pts", delta=cap_str)
    else:
        st.metric("Assets selected", "5 Drivers + 2 Teams")

st.divider()

# --- Selected Optimal Lineup -----------------------------------------------
st.subheader("Recommended Optimal Lineup")
st.caption("Globally optimal selection maximizing projected points under the budget cap.")

col_d, col_c = st.columns([3, 2])

with col_d:
    st.markdown("##### 🏎️ Selected Drivers (5)")
    d_rows = []
    for d in optimal["drivers"]:
        is_drs = (d["name"] == optimal["drs"])
        drs_badge = "⚡ 2x DRS Booster" if is_drs else "Regular"
        pts_val = f"{d['points'] * 2:.1f}" if is_drs else f"{d['points']:.1f}"
        d_rows.append({
            "Driver": d["name"],
            "Team": d["team"],
            "Price": f"${d['price']:.1f}M",
            "Role": drs_badge,
            "Projected Pts": pts_val,
            "Value (PPM)": f"{d['ppm']:.2f}",
        })
    df_disp_d = pd.DataFrame(d_rows)
    st.dataframe(df_disp_d, width="stretch", hide_index=True)

with col_c:
    st.markdown("##### 🏁 Selected Constructors (2)")
    c_rows = []
    for c in optimal["constructors"]:
        c_rows.append({
            "Constructor": c["name"],
            "Price": f"${c['price']:.1f}M",
            "Projected Pts": f"{c['points']:.1f}",
            "Value (PPM)": f"{c['ppm']:.2f}",
        })
    df_disp_c = pd.DataFrame(c_rows)
    st.dataframe(df_disp_c, width="stretch", hide_index=True)

if mode == "post" and actual_dream:
    st.markdown("---")
    act_pts = optimal.get("actual_points", 0.0)
    dream_pts = actual_dream.get("actual_points", 0.0)
    capture = (act_pts / dream_pts * 100.0) if dream_pts > 0 else 0.0
    st.success(
        f"**Post-Race Verdict:** The recommended lineup scored **{act_pts:.1f} actual fantasy points**, "
        f"capturing **{capture:.1f}%** of the theoretical Dream Team ceiling ({dream_pts:.1f} pts) "
        f"under the ${budget:.1f}M cap!", icon=":material/celebration:")

st.divider()

# --- Tabs for Deep Dive ----------------------------------------------------
tab_alts, tab_board, tab_backtest, tab_imp = st.tabs([
    "Alternative Lineups & Archetypes",
    "Asset Board (PPM & Pricing)",
    "Season Backtest & Performance",
    "Feature Importance"
])

# TAB 1: Alternatives & Archetypes
with tab_alts:
    st.subheader("Strategic Archetypes")
    st.caption("Different strategic roster allocations under the same budget cap.")

    arch_cols = st.columns(3)
    arch_data = [
        ("Mega-Constructors", archetypes.get("mega_constructors"),
         "Invests heavily in two top-tier constructors with value drivers."),
        ("Mega-Drivers", archetypes.get("mega_drivers"),
         "Invests in two premium drivers + DRS boost with budget constructors."),
        ("Max Efficiency (PPM)", archetypes.get("max_ppm"),
         "Maximizes total projected points per million dollars spent."),
    ]

    for col, (title, arch, desc) in zip(arch_cols, arch_data):
        with col:
            st.markdown(f"**{title}**")
            st.caption(desc)
            if arch:
                st.markdown(f"**Score:** `{arch['points']:.1f} pts` | **Cost:** `${arch['cost']:.1f}M`")
                st.markdown(f"**DRS Boost:** `{arch['drs']}`")
                d_str = ", ".join(d["name"] for d in arch["drivers"])
                c_str = ", ".join(c["name"] for c in arch["constructors"])
                st.markdown(f"**Drivers:** {d_str}")
                st.markdown(f"**Constructors:** {c_str}")

    st.markdown("---")
    st.subheader("Top Alternative Lineups")
    st.caption("Next best distinct lineups under the cap — helpful if you have transfers, price ties, or driver preferences.")

    alt_rows = []
    for i, a in enumerate(alternatives, start=2):
        d_str = ", ".join(d["name"] for d in a["drivers"])
        c_str = ", ".join(c["name"] for c in a["constructors"])
        alt_rows.append({
            "Rank": f"#{i}",
            "Projected Score": f"{a['points']:.1f} pts",
            "Cost": f"${a['cost']:.1f}M",
            "Remaining": f"${a['remaining']:.1f}M",
            "DRS Boost": a["drs"],
            "Drivers": d_str,
            "Constructors": c_str,
        })
    if alt_rows:
        st.dataframe(pd.DataFrame(alt_rows), width="stretch", hide_index=True)
    else:
        st.info("No alternative lineups found within the tight budget constraint.")

# TAB 2: Asset Board
with tab_board:
    st.subheader("Driver Board")
    st.caption("All drivers sorted by projected fantasy points and Points Per Million (PPM) value.")

    d_disp = d_board[[
        "driver", "team", "price", "pred_points", "ppm", "selected", "is_drs"
    ]].copy()
    d_disp.columns = ["Driver", "Team", "Price ($M)", "Projected Pts", "Value (PPM)", "Selected", "DRS Pick"]
    d_disp["Price ($M)"] = d_disp["Price ($M)"].map(lambda v: f"${v:.1f}M")
    d_disp["Selected"] = d_disp["Selected"].map(lambda v: "✅ Picked" if v else "")
    d_disp["DRS Pick"] = d_disp["DRS Pick"].map(lambda v: "⚡ 2x Boost" if v else "")

    st.dataframe(d_disp, width="stretch", hide_index=True)

    st.markdown("---")
    st.subheader("Constructor Board")
    st.caption("All constructors sorted by projected fantasy points and Points Per Million (PPM) value.")

    c_disp = c_board[[
        "team", "price", "pred_points", "ppm", "selected"
    ]].copy()
    c_disp.columns = ["Constructor", "Price ($M)", "Projected Pts", "Value (PPM)", "Selected"]
    c_disp["Price ($M)"] = c_disp["Price ($M)"].map(lambda v: f"${v:.1f}M")
    c_disp["Selected"] = c_disp["Selected"].map(lambda v: "✅ Picked" if v else "")

    st.dataframe(c_disp, width="stretch", hide_index=True)

# TAB 3: Season Backtest
with tab_backtest:
    st.subheader("Rolling Backtest Performance")
    st.caption("Leave-one-round-out backtest: each round is forecast using only previous rounds' data, "
               "comparing the model's recommended team vs the naive baseline and the dream team ceiling.")

    if records:
        b_df = pd.DataFrame(records)

        # Cumulative points chart
        cum_df = pd.DataFrame({
            "Round": b_df["round"],
            "Event": b_df["event"],
            "Model Recommended": b_df["model_pts"].cumsum(),
            "Naive Baseline": b_df["base_pts"].cumsum(),
            "Dream Team Ceiling": b_df["dream_pts"].cumsum(),
        })

        st.markdown("##### Cumulative Fantasy Points Over Season")
        chart_data = cum_df.melt(id_vars=["Round", "Event"],
                                 value_vars=["Model Recommended", "Naive Baseline", "Dream Team Ceiling"],
                                 var_name="Strategy", value_name="Cumulative Points")

        chart = alt.Chart(chart_data).mark_line(point=True).encode(
            x=alt.X("Round:O", title="Championship Round"),
            y=alt.Y("Cumulative Points:Q", title="Total Cumulative Fantasy Points"),
            color=alt.Color("Strategy:N", scale=alt.Scale(
                domain=["Model Recommended", "Naive Baseline", "Dream Team Ceiling"],
                range=["#00D2BE", "#999999", "#E10600"]
            )),
            tooltip=["Round", "Event", "Strategy", "Cumulative Points"]
        ).properties(height=350)
        st.altair_chart(chart, width="stretch")

        # Metrics overview
        m_tot = b_df["model_pts"].sum()
        b_tot = b_df["base_pts"].sum()
        d_tot = b_df["dream_pts"].sum()
        tot_cap = (m_tot / d_tot * 100.0) if d_tot > 0 else 0.0

        bm1, bm2, bm3 = st.columns(3)
        with bm1:
            st.metric("Total Model Fantasy Points", f"{m_tot:.0f} pts")
        with bm2:
            st.metric("Advantage Over Baseline", f"{m_tot - b_tot:+.0f} pts",
                      delta=f"{m_tot - b_tot:+.0f} pts", delta_color="normal")
        with bm3:
            st.metric("Points Captured of Dream Ceiling", f"{tot_cap:.1f}%")

        # Table
        disp_rec = pd.DataFrame({
            "Round": b_df["round"],
            "Event": b_df["event"],
            "Model Lineup Score": b_df["model_pts"].map(lambda v: f"{v:.0f}"),
            "Baseline Score": b_df["base_pts"].map(lambda v: f"{v:.0f}"),
            "Dream Team Ceiling": b_df["dream_pts"].map(lambda v: f"{v:.0f}"),
            "Capture %": b_df["capture_pct"].map(lambda v: f"{v:.1f}%" if pd.notna(v) else "-"),
            "Budget Used": b_df["budget_used"].map(lambda v: f"${v:.1f}M"),
            "DRS Booster": b_df["drs"],
        })
        st.dataframe(disp_rec, width="stretch", hide_index=True)
    else:
        st.info("Not enough completed rounds to run a rolling backtest.")

# TAB 4: Feature Importance
with tab_imp:
    st.subheader("Feature Importance")
    st.caption("Permutation importance: how much prediction error (MAE) increases when each feature is shuffled.")

    ic1, ic2 = st.columns(2)
    with ic1:
        st.markdown("**Delta Model (Form Deviation)**")
        if imp_delta is not None and not imp_delta.empty:
            st.bar_chart(imp_delta.set_index("feature")["importance"], horizontal=True,
                         x_label="MAE increase", y_label="")
    with ic2:
        st.markdown("**Direct Points Model**")
        if imp_direct is not None and not imp_direct.empty:
            st.bar_chart(imp_direct.set_index("feature")["importance"], horizontal=True,
                         x_label="MAE increase", y_label="")

# --- Diagnostics ------------------------------------------------------------
st.divider()
st.caption(f"Dataset: {features['round'].nunique()} rounds, {len(features)} driver-race records. "
           f"Features: {', '.join(predict_fantasy.FEATURES)}")
wc.tuned_params_expander(result, {"delta": "Delta", "direct": "Direct"})
wc.log_expander("Season data log", result["season_log"])
if result["prep_log"].strip():
    wc.log_expander("Weekend download log", result["prep_log"])
