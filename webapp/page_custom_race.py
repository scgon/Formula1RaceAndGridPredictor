"""Custom race prediction page: user-supplied qualifying order -> predicted finish.

This pipeline is fundamentally different from the others: it predicts a hypothetical
race from a user-supplied grid rather than a calendar round. Still, it follows
the same web-app conventions: sidebar controls, button-gated execution, fitted-model
cache reuse, and the standard render sections (table, diagram, backtest, importance).
"""

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

import predict_custom_race as custom
import webapp_common as wc

st.title("Custom Race Predictor")
st.caption("Build a hypothetical race: choose a reference season and circuit, assign "
           "drivers to qualifying positions using the dropdowns below. No duplicates allowed.")

years = custom.bundled_years()
if not years:
    st.error("No bundled season data is available.")
    st.stop()

with st.sidebar:
    st.header("Race setup")
    year = st.selectbox("Reference season", years, index=len(years) - 1,
                        key="custom_year")
    schedule = wc.get_schedule(year)
    circuits = [str(x) for x in schedule["EventName"].dropna().tolist()]
    data = custom.load_data(year)
    if not circuits:
        circuits = sorted(data["event"].dropna().unique())
    circuit = st.selectbox("Circuit", circuits, key="custom_circuit")
    profile = st.selectbox("Model profile", ["Fast", "Optimized"],
                           key="custom_profile")
    run = st.button("Predict custom race", type="primary", key="custom_race_run",
                    help="Execute the pipeline with the current settings — "
                         "changing settings never starts it automatically.")

# Driver roster for the reference season (latest completed round).
season_data = data[data["season"] == year]
latest_round = season_data["round"].max()
roster = (season_data[season_data["round"] == latest_round]
          .sort_values("grid")
          .drop_duplicates("driver"))
drivers = roster["driver"].dropna().astype(str).tolist()
# Replace Ocon (OCO) with Camara (CAM) for 2026 display
if year == 2026:
    drivers = ["CAM" if d == "OCO" else d for d in drivers]

# Championship points order across the whole season.
championship_points = (season_data.groupby("driver")["points"]
                       .sum()
                       .sort_values(ascending=False))
# Include all drivers from the roster; drivers with 0 points go at the end
championship_order = championship_points.index.tolist()
# Add any drivers from the roster who didn't score points (not in championship_points)
for d in drivers:
    if d not in championship_order:
        championship_order.append(d)
# Defensive: ensure championship_order has at least all drivers (fallback to roster order)
if len(championship_order) < len(drivers):
    for d in drivers:
        if d not in championship_order:
            championship_order.append(d)
# Replace Ocon (OCO) with Camara (CAM) in championship order for 2026 display
if year == 2026:
    championship_order = ["CAM" if d == "OCO" else d for d in championship_order]

# True qualifying order for the SELECTED circuit (the round whose EventName matches),
# not the most recent race.
true_quali_order = []
circuit_row = schedule[schedule["EventName"] == circuit]
if not circuit_row.empty:
    round_number = int(circuit_row["RoundNumber"].iloc[0])
    circuit_round_data = season_data[season_data["round"] == round_number]
    if not circuit_round_data.empty:
        true_quali_order = (circuit_round_data.sort_values("grid")["driver"]
                            .dropna()
                            .astype(str)
                            .tolist())
# Replace Ocon (OCO) with Camara (CAM) in true qualifying order for 2026
if year == 2026:
    true_quali_order = ["CAM" if d == "OCO" else d for d in true_quali_order]

PLACEHOLDER = "—"

# --- Apply any pending selection update BEFORE widgets are created ---
if "_custom_pending_selection" in st.session_state:
    pending = st.session_state.pop("_custom_pending_selection")
    st.session_state["custom_selection"] = pending
    # Set widget keys so dropdowns render with new values on this run
    for pos, driver in pending.items():
        st.session_state[f"custom_pos_{pos}"] = PLACEHOLDER if driver is None else driver

# Track selected drivers per position; start empty (reset when season/circuit changes).
# Use separate tracking keys to avoid modifying widget keys after instantiation.
_last_year = st.session_state.get("_custom_last_year")
_last_circuit = st.session_state.get("_custom_last_circuit")

if ("custom_selection" not in st.session_state
        or _last_year != year
        or _last_circuit != circuit):
    st.session_state["custom_selection"] = {pos: None for pos in range(1, len(drivers) + 1)}
    st.session_state["_custom_last_year"] = year
    st.session_state["_custom_last_circuit"] = circuit

# --- position dropdowns ---------------------------------------------------------
st.subheader("Qualifying order")
st.write("Pick a driver for each qualifying position. Each driver can appear only once — "
         "if you change one selection, the previously chosen driver is released for other positions.")

# Build available drivers for each position (exclude already selected)
available_for_positions = {}
for pos in range(1, len(drivers) + 1):
    already_selected = [st.session_state["custom_selection"][p] for p in range(1, len(drivers) + 1)
                        if st.session_state["custom_selection"][p] is not None and p != pos]
    available_for_positions[pos] = [PLACEHOLDER] + sorted(
        d for d in drivers if d not in already_selected)

cols = st.columns(min(len(drivers), 3))
new_selection = {}
for idx, pos in enumerate(range(1, len(drivers) + 1)):
    with cols[idx % 3]:
        current = st.session_state["custom_selection"].get(pos)
        if current is None or current not in available_for_positions[pos]:
            current = PLACEHOLDER
        choice = st.selectbox(
            f"P{pos}",
            options=available_for_positions[pos],
            index=available_for_positions[pos].index(current),
            key=f"custom_pos_{pos}",
        )
        new_selection[pos] = None if choice == PLACEHOLDER else choice

# Update session state if changed
if new_selection != st.session_state["custom_selection"]:
    st.session_state["custom_selection"] = new_selection

# Show progress
filled = [v for v in st.session_state["custom_selection"].values() if v is not None]
st.caption(f"{len(filled)} of {len(drivers)} positions assigned.")

# --- autofill buttons -----------------------------------------------------------
st.divider()
st.subheader("Autofill remaining positions")
col1, col2 = st.columns(2)
with col1:
    if st.button("Fill rest by championship", key="custom_autofill_champ",
                 help="Fill every unassigned position in current championship-points order"):
        new_sel = dict(st.session_state["custom_selection"])
        champ_idx = 0
        # Build a combined source: championship order first, then any remaining drivers from roster
        source = list(championship_order)
        for d in drivers:
            if d not in source:
                source.append(d)
        for pos in range(1, len(drivers) + 1):
            if new_sel[pos] is None:
                while (champ_idx < len(source)
                       and source[champ_idx] in new_sel.values()):
                    champ_idx += 1
                if champ_idx < len(source):
                    new_sel[pos] = source[champ_idx]
                    champ_idx += 1
        st.session_state["_custom_pending_selection"] = new_sel
        st.rerun()
with col2:
    if st.button("Fill rest by true quali", key="custom_autofill_quali",
                 help="Fill every unassigned position using the actual qualifying order "
                      "from the selected circuit"):
        new_sel = dict(st.session_state["custom_selection"])
        quali_idx = 0
        # Build a combined source: true quali order first, then any remaining drivers from roster
        source = list(true_quali_order)
        for d in drivers:
            if d not in source:
                source.append(d)
        for pos in range(1, len(drivers) + 1):
            if new_sel[pos] is None:
                while (quali_idx < len(source)
                       and source[quali_idx] in new_sel.values()):
                    quali_idx += 1
                if quali_idx < len(source):
                    new_sel[pos] = source[quali_idx]
                    quali_idx += 1
        st.session_state["_custom_pending_selection"] = new_sel
        st.rerun()

# Clear all selections button
if st.button("Clear selections", key="custom_clear_selections",
             help="Reset all driver assignments — start from scratch"):
    new_sel = {pos: None for pos in range(1, len(drivers) + 1)}
    st.session_state["_custom_pending_selection"] = new_sel
    st.rerun()

# Validate and run --------------------------------------------------------------
filled = [v for v in st.session_state["custom_selection"].values() if v is not None]
all_filled = len(filled) == len(drivers)
unique_check = len(filled) == len(set(filled))

# Convert profile to lowercase for the pipeline
profile_lower = "optimized" if profile == "Optimized" else "fast"

# Find the target round for the selected circuit
circuit_row = schedule[schedule["EventName"] == circuit]
target_round = int(circuit_row["RoundNumber"].iloc[0]) if not circuit_row.empty else None

if run:
    if target_round is None:
        st.error("Could not find the selected circuit in the schedule.")
    elif not all_filled:
        st.error("Please assign a driver to every qualifying position before running.")
    elif not unique_check:
        st.error("Duplicate drivers detected — each driver can appear only once "
                 "in the qualifying order.")
    else:
        try:
            with st.spinner("Training on bundled historical races and predicting "
                            "the custom grid..."):
                # Build features from the full data
                features = custom.build_features(data)
                # Convert display names (CAM) back to data names (OCO) for the pipeline
                qualifying_order = list(st.session_state["custom_selection"].values())
                if year == 2026:
                    qualifying_order = ["OCO" if d == "CAM" else d for d in qualifying_order]
                bundle = custom.custom_predictions(features, year, target_round, qualifying_order, profile=profile_lower)
                # Use the gain model as the main prediction
                gain = bundle["gain"]
                # Format for display
                pred = gain[["pred_pos", "driver", "team", "grid", "pred_finish"]].copy()
                pred.columns = ["predicted_position", "driver", "team", "grid", "predicted_finish"]
                # Convert prediction driver codes back to display names for 2026
                if year == 2026:
                    pred["driver"] = pred["driver"].apply(lambda d: "CAM" if d == "OCO" else d)
                result = {
                    "prediction": pred,
                    "circuit": circuit,
                    "training_rows": int(bundle["train"].shape[0]),
                }
            st.session_state["custom_race_result"] = result
        except Exception as exc:
            st.error(f"Custom prediction failed: {type(exc).__name__}: {exc}")

result = st.session_state.get("custom_race_result")
if result is None:
    st.info("Choose a circuit, assign drivers to qualifying positions (or use an "
            "autofill button), then press **Predict custom race**.")
    st.stop()

pred = result["prediction"]
colors = wc.team_colors(year)
display = pred[["predicted_position", "driver", "team", "grid", "predicted_finish"]].copy()
display.columns = ["Predicted finish", "Driver", "Team", "Qualifying", "Model score"]
styled = display.style.format({"Predicted finish": "P{:.0f}",
                               "Qualifying": "P{:.0f}",
                               "Model score": "{:.1f}"})
styled = styled.map(wc.position_css, subset=["Predicted finish", "Qualifying"])
styled = wc.style_driver_team_columns(styled, pred, colors)
st.subheader(f"Predicted finishing order — {result['circuit']}")
st.dataframe(styled, width="stretch", hide_index=True)
st.success(f"**Predicted podium:** {wc.drivers_md(pred.head(3)['driver'],
                                                  wc.team_by_driver(pred), colors)}")
st.caption(f"Trained on {result['training_rows']:,} historical race-driver rows. "
           f"Circuit-specific form is used when this circuit exists in the bundled history.")