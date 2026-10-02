"""Streamlit glue for the F1 prediction web app.

No pipeline logic lives here — this module only wires the two pipelines
(predict_race / predict_grid, both built on f1_common) into Streamlit:
stdout capture for live download logs, and cached wrappers around data
loading, feature building, backtests, final predictions and feature
importance so interacting with widgets does not restart minutes of work.

Cache keys include a per-page `data_version` (bumped by the "reload data"
button) and a `force_refresh` flag (the CLI --refresh equivalent).
"""

import io
import sys
from contextlib import contextmanager
from datetime import datetime
from pathlib import Path
from types import SimpleNamespace

import altair as alt
import pandas as pd
import streamlit as st

_HERE = Path(__file__).resolve().parent
for _p in (_HERE, _HERE.parent / "pipelines"):
    if str(_p) not in sys.path:
        sys.path.insert(0, str(_p))

import f1_common
import predict_race
import predict_grid

f1_common.setup()

MODULES = {"race": predict_race, "grid": predict_grid}

CURRENT_YEAR = datetime.now().year


class TargetUnavailable(Exception):
    """Raised when the CLI would SystemExit (e.g. qualifying not done yet)."""


# ---------------------------------------------------------------------------
# stdout capture
# ---------------------------------------------------------------------------

@contextmanager
def capture_stdout():
    """Collect everything printed inside the block into a StringIO.

    Nested captures tee their text into the parent capture (or the real
    console), so download progress stays visible on the server terminal.
    Cached pipeline stages capture their own logs and return them for display;
    no Streamlit element is ever touched inside a cached function.
    """
    buffer = io.StringIO()
    parent = sys.stdout

    class _Writer(io.TextIOBase):
        def write(self, text):
            buffer.write(text)
            if parent is not None:
                try:
                    if isinstance(parent, _Writer):
                        parent.write_raw(text)
                    else:
                        parent.write(text)
                except Exception:
                    pass
            return len(text)

        def write_raw(self, text):
            buffer.write(text)
            return len(text)

    sys.stdout = _Writer()
    try:
        yield buffer
    finally:
        sys.stdout = parent


# ---------------------------------------------------------------------------
# cached pipeline access
# ---------------------------------------------------------------------------

@st.cache_resource(show_spinner=False)
def get_schedule(year):
    import fastf1
    f1_common.setup()
    return fastf1.get_event_schedule(year, include_testing=False)


@st.cache_resource(show_spinner=False)
def load_season(kind, year, data_version, force_refresh=False):
    """Collect season data for one pipeline ("race" / "grid").

    Falls back to the previous year if the requested season has no completed
    rounds yet, mirroring the CLI. Returns {year, data, schedule, log}.
    """
    mod = MODULES[kind]
    with capture_stdout() as log:
        used_year = year
        schedule = get_schedule(used_year)
        data = mod.collect_season(used_year, schedule, refresh=force_refresh)
        if data.empty and used_year > 2018:
            used_year -= 1
            print(f"\nNo completed rounds in {year} yet, falling back to {used_year}")
            schedule = get_schedule(used_year)
            data = mod.collect_season(used_year, schedule, refresh=force_refresh)
    if data.empty:
        return {"year": used_year, "data": None, "schedule": schedule, "log": log.getvalue()}
    return {"year": used_year, "data": data, "schedule": schedule, "log": log.getvalue()}


def resolve_target(kind, selection, year, schedule, data):
    """Map a UI selection onto the pipeline's resolve_target.

    selection: None (auto), "next", or an integer round number.
    Returns (round, mode, event_name, log); raises TargetUnavailable with the
    CLI message when the round cannot be predicted.
    """
    mod = MODULES[kind]
    args = SimpleNamespace(
        next=selection == "next",
        predict_round=selection if isinstance(selection, int) else None,
    )
    with capture_stdout() as log:
        try:
            target, mode, event_name = mod.resolve_target(args, year, schedule, data)
        except SystemExit as exc:
            raise TargetUnavailable(str(exc.code)) from None
    return target, mode, event_name, log.getvalue()


@st.cache_resource(show_spinner=False)
def prepare_features(kind, year, data_version, force_refresh, target, mode, event_name):
    """Season data + (in pre mode) the upcoming round, turned into features."""
    mod = MODULES[kind]
    season = load_season(kind, year, data_version, force_refresh)
    data = season["data"]
    with capture_stdout() as log:
        if mode == "pre":
            if kind == "race":
                upcoming = mod.load_upcoming_round(year, target, event_name)
            else:
                completed = sorted(data["round"].unique())
                fallback = data[data["round"] == completed[-1]]
                upcoming = mod.load_upcoming_round(year, target, event_name, fallback)
            data = pd.concat([data[data["round"] != target], upcoming], ignore_index=True)
        features = mod.build_features(data)
    return {"features": features, "log": log.getvalue()}


@st.cache_resource(show_spinner=False)
def backtest_records(kind, year, data_version, force_refresh, target, mode,
                     event_name, min_train_rounds):
    mod = MODULES[kind]
    prep = prepare_features(kind, year, data_version, force_refresh, target, mode, event_name)
    return mod.backtest_records(prep["features"], min_train_rounds)


@st.cache_resource(show_spinner=False)
def final_predictions(kind, year, data_version, force_refresh, target, mode, event_name):
    mod = MODULES[kind]
    prep = prepare_features(kind, year, data_version, force_refresh, target, mode, event_name)
    return mod.final_predictions(prep["features"], target)


@st.cache_resource(show_spinner=False)
def importance_frame(kind, year, data_version, force_refresh, target, mode,
                     event_name, model_mode):
    mod = MODULES[kind]
    pred = final_predictions(kind, year, data_version, force_refresh, target, mode, event_name)
    return mod.importance_frame(pred, model_mode)


# ---------------------------------------------------------------------------
# UI helpers
# ---------------------------------------------------------------------------

def season_controls(kind):
    """Sidebar season selector + data controls. Returns (year, data_version, force_refresh)."""
    version_key = f"{kind}_data_version"
    if version_key not in st.session_state:
        st.session_state[version_key] = 0
    year = st.number_input(
        "Season", min_value=2018, max_value=CURRENT_YEAR + 1, value=CURRENT_YEAR,
        help="First use of a different season downloads its full history (~several minutes).",
        key=f"{kind}_season",
    )
    force_refresh = st.checkbox(
        "Force full re-download (`--refresh`)",
        help="Ignore the local CSV cache and re-fetch every session of the season. Slow.",
        key=f"{kind}_refresh",
    )
    if st.button("Reload season data", key=f"{kind}_reload",
                 help="Re-check fastf1 for newly completed rounds (only new rounds are downloaded)."):
        st.session_state[version_key] += 1
    return int(year), st.session_state[version_key], force_refresh


def target_selectbox(kind, schedule, completed_rounds):
    """Round selector. Returns None (auto), "next", or a round number."""
    label = "Target race" if kind == "race" else "Target qualifying"
    options = {"Auto (recommended)": None, "Next on the calendar": "next"}
    for _, row in schedule.iterrows():
        rn = int(row["RoundNumber"])
        state = "completed" if rn in completed_rounds else "upcoming"
        options[f"Round {rn} — {row['EventName']} ({state})"] = rn
    choice = st.selectbox(label, list(options), key=f"{kind}_target",
                          help="Auto follows the same logic as the CLI: predict the next "
                               "event if it can be predicted, otherwise review the last completed round.")
    return options[choice]


def log_expander(title, text):
    with st.expander(title):
        st.code(text.strip() or "(no output)", language=None)


def page_link(path, label, icon):
    """st.page_link that degrades to a caption when the page is executed
    standalone (no st.navigation context, e.g. AppTest.from_file)."""
    try:
        st.page_link(path, label=label, icon=icon)
    except Exception:
        st.caption(f"**{label}** — open via the sidebar navigation")


# ---------------------------------------------------------------------------
# colors
# ---------------------------------------------------------------------------

@st.cache_resource(show_spinner=False)
def team_colors(year):
    """Team name -> hex color, taken straight from fastf1 session results
    (the same source the notebooks use). Most recent completed round wins."""
    import fastf1
    f1_common.setup()
    rounds = f1_common.completed_rounds(get_schedule(year))
    for rn, _name in reversed(rounds):
        try:
            session = fastf1.get_session(year, rn, "R")
            session.load(laps=False, telemetry=False, weather=False, messages=False)
            colors = {t: "#" + c for t, c in
                      zip(session.results["TeamName"], session.results["TeamColor"])
                      if isinstance(c, str) and c}
            if colors:
                return colors
        except Exception:
            continue
    return {}


def readable_color(hex_color, fallback="#999999"):
    """Team color adjusted to stay readable as text on a white background."""
    if not hex_color:
        hex_color = fallback
    try:
        r, g, b = (int(hex_color[i:i + 2], 16) for i in (1, 3, 5))
    except (ValueError, IndexError):
        return fallback
    luminance = 0.299 * r + 0.587 * g + 0.114 * b
    if luminance > 170:
        factor = 0.5
    elif luminance > 135:
        factor = 0.72
    else:
        factor = 1.0
    return "#{:02x}{:02x}{:02x}".format(int(r * factor), int(g * factor), int(b * factor))


PODIUM_STYLES = {
    1: "background-color: #FFD700; color: #4a3200; font-weight: 600",   # gold
    2: "background-color: #C0C0C0; color: #333333; font-weight: 600",   # silver
    3: "background-color: #CD7F32; color: #ffffff; font-weight: 600",   # bronze
}


def position_css(value):
    """Cell style for a position column: gold/silver/bronze for P1/P2/P3."""
    if pd.isna(value):
        return ""
    try:
        return PODIUM_STYLES.get(int(value), "")
    except (TypeError, ValueError):
        return ""


def team_css(team, colors, bold=False):
    css = f"color: {readable_color(colors.get(team))}"
    if bold:
        css += "; font-weight: 700"
    return css


def bold_row_style(row):
    """Bold every cell of a row (for Total rows in points tables)."""
    return ["font-weight: 700"] * len(row)


# ---------------------------------------------------------------------------
# controls
# ---------------------------------------------------------------------------

def first_backtest_control(kind, completed):
    """Sidebar selectbox picking the first round the backtest predicts.

    All completed rounds before it are the training set for the first
    backtest model. Returns the equivalent `min_train_rounds` for the
    pipeline (number of completed rounds before the selection).
    """
    options = completed[1:]
    if not options:
        return 0
    default = options[4] if len(options) > 4 else options[0]
    choice = st.selectbox(
        "First backtest round", options, index=options.index(default),
        key=f"{kind}_first_backtest",
        format_func=lambda rn: f"Round {rn}",
        help="The backtest predicts every completed round from this one onwards; "
             "all rounds before it train the first model.",
    )
    return sum(1 for r in completed if r < choice)


# ---------------------------------------------------------------------------
# notebook-style charts (Streamlit-native)
# ---------------------------------------------------------------------------

def predicted_order_chart(rows, cut_lines, colors, slot_label="position"):
    """Notebook-style 'final predicted grids' diagram as a native Streamlit
    chart (Altair, rendered with theme="streamlit" so it matches the app):
    one lane per model, team-colored driver codes placed at their predicted
    slot, dashed reference lines at the given cuts.

    rows: list of (label, DataFrame, position_column), top to bottom.
    cut_lines: {position: label} for the dashed reference lines.
    colors: team -> hex color map.
    """
    lane_names = [label for label, _, _ in rows]
    pieces = []
    for label, frame, col in rows:
        piece = frame[["driver", "team", col]].dropna(subset=[col]).rename(columns={col: "position"})
        piece["lane"] = label
        pieces.append(piece)
    data = pd.concat(pieces, ignore_index=True)
    data["position"] = data["position"].astype(float)
    n_slots = int(data["position"].max())
    teams = list(dict.fromkeys(data["team"]))
    scale = alt.Scale(domain=teams, range=[colors.get(t, "#999999") for t in teams])
    lane_order = list(reversed(lane_names))  # vega y-domain runs bottom -> top

    lanes = alt.Chart(data).mark_text(fontSize=12, fontWeight="bold").encode(
        x=alt.X("position:Q", title=slot_label,
                scale=alt.Scale(domain=[0.3, n_slots + 0.7]),
                axis=alt.Axis(values=list(range(1, n_slots + 1)), tickMinStep=1)),
        y=alt.Y("lane:N", title=None, sort=lane_order),
        text="driver:N",
        color=alt.Color("team:N", scale=scale, legend=None),
        tooltip=["lane:N", "driver:N", "team:N",
                 alt.Tooltip("position:Q", title=slot_label, format=".0f")],
    )

    if cut_lines:
        cuts = pd.DataFrame({"position": list(cut_lines.keys()),
                             "label": list(cut_lines.values()),
                             "lane": lane_names[0]})
        rules = alt.Chart(cuts).mark_rule(color="#888888", strokeDash=[5, 5]).encode(x="position:Q")
        cut_labels = alt.Chart(cuts).mark_text(fontSize=9, color="#888888", dy=-13).encode(
            x="position:Q", y=alt.Y("lane:N", sort=lane_order), text="label:N")
        return rules + lanes + cut_labels
    return lanes
