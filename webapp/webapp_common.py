"""Streamlit glue for the F1 prediction web app.

No pipeline logic lives here — this module only wires the two pipelines
(predict_race / predict_grid, both built on f1_common) into Streamlit.

Execution is button-gated: changing a widget never starts the pipeline.
Pressing **Run prediction** executes the whole pipeline once and stores a
render bundle in st.session_state; every rerun afterwards just renders from
that bundle. **Reload season data** refreshes the data layer only (it can
download newly completed rounds, or a full re-download when the force
checkbox is set — the checkbox alone never triggers a download).

Because the data layer is not Streamlit-cached, collect/download progress is
streamed live into a log element: collect_season prints the round it is
currently fetching (round number + event name) before each download.
"""

import io
import sys
import time
from contextlib import contextmanager
from datetime import datetime
from pathlib import Path
from types import SimpleNamespace

import altair as alt
import pandas as pd
import streamlit as st
from fastf1.exceptions import RateLimitExceededError

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
def capture_stdout(log_element=None, refresh_secs=0.3):
    """Collect everything printed inside the block into a StringIO.

    If log_element is an st.empty placeholder, its content is refreshed live
    (throttled) while the block runs. Nested captures tee their text into
    their parent (or the real console), so progress printed by inner helpers
    of an outer capture still streams into the outer log element.
    """
    buffer = io.StringIO()
    parent = sys.stdout

    class _Writer(io.TextIOBase):
        def __init__(self):
            self._last = 0.0

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
            self._update_element()
            return len(text)

        def write_raw(self, text):
            buffer.write(text)
            self._update_element()
            return len(text)

        def _update_element(self):
            if log_element is not None:
                now = time.time()
                if now - self._last >= refresh_secs:
                    self._last = now
                    try:
                        log_element.code(buffer.getvalue(), language=None)
                    except Exception:
                        pass

    sys.stdout = _Writer()
    try:
        yield buffer
    finally:
        sys.stdout = parent
        if log_element is not None:
            text = buffer.getvalue()
            try:
                if text:
                    log_element.code(text, language=None)
            except Exception:
                pass


# ---------------------------------------------------------------------------
# cached lookups (no elements, no downloads of session data)
# ---------------------------------------------------------------------------

@st.cache_resource(show_spinner=False)
def get_schedule(year):
    import fastf1
    f1_common.setup()
    return fastf1.get_event_schedule(year, include_testing=False)


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


def completed_round_numbers(schedule):
    """Calendar-based completed rounds (race started >COMPLETION_BUFFER ago)."""
    return [rn for rn, _name in f1_common.completed_rounds(schedule)]


def bundled_years():
    """Years whose season CSVs are bundled in the repo (both pipelines present).

    Selecting a bundled year never triggers a bulk season download — only
    rounds completed since the CSV snapshot (plus the upcoming round's
    sessions) are fetched, which keeps the app far below the F1 API's hard
    limit of 500 uncached calls per hour.
    """
    years = set()
    for path in f1_common.DATA_DIR.glob("season_*.csv"):
        try:
            year = int(path.stem.split("_")[1])
        except (IndexError, ValueError):
            continue
        if (f1_common.DATA_DIR / f"quali_season_{year}.csv").exists():
            years.add(year)
    return sorted(years)


# ---------------------------------------------------------------------------
# data + pipeline execution (uncached — runs only on button press)
# ---------------------------------------------------------------------------

def load_season(kind, year, force_refresh=False):
    """Collect season data for one pipeline ("race" / "grid"), reusing the
    local CSV cache; falls back to the previous year if the requested season
    has no completed rounds yet, mirroring the CLI. Returns
    {year, data, schedule, log}; data is None when nothing is available."""
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


def prepare_features(kind, season, target, mode, event_name):
    """Season data + (in pre mode) the upcoming round, turned into features."""
    mod = MODULES[kind]
    data = season["data"]
    with capture_stdout() as log:
        if mode == "pre":
            if kind == "race":
                upcoming = mod.load_upcoming_round(season["year"], target, event_name)
            else:
                completed = sorted(data["round"].unique())
                fallback = data[data["round"] == completed[-1]]
                upcoming = mod.load_upcoming_round(season["year"], target, event_name, fallback)
            data = pd.concat([data[data["round"] != target], upcoming], ignore_index=True)
        features = mod.build_features(data)
    return {"features": features, "log": log.getvalue()}


def run_pipeline(kind, year, data_version, force_refresh, selection, min_train,
                 profile="fast"):
    """Execute the full pipeline once with the given settings.

    Returns a render bundle for the page (stored in st.session_state by the
    caller), or {"error": message} when the run failed. Download progress is
    streamed live into a log element while data is collected.
    """
    mod = MODULES[kind]
    label = "race" if kind == "race" else "qualifying"
    main_mode = "gain" if kind == "race" else "anchor"
    tune_note = " (tuning hyperparameters per model)" if profile == "optimized" else ""
    log_box = st.empty()
    try:
        # one capture spans the whole run: download progress, then the
        # per-model training lines the pipelines print during the backtest
        # and final-model phases, all streamed live into the log element.
        with capture_stdout(log_box, refresh_secs=0.3):
            season = load_season(kind, year, force_refresh)
            if season["data"] is None:
                return {"error": f"No {label} data available for this or the previous season."}
            used_year = season["year"]
            target, mode, event_name, resolve_log = resolve_target(
                kind, selection, used_year, season["schedule"], season["data"])
            prep = prepare_features(kind, season, target, mode, event_name)
            features = prep["features"]
            with st.spinner(f"Running rolling backtest (one model pair per round){tune_note}..."):
                records = mod.backtest_records(features, min_train, profile)
            with st.spinner("Training final models and computing permutation importance..."):
                pred = mod.final_predictions(features, target, profile)
                imp_main = mod.importance_frame(pred, main_mode)
                imp_direct = mod.importance_frame(pred, "direct")
        return {
            "settings": (year, data_version, force_refresh, selection, min_train, profile),
            "requested_year": year,
            "year": used_year,
            "target": target,
            "mode": mode,
            "event": event_name,
            "resolve_log": resolve_log,
            "season_log": season["log"],
            "prep_log": prep["log"],
            "features": features,
            "records": records,
            "main": pred[main_mode],
            "train_rounds": int(pred["train"]["round"].nunique()),
            "imp_main": imp_main,
            "imp_direct": imp_direct,
            "tuned_params": ({main_mode: pred[f"{main_mode}_params"],
                              "direct": pred["direct_params"]}
                             if profile == "optimized" else None),
        }
    except TargetUnavailable as exc:
        return {"error": str(exc)}
    except RateLimitExceededError:
        return {"error": "The F1 data API rate limit was hit (500 calls per hour) — "
                         "this clears within the hour. Rounds fetched so far are saved; "
                         "press Run prediction again later."}
    except Exception as exc:  # download failures etc.
        return {"error": f"Pipeline failed: {type(exc).__name__}: {exc}"}
    finally:
        log_box.empty()


# ---------------------------------------------------------------------------
# UI helpers
# ---------------------------------------------------------------------------

def season_inputs(kind):
    """Sidebar season + data controls. Returns (year, force_refresh, reload_pressed).

    Only seasons with bundled CSVs are selectable, so public visitors can't
    trigger a full-season download (~450 API calls) by browsing years — the
    scheduled data-refresh workflow keeps the bundle current. The year after
    the latest bundled one is offered for next-season previews (the app then
    trains on the previous season's data, as the CLI fallback does).
    Nothing here executes the pipeline: the force checkbox only takes effect
    on the next Reload/Run press, and Reload refreshes the data layer only.
    """
    available = bundled_years()
    if available:
        options = sorted(set(available) | {max(available) + 1})
        default = CURRENT_YEAR if CURRENT_YEAR in options else max(options)
        help_text = ("Seasons with bundled historical data (kept current by a scheduled "
                     "job). The year after the last bundled season predicts upcoming "
                     "races using the previous season's training data.")
    else:
        options = [CURRENT_YEAR]
        default = CURRENT_YEAR
        help_text = "First use of a season downloads its full history (~several minutes)."
    year = st.selectbox(
        "Season", options, index=options.index(default), key=f"{kind}_season",
        help=help_text,
    )
    force_refresh = st.checkbox(
        "Force full re-download (`--refresh`)",
        help="Ignore the local CSV cache and re-fetch every session. Applied on the next "
             "Reload season data or Run prediction press — checking this alone downloads nothing.",
        key=f"{kind}_refresh",
    )
    reload_pressed = st.button(
        "Reload season data", key=f"{kind}_reload",
        help="Re-check fastf1 for newly completed rounds (only new rounds are downloaded, "
             "everything when the force checkbox is set). Press Run prediction afterwards "
             "to refresh the results.")
    return int(year), force_refresh, reload_pressed


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


def first_backtest_control(kind, completed):
    """Sidebar selectbox picking the first round the backtest predicts.

    "Auto (recommended)" — the default — applies the pipeline default
    (MIN_TRAIN_ROUNDS completed rounds of training data before the first
    predicted round). An explicit round starts the backtest there, with all
    completed rounds before it training the first model. The earliest
    explicit round differs per pipeline: the race backtest can predict
    round 2 (trained on round 1 only), while the qualifying anchor model
    needs each driver's previous quali result — which round-1 entrants
    don't have — so qualifying backtests start at round 3. Returns the
    equivalent `min_train_rounds` for the pipeline.
    """
    skip = 2 if kind == "grid" else 1
    explicit = completed[skip:]
    options = {"Auto (recommended)": None}
    for rn in explicit:
        options[f"Round {rn}"] = rn
    help_text = (f"Auto starts at the first round with the pipeline default of "
                 f"{f1_common.MIN_TRAIN_ROUNDS} completed rounds before it. An explicit "
                 "round starts the backtest there; all rounds before it train the "
                 "first model.")
    if kind == "grid":
        help_text += (" Round 2 cannot be predicted: the anchor model needs each "
                      "driver's previous qualifying result.")
    choice = st.selectbox("First backtest round", list(options),
                          key=f"{kind}_first_backtest", help=help_text)
    selected = options[choice]
    if selected is None:
        return f1_common.MIN_TRAIN_ROUNDS
    return sum(1 for r in completed if r < selected)


def model_profile_input(kind):
    """Sidebar selectbox for the model profile (the CLI --model flag).

    "Fast" keeps the project's fixed hyperparameters; "Optimized" runs a
    small hyperparameter search per trained model (minimizing
    cross-validated MAE over the training rounds) — usually a better MAE,
    but the backtest takes noticeably longer.
    """
    options = ("Fast", "Optimized")
    choice = st.selectbox(
        "Model profile", options, key=f"{kind}_model_profile",
        help="Fast: fixed hyperparameters, quickest run. Optimized: every model tunes its "
             "hyperparameters by minimizing cross-validated MAE on the rounds it trains on — "
             "better predictions on average, but the run takes longer.")
    return "optimized" if choice == "Optimized" else "fast"


def tuned_params_expander(result, labels):
    """Expander listing the hyperparameters the optimized profile picked for
    the final models (rendered only for optimized runs)."""
    params = result.get("tuned_params")
    if not params:
        return
    with st.expander("Tuned hyperparameters (this run)"):
        st.caption("Chosen per model by minimizing cross-validated MAE over the training "
                   "rounds; the fast profile's fixed values were always a candidate, so the "
                   "search never scores worse than Fast on that validation.")
        untuned = "fast defaults (no clearly better combination found)"
        for key, label in labels.items():
            picked = f1_common.format_params(params.get(key)) or untuned
            st.markdown(f"**{label} model**: {picked}")


def log_expander(title, text):
    with st.expander(title):
        st.code(text.strip() or "(no output)", language=None)


def degenerate_training_note(order_desc):
    """Caption flagging a final prediction whose models trained on a single
    round: with the project's model settings, one round of rows is too few
    for a single tree split, so the prediction basically reproduces
    `order_desc` (the baseline)."""
    st.caption(
        f":material/info: This round's models were trained on a single round of data — "
        f"too few rows for the model to make a single tree split, so the predictions "
        f"below basically reproduce {order_desc}.")


def degenerate_backtest_note(bt, train_cols, order_desc):
    """Same caveat as degenerate_training_note, for backtest results: flags
    every round whose models trained on fewer than two rounds."""
    flagged = sorted(int(r) for r in bt.loc[(bt[train_cols] < 2).any(axis=1), "round"])
    if not flagged:
        return
    rounds_txt = ", ".join(f"round {r}" for r in flagged)
    were = "was" if len(flagged) == 1 else "were"
    st.caption(
        f":material/info: {rounds_txt} {were} predicted by models trained on a single round "
        f"of data — too few rows for the model to make a single tree split, so those "
        f"predictions basically reproduce {order_desc}.")


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


def zero_error_css(value):
    """Cell style for an error column: green when the prediction was exact."""
    if pd.isna(value):
        return ""
    try:
        return "background-color: #b7f0c8; color: #0b5c2a; font-weight: 600" if float(value) == 0 else ""
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
# charts (Streamlit-native)
# ---------------------------------------------------------------------------

def predicted_order_chart(rows, cut_lines, colors, slot_label="position"):
    """Notebook-style 'final predicted grids' diagram as a native Streamlit
    chart (Altair, rendered with theme="streamlit" so it matches the app):
    one lane per model, team-colored driver codes placed at their predicted
    slot, dashed reference lines at the given cuts.

    rows: list of (label, DataFrame, position_column), top to bottom
    (the first entry renders as the bottom lane).
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
    lane_order = list(reversed(lane_names))  # first list entry = bottom lane

    lanes = alt.Chart(data).mark_text(fontSize=16, fontWeight="bold").encode(
        x=alt.X("position:Q", title=slot_label,
                scale=alt.Scale(domain=[0.3, n_slots + 0.7]),
                axis=alt.Axis(values=list(range(1, n_slots + 1)), tickMinStep=1)),
        y=alt.Y("lane:N", title=None, sort=lane_order,
                axis=alt.Axis(labelFontSize=13)),
        text="driver:N",
        color=alt.Color("team:N", scale=scale, legend=None),
        tooltip=["lane:N", "driver:N", "team:N",
                 alt.Tooltip("position:Q", title=slot_label, format=".0f")],
    ).properties(height=30 + 45 * len(rows))

    if cut_lines:
        cuts = pd.DataFrame({"position": list(cut_lines.keys()),
                             "label": list(cut_lines.values()),
                             "lane": lane_names[0]})
        rules = alt.Chart(cuts).mark_rule(color="#888888", strokeDash=[5, 5]).encode(x="position:Q")
        cut_labels = alt.Chart(cuts).mark_text(fontSize=10, color="#888888", dy=-15).encode(
            x="position:Q", y=alt.Y("lane:N", sort=lane_order), text="label:N")
        return rules + lanes + cut_labels
    return lanes
