"""Homepage for the F1 Race & Grid Predictor web app."""

import sys
from pathlib import Path

_HERE = Path(__file__).resolve().parent
for _p in (_HERE, _HERE.parent / "pipelines"):
    if str(_p) not in sys.path:
        sys.path.insert(0, str(_p))

import pandas as pd
import streamlit as st

import f1_common
import webapp_common as wc

st.title("Formula 1 Race & Grid Predictor")
st.caption("Machine-learning predictions for F1 race results and qualifying grids, "
           "built on [fastf1](https://github.com/theOehrly/fastf1) data and gradient boosting.")

# --- next race banner -----------------------------------------------------
try:
    schedule = wc.get_schedule(wc.CURRENT_YEAR)
    now = f1_common.utc_now()
    upcoming = None
    for _, row in schedule.iterrows():
        race_utc = f1_common.session_utc(row, "Race")
        if race_utc is not None and race_utc > now:
            upcoming = (row, race_utc)
            break
    if upcoming is not None:
        row, race_utc = upcoming
        days = (race_utc - now).days
        when = f"in {days} day{'s' if days != 1 else ''}" if days >= 0 else "today"
        st.info(f"**Next race:** Round {int(row['RoundNumber'])} — {row['EventName']} "
                f"({row.get('Country', '')}), race on {race_utc:%A %d %B %Y} ({when}, UTC).",
                icon=":material/calendar_month:")
    else:
        st.info(f"The {wc.CURRENT_YEAR} season is over — predictions can be reviewed on past rounds.",
                icon=":material/flag:")
except Exception:
    st.caption("Race calendar unavailable (offline) — predictions still work from cached data.")

# --- the pipelines ---------------------------------------------------------
st.subheader("Three pipelines, six models plus four milestone classifiers")

race_col, grid_col, extras_col = st.columns(3)

with race_col:
    st.markdown("##### :material/sports_score: Race prediction")
    st.markdown(
        "Predicts the **finishing order** of a Grand Prix — after qualifying with both "
        "models, or before it with the direct model alone (no grid to anchor the gain "
        "model to yet).\n\n"
        "- **Gain model** — predicts positions gained/lost vs the grid (finish − grid)\n"
        "- **Direct model** — predicts the absolute finishing position\n"
        "- **Baseline** — simply keeping grid order\n"
    )
    with st.expander("Race features"):
        st.markdown(
            "- Grid position, quali gap to pole, team quali gap\n"
            "- FP1–FP3 pace: best-lap and long-run stint deltas, lap counts\n"
            "- Sprint qualifying & sprint race results (sprint weekends)\n"
            "- Driver form: last finish, 3-race rolling and season average\n"
            "- Team form: 3-race rolling and season average\n"
            "- Championship points (driver & team) before the round\n"
            "- Team identity (categorical)"
        )
    wc.page_link("webapp/page_race.py", label="Open race prediction", icon=":material/sports_score:")

with grid_col:
    st.markdown("##### :material/timer: Qualifying prediction")
    st.markdown(
        "Predicts the **qualifying classification** before it runs.\n\n"
        "- **Anchor model** — predicts the change vs each driver's previous quali result\n"
        "- **Direct model** — predicts the absolute quali position\n"
        "- **Baseline** — repeating the previous qualifying order (persistence)\n"
    )
    with st.expander("Qualifying features"):
        st.markdown(
            "- FP1/FP2 pace, plus FP3 and sprint-race results once they have run\n"
            "- Sprint qualifying position & gap (sprint weekends)\n"
            "- Previous quali result, 3-round and season quali form\n"
            "- Team quali form and championship points\n"
            "- Team identity (categorical)\n\n"
            "Sessions beyond FP1/FP2 are optional: predictions before they run rely on "
            "Friday practice and historical form."
        )
    wc.page_link("webapp/page_quali.py", label="Open qualifying prediction", icon=":material/timer:")

with extras_col:
    st.markdown("##### :material/emoji_events: Milestones & extras")
    st.markdown(
        "Four small classifiers, each predicting one weekend milestone — pick one and "
        "only it runs.\n\n"
        "- **Pole position** — day-before-quali information (FP1/FP2, sprint quali, quali form)\n"
        "- **Race winner** — from the grid, practice pace, sprint results and form\n"
        "- **First retirement** — adds driver/team reliability history\n"
        "- **Fastest lap** — one-lap pace, race pace and past fastest laps\n"
        "- **Baselines** — best quali form, grid P1, most retirements, most fastest laps\n"
    )
    with st.expander("Milestone models"):
        st.markdown(
            "Each model scores every driver with a probability; the top pick is the "
            "prediction. First-retirement and fastest-lap targets come from race lap "
            "data, so this pipeline keeps its own season files (`extras_season_*.csv`)."
        )
    wc.page_link("webapp/page_extras.py", label="Open milestones & extras", icon=":material/emoji_events:")

st.divider()

# --- methodology -----------------------------------------------------------
st.subheader("How models are evaluated")
st.markdown(
    "Every pipeline runs a **rolling backtest** over the season: for each completed round, "
    "the models are retrained on all earlier rounds only, then predict that round. "
    "Reported metrics: mean absolute position error (MAE), podium hit rate, pole/winner "
    "hit rate and rank correlation — always compared against the naive baseline. "
    "Models are `HistGradientBoostingRegressor` (the two order pipelines) or "
    "`HistGradientBoostingClassifier` (the milestone classifiers) with a fixed seed: "
    "identical data gives identical predictions.\n\n"
    "Each prediction page offers two **model profiles**: *Fast* (fixed hyperparameters, "
    "quickest run) and *Optimized* (every model tunes its hyperparameters by minimizing "
    "cross-validated error — MAE for the order models, log loss for the milestone "
    "classifiers — on the rounds it trains on; usually better, but a slower run)."
)

# --- cached data status ----------------------------------------------------
st.subheader("Cached data")
rows = []
for kind, pattern, target in (("Race", "season_*.csv", "finish"),
                              ("Qualifying", "quali_season_*.csv", "quali_pos"),
                              ("Extras", "extras_season_*.csv", "finish")):
    for path in sorted(f1_common.DATA_DIR.glob(pattern)):
        try:
            df = pd.read_csv(path)
            rows.append({
                "Pipeline": kind,
                "Season": path.stem.split("_")[-1],
                "Rounds": df["round"].nunique(),
                "Records": len(df),
                "Last updated": pd.Timestamp(path.stat().st_mtime, unit="s").strftime("%Y-%m-%d %H:%M"),
            })
        except Exception:
            continue
if rows:
    st.table(pd.DataFrame(rows))
    st.caption("Season CSVs are bundled with the app and kept current by a scheduled job; "
               "the fastf1 session cache lives in `cache/` (not tracked). "
               "Each prediction page has a *Reload season data* button to pick up newly completed rounds.")
else:
    st.warning("No season data cached yet — run either prediction page once to download it.")

with st.expander("Prefer the command line?"):
    st.markdown(
        "Both pipelines also run as command-line scripts. See the "
        "[GitHub README](https://github.com/scgon/Formula1RaceAndGridPredictor#quick-start-cli) "
        "for setup, dependencies and the full list of flags. This hosted app runs on a free "
        "shared cloud container — if it feels throttled or hits rate limits, running the "
        "project locally (`python -m streamlit run app.py`) is faster and avoids them."
    )
