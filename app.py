"""Entry point for the Streamlit web app.

Run from the repo root (see the GitHub README for setup):
    python -m streamlit run app.py
"""

import streamlit as st

st.set_page_config(
    page_title="F1 Race & Grid Predictor",
    page_icon="🏁",
    layout="wide",
    initial_sidebar_state="expanded",
)

home = st.Page("webapp/page_home.py", title="Home", icon=":material/home:", default=True)
race = st.Page("webapp/page_race.py", title="Race prediction", icon=":material/sports_score:")
quali = st.Page("webapp/page_quali.py", title="Qualifying prediction", icon=":material/timer:")
extras = st.Page("webapp/page_extras.py", title="Milestones & extras", icon=":material/emoji_events:")

pg = st.navigation({"": [home], "Predictions": [race, quali, extras]})
pg.run()
