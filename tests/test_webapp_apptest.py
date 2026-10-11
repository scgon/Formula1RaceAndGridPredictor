"""Streamlit AppTest runs: every page must render without errors, and each
prediction page must complete a full Run (data -> resolve -> features ->
backtest -> final models -> importance -> render) off the bundled CSVs.

The Run tests pin a completed round — the newest one present in the
bundled season CSV — so they are deterministic: a post-mode round resolves
from the tracked CSVs alone and downloads nothing beyond the season
schedule (one tiny call — these tests need network access, which is why
they carry the "slow" marker; run `-m "not slow"` for the offline-only
fast subset).
"""

from pathlib import Path
from datetime import datetime

import pandas as pd
import pytest
from streamlit.testing.v1 import AppTest

import webapp_common as wc

REPO_ROOT = Path(__file__).resolve().parents[1]
PAGES = REPO_ROOT / "webapp"

# target-selectbox key -> the season CSV the page's Run resolves from
SEASON_CSV = {"race_target": "season_{year}.csv",
              "grid_target": "quali_season_{year}.csv",
              "extras_target": "extras_season_{year}.csv"}


def _completed_target_label(at, key):
    """Label of the newest round a Run can actually resolve: completed on
    the calendar AND present in the bundled season CSV.

    A round counts as completed 3 hours after its race start, but the data
    providers lag behind the session — race results reach the results API
    quickly, the lap data the extras pipeline needs much later — so for
    hours after every race the newest completed round is missing from its
    CSV and cannot be fetched either. A Run pinned to such a round renders
    the designed "retry in a few hours" error instead of a prediction,
    which would fail the suite for no code reason; pinning to the CSV
    keeps the Run resolving from the tracked data alone. Round 1 can never
    be predicted (no training data before it) — the later the round, the
    more fully-trained the backtest it exercises."""
    kind = key.removesuffix("_target")
    year = int(at.sidebar.selectbox(key=f"{kind}_season").value)
    path = REPO_ROOT / "data" / SEASON_CSV[key].format(year=year)
    if not path.exists():
        pytest.skip(f"no bundled {kind} season CSV for {year} yet")
    have = set(pd.read_csv(path, usecols=["round"])["round"])
    labels = [o for o in at.sidebar.selectbox(key=key).options
              if "(completed)" in o]
    runnable = [o for o in labels if int(o.split()[1]) in have]
    if not runnable:
        pytest.skip(f"no completed round of the {year} {kind} season is bundled yet")
    return runnable[-1]


def test_home_page_renders():
    at = AppTest.from_file(str(PAGES / "page_home.py"), default_timeout=300)
    at.run()
    assert not at.exception
    assert at.title[0].value.startswith("Formula 1")


@pytest.mark.slow
def test_race_page_full_run_on_a_completed_round():
    at = AppTest.from_file(str(PAGES / "page_race.py"), default_timeout=900)
    at.run()
    assert not at.exception
    at.sidebar.selectbox(key="race_target").set_value(
        _completed_target_label(at, "race_target"))
    at.sidebar.button(key="race_run").click()
    at.run()
    assert not at.exception and not at.error
    assert any("Review" in s.value for s in at.subheader)
    assert at.dataframe  # the prediction table rendered


@pytest.mark.slow
def test_quali_page_full_run_on_a_completed_round():
    at = AppTest.from_file(str(PAGES / "page_quali.py"), default_timeout=900)
    at.run()
    assert not at.exception
    at.sidebar.selectbox(key="grid_target").set_value(
        _completed_target_label(at, "grid_target"))
    at.sidebar.button(key="grid_run").click()
    at.run()
    assert not at.exception and not at.error
    assert any("Review" in s.value for s in at.subheader)
    assert at.dataframe  # the prediction table rendered


@pytest.mark.slow
def test_extras_page_full_run_on_a_completed_round():
    at = AppTest.from_file(str(PAGES / "page_extras.py"), default_timeout=900)
    at.run()
    assert not at.exception
    at.sidebar.selectbox(key="extras_target").set_value(
        _completed_target_label(at, "extras_target"))
    at.sidebar.button(key="extras_run").click()
    at.run()
    assert not at.exception and not at.error
    assert any("Review" in s.value for s in at.subheader)
    assert at.dataframe  # the probability table rendered


@pytest.mark.slow
def test_extras_champion_page_full_run_on_a_completed_season():
    # a past-calendar extras season: its championship is decided, so the
    # champion model's backtest is scored end to end (a current-year run
    # would render too, but every hit would be an unscored "?")
    complete = [y for y in wc.bundled_years("extras") if y < datetime.now().year]
    if not complete:
        pytest.skip("no completed extras season is bundled")
    season = max(complete)
    at = AppTest.from_file(str(PAGES / "page_extras.py"), default_timeout=900)
    at.run()
    assert not at.exception
    at.sidebar.selectbox(key="extras_season").set_value(season)
    at.sidebar.selectbox(key="extras_milestone").set_value("Championship winner")
    at.run()  # re-render the target selectbox with the chosen season's rounds
    at.sidebar.selectbox(key="extras_target").set_value(
        _completed_target_label(at, "extras_target"))
    at.sidebar.button(key="extras_run").click()
    at.run()
    assert not at.exception and not at.error
    assert any("Review" in s.value for s in at.subheader)
    assert at.dataframe  # the probability table rendered
    assert any("the model's call was a" in i.value for i in at.info)  # scored


@pytest.mark.slow
def test_championship_page_full_run_on_a_completed_season():
    # a completed season: the forecast resolves to post mode (the final
    # standings are known), so the render is fully scored and deterministic
    # apart from the schedule fetch; the backtest (leave-one-season-out over
    # every complete bundled season) is the multi-minute part of the run
    complete = [y for y in wc.bundled_years("championship") if y < datetime.now().year]
    if not complete:
        pytest.skip("no completed championship season is bundled")
    season = max(complete)
    at = AppTest.from_file(str(PAGES / "page_championship.py"), default_timeout=900)
    at.run()
    assert not at.exception
    at.sidebar.selectbox(key="championship_season").set_value(season)
    at.sidebar.selectbox(key="championship_championship").set_value(
        "Drivers' championship")
    at.run()  # re-render the snapshot selectbox with the chosen season's rounds
    at.sidebar.button(key="championship_run").click()
    at.run()
    assert not at.exception and not at.error
    assert any("Review" in s.value for s in at.subheader)
    assert at.dataframe  # the projected standings table rendered
    standings = at.dataframe[0].value
    for col in ("Final points — remaining", "Pts/round — remaining",
                "Final points — direct", "Pts/round — direct",
                "Baseline points", "Avg pts/round"):
        assert col in standings.columns
    # the auto snapshot of a complete season is its final round — nothing is
    # left to forecast, so the implied-pace columns come out blank while the
    # season average (the rate the run-rate baseline extrapolates) never is
    assert standings["Pts/round — remaining"].isna().all()
    assert standings["Pts/round — direct"].isna().all()
    assert standings["Avg pts/round"].notna().all()


@pytest.mark.slow
def test_championship_page_first_backtest_season_setting():
    # the *First backtest season* selectbox wired end-to-end: an explicit
    # season scopes the backtest to that season and later, and the scope
    # caption renders. A late *First backtest round* keeps only a couple of
    # forecasts, so this full Run stays fast.
    complete = [y for y in wc.bundled_years("championship") if y < datetime.now().year]
    if not complete:
        pytest.skip("no completed championship season is bundled")
    season = max(complete)
    at = AppTest.from_file(str(PAGES / "page_championship.py"), default_timeout=900)
    at.run()
    assert not at.exception
    at.sidebar.selectbox(key="championship_season").set_value(season)
    at.sidebar.selectbox(key="championship_championship").set_value(
        "Drivers' championship")
    at.run()  # re-render the round/season selectboxes with the chosen season
    round_labels = [o for o in at.sidebar.selectbox(key="championship_first_backtest").options
                    if o.startswith("Round ")]
    at.sidebar.selectbox(key="championship_first_backtest").set_value(round_labels[-2])
    at.sidebar.selectbox(key="championship_first_season").set_value(str(season))
    at.sidebar.button(key="championship_run").click()
    at.run()
    assert not at.exception and not at.error
    assert any("First backtest season" in c.value for c in at.caption)
    assert any("excluded from the evaluation" in c.value for c in at.caption)
