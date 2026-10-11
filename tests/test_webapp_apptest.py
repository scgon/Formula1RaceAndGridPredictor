"""Streamlit AppTest runs: every page must render without errors, and each
prediction page must complete a full Run (data -> resolve -> features ->
backtest -> final models -> importance -> render) off the bundled CSVs.

The Run tests pin a completed round so they are deterministic: a post-mode
round resolves from the tracked CSVs alone and downloads nothing beyond
the season schedule (one tiny call — these tests need network access,
which is why they carry the "slow" marker; run `-m "not slow"` for the
offline-only fast subset).
"""

from pathlib import Path
from datetime import datetime

import pytest
from streamlit.testing.v1 import AppTest

import webapp_common as wc

REPO_ROOT = Path(__file__).resolve().parents[1]
PAGES = REPO_ROOT / "webapp"


def _completed_target_label(at, key):
    selectbox = at.sidebar.selectbox(key=key)
    labels = [o for o in selectbox.options if "(completed)" in o]
    # the LAST completed round: round 1 can never be predicted (no training
    # data before it), and later rounds exercise a fully-trained backtest
    return labels[-1]


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


