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

import pytest
from streamlit.testing.v1 import AppTest

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
