"""Offline unit tests for webapp/webapp_common.py — the pure UI glue:
stdout capture, the schedule TTL cache, team-color helpers and the
Streamlit-context fail-opens. No Streamlit run context and no network
needed (the schedule tests monkeypatch the fetch)."""

import re
import time

import pandas as pd
import pytest

import f1_common
import webapp_common as wc


# --- stdout capture -----------------------------------------------------------

def test_capture_stdout_collects_prints():
    with wc.capture_stdout() as log:
        print("hello")
    assert log.getvalue() == "hello\n"


def test_capture_stdout_nests_and_tees_into_the_parent():
    with wc.capture_stdout() as outer:
        with wc.capture_stdout() as inner:
            print("inner text")
        print("outer text")
    assert "inner text" in inner.getvalue()
    assert "outer text" in outer.getvalue()
    # the parent capture sees everything printed inside the nested one too
    assert "inner text" in outer.getvalue()


# --- schedule TTL cache ---------------------------------------------------------

def test_get_schedule_serves_a_fresh_cache_entry(monkeypatch):
    fake = pd.DataFrame({"RoundNumber": [1], "EventName": ["Fake GP"]})
    monkeypatch.setattr(wc, "_schedule_cache", {2077: [fake, time.monotonic()]})
    assert wc.get_schedule(2077) is fake


def test_get_schedule_serves_the_last_good_copy_when_the_refresh_fails(monkeypatch):
    fake = pd.DataFrame({"RoundNumber": [1], "EventName": ["Fake GP"]})
    monkeypatch.setattr(wc, "_schedule_cache",
                        {2077: [fake, time.monotonic() - wc.SCHEDULE_TTL - 10]})
    import fastf1

    def _boom(*args, **kwargs):
        raise RuntimeError("offline")

    monkeypatch.setattr(fastf1, "get_event_schedule", _boom)
    assert wc.get_schedule(2077) is fake


# --- team colors ---------------------------------------------------------------

def test_bundled_team_colors_render_from_the_bundle():
    years = pd.read_csv(f1_common.TEAM_COLORS_CSV)["year"].unique()
    colors = wc._bundled_team_colors(int(years.max()))
    assert colors
    for color in colors.values():
        assert re.fullmatch(r"#[0-9a-fA-F]{6}", color)


def test_expand_team_aliases_adds_every_historical_name():
    colors = wc.expand_team_aliases({"Red Bull": "#1234AB"})
    assert colors["Red Bull"] == "#1234AB"
    assert colors["Red Bull Racing"] == "#1234AB"
    assert "Aston Martin" not in colors  # lineage without a mapped team untouched

    colors = wc.expand_team_aliases({"RB": "#AA0000"})
    assert colors["Racing Bulls"] == "#AA0000"
    assert colors["Toro Rosso"] == "#AA0000"
    assert "Red Bull" not in colors  # a different lineage stays out


def test_bundled_years_follow_the_csvs():
    years = wc.bundled_years()
    assert years == sorted(years)
    assert years  # the repo bundles at least one season
    assert years[-1] >= 2026
    for year in years:
        assert (f1_common.DATA_DIR / f"season_{year}.csv").exists()
        assert (f1_common.DATA_DIR / f"quali_season_{year}.csv").exists()
    extras = wc.bundled_years("extras")
    assert extras
    assert set(extras) <= set(years)
    for year in extras:
        assert (f1_common.DATA_DIR / f"extras_season_{year}.csv").exists()


def test_team_by_driver_last_row_wins():
    frame = pd.DataFrame({"driver": ["A", "A", "B"],
                          "team": ["Team X", "Team Y", "Team Z"]})
    assert wc.team_by_driver(frame) == {"A": "Team Y", "B": "Team Z"}


def test_style_driver_team_columns_applies_to_both_columns():
    source = pd.DataFrame({"driver": ["A", "B"], "team": ["T1", "T2"],
                           "pred_pos": [1, 2]})
    display = source.rename(columns={"driver": "Driver", "team": "Team"})
    styled = wc.style_driver_team_columns(display.style, source,
                                          {"T1": "#ff0000", "T2": "#0000ff"})
    html = styled.to_html()
    assert html.count("color:") >= 2  # one team-colored cell per column


def test_driver_md_emits_the_color_directive():
    text = wc.driver_md("Norris", {"Norris": "McLaren"}, {"McLaren": "#ff8000"})
    assert text == ':color[Norris]{foreground="' + wc.readable_color("#ff8000") + '"}'


# --- readable colors and cell styles ---------------------------------------------

def test_readable_color_passes_high_contrast_through():
    assert wc.readable_color("#000000") == "#000000"


def test_readable_color_adjusts_low_contrast_for_light_mode():
    adjusted = wc.readable_color("#ffff00")  # pure yellow fails on white
    assert adjusted != "#ffff00"
    assert wc._contrast(wc._parse_hex(adjusted),
                        wc.LIGHT_BACKGROUND) >= wc.CONTRAST_TARGET


def test_readable_color_adjusts_for_dark_mode():
    adjusted = wc.readable_color("#404040", dark=True)
    assert adjusted != "#404040"
    assert wc._contrast(wc._parse_hex(adjusted),
                        wc.DARK_BACKGROUND) >= wc.CONTRAST_TARGET


def test_readable_color_falls_back_on_garbage_input():
    # the gray fallback is itself readability-adjusted like any team color
    for garbage in ("not-a-color", None):
        assert wc.readable_color(garbage) == wc.readable_color("#999999")
        assert re.fullmatch(r"#[0-9a-f]{6}", wc.readable_color(garbage))


def test_position_css_styles_the_podium():
    assert "FFD700" in wc.position_css(1)
    assert "C0C0C0" in wc.position_css(2.0)
    assert "CD7F32" in wc.position_css(3)
    assert wc.position_css(4) == ""
    assert wc.position_css("DNF") == ""


def test_zero_error_css_flags_exact_predictions():
    assert "b7f0c8" in wc.zero_error_css(0)
    assert wc.zero_error_css(0.0) == wc.zero_error_css(0)
    assert wc.zero_error_css(2) == ""


def test_probability_css_ramps_with_probability():
    low = wc.probability_css(0.1, dark=False)
    high = wc.probability_css(0.9, dark=False)
    for css in (low, high):
        assert "background-color: rgba(" in css
        assert "color: #" in css
    alpha_low = float(low.split("rgba(")[1].rsplit(")", 1)[0].split(", ")[3])
    alpha_high = float(high.split("rgba(")[1].rsplit(")", 1)[0].split(", ")[3])
    assert alpha_low < alpha_high  # the fill opacity tracks the probability


def test_css_helpers_handle_nan():
    assert wc.probability_css(float("nan")) == ""
    assert wc.position_css(float("nan")) == ""
    assert wc.zero_error_css(float("nan")) == ""


# --- Streamlit context fail-opens -------------------------------------------------

def test_context_helpers_fail_open_outside_a_script_run():
    assert wc.on_community_cloud() is False
    assert wc.theme_is_dark() is False


# --- charts ------------------------------------------------------------------------

def test_predicted_order_chart_builds_offline():
    gain = pd.DataFrame({"driver": ["ALO", "HAM"], "team": ["A", "B"],
                         "pred_pos": [1.0, 2.0]})
    actual = pd.DataFrame({"driver": ["HAM", "ALO"], "team": ["B", "A"],
                           "finish": [1.0, 2.0]})
    chart = wc.predicted_order_chart(
        [("Gain model", gain, "pred_pos"), ("Actual result", actual, "finish")],
        cut_lines={3.5: "podium", 10.5: "points"},
        colors={"A": "#ff0000", "B": "#0000ff"})
    assert chart.to_dict()  # the layered chart compiles


def test_probability_chart_builds_offline():
    table = pd.DataFrame({"driver": ["A", "B", "C"], "team": ["T1", "T2", "T1"],
                          "probability": [0.5, 0.3, 0.2]})
    chart = wc.probability_chart(table, {"T1": "#ff0000", "T2": "#0000ff"})
    assert chart.to_dict()
