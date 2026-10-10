"""Schema and sanity checks for the bundled data/ CSVs — what a fresh
container (and every CI run) resolves from. A schema drift here is what
makes collect_season discard a season CSV and re-download it, and a broken
bundle breaks every cold app start, so the tests reuse the exact same
required-column definition as scripts/refresh_data.py."""

import re

import pandas as pd

import f1_common
import refresh_data  # scripts/refresh_data.py, on sys.path via conftest

BUNDLED_KINDS = {"season_": "race", "quali_season_": "grid",
                 "extras_season_": "extras"}


def _bundled(prefix):
    return sorted(f1_common.DATA_DIR.glob(f"{prefix}*.csv"))


def test_the_repo_bundles_season_data():
    assert _bundled("season_")
    assert _bundled("quali_season_")
    assert _bundled("extras_season_")


def test_every_bundled_season_has_the_required_schema():
    for prefix, kind in BUNDLED_KINDS.items():
        for path in _bundled(prefix):
            frame = pd.read_csv(path, dtype={"driver_number": str})
            missing = refresh_data.required_columns(kind) - set(frame.columns)
            assert not missing, f"{path.name}: missing {sorted(missing)}"


def test_bundled_seasons_have_full_grids():
    for prefix in BUNDLED_KINDS:
        for path in _bundled(prefix):
            frame = pd.read_csv(path)
            counts = frame.groupby("round").size()
            assert len(counts) > 0, path.name
            smallest = counts.idxmin()
            assert counts.min() >= 10, \
                f"{path.name}: round {smallest} looks truncated ({counts.min()} rows)"


def test_team_colors_bundle_is_wellformed():
    frame = pd.read_csv(f1_common.TEAM_COLORS_CSV)
    assert list(frame.columns) == ["year", "team", "color"]
    assert frame["color"].str.fullmatch(r"#[0-9a-fA-F]{6}").all()
    per_year = frame.groupby("year").size()
    assert per_year.min() >= 8  # an F1 grid has ten teams
    # every bundled season must be covered: a cold container renders colors
    # for any selectable year with zero fastf1 API calls
    for path in _bundled("season_"):
        year = int(path.stem.split("_")[1])
        assert year in set(per_year.index), f"{path.name} has no bundled colors"
