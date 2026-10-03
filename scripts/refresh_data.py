"""Refresh the bundled season CSVs (data/*.csv).

The web app and both CLI pipelines reuse these files, so a fresh container
never re-downloads whole seasons (fastf1 hard-stops at 500 uncached API
calls/hour, which a cold full-season download would exceed). The scheduled
GitHub workflow (.github/workflows/refresh-data.yml) runs this script to
keep the bundled data current; it can also be run locally for one-off bulk
generation of past seasons, e.g.:

    python scripts/refresh_data.py --years 2018 2019 2020 --wait-on-limit

Per season the race pipeline is collected before the grid pipeline so the
grid pipeline's session loads are served from the fastf1 cache the race
pipeline just filled. With --wait-on-limit, hitting the rate limit makes
collection sleep and retry instead of skipping rounds, so bulk generation
takes longer but completes on its own. Exits non-zero when rounds are still
missing (progress is saved either way; simply run again later).
"""

import argparse
import sys
from datetime import datetime
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "pipelines"))

import pandas as pd
import fastf1

import f1_common
import predict_grid
import predict_race

CSV_NAMES = {"race": "season_{year}.csv", "grid": "quali_season_{year}.csv"}
MODULES = {"race": predict_race, "grid": predict_grid}


def missing_rounds(kind, year, schedule):
    """Completed rounds of `year` that the pipeline's season CSV is missing."""
    path = f1_common.DATA_DIR / CSV_NAMES[kind].format(year=year)
    completed = {rn for rn, _ in f1_common.completed_rounds(schedule)}
    if not completed:
        return []
    have = set()
    if path.exists():
        frame = pd.read_csv(path, dtype={"driver_number": str})
        if "round" in frame.columns:
            have = set(frame["round"].unique())
    return sorted(completed - have)


def collect_year(year, wait_on_limit):
    schedule = fastf1.get_event_schedule(year, include_testing=False)
    if not f1_common.completed_rounds(schedule):
        print(f"Season {year}: no completed rounds yet — nothing to collect")
        return True
    ok = True
    for kind in ("race", "grid"):
        left = missing_rounds(kind, year, schedule)
        if not left:
            print(f"Season {year} {kind}: already complete")
            continue
        print(f"Season {year} {kind}: {len(left)} round(s) to fetch")
        MODULES[kind].collect_season(year, schedule, rate_limit_wait=wait_on_limit)
        left = missing_rounds(kind, year, schedule)
        if left:
            print(f"Season {year} {kind}: {len(left)} round(s) still missing — run again later")
            ok = False
        else:
            print(f"Season {year} {kind}: complete")
    return ok


def main():
    parser = argparse.ArgumentParser(
        description="Refresh the bundled season CSVs (data/*.csv) for both pipelines.")
    parser.add_argument("--years", type=int, nargs="+", default=None,
                        help="seasons to collect (default: previous and current season)")
    parser.add_argument("--wait-on-limit", action="store_true",
                        help="sleep and retry when the F1 API rate limit is hit, "
                             "instead of skipping the remaining rounds")
    args = parser.parse_args()

    f1_common.setup()
    years = args.years or [datetime.now().year - 1, datetime.now().year]
    wait = f1_common.RATE_LIMIT_WAIT_SECS if args.wait_on_limit else None
    ok = True
    for year in sorted(set(years)):
        ok = collect_year(year, wait) and ok
    if not ok:
        print("\nSome rounds are still missing — progress is saved; run again later.")
        sys.exit(1)


if __name__ == "__main__":
    main()
