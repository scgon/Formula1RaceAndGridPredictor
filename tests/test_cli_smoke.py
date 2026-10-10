"""The cheap import/argparse smoke test: --help exercises every module's
full import chain (fastf1, sklearn, the shared machinery) without loading
any session data, exactly as documented in AGENTS.md."""

import subprocess
import sys
from pathlib import Path

import pytest

REPO_ROOT = Path(__file__).resolve().parents[1]

PIPELINES = [
    "pipelines/predict_race.py",
    "pipelines/predict_grid.py",
    "pipelines/predict_extras.py",
]


@pytest.mark.parametrize("script", PIPELINES)
def test_pipeline_cli_help(script):
    result = subprocess.run(
        [sys.executable, "-u", script, "--help"],
        capture_output=True, text=True, cwd=REPO_ROOT, timeout=120,
    )
    assert result.returncode == 0, result.stderr
    assert "usage:" in result.stdout
