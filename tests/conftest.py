"""Test bootstrap: import the project modules the same way the pages and
the CLIs resolve them — the project is not an installed package, so the
repo root, pipelines/, webapp/ and scripts/ go on sys.path (mirroring the
pages' own bootstrap in webapp/)."""

import sys
from pathlib import Path

import pytest

REPO_ROOT = Path(__file__).resolve().parents[1]

for _p in (REPO_ROOT, REPO_ROOT / "pipelines", REPO_ROOT / "webapp",
           REPO_ROOT / "scripts"):
    if str(_p) not in sys.path:
        sys.path.insert(0, str(_p))

import f1_common
import predict_fantasy


@pytest.fixture(autouse=True)
def _isolated_model_cache(tmp_path):
    """Every pipeline fit goes through f1_common's fitted-model cache;
    point it at a per-test directory so the suite never reads or writes the
    repo's real cache/models (the cache's own behavior is tested with this
    same module attribute in test_f1_common)."""
    f1_common.MODEL_CACHE_DIR = tmp_path / "models"
    yield


@pytest.fixture(autouse=True)
def _no_history_pool():
    """The fantasy pipeline's cross-season training pool reads the bundled
    data/season_*.csv files; disable it so the suite stays offline and
    trains only on the synthetic frames each test builds."""
    predict_fantasy.HISTORY_POOL_SEASONS = ()
    yield
