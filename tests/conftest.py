"""Test bootstrap: import the project modules the same way the pages and
the CLIs resolve them — the project is not an installed package, so the
repo root, pipelines/, webapp/ and scripts/ go on sys.path (mirroring the
pages' own bootstrap in webapp/)."""

import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]

for _p in (REPO_ROOT, REPO_ROOT / "pipelines", REPO_ROOT / "webapp",
           REPO_ROOT / "scripts"):
    if str(_p) not in sys.path:
        sys.path.insert(0, str(_p))
