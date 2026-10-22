"""Shared pytest hooks.

``@pytest.mark.data("relative/path", ...)`` marks a test that needs project data which is not
part of the source tree (frozen runs, track cards, benchmark outputs, analysed audio). Paths are
relative to the data root (``$MUSICMIX_DATA``, default: the checkout); the test is skipped when
any of them is missing, and the whole group can be deselected with ``-m "not data"``.
"""
from __future__ import annotations

from pathlib import Path
import sys

import pytest

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT / "src") not in sys.path:
    sys.path.insert(0, str(ROOT / "src"))

from musicmix.paths import data_root  # noqa: E402


def pytest_collection_modifyitems(config, items):
    base = data_root()
    for item in items:
        for marker in item.iter_markers(name="data"):
            missing = [p for p in marker.args if not ((base / p).exists() or (ROOT / p).exists())]
            if missing:
                item.add_marker(pytest.mark.skip(reason=f"dati di progetto assenti: {', '.join(missing)}"))
                break
