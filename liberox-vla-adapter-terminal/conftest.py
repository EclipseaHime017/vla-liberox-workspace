from __future__ import annotations

import sys
from pathlib import Path

import pytest


PROJECT_ROOT = Path(__file__).resolve().parent
for path in (PROJECT_ROOT, PROJECT_ROOT / "scripts"):
    if str(path) not in sys.path:
        sys.path.insert(0, str(path))


@pytest.fixture(autouse=True)
def _isolated_digest_cache(tmp_path_factory, monkeypatch):
    """Model digests computed by tests must never reach models/.cache/digests."""
    monkeypatch.setenv("VLA_DIGEST_CACHE_ROOT", str(tmp_path_factory.mktemp("digests")))
