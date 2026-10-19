"""Also isolate cache writes when backend/pyproject.toml is the pytest root."""
import sys
from pathlib import Path

import pytest

PLATFORM_ROOT = Path(__file__).resolve().parents[2]
for path in (PLATFORM_ROOT, PLATFORM_ROOT / "scripts"):
    if str(path) not in sys.path:
        sys.path.insert(0, str(path))


@pytest.fixture(autouse=True)
def _isolated_digest_cache(tmp_path_factory, monkeypatch):
    monkeypatch.setenv("VLA_DIGEST_CACHE_ROOT", str(tmp_path_factory.mktemp("digests")))
