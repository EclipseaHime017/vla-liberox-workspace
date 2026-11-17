"""The installer must import its pinned revision before third-party packages exist."""
import subprocess
import sys
import importlib.util
import types

import pytest

from vla_rynn_iql.config import PROJECT_ROOT


def test_installer_revision_import_needs_only_the_standard_library():
    code = f"""
import sys
sys.path.insert(0, {str(PROJECT_ROOT / 'src')!r})
from vla_rynn_iql.pi05_assets import OPENPI_COMMIT
assert len(OPENPI_COMMIT) == 40
assert not any(name in sys.modules for name in ('yaml', 'numpy', 'torch'))
"""
    subprocess.run([sys.executable, "-S", "-c", code], check=True)


def installer(monkeypatch):
    # Only parsing TOML needs Python 3.11; this unit isolates the version guard.
    if importlib.util.find_spec("tomllib") is None:
        monkeypatch.setitem(sys.modules, "tomllib", types.ModuleType("tomllib"))
    spec = importlib.util.spec_from_file_location("setup_pi05", PROJECT_ROOT / "scripts/setup_pi05.py")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_preserved_package_must_match_the_locked_version(monkeypatch):
    module = installer(monkeypatch)
    monkeypatch.setattr(module.importlib.metadata, "version", lambda name: "0.23.1")
    lock = {"package": [{"name": "rerun-sdk", "version": "0.23.1"}]}
    assert module.preserved_packages(["rerun-sdk"], lock) == ["--no-install-package", "rerun-sdk"]
    with pytest.raises(ValueError, match="No unique locked distribution"):
        module.preserved_packages(["unknown"], lock)
    monkeypatch.setattr(module.importlib.metadata, "version", lambda name: "0.24.0")
    with pytest.raises(ValueError, match="upstream lock requires"):
        module.preserved_packages(["rerun-sdk"], lock)
