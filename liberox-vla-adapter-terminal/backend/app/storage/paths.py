"""Shared, dependency-free recording relocation support."""
import importlib.util
import sys
from pathlib import Path

_NAME = "_vla_storage_paths"
if _NAME not in sys.modules:
    _path = Path(__file__).resolve().parents[4] / "vla-adapter-rynn-iql/src/vla_rynn_iql/storage_paths.py"
    _spec = importlib.util.spec_from_file_location(_NAME, _path)
    _module = importlib.util.module_from_spec(_spec)
    sys.modules[_NAME] = _module
    _spec.loader.exec_module(_module)

storage_path = sys.modules[_NAME].storage_path
storage_lease = sys.modules[_NAME].storage_lease
clear_storage_cache = sys.modules[_NAME].clear_storage_cache
