"""Reuse the training suite's data fixtures without copying its setup logic."""
import importlib.util
from pathlib import Path
import sys

PROJECT = Path(__file__).resolve().parents[2]
for path in (PROJECT, PROJECT / "src", PROJECT / "tests"):
    sys.path.insert(0, str(path))

spec = importlib.util.spec_from_file_location("_hpc_training_fixtures", PROJECT / "tests/conftest.py")
fixtures = importlib.util.module_from_spec(spec)
spec.loader.exec_module(fixtures)
_episode = fixtures._episode
configured = fixtures.configured
isolated_model_digest_cache = fixtures.isolated_model_digest_cache
