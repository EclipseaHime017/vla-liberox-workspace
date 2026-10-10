"""Default checkout locations and optional evaluator discovery stay in sync."""

import json
from pathlib import Path
from types import SimpleNamespace

import pytest
import yaml

from backend.app.services.offline_job_service import OfflineJobService


WORKSPACE = Path(__file__).resolve().parents[3]


@pytest.mark.parametrize("filename,section,expected", [
    ("configs/config.yaml", None,
     {"vla_root": "VLA-Adapter", "liberox_root": "LIBERO-X"}),
    ("vla-adapter-rynn-iql/configs/runtime.yaml", "paths",
     {"vla_adapter_root": "VLA-Adapter", "libero_x_root": "LIBERO-X", "rynnvalue_root": "RynnValue"}),
    ("vla-adapter-rynn-iql/configs/inference.yaml", "paths",
     {"vla_adapter_root": "VLA-Adapter", "libero_x_root": "LIBERO-X"}),
    ("vla-adapter-robometer/configs/robometer_evaluation.yaml", "paths",
     {"robometer_root": "Robometer"}),
])
def test_default_checkouts_live_under_third_party(filename, section, expected):
    path = WORKSPACE / filename
    config = yaml.safe_load(path.read_text())
    paths = config[section] if section else config
    for key, name in expected.items():
        assert (path.parent / paths[key]).resolve() == WORKSPACE / "third_party" / name


@pytest.mark.parametrize("absolute", [False, True])
def test_robometer_discovery_uses_its_configured_checkout(tmp_path, monkeypatch, absolute):
    root = tmp_path / "integration"
    config = root / "configs" / "robometer_evaluation.yaml"
    config.parent.mkdir(parents=True)
    checkout = tmp_path / "third_party" / "custom-robometer"
    package = checkout / "robometer" / "__init__.py"
    package.parent.mkdir(parents=True)
    package.touch()
    config.write_text(yaml.safe_dump({"paths": {
        "robometer_root": str(checkout) if absolute else "../../third_party/custom-robometer",
    }}))
    jobs = object.__new__(OfflineJobService)
    jobs.ui_config = SimpleNamespace(robometer_root=root, robometer_environment="robometer-reward")
    monkeypatch.setattr("backend.app.services.offline_job_service.subprocess.run", lambda *a, **kw:
                        SimpleNamespace(stdout=json.dumps({"envs": ["/envs/robometer-reward"]})))
    assert jobs.evaluator_capabilities()["robometer"] == {"available": True, "reason": None}
    package.unlink()
    result = jobs.evaluator_capabilities()
    assert result["rynnvalue"]["available"]
    assert not result["robometer"]["available"]
    assert str(checkout) in result["robometer"]["reason"]


def test_missing_robometer_config_does_not_disable_rynnvalue(tmp_path):
    jobs = object.__new__(OfflineJobService)
    jobs.ui_config = SimpleNamespace(robometer_root=tmp_path)
    result = jobs.evaluator_capabilities()
    assert result["rynnvalue"]["available"]
    assert not result["robometer"]["available"]
    assert "config" in result["robometer"]["reason"]
