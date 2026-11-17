"""Default checkout locations and optional evaluator discovery stay in sync."""

import json
from pathlib import Path
from types import SimpleNamespace

import pytest
import yaml

from studio.services.offline_job_service import OfflineJobService


WORKSPACE = Path(__file__).resolve().parents[2]


@pytest.mark.parametrize("filename,section,expected", [
    ("configs/studio/studio.yaml", None,
     {"vla_root": "VLA-Adapter", "liberox_root": "LIBERO-X"}),
    ("configs/runtime.yaml", "paths",
     {"vla_adapter_root": "VLA-Adapter", "libero_x_root": "LIBERO-X", "rynnvalue_root": "RynnValue"}),
    ("configs/inference.yaml", "paths",
     {"vla_adapter_root": "VLA-Adapter", "libero_x_root": "LIBERO-X"}),
    ("configs/rewards/robometer.yaml", "paths",
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
    config = root / "configs" / "rewards" / "robometer.yaml"
    config.parent.mkdir(parents=True)
    checkout = tmp_path / "third_party" / "custom-robometer"
    package = checkout / "robometer" / "__init__.py"
    package.parent.mkdir(parents=True)
    package.touch()
    config.write_text(yaml.safe_dump({"paths": {
        "robometer_root": str(checkout) if absolute else "../../../third_party/custom-robometer",
    }}))
    jobs = object.__new__(OfflineJobService)
    jobs.ui_config = SimpleNamespace(workspace_root=root, robometer_environment="robometer-reward")
    monkeypatch.setattr("studio.services.offline_job_service.subprocess.run", lambda *a, **kw:
                        SimpleNamespace(stdout=json.dumps({"envs": ["/envs/robometer-reward"]})))
    assert jobs.evaluator_capabilities()["robometer"] == {"available": True, "reason": None}
    package.unlink()
    result = jobs.evaluator_capabilities()
    assert result["rynnvalue"]["available"]
    assert not result["robometer"]["available"]
    assert str(checkout) in result["robometer"]["reason"]


def test_missing_robometer_config_does_not_disable_rynnvalue(tmp_path):
    jobs = object.__new__(OfflineJobService)
    jobs.ui_config = SimpleNamespace(workspace_root=tmp_path)
    result = jobs.evaluator_capabilities()
    assert result["rynnvalue"]["available"]
    assert not result["robometer"]["available"]
    assert "config" in result["robometer"]["reason"]
