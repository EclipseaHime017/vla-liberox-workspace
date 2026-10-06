from __future__ import annotations

import base64
import io
import json
import queue
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest
import yaml

from backend.app.policies.catalog import PolicyCatalog, _sha256, _stable_hash
from backend.app.policies.pi05 import Pi05PolicyProvider
from backend.app.policies.pi05_catalog import assets, configured_model
from backend.app.services.policy_management_service import PolicyManagementService


def checkpoint(tmp_path):
    root = tmp_path / "pi05"
    norm = root / assets().NORM_FILE
    norm.parent.mkdir(parents=True)
    norm.write_text("{}")
    (root / "model.safetensors").write_bytes(b"fake-weights-for-contract-test")
    identity = {"schema_version": 1, "family": "pi05", "openpi_commit": assets().OPENPI_COMMIT,
                "source": assets().OFFICIAL_CHECKPOINT, "config_name": "pi05_libero",
                "native_action_horizon": 10, "native_action_dim": 32, "discrete_state_input": False,
                "files": {name: _sha256(root / name) for name in ("model.safetensors", assets().NORM_FILE)}}
    (root / assets().IDENTITY_FILE).write_text(json.dumps(identity))
    settings = {**configured_model(), "base_checkpoint": str(root)}
    return settings, identity


def test_catalog_lists_without_loading_and_revalidates_selected_assets(tmp_path):
    settings, _ = checkpoint(tmp_path)
    catalog = PolicyCatalog(tmp_path / "registry", "unused-vla", "libero_object", pi05_model=settings)
    assert [item["family"] for item in catalog.list_policies()] == ["vla_adapter", "pi05"]
    selected = catalog.select("pi05-libero-base")
    assert selected.base_revision and selected.is_base
    (Path(settings["base_checkpoint"]) / assets().NORM_FILE).write_text("changed")
    catalog.list_policies()  # Lists are lightweight; selection performs content validation.
    with pytest.raises(ValueError, match="hash mismatch"):
        catalog.select("pi05-libero-base")


def test_pi_overlay_roundtrip_and_base_protection(tmp_path):
    settings, identity = checkpoint(tmp_path)
    registry = tmp_path / "registry"
    directory = registry / "pi-test"
    directory.mkdir(parents=True)
    (directory / "actor.pt").write_bytes(b"fake-deltas")
    (directory / "pi05_identity.json").write_text(json.dumps(identity))
    compatibility = {"base_checkpoint": settings["base_checkpoint"], "stats_key": settings["stats_key"],
                     "action_horizon": 8, "action_dim": 7, "proprio_dim": 8}
    payload = {"schema_version": 4, "family": "pi05", "algorithm": "bc", "model_config": settings,
               "policy_id": "pi-test", "label": "test", "actor": "actor.pt", "base_identity": "pi05_identity.json",
               "dataset_sha256": "d" * 64, "reward_sha256": None, "training_step": 1,
               "component_sha256": {key: _sha256(directory / name) for key, name in
                                    (("actor", "actor.pt"), ("base_identity", "pi05_identity.json"))},
               **compatibility, "compatibility_sha256": _stable_hash(compatibility)}
    (directory / "policy.yaml").write_text(yaml.safe_dump(payload))
    catalog = PolicyCatalog(registry, "unrelated-vla", "libero_object", pi05_model=settings)
    entry = catalog.select("pi-test")
    assert entry.family == "pi05" and entry.action_head is None and entry.actor.is_file()
    assert catalog.entry("pi-test").content_sha256 == entry.content_sha256
    management = object.__new__(PolicyManagementService)
    management.catalog = catalog
    management._entry = catalog.entry
    with pytest.raises(Exception, match="read-only"):
        management._assert_mutable("pi05-libero-base")
    (directory / "actor.pt").write_bytes(b"changed")
    catalog.refresh()
    with pytest.raises(ValueError, match="hash mismatch"):
        catalog.entry("pi-test")


def test_checkpoint_binds_conversion_precision(tmp_path):
    settings, identity = checkpoint(tmp_path)
    root = Path(settings["base_checkpoint"])
    original = assets().identity_digest(assets().checkpoint_identity(root))
    identity["conversion_precision"] = "bfloat16"
    (root / assets().IDENTITY_FILE).write_text(json.dumps(identity))
    assert assets().identity_digest(assets().checkpoint_identity(root)) != original
    identity["conversion_precision"] = "unknown"
    (root / assets().IDENTITY_FILE).write_text(json.dumps(identity))
    with pytest.raises(ValueError, match="conversion precision"):
        assets().checkpoint_identity(root)


def test_native_inputs_camera_mask_codec_and_horizon(monkeypatch):
    provider = Pi05PolicyProvider(None, None, None)
    provider.process = SimpleNamespace(poll=lambda: None)
    provider.current_policy_entry = object()
    agent = np.arange(12, dtype=np.uint8).reshape(2, 2, 3)
    obs = {"agentview_image": agent, "robot0_eye_in_hand_image": agent + 1,
           "robot0_eef_pos": [1, 2, 3], "robot0_eef_quat": [0, 0, 0, 1], "robot0_gripper_qpos": [.1, .2]}
    captured = {}
    def send(request):
        captured.update(request)
        return {"actions": np.ones((10, 7)).tolist()}
    monkeypatch.setattr(provider, "_send", send)
    assert provider.predict(obs, "test", ("robot0_eye_in_hand",)).shape == (8, 7)
    with np.load(io.BytesIO(base64.b64decode(captured["arrays"]))) as inputs:
        assert np.array_equal(inputs["observation/image"], agent[::-1, ::-1])
        assert not inputs["observation/wrist_image"].any()
        assert np.allclose(inputs["observation/state"], [1, 2, 3, 0, 0, 0, .1, .2])
    assert np.array_equal(provider.process_action(np.array([2, -2, 0, 0, 0, 0, -.4])), [1, -1, 0, 0, 0, 0, np.float32(-.4)])


def test_worker_timeout_unloads_and_does_not_fallback(monkeypatch):
    provider = Pi05PolicyProvider(None, None, None)
    provider.responses = queue.Queue()
    called = []
    monkeypatch.setattr(provider, "unload", lambda: called.append(True))
    with pytest.raises(TimeoutError):
        provider._receive(.001)
    assert called == [True]


def test_same_weights_cache_hit_keeps_selected_policy_identity():
    selected = SimpleNamespace(family="pi05", policy_id="copy", label="Copy", content_sha256="a" * 64)
    provider = Pi05PolicyProvider(None, None, SimpleNamespace(select=lambda _: selected))
    provider.process = SimpleNamespace(poll=lambda: None)
    provider.current_policy_entry = SimpleNamespace(content_sha256=selected.content_sha256)
    provider.current_policy_id = "original"
    provider.load(8, "copy", expected_content_sha256=selected.content_sha256)
    assert provider.current_policy_id == "copy"
    assert provider.current_policy_entry is selected


def test_ui_training_pins_weights_and_selects_pi_environment(tmp_path, monkeypatch):
    from test_dataset_reward_versions import setup_jobs
    jobs, dataset = setup_jobs(tmp_path / "jobs")
    settings, identity = checkpoint(tmp_path)
    load = jobs._load_base_config
    def config(algorithm=None, model_family=None):
        raw = load(algorithm, model_family)
        if model_family == "pi05":
            raw["model"] = settings.copy()
        return raw
    monkeypatch.setattr(jobs, "_load_base_config", config)
    job = jobs.start_training(dataset["id"], {"algorithm": "bc", "model_family": "pi05"})
    sealed = yaml.safe_load(Path(job["config_path"]).read_text())
    assert sealed["model"]["base_revision"] == assets().identity_digest(identity)
    assert next(stage for stage in job["stages"] if stage["id"] == "train")["environment"] == "pi05"
