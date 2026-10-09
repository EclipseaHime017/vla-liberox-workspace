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
from backend.app.policies.pi05_catalog import assets
from backend.app.policies.registry import configured_models
from backend.app.services.policy_management_service import PolicyManagementService


def checkpoint(tmp_path, base_id="pi05-libero-base"):
    base = assets().base_model(base_id)
    root = tmp_path / base_id
    norm = root / base.norm_file
    norm.parent.mkdir(parents=True)
    norm.write_text("{}")
    (root / "model.safetensors").write_bytes(b"fake-weights-for-contract-test")
    identity = {"schema_version": 1, "family": "pi05", "openpi_commit": assets().OPENPI_COMMIT,
                "source": base.source, "config_name": base.config_name,
                "native_action_horizon": 10, "native_action_dim": 32, "discrete_state_input": False,
                "files": {name: _sha256(root / name) for name in ("model.safetensors", base.norm_file)}}
    if base.revision:
        identity["source_revision"] = base.revision
    (root / assets().IDENTITY_FILE).write_text(json.dumps(identity))
    settings = {**configured_models()[base_id], "base_checkpoint": str(root)}
    return settings, identity


def test_catalog_lists_without_loading_and_revalidates_selected_assets(tmp_path):
    settings, _ = checkpoint(tmp_path)
    catalog = PolicyCatalog(tmp_path / "registry", base_models={settings["base_id"]: settings})
    assert [item["family"] for item in catalog.list_policies()] == ["pi05"]
    selected = catalog.select("pi05-libero-base")
    assert selected.base_revision and selected.is_base
    (Path(settings["base_checkpoint"]) / assets().base_model(settings["base_id"]).norm_file).write_text("changed")
    catalog.list_policies()  # Lists are lightweight; selection performs content validation.
    with pytest.raises(ValueError, match="hash mismatch"):
        catalog.select("pi05-libero-base")


@pytest.mark.parametrize("base_id,legacy,schema", [
    ("pi05-libero-base", False, 5), ("pi05-liberox-base", False, 5),
    ("pi05-libero-base", False, 4), ("pi05-libero-base", True, 4),
    ("pi05-liberox-base", False, 4),
])
def test_pi_overlay_roundtrip_and_base_protection(tmp_path, base_id, legacy, schema):
    settings, identity = checkpoint(tmp_path, base_id)
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
    if legacy:
        payload["model_config"] = {key: value for key, value in settings.items() if key != "base_id"}
    if schema == 5:
        from backend.app.policies.registry import model_module
        payload.update(schema_version=5, parent=model_module("model_artifacts").parent_snapshot(
            settings, assets().identity_digest(identity)))
    (directory / "policy.yaml").write_text(yaml.safe_dump(payload))
    catalog = PolicyCatalog(registry, base_models={base_id: settings})
    entry = catalog.select("pi-test")
    assert entry.family == "pi05" and entry.action_head is None and entry.actor.is_file()
    assert catalog.entry("pi-test").content_sha256 == entry.content_sha256
    assert (entry.parent is not None) == (schema == 5)
    if legacy and schema == 4:
        old_hash = _stable_hash({"family": "pi05", "base_checkpoint": settings["base_checkpoint"],
                                "stats_key": settings["stats_key"], "model_config": payload["model_config"],
                                "component_sha256": payload["component_sha256"],
                                "base_revision": assets().identity_digest(identity)})
        assert entry.content_sha256 == old_hash
    management = object.__new__(PolicyManagementService)
    management.catalog = catalog
    management._entry = catalog.entry
    with pytest.raises(Exception, match="read-only"):
        management._assert_mutable(base_id)
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
    provider.current_policy_entry = SimpleNamespace(settings=configured_models()["pi05-libero-base"])
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


@pytest.mark.parametrize("base_id", ["pi05-libero-base", "pi05-liberox-base"])
def test_ui_training_pins_weights_and_selects_pi_environment(tmp_path, monkeypatch, base_id):
    from test_dataset_reward_versions import setup_jobs
    jobs, dataset = setup_jobs(tmp_path / "jobs")
    settings, identity = checkpoint(tmp_path, base_id)
    load = jobs._load_base_config
    def config(algorithm=None, model_family=None):
        raw = load(algorithm, model_family)
        if model_family == "pi05":
            raw["model"] = settings.copy()
        return raw
    monkeypatch.setattr(jobs, "_load_base_config", config)
    job = jobs.start_training(dataset["id"], {"algorithm": "bc", "model_family": "pi05", "model_base_id": base_id})
    sealed = yaml.safe_load(Path(job["config_path"]).read_text())
    assert sealed["model"]["base_revision"] == assets().identity_digest(identity)
    assert next(stage for stage in job["stages"] if stage["id"] == "train")["environment"] == "pi05"


def test_multiple_bases_are_lightweight_and_wrong_source_is_rejected(tmp_path, monkeypatch):
    official, _ = checkpoint(tmp_path)
    liberox, identity = checkpoint(tmp_path, "pi05-liberox-base")
    catalog = PolicyCatalog(tmp_path / "registry",
                            base_models={item["base_id"]: item for item in (official, liberox)})
    monkeypatch.setattr(catalog, "_component_sha256", lambda _: pytest.fail("listing hashed a model"))
    assert [item["policy_id"] for item in catalog.list_policies()] == ["pi05-libero-base", "pi05-liberox-base"]
    monkeypatch.undo()
    chosen = catalog.select("pi05-liberox-base")
    assert chosen.is_base and chosen.base_revision == assets().identity_digest(identity)
    assert chosen.stats_key == "meituan/LIBERO-X"
    # A valid official checkpoint is still the wrong checkpoint for LIBERO-X.
    wrong = {**liberox, "base_checkpoint": official["base_checkpoint"]}
    bad_catalog = PolicyCatalog(tmp_path / "registry", base_models={wrong["base_id"]: wrong})
    with pytest.raises(ValueError, match="identity does not match"):
        bad_catalog.select("pi05-liberox-base")
    root = Path(liberox["base_checkpoint"])
    identity["source_revision"] = "0" * 40
    (root / assets().IDENTITY_FILE).write_text(json.dumps(identity))
    with pytest.raises(ValueError, match="identity does not match"):
        catalog.select("pi05-liberox-base")


def test_training_defaults_select_liberox_without_loading_weights(tmp_path):
    from test_dataset_reward_versions import setup_jobs
    jobs, dataset = setup_jobs(tmp_path)
    defaults = jobs.defaults(dataset["id"], algorithm="bc", model_family="pi05", model_base_id="pi05-liberox-base")
    assert defaults["model"]["model_base_id"] == "pi05-liberox-base"
    with pytest.raises(ValueError):
        jobs.defaults(algorithm="bc", model_family="vla_adapter", model_base_id="pi05-liberox-base")
