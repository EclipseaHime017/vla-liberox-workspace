from __future__ import annotations

import hashlib
import json
import threading
from pathlib import Path
from types import SimpleNamespace

import pytest
import torch
import yaml

from backend.app.policies.catalog import PolicyCatalog
from backend.app.policies.vla_adapter import VLAAdapterPolicyProvider
from backend.app.services.policy_management_service import PolicyManagementService


BASE = "VLA-Adapter/LIBERO-Object-Pro"
STATS = "libero_object_no_noops"


def _sha(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _stable(value) -> str:
    return hashlib.sha256(
        json.dumps(value, sort_keys=True, separators=(",", ":")).encode()
    ).hexdigest()


def _overlay(root: Path, policy_id: str = "trained") -> Path:
    directory = root / policy_id
    directory.mkdir(parents=True)
    action = directory / "action_head.pt"
    proprio = directory / "proprio_projector.pt"
    torch.save(torch.nn.Linear(2, 2).state_dict(), action)
    torch.save(torch.nn.Linear(2, 2).state_dict(), proprio)
    compatibility = {
        "base_checkpoint": BASE,
        "stats_key": STATS,
        "action_horizon": 8,
        "action_dim": 7,
        "proprio_dim": 8,
    }
    manifest = {
        "schema_version": 1,
        "policy_id": policy_id,
        "label": "Rynn IQL test",
        **compatibility,
        "action_head": action.name,
        "proprio_projector": proprio.name,
        "dataset_sha256": "d" * 64,
        "reward_sha256": "e" * 64,
        "training_step": 20,
        "component_sha256": {
            "action_head": _sha(action),
            "proprio_projector": _sha(proprio),
        },
        "compatibility_sha256": _stable(compatibility),
    }
    path = directory / "policy.yaml"
    path.write_text(yaml.safe_dump(manifest, sort_keys=False), encoding="utf-8")
    return path


def test_policy_catalog_validates_and_lists_overlay(tmp_path: Path):
    _overlay(tmp_path)
    catalog = PolicyCatalog(tmp_path, BASE, "libero_object")
    policies = catalog.list_policies()
    assert [item["policy_id"] for item in policies] == ["base", "trained"]
    assert catalog.entry("trained").stats_key == STATS
    with pytest.raises(ValueError, match="Unknown policy_id"):
        catalog.entry("missing")


def test_bc_overlay_and_legacy_iql_are_both_discoverable(tmp_path: Path):
    _overlay(tmp_path, "legacy-iql")
    path = _overlay(tmp_path, "bc-test")
    raw = yaml.safe_load(path.read_text())
    raw.update(schema_version=2, algorithm="bc", reward_sha256=None)
    path.write_text(yaml.safe_dump(raw))
    catalog = PolicyCatalog(tmp_path, BASE, "libero_object")
    assert catalog.entry("bc-test").public()["algorithm"] == "bc"
    assert catalog.entry("legacy-iql").public()["algorithm"] == "iql"
    raw["reward_sha256"] = "e" * 64
    path.write_text(yaml.safe_dump(raw))
    catalog.refresh()
    with pytest.raises(ValueError, match="must not reference rewards"):
        catalog.entry("bc-test")


def test_policy_catalog_rejects_component_tampering(tmp_path: Path):
    manifest = _overlay(tmp_path)
    (manifest.parent / "action_head.pt").write_bytes(b"tampered")
    catalog = PolicyCatalog(tmp_path, BASE, "libero_object")
    assert [item["policy_id"] for item in catalog.list_policies()] == ["base"]
    with pytest.raises(ValueError, match="hash mismatch"):
        catalog.entry("trained")


def test_provider_switches_components_and_restores_base(tmp_path: Path):
    manifest = _overlay(tmp_path)
    catalog = PolicyCatalog(tmp_path, BASE, "libero_object")
    provider = object.__new__(VLAAdapterPolicyProvider)
    provider.runtime = SimpleNamespace(torch=torch)
    provider.components = SimpleNamespace(
        action_head=torch.nn.Linear(2, 2),
        proprio_projector=torch.nn.Linear(2, 2),
    )
    provider._base_action_head = provider._cpu_state(provider.components.action_head)
    provider._base_proprio_projector = provider._cpu_state(
        provider.components.proprio_projector
    )
    provider.current_policy_id = "base"
    original = provider.components.action_head.weight.detach().clone()
    provider._apply_policy(catalog.entry("trained"))
    assert provider.current_policy_id == "trained"
    assert not torch.equal(provider.components.action_head.weight, original)
    provider._apply_policy(catalog.entry("base"))
    assert provider.current_policy_id == "base"
    assert torch.equal(provider.components.action_head.weight, original)


def test_provider_revalidates_overlay_at_load_boundary(tmp_path: Path):
    manifest = _overlay(tmp_path)
    catalog = PolicyCatalog(tmp_path, BASE, "libero_object")
    provider = object.__new__(VLAAdapterPolicyProvider)
    provider.runtime = SimpleNamespace(torch=torch)
    provider.catalog = catalog
    provider.components = SimpleNamespace(
        action_head=torch.nn.Linear(2, 2),
        proprio_projector=torch.nn.Linear(2, 2),
    )
    provider.cfg = SimpleNamespace(num_open_loop_steps=8)
    provider.current_policy_id = "base"
    provider._lock = threading.RLock()
    (manifest.parent / "action_head.pt").write_bytes(b"changed-after-draft")
    with pytest.raises(ValueError, match="hash mismatch"):
        provider.load(8, "trained")


def test_policy_management_renames_copies_and_deletes_overlay(tmp_path: Path):
    _overlay(tmp_path)
    catalog = PolicyCatalog(tmp_path, BASE, "libero_object")
    manager = SimpleNamespace(
        policy_catalog=catalog, active_session_id=None, draft=None,
    )
    current = PolicyManagementService(manager)
    renamed = current.rename("trained", "Grasp tuning v2")
    assert renamed["label"] == "Grasp tuning v2"
    copied = current.copy("trained", "Grasp tuning copy")
    assert copied["policy_id"] != "trained"
    assert (tmp_path / copied["policy_id"] / "policy.yaml").is_file()
    assert current.delete(copied["policy_id"], copied["policy_id"]) == {
        "deleted": copied["policy_id"]
    }
    assert [item["policy_id"] for item in current.list()] == ["base", "trained"]
    with pytest.raises(Exception, match="read-only"):
        current.delete("base", "base")


def test_queued_overlay_mutations_are_locked_and_rejected(tmp_path: Path):
    from contextlib import contextmanager

    manifest = _overlay(tmp_path)
    catalog = PolicyCatalog(tmp_path, BASE, "libero_object")
    manager = SimpleNamespace(policy_catalog=catalog, active_session_id=None, draft=None)
    held = False

    @contextmanager
    def lock():
        nonlocal held
        held = True
        try:
            yield
        finally:
            held = False

    def jobs_list():
        assert held
        return [{"status": "QUEUED", "parameters": {"policy_id": "trained"}}]

    jobs = SimpleNamespace(list=jobs_list)
    current = PolicyManagementService(manager, jobs)
    before = manifest.read_bytes()
    jobs.lock = lock()
    with pytest.raises(Exception, match="active job"):
        current.rename("trained", "changed")
    jobs.lock = lock()
    with pytest.raises(Exception, match="active job"):
        current.delete("trained", "trained")
    assert manifest.read_bytes() == before


def _adapted_overlay(root, policy_id="adapted"):
    path = _overlay(root, policy_id)
    payload = yaml.safe_load(path.read_text())
    backbone = torch.nn.Linear(2, 2)
    torch.save(backbone.state_dict(), path.parent / "backbone.pt")
    payload.update(schema_version=3, algorithm="bc", reward_sha256=None, backbone="backbone.pt",
        model_config={"family": "vla_adapter", "backbone": "lora", "action_head": "train",
                      "proprio_projector": "frozen", "lora": {"rank": 32, "alpha": 64, "dropout": 0}})
    payload["component_sha256"]["backbone"] = _sha(path.parent / "backbone.pt")
    path.write_text(yaml.safe_dump(payload))
    return path


def test_adapted_policy_copy_contains_backbone_and_validates_hash(tmp_path):
    path = _adapted_overlay(tmp_path)
    catalog = PolicyCatalog(tmp_path, BASE, "libero_object")
    manager = SimpleNamespace(policy_catalog=catalog, active_session_id=None, draft=None)
    service = PolicyManagementService(manager)
    detail = service.detail("adapted")
    assert {item["name"] for item in detail["components"]} == {"backbone", "action_head", "proprio_projector"}
    copied = service.copy("adapted", "LoRA copy")
    assert catalog.entry(copied["policy_id"]).backbone.read_bytes() == (path.parent / "backbone.pt").read_bytes()
    (path.parent / "backbone.pt").write_bytes(b"corrupt")
    catalog.refresh()
    with pytest.raises(ValueError, match="backbone hash mismatch"):
        catalog.entry("adapted")


def test_switching_adapted_backbones_reloads_base_but_frozen_overlays_reuse_it(tmp_path, monkeypatch):
    import copy
    from backend.app.policies import vla_adapter
    path = _adapted_overlay(tmp_path)
    _overlay(tmp_path, "head-only")
    catalog = PolicyCatalog(tmp_path, BASE, "libero_object")
    initial = SimpleNamespace(model=torch.nn.Linear(2, 2), action_head=torch.nn.Linear(2, 2),
                              proprio_projector=torch.nn.Linear(2, 2))
    loads = []

    def build(*_):
        loads.append(True)
        return SimpleNamespace(num_open_loop_steps=8), copy.deepcopy(initial)

    monkeypatch.setattr(vla_adapter.direct, "load_policy_runtime", lambda *_: None)
    monkeypatch.setattr(vla_adapter.direct, "build_model", build)
    monkeypatch.setattr(vla_adapter, "replace", lambda cfg, **kwargs: cfg)
    provider = VLAAdapterPolicyProvider(SimpleNamespace(torch=torch), SimpleNamespace(), catalog)
    provider.load(8, "base")
    provider.load(8, "head-only")
    assert len(loads) == 1
    provider.load(8, "adapted")
    assert len(loads) == 2
    expected = torch.load(path.parent / "backbone.pt", weights_only=True)
    assert torch.equal(provider.components.model.weight, expected["weight"])
    provider.load(8, "adapted")
    assert len(loads) == 2
    provider.load(8, "base")
    assert len(loads) == 3
    assert torch.equal(provider.components.model.weight, initial.model.weight)
    provider.load(8, "head-only")
    assert len(loads) == 3


@pytest.mark.parametrize("invalid", ["", False, 0, None])
def test_adapted_overlay_cannot_silently_omit_backbone(tmp_path, invalid):
    path = _adapted_overlay(tmp_path)
    payload = yaml.safe_load(path.read_text())
    payload["backbone"] = invalid
    payload["component_sha256"].pop("backbone")
    path.write_text(yaml.safe_dump(payload))
    catalog = PolicyCatalog(tmp_path, BASE, "libero_object")
    with pytest.raises(ValueError, match="backbone"):
        catalog.entry("adapted")


def test_large_model_copy_and_bootstrap_use_threadpool(monkeypatch):
    import asyncio
    from backend.app.api import policies, runs
    from backend.app.api.models import CopyPolicyRequest
    calls = []

    async def offload(function, *args):
        calls.append(function.__name__)
        return function(*args)

    class Service:
        def copy(self, policy_id, label):
            return {"policy_id": policy_id, "label": label}

        def bootstrap(self):
            return {"policies": []}

        def evaluator_capabilities(self):
            return {}

    service = Service()
    monkeypatch.setattr(policies, "run_in_threadpool", offload)
    monkeypatch.setattr(runs, "run_in_threadpool", offload)
    monkeypatch.setattr(policies, "_service", lambda _: service)
    monkeypatch.setattr(runs, "service", lambda _: service)
    request = SimpleNamespace(app=SimpleNamespace(state=SimpleNamespace(offline_job_service=service)))

    async def exercise():
        assert (await policies.copy_model("adapted", CopyPolicyRequest(label="copy"), request))["label"] == "copy"
        assert (await runs.bootstrap(request))["policies"] == []

    asyncio.run(exercise())
    assert calls == ["copy", "bootstrap", "evaluator_capabilities"]
