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
