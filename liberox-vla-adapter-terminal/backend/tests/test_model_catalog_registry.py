from dataclasses import replace
from pathlib import Path

import pytest
import yaml

from backend.app.policies.catalog import PolicyCatalog, _stable_hash
from backend.app.policies.registry import model_module
from test_policy_overlay import _overlay, BASE, STATS
from test_pi05_policy import checkpoint


def test_multiple_vla_bases_resolve_their_own_assets(tmp_path, monkeypatch):
    registry = model_module("base_models")
    definitions = {}
    for identifier in ("base", "other-vla"):
        root = tmp_path / identifier
        root.mkdir()
        (root / "config.json").write_text("{}")
        (root / "model.safetensors").write_bytes(identifier.encode())
        registration = replace(registry.BASE_MODELS["base"], id=identifier, checkpoint=str(root), stats_key=identifier)
        monkeypatch.setitem(registry.BASE_MODELS, identifier, registration)
        definitions[identifier] = registry.registered_defaults("vla_adapter", identifier)
    catalog = PolicyCatalog(tmp_path / "policies", base_models=definitions)
    first = catalog.select("base")
    second = catalog.select("other-vla")
    assert first.base_revision != second.base_revision
    assert second.settings["base_id"] == "other-vla"
    assert second.stats_key == "other-vla"
    assert catalog.entry("base").content_sha256 == first.content_sha256
    (Path(second.base_checkpoint) / "model.safetensors").write_bytes(b"new")
    assert catalog.select("other-vla").content_sha256 != second.content_sha256


def test_legacy_partial_vla_settings_and_identity(tmp_path):
    path = _overlay(tmp_path)
    raw = yaml.safe_load(path.read_text())
    raw.update(schema_version=3, algorithm="iql", backbone=None,
               model_config={"family": "vla_adapter", "backbone": "frozen", "stats_key": "libero_object"})
    path.write_text(yaml.safe_dump(raw))
    entry = PolicyCatalog(tmp_path, base_models={
        "base": {"base_checkpoint": BASE, "stats_key": "libero_object"}}).entry("trained")
    assert entry.settings["stats_key"] == STATS
    assert entry.settings["base_checkpoint"] == BASE
    assert entry.content_sha256 == _stable_hash({
        "family": "vla_adapter", "base_checkpoint": BASE, "stats_key": STATS,
        "model_config": raw["model_config"], "component_sha256": raw["component_sha256"], "base_revision": None})


def test_child_selection_does_not_rebase_registered_parent(tmp_path):
    path = _overlay(tmp_path)
    raw = yaml.safe_load(path.read_text())
    settings = model_module("models").model_config({"model": {"stats_key": STATS}})
    parent = model_module("model_artifacts").parent_snapshot(settings, "b" * 40)
    raw.update(schema_version=5, family="vla_adapter", algorithm="iql", backbone=None,
               model_config=settings, parent=parent)
    path.write_text(yaml.safe_dump(raw))
    catalog = PolicyCatalog(tmp_path, base_models={
        "base": {"base_checkpoint": BASE, "stats_key": "libero_object"}},
        base_revisions={"base": "a" * 40})
    before = catalog.entry("base").content_sha256
    selected = catalog.select("trained")
    assert selected.base_revision == "b" * 40
    assert selected.settings["base_revision"] == "b" * 40
    assert catalog.select("base").base_revision == "a" * 40
    assert catalog.entry("base").content_sha256 == before


@pytest.mark.parametrize("legacy", [True, False])
def test_pi_base_snapshot_reconstruction_preserves_hash(tmp_path, legacy):
    from backend.app.policies.snapshots import snapshot, catalog_from_snapshot
    settings, _ = checkpoint(tmp_path)
    if legacy:
        settings.pop("base_id")
    catalog = PolicyCatalog(tmp_path / "policies", base_models={"pi05-libero-base": settings})
    original = catalog.select("pi05-libero-base")
    frozen = snapshot(original)
    if legacy:
        frozen["model_config"].pop("base_id")
        expected = _stable_hash({"family": original.family, "base_checkpoint": original.base_checkpoint,
            "stats_key": original.stats_key, "model_config": frozen["model_config"],
            "component_sha256": {}, "base_revision": original.base_revision})
        assert frozen["content_sha256"] == expected
    reconstructed = catalog_from_snapshot(frozen, tmp_path / "empty")
    assert reconstructed.select(original.policy_id).content_sha256 == original.content_sha256


def test_model_module_pins_local_training_assets(tmp_path):
    root = tmp_path / "base"
    root.mkdir()
    (root / "model.safetensors").write_bytes(b"first")
    settings = model_module("model_assets").pin_model({"base_checkpoint": str(root)})
    assert settings["base_revision"] and settings["contract"]["io"]["action_dim"] == 7
    (root / "model.safetensors").write_bytes(b"second")
    with pytest.raises(ValueError, match="changed"):
        model_module("model_assets").pin_model(settings)
