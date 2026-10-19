from dataclasses import replace
import copy
from pathlib import Path

import pytest
import yaml

from backend.app.policies.catalog import PolicyCatalog, _stable_hash
from backend.app.policies.registry import model_module
from test_policy_overlay import _overlay, BASE, STATS
from test_pi05_policy import checkpoint


def test_catalog_exposes_independent_variant_execution_defaults(tmp_path, monkeypatch):
    registry = model_module("base_models")
    definitions = {}
    for identifier, steps in (("pi05-libero-base", 5), ("pi05-liberox-base", 9)):
        base = registry.BASE_MODELS[identifier]
        monkeypatch.setitem(registry.BASE_MODELS, identifier,
                            replace(base, inference={"open_loop_steps": steps}))
        definitions[identifier] = model_module("models").base_model_config(identifier)
    catalog = PolicyCatalog(tmp_path / "policies", base_models=definitions)
    policies = {entry["policy_id"]: entry for entry in catalog.list_policies()}
    for identifier, steps in (("pi05-libero-base", 5), ("pi05-liberox-base", 9)):
        assert policies[identifier]["inference"] == {"open_loop_steps": steps, "max_open_loop_steps": 10}
        assert catalog.entry(identifier).resolve_open_loop_steps(None) == steps


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


def test_short_supervision_child_loads_through_catalog_and_cli(tmp_path):
    path = _overlay(tmp_path)
    raw = yaml.safe_load(path.read_text())
    settings = model_module("models").model_config({"model": {"stats_key": STATS}})
    settings["contract"] = model_module("base_models").model_contract(settings)
    settings["contract"]["io"]["replay_horizon"] = 4
    parent = model_module("model_artifacts").parent_snapshot(settings, "b" * 40)
    raw.update(schema_version=5, family="vla_adapter", algorithm="iql", backbone=None,
               model_config=settings, parent=parent, action_horizon=4)
    raw["compatibility_sha256"] = _stable_hash({key: raw[key] for key in (
        "base_checkpoint", "stats_key", "action_horizon", "action_dim", "proprio_dim")})
    path.write_text(yaml.safe_dump(raw))
    catalog = PolicyCatalog(tmp_path, base_models={
        "base": {"base_checkpoint": BASE, "stats_key": "libero_object"}}, base_revisions={"base": "b" * 40})
    entry = catalog.select("trained")
    assert entry.public()["io"]["replay_horizon"] == 4
    assert entry.public()["io"]["native_action_horizon"] == 8
    adapter = model_module("vla_adapter")
    adapter.validate_overlay(adapter.load_overlay(path), BASE, STATS)


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


@pytest.mark.parametrize("identifier", ["base", "pi05-libero-base", "pi05-liberox-base"])
@pytest.mark.parametrize("legacy_snapshot", [True, False])
def test_orientation_changes_behavior_identity_but_not_weights_or_frozen_snapshot(tmp_path, monkeypatch, identifier, legacy_snapshot):
    from backend.app.policies.snapshots import snapshot, catalog_from_snapshot
    registry = model_module("base_models")
    settings = model_module("models").base_model_config(identifier)
    catalog = PolicyCatalog(tmp_path, base_models={identifier: settings}, base_revisions={identifier: "a" * 64})
    original = catalog.entry(identifier)
    before = original.content_sha256
    frozen = snapshot(original)
    if legacy_snapshot:
        frozen.pop("model_contract")
    base = registry.BASE_MODELS[identifier]
    contract = copy.deepcopy(base.contract)
    contract["io"].update(image_flip_vertical=True, image_flip_horizontal=False)
    monkeypatch.setitem(registry.BASE_MODELS, identifier, replace(base, contract=contract))
    catalog.refresh()
    changed = catalog.entry(identifier)
    assert changed.content_sha256 != before
    assert changed.base_revision == original.base_revision
    assert changed.public()["io"]["image_flip_horizontal"] is False
    # Reconstruction is tested without loading any model files.
    monkeypatch.setattr(PolicyCatalog, "select", PolicyCatalog.entry)
    restored = catalog_from_snapshot(frozen, tmp_path / "empty")
    assert restored.entry(identifier).content_sha256 == before
    assert restored.entry(identifier).public()["io"]["image_flip_horizontal"] is True
    frozen["model_contract"] = copy.deepcopy(contract)
    frozen["model_contract"]["io"].update(image_flip_vertical=False, image_flip_horizontal=False)
    with pytest.raises(ValueError, match="changed"):
        catalog_from_snapshot(frozen, tmp_path / "empty")


@pytest.mark.parametrize("schema", [1, 3, 5])
def test_old_and_frozen_vla_child_keep_orientation_when_parent_registration_changes(tmp_path, monkeypatch, schema):
    path = _overlay(tmp_path)
    raw = yaml.safe_load(path.read_text())
    settings = model_module("models").model_config({"model": {"stats_key": STATS}})
    if schema in (3, 5):
        raw.update(schema_version=schema, algorithm="iql", backbone=None, model_config=settings)
    if schema == 5:
        raw.update(family="vla_adapter", parent=model_module("model_artifacts").parent_snapshot(settings, "a" * 64))
    path.write_text(yaml.safe_dump(raw))
    catalog = PolicyCatalog(tmp_path, base_models={"base": {"base_checkpoint": BASE, "stats_key": "libero_object"}})
    original = catalog.entry("trained")
    registry = model_module("base_models")
    base = registry.BASE_MODELS["base"]
    contract = copy.deepcopy(base.contract)
    contract["io"].update(image_flip_vertical=False, image_flip_horizontal=True)
    monkeypatch.setitem(registry.BASE_MODELS, "base", replace(base, contract=contract))
    catalog.refresh()
    restored = catalog.entry("trained")
    assert restored.content_sha256 == original.content_sha256
    assert restored.public()["io"]["image_flip_vertical"] is True
    assert restored.public()["io"]["image_flip_horizontal"] is True
    # The independent CLI must resolve the same legacy/frozen contract.
    cli_overlay = model_module("vla_adapter").load_overlay(path)
    assert cli_overlay.model_config["contract"]["io"]["image_flip_vertical"] is True
    assert cli_overlay.model_config["contract"]["io"]["image_flip_horizontal"] is True
    assert yaml.safe_load(path.read_text()) == raw
