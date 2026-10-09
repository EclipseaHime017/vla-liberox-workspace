from dataclasses import replace
import copy

import pytest
import yaml

from vla_rynn_iql import base_models, models
from vla_rynn_iql.model_artifacts import parent_snapshot, decode_child


def child(family="vla_adapter"):
    settings = models.model_config({"model": {"family": family}})
    parent = parent_snapshot(settings, "a" * 64)
    return {"schema_version": 5, "family": family, "model_config": settings, "parent": parent,
            "base_checkpoint": settings["base_checkpoint"], "stats_key": settings["stats_key"],
            "action_horizon": 8, "action_dim": 7, "proprio_dim": 8}


def test_registry_is_lightweight_and_drives_every_base():
    for base in base_models.BASE_MODELS.values():
        settings = models.base_model_config(base.id)
        assert settings["base_id"] == base.id
        assert settings["family"] == base.family
        assert settings["stats_key"] == base.stats_key
        assert base_models.model_contract(settings) == base.contract


def family_files():
    return {name: yaml.safe_load(family.config_path.read_text())
            for name, family in base_models.MODEL_FAMILIES.items()}


def write_registry(root, families):
    root.mkdir(parents=True, exist_ok=True)
    index = {"schema_version": 1, "models": {name: f"{name}.yaml" for name in families}}
    for name, values in families.items():
        (root / f"{name}.yaml").write_text(yaml.safe_dump(values))
    path = root / "registry.yaml"
    path.write_text(yaml.safe_dump(index, sort_keys=False))
    return path


@pytest.mark.parametrize("change", [
    lambda raw: raw["pi05"].update(extra=True),
    lambda raw: raw["pi05"]["io"].update(action_dim=8),
    lambda raw: raw["vla_adapter"]["variants"]["base"].update(revision="main"),
    lambda raw: raw["vla_adapter"]["variants"]["base"].update(family="unknown"),
    lambda raw: raw["vla_adapter"]["variants"]["base"].update(preset="obsolete"),
    lambda raw: raw["pi05"].update(model_name="different"),
    lambda raw: raw["pi05"].update(variants=[]),
    lambda raw: raw["pi05"].update(default_variant="base"),
])
def test_registry_rejects_invalid_contracts(tmp_path, change):
    raw = family_files()
    change(raw)
    path = write_registry(tmp_path, raw)
    with pytest.raises(ValueError):
        base_models.load_registry(path)


@pytest.mark.parametrize("models", [{}, [], {"pi05": "../pi05.yaml"}, {"pi05": {"label": "inline"}}])
def test_registry_index_requires_family_files(tmp_path, models):
    path = tmp_path / "registry.yaml"
    path.write_text(yaml.safe_dump({"schema_version": 1, "models": models}))
    with pytest.raises(ValueError):
        base_models.load_registry(path)


def test_registry_rejects_duplicate_keys(tmp_path):
    path = tmp_path / "registry.yaml"
    path.write_text("schema_version: 1\nschema_version: 1\nmodels: {}\n")
    with pytest.raises(yaml.constructor.ConstructorError, match="duplicate"):
        base_models.load_registry(path)


def test_base_ids_are_unique_across_families(tmp_path):
    raw = family_files()
    raw["pi05"]["variants"]["base"] = raw["pi05"]["variants"].pop("pi05-libero-base")
    path = write_registry(tmp_path, raw)
    with pytest.raises(ValueError, match="Duplicate base model ID"):
        base_models.load_registry(path)


def test_new_base_inherits_family_without_vendor_or_business_registration(tmp_path, monkeypatch):
    raw = family_files()
    raw["vla_adapter"]["variants"]["new-variant"] = {
        "label": "New variant", "source": "publisher/weights", "revision": None,
        "checkpoint": "./weights", "stats_key": "new_stats",
    }
    path = write_registry(tmp_path / "configs/models", raw)
    families, bases = base_models.load_registry(path)
    added = bases["new-variant"]
    assert added.family == "vla_adapter"
    assert added.contract == families[added.family].contract
    assert added.checkpoint == str(path.parent / "weights")
    monkeypatch.setitem(base_models.BASE_MODELS, added.id, added)
    settings = models.base_model_config(added.id)
    assert settings["base_checkpoint"] == added.checkpoint
    assert settings["stats_key"] == added.stats_key
    family = next(item for item in models.model_catalog() if item["id"] == added.family)
    assert any(item["id"] == added.id for item in family["bases"])


def test_family_selection_does_not_leak_default_variant_assets():
    from vla_rynn_iql.config import load_train_config
    for base in base_models.BASE_MODELS.values():
        config = load_train_config(family=base.family, overrides={"model": {"base_id": base.id}})
        settings = config.section("model")
        assert settings["base_id"] == base.id
        assert settings["base_checkpoint"] == base.checkpoint
        assert settings["stats_key"] == base.stats_key


@pytest.mark.parametrize("family,schema", [("vla_adapter", 3), ("pi05", 4)])
def test_child_inherits_frozen_parent_not_live_registry(monkeypatch, family, schema):
    raw = child(family)
    parent = raw["parent"]
    old = base_models.BASE_MODELS[parent["id"]]
    contract = copy.deepcopy(old.contract)
    if family == "pi05":
        contract["io"]["native_action_horizon"] = 20
    monkeypatch.setitem(base_models.BASE_MODELS, old.id, replace(old, stats_key="new/stats", contract=contract))
    decoded, frozen = decode_child(raw)
    assert decoded["schema_version"] == schema
    assert decoded["model_config"]["stats_key"] == parent["stats_key"]
    assert decoded["model_config"]["base_revision"] == parent["revision"]
    assert decoded["model_config"]["contract"] == parent["contract"]
    assert frozen == parent


@pytest.mark.parametrize("change", [
    lambda raw: raw["parent"].update(revision="b" * 64),
    lambda raw: raw["model_config"].update(base_revision="b" * 64),
    lambda raw: raw.update(family="pi05"),
    lambda raw: raw.update(action_dim=8),
    lambda raw: raw["model_config"].update(stats_key="wrong"),
    lambda raw: raw["model_config"].update(base_id="pi05-libero-base"),
])
def test_child_rejects_identity_conflicts(change):
    raw = child()
    change(raw)
    with pytest.raises(ValueError):
        decode_child(raw)


def test_yaml_vla_revision_is_used_without_network(monkeypatch):
    from vla_rynn_iql.checkpoint_assets import resolve_revision
    base = base_models.BASE_MODELS["base"]
    monkeypatch.setitem(base_models.BASE_MODELS, "base", replace(base, revision="a" * 40))
    settings = models.model_config({})
    assert resolve_revision(settings["base_checkpoint"], settings["base_revision"]) == "a" * 40


def test_downloaded_weights_keep_source_commit_separate_from_local_hash(tmp_path, monkeypatch):
    from vla_rynn_iql.model_assets import pin_model
    weights = tmp_path / "downloaded"
    weights.mkdir()
    (weights / "model.safetensors").write_bytes(b"local weights")
    base = base_models.BASE_MODELS["base"]
    monkeypatch.setitem(base_models.BASE_MODELS, base.id,
                        replace(base, checkpoint=str(weights), revision="a" * 40))
    settings = models.base_model_config(base.id)
    assert settings["base_revision"] is None
    pinned = pin_model(settings)
    assert len(pinned["base_revision"]) == 64
    parent = parent_snapshot(pinned, pinned["base_revision"])
    assert parent["source_revision"] == "a" * 40
    assert parent["revision"] == pinned["base_revision"]
    (weights / "model.safetensors").write_bytes(b"changed weights")
    with pytest.raises(ValueError, match="changed"):
        pin_model(pinned)


def test_legacy_manifest_is_not_rewritten():
    for schema in (1, 2, 3, 4):
        raw = {"schema_version": schema, "model_config": {"backbone": "frozen"}}
        decoded, parent = decode_child(raw)
        assert decoded is raw and parent is None
