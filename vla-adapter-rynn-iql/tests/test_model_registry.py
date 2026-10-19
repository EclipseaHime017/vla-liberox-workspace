from dataclasses import replace
import copy

import pytest
import yaml

from vla_rynn_iql import base_models, models
from vla_rynn_iql.model_artifacts import parent_snapshot, decode_child
from vla_rynn_iql.image_transforms import ORIENTATIONS, image_orientation


def child(family="vla_adapter"):
    settings = models.model_config({"model": {"family": family}})
    parent = parent_snapshot(settings, "a" * 64)
    return {"schema_version": 5, "family": family, "model_config": settings, "parent": parent,
            "base_checkpoint": settings["base_checkpoint"], "stats_key": settings["stats_key"],
            "action_horizon": parent["contract"]["io"]["replay_horizon"], "action_dim": 7, "proprio_dim": 8}


def test_registry_is_lightweight_and_drives_every_base():
    for base in base_models.BASE_MODELS.values():
        settings = models.base_model_config(base.id)
        assert settings["base_id"] == base.id
        assert settings["family"] == base.family
        assert settings["stats_key"] == base.stats_key
        assert base_models.model_contract(settings) == base.contract


@pytest.mark.parametrize("base_id,default,maximum", [
    ("base", 8, 8), ("pi05-libero-base", 5, 10), ("pi05-liberox-base", 10, 10),
])
def test_execution_steps_are_separate_from_prediction_and_training(base_id, default, maximum):
    settings = models.base_model_config(base_id)
    assert base_models.execution_config(settings) == {
        "open_loop_steps": default, "max_open_loop_steps": maximum,
    }
    assert base_models.resolve_open_loop_steps(settings, None) == default
    assert base_models.resolve_open_loop_steps(settings, maximum) == maximum
    supervision = base_models.model_contract(settings)["io"]["replay_horizon"]
    assert supervision == maximum
    # Trained children do not execute the unsupervised tail by default.
    assert base_models.execution_config(settings, trained=True) == {
        "open_loop_steps": min(default, supervision), "max_open_loop_steps": maximum,
    }
    assert base_models.resolve_open_loop_steps(settings, maximum, trained=True) == maximum
    with pytest.raises(ValueError, match="open_loop_steps"):
        base_models.resolve_open_loop_steps(settings, maximum + 1)


@pytest.mark.parametrize("orientation", ["none", "flip_vertical", "flip_horizontal", "rotate_180"])
@pytest.mark.parametrize("family", ["vla_adapter", "pi05"])
def test_variant_image_direction_overrides_family_and_is_frozen_in_child(tmp_path, monkeypatch, family, orientation):
    raw = family_files()
    identifier = raw[family]["default_variant"]
    vertical, horizontal = ORIENTATIONS[orientation]
    raw[family]["variants"][identifier]["io"] = {
        "image_flip_vertical": vertical, "image_flip_horizontal": horizontal}
    _, bases = base_models.load_registry(write_registry(tmp_path, raw))
    monkeypatch.setitem(base_models.BASE_MODELS, identifier, bases[identifier])
    settings = models.base_model_config(identifier)
    assert image_orientation(base_models.model_contract(settings)["io"]) == orientation
    exported = child(family)
    changed = copy.deepcopy(bases[identifier].contract)
    changed["io"]["image_flip_horizontal"] = not horizontal
    monkeypatch.setitem(base_models.BASE_MODELS, identifier, replace(bases[identifier], contract=changed))
    decoded, _ = decode_child(exported)
    assert image_orientation(base_models.model_contract(decoded["model_config"])["io"]) == orientation
    assert image_orientation(raw[family]["io"]) == "rotate_180"


def test_variant_can_override_only_one_axis(tmp_path):
    raw = family_files()
    raw["pi05"]["variants"]["pi05-liberox-base"]["io"] = {"image_flip_horizontal": False}
    _, bases = base_models.load_registry(write_registry(tmp_path, raw))
    assert image_orientation(bases["pi05-liberox-base"].contract["io"]) == "flip_vertical"
    assert image_orientation(bases["pi05-libero-base"].contract["io"]) == "rotate_180"


@pytest.mark.parametrize("value", [None, 0, 1, "true", "false", []])
@pytest.mark.parametrize("key", ["image_flip_vertical", "image_flip_horizontal"])
def test_image_flips_require_real_booleans(tmp_path, key, value):
    raw = family_files()
    raw["pi05"]["variants"]["pi05-libero-base"]["io"] = {key: value}
    with pytest.raises(ValueError, match="must be booleans"):
        base_models.load_registry(write_registry(tmp_path, raw))


def test_legacy_family_and_new_partial_variant_can_be_read_together(tmp_path):
    raw = family_files()
    raw["pi05"]["io"].pop("image_flip_vertical")
    raw["pi05"]["io"].pop("image_flip_horizontal")
    raw["pi05"]["io"]["image_orientation"] = "rotate_180"
    raw["pi05"]["variants"]["pi05-liberox-base"]["io"] = {"image_flip_horizontal": False}
    _, bases = base_models.load_registry(write_registry(tmp_path, raw))
    assert image_orientation(bases["pi05-liberox-base"].contract["io"]) == "flip_vertical"
    assert image_orientation(bases["pi05-libero-base"].contract["io"]) == "rotate_180"


def test_old_frozen_contract_keeps_its_hash_and_matches_new_boolean_encoding():
    from vla_rynn_iql.io import stable_hash
    from vla_rynn_iql.model_artifacts import parent_identity
    from vla_rynn_iql.image_transforms import canonical_contract

    raw = child()
    current = copy.deepcopy(raw["parent"])
    old_io = raw["parent"]["contract"]["io"]
    old_io.pop("image_flip_vertical")
    old_io.pop("image_flip_horizontal")
    old_io["image_orientation"] = "rotate_180"
    raw["parent"]["sha256"] = stable_hash({key: value for key, value in raw["parent"].items() if key != "sha256"})
    before = copy.deepcopy(raw)
    decoded, parent = decode_child(raw)
    assert raw == before
    assert parent == before["parent"]
    assert decoded["model_config"]["contract"] == before["parent"]["contract"]
    assert canonical_contract(decoded["model_config"]["contract"]) == current["contract"]
    assert parent_identity(parent) == parent_identity(current)


def test_image_orientation_and_flip_flags_cannot_be_combined(tmp_path):
    raw = family_files()
    raw["pi05"]["variants"]["pi05-libero-base"]["io"] = {
        "image_orientation": "rotate_180", "image_flip_vertical": True}
    with pytest.raises(ValueError):
        base_models.load_registry(write_registry(tmp_path, raw))


@pytest.mark.parametrize("orientation", [None, True, [], "mirror", 180])
def test_variant_image_direction_is_strict(tmp_path, orientation):
    raw = family_files()
    raw["pi05"]["variants"]["pi05-libero-base"]["io"] = {"image_orientation": orientation}
    with pytest.raises(ValueError, match="image_orientation"):
        base_models.load_registry(write_registry(tmp_path, raw))


@pytest.mark.parametrize("value", [True, 0, -1, 11, 2.5, "5"])
@pytest.mark.parametrize("base_id", ["pi05-libero-base", "pi05-liberox-base"])
def test_registry_rejects_invalid_execution_steps(tmp_path, base_id, value):
    raw = family_files()
    raw["pi05"]["variants"][base_id]["inference"]["open_loop_steps"] = value
    with pytest.raises(ValueError, match="open_loop_steps"):
        base_models.load_registry(write_registry(tmp_path, raw))


def test_variant_owns_execution_defaults_without_changing_weight_identity(tmp_path, monkeypatch):
    raw = family_files()
    raw["pi05"]["variants"]["pi05-libero-base"]["inference"] = {"open_loop_steps": 10}
    _, bases = base_models.load_registry(write_registry(tmp_path, raw))
    assert bases["pi05-liberox-base"].inference == {"open_loop_steps": 10}
    base = bases["pi05-libero-base"]
    assert base.inference == {"open_loop_steps": 10}
    original = base_models.BASE_MODELS[base.id]
    settings = models.base_model_config(base.id)
    snapshot = parent_snapshot(settings, "a" * 64)
    monkeypatch.setitem(base_models.BASE_MODELS, base.id, replace(original, inference=base.inference))
    assert models.base_model_config(base.id) == settings
    assert parent_snapshot(settings, "a" * 64) == snapshot
    assert base_models.execution_config(settings)["open_loop_steps"] == 10
    decoded, _ = decode_child(child("pi05"))
    assert base_models.execution_config(decoded["model_config"])["open_loop_steps"] == 10


def test_execution_limit_follows_variant_contract_not_a_family_constant(tmp_path, monkeypatch):
    raw = family_files()
    raw["pi05"]["variants"]["pi05-liberox-base"].update(
        io={"native_action_horizon": 50}, inference={"open_loop_steps": 25})
    _, bases = base_models.load_registry(write_registry(tmp_path, raw))
    base = bases["pi05-liberox-base"]
    monkeypatch.setitem(base_models.BASE_MODELS, base.id, base)
    settings = models.base_model_config(base.id)
    assert base_models.execution_config(settings) == {"open_loop_steps": 25, "max_open_loop_steps": 50}
    assert base_models.resolve_open_loop_steps(settings, 50) == 50
    assert base_models.execution_config(models.base_model_config("pi05-libero-base")) == {
        "open_loop_steps": 5, "max_open_loop_steps": 10,
    }


def test_child_execution_limit_uses_frozen_native_horizon_not_live_registry_or_replay(monkeypatch):
    raw = child("pi05")
    original = base_models.BASE_MODELS[raw["parent"]["id"]]
    contract = copy.deepcopy(original.contract)
    contract["io"]["native_action_horizon"] = 50
    monkeypatch.setitem(base_models.BASE_MODELS, original.id,
                        replace(original, contract=contract, inference={"open_loop_steps": 25}))
    decoded, _ = decode_child(raw)
    assert base_models.execution_config(decoded["model_config"]) == {
        "open_loop_steps": 10, "max_open_loop_steps": 10,
    }
    with pytest.raises(ValueError, match=r"\[1, 10\]"):
        base_models.resolve_open_loop_steps(decoded["model_config"], 11)


@pytest.mark.parametrize("change", [
    lambda raw: raw["pi05"].update(inference={"open_loop_steps": 5}),
    lambda raw: raw["pi05"]["variants"]["pi05-libero-base"].pop("inference"),
    lambda raw: raw["pi05"]["variants"]["pi05-liberox-base"].update(inference={}),
    lambda raw: raw["pi05"]["variants"]["pi05-liberox-base"].update(inference=None),
    lambda raw: raw["pi05"]["variants"]["pi05-liberox-base"]["inference"].update(extra=1),
    lambda raw: raw["pi05"]["variants"]["pi05-liberox-base"].update(
        io={"native_action_horizon": 8}, inference={"open_loop_steps": 9}),
])
def test_execution_configuration_is_required_per_variant(tmp_path, change):
    raw = family_files()
    change(raw)
    with pytest.raises(ValueError):
        base_models.load_registry(write_registry(tmp_path, raw))


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
        "inference": {"open_loop_steps": 4},
    }
    path = write_registry(tmp_path / "configs/models", raw)
    families, bases = base_models.load_registry(path)
    added = bases["new-variant"]
    assert added.family == "vla_adapter"
    assert added.contract == families[added.family].contract
    assert added.inference == {"open_loop_steps": 4}
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
