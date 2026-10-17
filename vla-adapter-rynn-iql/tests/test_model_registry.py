from dataclasses import replace
import copy
import subprocess
import sys
from types import ModuleType

import pytest
import yaml

from vla_rynn_iql import base_models, models
from vla_rynn_iql.image_transforms import ORIENTATIONS, image_orientation, canonical_contract
from vla_rynn_iql.io import stable_hash
from vla_rynn_iql.model_artifacts import parent_snapshot, decode_child, parent_identity


def write_registry(root, families):
    root.mkdir(parents=True, exist_ok=True)
    index = {"schema_version": 1, "models": {name: f"{name}.yaml" for name in families}}
    for name, values in families.items():
        (root / f"{name}.yaml").write_text(yaml.safe_dump(values))
    path = root / "registry.yaml"
    path.write_text(yaml.safe_dump(index, sort_keys=False))
    return path


def family_files():
    return {name: yaml.safe_load(family.config_path.read_text())
            for name, family in base_models.MODEL_FAMILIES.items()}


@pytest.fixture(autouse=True)
def isolated_registry(tmp_path, monkeypatch):
    families = family_files()
    for name, entry in families.items():
        entry["default_variant"] = f"{name}-first"
        entry["variants"] = {
            f"{name}-{suffix}": {
                "label": f"Fixture {suffix}", "source": f"publisher/{name}-{suffix}",
                "revision": "a" * 40, "checkpoint": f"./weights/{suffix}", "stats_key": f"stats_{suffix}",
                "observation": {"image_flip_vertical": True, "image_flip_horizontal": True},
                "inference": {"open_loop_steps": min(index, entry["io"]["native_action_horizon"])},
            } for index, suffix in enumerate(("first", "second"), 1)
        }
    path = write_registry(tmp_path / "configs/models", families)
    definitions, variants = base_models.load_registry(path)
    # Keep shared mappings intact: model/config/artifact consumers hold references.
    for mapping, replacement in ((base_models.MODEL_FAMILIES, definitions), (base_models.BASE_MODELS, variants)):
        for key in list(mapping):
            monkeypatch.delitem(mapping, key)
        for key, value in replacement.items():
            monkeypatch.setitem(mapping, key, value)
    return path


def child(family=None):
    settings = models.model_config({"model": {"family": family or models.DEFAULT_FAMILY}})
    parent = parent_snapshot(settings, "a" * 64)
    io = parent["contract"]["io"]
    return {"schema_version": 5, "family": settings["family"], "model_config": settings, "parent": parent,
            "base_checkpoint": settings["base_checkpoint"], "stats_key": settings["stats_key"],
            "action_horizon": io["replay_horizon"], "action_dim": io["action_dim"], "proprio_dim": io["proprio_dim"]}


def test_all_variants_preserve_declared_assets_and_execution_defaults():
    for family in base_models.MODEL_FAMILIES.values():
        raw = yaml.safe_load(family.config_path.read_text())
        for identifier, declared in raw["variants"].items():
            base = base_models.base_model(identifier)
            settings = models.base_model_config(identifier)
            assert settings["base_id"] == identifier
            assert settings["family"] == family.id
            assert settings["stats_key"] == declared["stats_key"]
            assert base.source == declared["source"] and base.revision == declared["revision"]
            assert base.checkpoint == str((family.config_path.parent / declared["checkpoint"]).resolve())
            assert base_models.model_contract(settings) == {**family.contract, "observation": declared["observation"]}
            assert base_models.model_observation(settings) == declared["observation"]
            default = declared["inference"]["open_loop_steps"]
            maximum = family.contract["io"]["native_action_horizon"]
            assert base_models.execution_config(settings) == {"open_loop_steps": default, "max_open_loop_steps": maximum}
            assert base_models.resolve_open_loop_steps(settings, None) == default
            assert base_models.resolve_open_loop_steps(settings, maximum) == maximum
            with pytest.raises(ValueError, match="open_loop_steps"):
                base_models.resolve_open_loop_steps(settings, maximum + 1)


def test_registry_imports_without_loading_model_runtimes():
    subprocess.run([sys.executable, "-B", "-c", "import sys; import vla_rynn_iql.models; "
                    "assert not {'torch', 'transformers', 'openpi', 'prismatic'} & sys.modules.keys()"],
                   check=True, capture_output=True, text=True)


def test_default_family_is_not_changed_by_registry_order(monkeypatch):
    definitions = dict(reversed(list(base_models.MODEL_FAMILIES.items())))
    monkeypatch.setattr(base_models, "MODEL_FAMILIES", definitions)
    definition = definitions[base_models.DEFAULT_FAMILY]
    assert base_models.model_contract({}) == base_models.base_model(definition.default_base).contract


@pytest.mark.parametrize("orientation", list(ORIENTATIONS))
def test_variant_overrides_and_child_inheritance_are_family_independent(tmp_path, monkeypatch, orientation):
    raw = family_files()
    vertical, horizontal = ORIENTATIONS[orientation]
    for family, entry in raw.items():
        identifier = entry["default_variant"]
        entry["variants"][identifier]["observation"] = {
            "image_flip_vertical": vertical, "image_flip_horizontal": horizontal}
    _, variants = base_models.load_registry(write_registry(tmp_path / "updated", raw))
    for identifier, variant in variants.items():
        monkeypatch.setitem(base_models.BASE_MODELS, identifier, variant)
    for family in raw:
        exported = child(family)
        original = base_models.base_model(exported["parent"]["id"])
        assert image_orientation(original.contract["observation"]) == orientation
        changed = copy.deepcopy(original.contract)
        changed["observation"]["image_flip_horizontal"] = not horizontal
        monkeypatch.setitem(base_models.BASE_MODELS, original.id, replace(original, contract=changed))
        decoded, parent = decode_child(exported)
        assert image_orientation(decoded["model_config"]["contract"]["observation"]) == orientation
        assert parent == exported["parent"]


@pytest.mark.parametrize("value", [None, 0, 1, "true", "false", []])
@pytest.mark.parametrize("key", ["image_flip_vertical", "image_flip_horizontal"])
def test_flip_flags_require_boolean_values(tmp_path, key, value):
    raw = family_files()
    entry = next(iter(raw.values()))
    entry["variants"][entry["default_variant"]]["observation"][key] = value
    with pytest.raises(ValueError, match="must be booleans"):
        base_models.load_registry(write_registry(tmp_path / "invalid", raw))


def test_variants_have_independent_observation_without_family_defaults(tmp_path, monkeypatch):
    raw = family_files()
    for entry in raw.values():
        entry["variants"][entry["default_variant"]]["observation"]["image_flip_horizontal"] = False
    _, variants = base_models.load_registry(write_registry(tmp_path / "independent", raw))
    for identifier, base in variants.items():
        monkeypatch.setitem(base_models.BASE_MODELS, identifier, base)
    for family, entry in raw.items():
        assert image_orientation(base_models.model_observation(models.base_model_config(entry["default_variant"]))) == "flip_vertical"
        assert image_orientation(base_models.model_observation(models.base_model_config(f"{family}-second"))) == "rotate_180"
        assert not (set(entry["io"]) & {"image_flip_vertical", "image_flip_horizontal", "image_orientation"})


def test_editing_one_variant_never_changes_another_variants_contract(tmp_path):
    raw = family_files()
    original_families, original_variants = base_models.load_registry(write_registry(tmp_path / "registry", raw))
    for family, values in raw.items():
        for identifier in values["variants"]:
            changed = copy.deepcopy(raw)
            flags = changed[family]["variants"][identifier]["observation"]
            flags["image_flip_horizontal"] = not flags["image_flip_horizontal"]
            families, variants = base_models.load_registry(write_registry(tmp_path / "registry", changed))
            assert all(definition.contract == original_families[name].contract for name, definition in families.items())
            for candidate, variant in variants.items():
                assert (variant.contract != original_variants[candidate].contract) == (candidate == identifier)
                assert variant.stats_key == original_variants[candidate].stats_key
                assert variant.checkpoint == original_variants[candidate].checkpoint
    for identifier, variant in original_variants.items():
        settings = models.base_model_config(identifier)
        contract = base_models.model_contract(settings)
        contract["observation"]["image_flip_horizontal"] = False
        assert base_models.model_contract(settings) == variant.contract


@pytest.mark.parametrize("orientation", [None, True, [], "mirror", 180])
def test_legacy_orientation_is_strict(orientation):
    settings = models.base_model_config(next(iter(base_models.BASE_MODELS)))
    contract = base_models.model_contract(settings)
    contract.pop("observation")
    contract["io"]["image_orientation"] = orientation
    with pytest.raises(ValueError, match="image_orientation"):
        base_models.validate_contract(contract, settings["family"])


def test_builtin_adapters_reject_unsupported_image_sizes(tmp_path):
    for family, entry in family_files().items():
        entry["io"]["image_size"] = 448
        with pytest.raises(ValueError, match="Unsupported model contract"):
            base_models.load_registry(write_registry(tmp_path / family, {family: entry}))


@pytest.mark.parametrize("encoding", ["flags", "orientation"])
def test_old_frozen_contract_retains_identity(encoding):
    raw = child()
    current = copy.deepcopy(raw["parent"])
    io = raw["parent"]["contract"]["io"]
    observation = raw["parent"]["contract"].pop("observation")
    io.update(observation if encoding == "flags" else {"image_orientation": image_orientation(observation)})
    raw["parent"]["sha256"] = stable_hash({key: value for key, value in raw["parent"].items() if key != "sha256"})
    before = copy.deepcopy(raw)
    decoded, parent = decode_child(raw)
    assert raw == before and parent == before["parent"]
    assert decoded["model_config"]["contract"] == before["parent"]["contract"]
    assert canonical_contract(decoded["model_config"]["contract"]) == current["contract"]
    assert parent_identity(parent) == parent_identity(current)


@pytest.mark.parametrize("value", [True, 0, -1, 10000, 2.5, "5"])
def test_execution_steps_are_validated_for_all_variants(tmp_path, value):
    raw = family_files()
    for entry in raw.values():
        entry["variants"][entry["default_variant"]]["inference"]["open_loop_steps"] = value
    with pytest.raises(ValueError, match="open_loop_steps"):
        base_models.load_registry(write_registry(tmp_path / "invalid", raw))


def test_execution_defaults_do_not_change_parent_identity(monkeypatch):
    for family in base_models.MODEL_FAMILIES.values():
        identifier = family.default_base
        original = base_models.base_model(identifier)
        settings = models.base_model_config(identifier)
        before = parent_snapshot(settings, "a" * 64)
        maximum = original.contract["io"]["native_action_horizon"]
        monkeypatch.setitem(base_models.BASE_MODELS, identifier, replace(original, inference={"open_loop_steps": maximum}))
        assert models.base_model_config(identifier) == settings
        assert parent_snapshot(settings, "a" * 64) == before
        assert base_models.execution_config(settings)["open_loop_steps"] == maximum


@pytest.mark.parametrize("change", [
    lambda entry: entry.update(extra=True),
    lambda entry: entry["io"].update(action_dim=99),
    lambda entry: entry.update(default_variant="missing"),
    lambda entry: entry.update(model_name="different"),
    lambda entry: entry.update(adapter="missing"),
    lambda entry: entry.update(variants=[]),
    lambda entry: entry["variants"][entry["default_variant"]].update(revision="main"),
    lambda entry: entry["variants"][entry["default_variant"]].update(preset="obsolete"),
    lambda entry: entry["variants"][entry["default_variant"]].pop("inference"),
    lambda entry: entry["variants"][entry["default_variant"]].update(inference=None),
    lambda entry: entry["variants"][entry["default_variant"]].update(inference={}),
    lambda entry: entry.update(inference={"open_loop_steps": 1}),
    lambda entry: entry["variants"][entry["default_variant"]]["inference"].update(extra=1),
    lambda entry: entry["variants"][entry["default_variant"]].update(io={"image_orientation": "rotate_180", "image_flip_vertical": True}),
    lambda entry: entry["io"].update(image_flip_vertical=True),
    lambda entry: entry.update(observation={"image_flip_vertical": True, "image_flip_horizontal": True}),
    lambda entry: entry["variants"][entry["default_variant"]].pop("observation"),
    lambda entry: entry["variants"][entry["default_variant"]].update(observation={"image_flip_vertical": True}),
    lambda entry: entry["variants"][entry["default_variant"]].update(observation=None),
    lambda entry: entry["variants"][entry["default_variant"]]["observation"].update(extra=True),
    lambda entry: entry["variants"][entry["default_variant"]].update(observation={"image_orientation": "rotate_180"}),
])
def test_registry_rejects_invalid_contracts(tmp_path, change):
    raw = family_files()
    change(next(iter(raw.values())))
    with pytest.raises(ValueError):
        base_models.load_registry(write_registry(tmp_path / "invalid", raw))


@pytest.mark.parametrize("entries", [{}, [], {"example": "../outside.yaml"}, {"example": {"label": "inline"}}])
def test_index_requires_family_files(tmp_path, entries):
    path = tmp_path / "registry.yaml"
    path.write_text(yaml.safe_dump({"schema_version": 1, "models": entries}))
    with pytest.raises(ValueError):
        base_models.load_registry(path)


def test_duplicate_yaml_keys_are_rejected(tmp_path):
    path = tmp_path / "registry.yaml"
    path.write_text("schema_version: 1\nschema_version: 1\nmodels: {}\n")
    with pytest.raises(yaml.constructor.ConstructorError, match="duplicate"):
        base_models.load_registry(path)


def test_variant_ids_are_unique_across_families(tmp_path):
    raw = family_files()
    entries = list(raw.values())
    duplicate = entries[0]["default_variant"]
    entries[1]["variants"][duplicate] = next(iter(entries[1]["variants"].values()))
    with pytest.raises(ValueError, match="Duplicate base model ID"):
        base_models.load_registry(write_registry(tmp_path / "duplicate", raw))


def test_variant_rename_needs_no_runtime_changes(tmp_path, monkeypatch):
    raw = family_files()
    family, entry = next(iter(raw.items()))
    previous = entry["default_variant"]
    renamed = "any-publisher-any-model-max"
    entry["variants"][renamed] = entry["variants"].pop(previous)
    entry["default_variant"] = renamed
    definitions, variants = base_models.load_registry(write_registry(tmp_path / "renamed", raw))
    monkeypatch.setitem(base_models.MODEL_FAMILIES, family, definitions[family])
    monkeypatch.setitem(base_models.BASE_MODELS, renamed, variants[renamed])
    settings = models.base_model_config(renamed)
    assert settings["base_id"] == renamed
    assert settings["stats_key"] == variants[renamed].stats_key
    assert models.model_backend({"model": settings}).__name__ == definitions[family].backend
    listed = next(item for item in models.model_catalog() if item["id"] == family)
    assert any(item["id"] == renamed for item in listed["bases"])


def test_training_selection_uses_each_variants_assets():
    from vla_rynn_iql.config import load_train_config
    for base in base_models.BASE_MODELS.values():
        config = load_train_config(family=base.family, overrides={"model": {"base_id": base.id}})
        assert config.section("model")["base_checkpoint"] == base.checkpoint
        assert config.section("model")["stats_key"] == base.stats_key


@pytest.mark.parametrize("change", [
    lambda raw: raw["parent"].update(revision="b" * 64),
    lambda raw: raw["model_config"].update(base_revision="b" * 64),
    lambda raw: raw.update(action_dim=99),
    lambda raw: raw["model_config"].update(stats_key="wrong"),
])
def test_child_rejects_identity_conflicts(change):
    raw = child()
    change(raw)
    with pytest.raises(ValueError):
        decode_child(raw)


def test_child_retains_frozen_normalization_after_registry_changes(monkeypatch):
    for family in base_models.MODEL_FAMILIES:
        raw = child(family)
        old = base_models.base_model(raw["parent"]["id"])
        monkeypatch.setitem(base_models.BASE_MODELS, old.id, replace(old, stats_key="new_stats"))
        decoded, parent = decode_child(raw)
        assert decoded["model_config"]["stats_key"] == raw["parent"]["stats_key"]
        assert decoded["model_config"]["base_revision"] == raw["parent"]["revision"]
        assert parent == raw["parent"]


def test_local_identity_is_separate_from_source_revision(tmp_path, monkeypatch):
    from vla_rynn_iql.model_assets import pin_model
    base = next(base for base in base_models.BASE_MODELS.values()
                if base_models.family_adapter(base.family).PIN_SOURCE_REVISION)
    remote = replace(base, checkpoint="publisher/fixture", revision="a" * 40)
    monkeypatch.setitem(base_models.BASE_MODELS, base.id, remote)
    assert models.base_model_config(base.id)["base_revision"] == remote.revision
    weights = tmp_path / "weights"
    weights.mkdir()
    (weights / "model.safetensors").write_bytes(b"fixture weights")
    monkeypatch.setitem(base_models.BASE_MODELS, base.id, replace(remote, checkpoint=str(weights)))
    settings = models.base_model_config(base.id)
    assert settings["base_revision"] is None
    pinned = pin_model(settings)
    parent = parent_snapshot(pinned, pinned["base_revision"])
    assert parent["source_revision"] == remote.revision
    assert len(parent["revision"]) == 64 and parent["revision"] != remote.revision
    (weights / "model.safetensors").write_bytes(b"changed fixture weights")
    with pytest.raises(ValueError, match="changed"):
        pin_model(pinned)


def test_legacy_manifests_are_not_rewritten():
    for schema in (1, 2, 3, 4):
        raw = {"schema_version": schema, "model_config": {"backbone": "frozen"}}
        decoded, parent = decode_child(raw)
        assert decoded is raw and parent is None


def test_third_family_uses_registered_adapter_without_runtime_branches(tmp_path, monkeypatch):
    template = next(iter(base_models.MODEL_FAMILIES.values())).implementation
    spec = ModuleType("fixture_model_spec")
    spec.__dict__.update({key: value for key, value in vars(template).items() if not key.startswith("__")})
    spec.BACKEND = "fixture_model_backend"
    spec.POLICY_ADAPTER = "fixture_policy_adapter"
    spec.TRAINING_MODES = ("frozen",)
    spec.DEFAULT_KEYS = {"environment", "backbone", "base_revision", "gain"}
    spec.PARAMETER_PATHS = {"model_gain": ("gain",)}
    spec.TRAINING_UI = {"fields": [{"name": "model_gain", "label": "Gain", "type": "number"}]}
    spec.configure = lambda settings, **kwargs: settings
    spec.signature = lambda settings: settings
    spec.validate_contract = lambda io, architecture: io["image_size"] == 384 and io["native_action_horizon"] >= 8
    spec.MANIFEST_SCHEMA = 6
    spec.MANIFEST_FAMILY_FIELD = True
    backend = ModuleType(spec.BACKEND)
    backend.actor_losses = lambda batch: [1.0] * len(batch)
    monkeypatch.setitem(sys.modules, spec.__name__, spec)
    monkeypatch.setitem(sys.modules, backend.__name__, backend)
    monkeypatch.setitem(base_models.ADAPTERS, "fixture-adapter", spec.__name__)
    raw = family_files()
    entry = copy.deepcopy(next(iter(raw.values())))
    entry.update(model_name="independent-family", adapter="fixture-adapter", default_variant="custom-checkpoint",
                 backbone_modes=["frozen"], defaults={"environment": "test", "backbone": "frozen", "base_revision": None, "gain": 2})
    entry["variants"] = {"custom-checkpoint": next(iter(entry["variants"].values()))}
    entry["io"].update(image_size=384, native_action_horizon=16)
    entry["variants"]["custom-checkpoint"]["inference"]["open_loop_steps"] = 12
    raw["independent-family"] = entry
    definitions, variants = base_models.load_registry(write_registry(tmp_path / "third", raw))
    monkeypatch.setitem(base_models.MODEL_FAMILIES, "independent-family", definitions["independent-family"])
    monkeypatch.setitem(base_models.BASE_MODELS, "custom-checkpoint", variants["custom-checkpoint"])
    settings = models.base_model_config("custom-checkpoint")
    assert settings["family"] == "independent-family" and settings["gain"] == 2
    from vla_rynn_iql.config import load_train_config
    for method in ("bc", "iql"):
        config = load_train_config(method=method, family="independent-family")
        assert config.section("model") == settings
    assert models.model_backend({"model": settings}).actor_losses([1, 2]) == [1.0, 1.0]
    assert models.model_catalog()[-1]["training_ui"]["fields"][0]["default"] == 2
    exported = child("independent-family")
    original = variants["custom-checkpoint"]
    assert base_models.resolve_open_loop_steps(settings, 16) == 16
    assert base_models.model_contract(settings)["io"]["replay_horizon"] == 8
    changed = copy.deepcopy(original.contract)
    changed["io"]["native_action_horizon"] = 32
    monkeypatch.setitem(base_models.BASE_MODELS, original.id,
                        replace(original, contract=changed, inference={"open_loop_steps": 24}))
    decoded, _ = decode_child(exported)
    assert decoded["schema_version"] == 6 and decoded["family"] == "independent-family"
    assert base_models.execution_config(decoded["model_config"]) == {"open_loop_steps": 16, "max_open_loop_steps": 16}
    with pytest.raises(ValueError, match="open_loop_steps"):
        base_models.resolve_open_loop_steps(decoded["model_config"], 17)
