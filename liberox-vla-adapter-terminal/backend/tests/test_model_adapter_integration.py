"""A synthetic third family exercises platform boundaries without model weights."""
import copy
import json
import sys
from dataclasses import replace
from pathlib import Path
from types import ModuleType, SimpleNamespace

import pytest
import yaml

from backend.app.policies.catalog import PolicyCatalog, PolicyEntry, _sha256, _stable_hash
from backend.app.policies.registry import model_module, configured_models
from backend.app.policies.router import PolicyProvider
from backend.app.policies.snapshots import snapshot, catalog_from_snapshot
from backend.app.services.offline_job_service import OfflineJobService
from backend.app.services.policy_management_service import PolicyManagementService


def test_platform_family_adapters_share_the_local_registry_namespace():
    definitions = model_module("base_models")
    for family in definitions.MODEL_FAMILIES.values():
        assert family.implementation.__package__ == f"{definitions.__package__}.model_families"


@pytest.fixture
def adapter(tmp_path, monkeypatch):
    definitions = model_module("base_models")
    family, adapter_id, variant = "fixture-family", "fixture-driver", "arbitrary-checkpoint-max"
    template = next(iter(definitions.MODEL_FAMILIES.values()))
    spec = ModuleType("fixture_family_spec")
    spec.__dict__.update({key: value for key, value in vars(template.implementation).items() if not key.startswith("__")})
    spec.BACKEND = "fixture_family_backend"
    spec.POLICY_ADAPTER = "fixture_family_policy"
    spec.TRAINING_MODES = ("frozen",)
    spec.DEFAULT_KEYS = {"environment", "backbone", "base_revision", "gain"}
    spec.PARAMETER_PATHS = {"model_gain": ("gain",)}
    spec.TRAINING_UI = {"fields": [{"name": "model_gain", "label": "Gain", "type": "number"}],
                        "components": [{"label": "Fixture actor", "description": "Synthetic"}]}
    spec.MANIFEST_SCHEMA, spec.MANIFEST_FAMILY_FIELD = 6, True
    spec.RECORDING_ACTION_CODEC, spec.EXPORT_PREFIX = "libero_env_v1", "fixture"
    spec.configure = lambda settings, **kwargs: settings
    spec.signature = lambda settings: settings
    spec.resolve_revision = lambda settings, hash_file=None, **kwargs: (hash_file or _sha256)(Path(settings["base_checkpoint"]) / "base.bin")
    root = tmp_path / "model"
    root.mkdir()
    (root / "base.bin").write_bytes(b"fixture base, not model weights")
    definition = replace(template, id=family, label="Fixture family", adapter=adapter_id, default_base=variant,
        defaults={"environment": "test", "backbone": "frozen", "base_revision": None, "gain": 2},
        backbone_modes=("frozen",), config_path=tmp_path / "fixture.yaml")
    base = replace(next(iter(definitions.BASE_MODELS.values())), id=variant, family=family,
                   label="Fixture checkpoint", checkpoint=str(root), source="fixture/weights", revision=None,
                   contract={**copy.deepcopy(definition.contract),
                             "observation": {"image_flip_vertical": True, "image_flip_horizontal": True}},
                   stats_key="fixture-stats", inference={"open_loop_steps": 2})
    monkeypatch.setitem(definitions.ADAPTERS, adapter_id, spec.__name__)
    monkeypatch.setitem(definitions.MODEL_FAMILIES, family, definition)
    monkeypatch.setitem(definitions.BASE_MODELS, variant, base)
    monkeypatch.setitem(sys.modules, spec.__name__, spec)

    backend = ModuleType(spec.BACKEND)
    def export_actor(components, settings, directory, **kwargs):
        path = directory / "weights/actor.bin"
        path.parent.mkdir()
        path.write_bytes(b"fixture actor")
        return {"schema_version": 6, "family": family, "weights": str(path.relative_to(directory)),
                "component_sha256": {"weights": _sha256(path)}}
    backend.export_actor = export_actor
    backend.actor_losses = lambda batch: [1.] * len(batch)
    monkeypatch.setitem(sys.modules, backend.__name__, backend)

    policy = ModuleType(spec.POLICY_ADAPTER)
    policy.PROVIDER = f"{policy.__name__}:Provider"
    policy.ARTIFACTS = policy.SNAPSHOT_FIELDS = ("weights",)
    policy.FORBIDDEN_FIELDS = ("action_head", "proprio_projector", "backbone", "actor", "base_identity")
    policy.base_settings = lambda identifier, settings: settings
    policy.identity_settings = lambda entry: entry.model_config
    policy.resolve_revision = lambda catalog, entry: spec.resolve_revision(entry.settings, catalog._component_sha256)
    def load_overlay(catalog, manifest, raw, *, parent=None):
        path = manifest.parent / raw["weights"]
        assert raw["schema_version"] == spec.MANIFEST_SCHEMA
        if catalog._component_sha256(path) != raw["component_sha256"]["weights"]:
            raise ValueError("Fixture weights changed")
        return PolicyEntry(policy_id=raw["policy_id"], label=raw["label"], family=family,
            base_checkpoint=raw["base_checkpoint"], stats_key=raw["stats_key"],
            manifest=manifest.resolve(), action_head=None, proprio_projector=None,
            training_step=raw["training_step"], compatibility_sha256=raw["compatibility_sha256"],
            model_config=raw["model_config"], component_sha256=raw["component_sha256"],
            base_revision=parent["revision"], parent=parent, algorithm=raw["algorithm"], artifacts={"weights": path.resolve()})
    policy.load_overlay = load_overlay
    class Provider:
        def __init__(self, runtime, eval_config, catalog):
            self.catalog, self.loaded = catalog, False
            self.current_policy_entry = self.current_policy_id = None
        def load(self, steps, policy_id, *, expected_content_sha256=None):
            self.current_policy_entry = self.catalog.select(policy_id)
            assert self.current_policy_entry.content_sha256 == expected_content_sha256
            self.current_policy_id, self.loaded = policy_id, True
        def unload(self):
            self.loaded = False
        def metadata(self):
            return {"provider": "fixture"}
    policy.Provider = Provider
    monkeypatch.setitem(sys.modules, policy.__name__, policy)
    settings = model_module("models").base_model_config(variant)
    return SimpleNamespace(family=family, variant=variant, settings=settings, spec=spec)


def test_new_family_uses_same_config_catalog_snapshot_and_provider_pipeline(tmp_path, adapter):
    models = model_module("models")
    assert models.model_backend({"model": adapter.settings}).actor_losses([1, 2]) == [1., 1.]
    assert models.model_parameters({"model": adapter.settings})["model_gain"] == 2
    # A family supporting only frozen mode must not be forced into full tuning by the service.
    OfflineJobService._validate_training_parameters({"algorithm": "bc", "model_family": adapter.family,
        "model_base_id": adapter.variant, "model_gain": 3}, models)
    assert adapter.variant in configured_models()
    pinned = model_module("model_assets").pin_model(adapter.settings)
    catalog = PolicyCatalog(tmp_path / "registry", base_models={adapter.variant: pinned})
    entry = catalog.select(adapter.variant)
    frozen = snapshot(entry)
    restored = catalog_from_snapshot(frozen, tmp_path / "empty")
    assert restored.select(adapter.variant).content_sha256 == entry.content_sha256
    provider = PolicyProvider(None, None, restored)
    provider.load(2, adapter.variant, expected_content_sha256=entry.content_sha256)
    assert provider.loaded and provider.current_policy_id == adapter.variant
    assert provider.metadata()["recording_action_codec"] == adapter.spec.RECORDING_ACTION_CODEC


def test_new_family_exports_custom_artifacts_and_preserves_frozen_parent(tmp_path, adapter):
    LoadedConfig = model_module("config").LoadedConfig
    _publish_overlay = model_module("training")._publish_overlay
    pinned = model_module("model_assets").pin_model(adapter.settings)
    parent = model_module("model_artifacts").parent_snapshot(pinned, pinned["base_revision"])
    components = SimpleNamespace(parent=parent, stats_key=pinned["stats_key"])
    config = LoadedConfig(tmp_path / "effective.yaml", {"model": pinned, "training": {"method": "bc"}, "reward": {}})
    checkpoint = tmp_path / "run/checkpoints/step-1"
    checkpoint.mkdir(parents=True)
    registry = tmp_path / "registry"
    manifest = _publish_overlay(checkpoint, registry, 1, config, components, {"dataset_sha256": "d" * 64}, None)
    raw = yaml.safe_load(manifest.read_text())
    assert raw["schema_version"] == 5 and raw["family"] == adapter.family
    catalog = PolicyCatalog(registry, base_models={adapter.variant: pinned})
    entry = catalog.select(raw["policy_id"])
    assert entry.artifact_path("weights").is_file()
    assert entry.parent == parent
    assert catalog_from_snapshot(snapshot(entry), tmp_path / "empty").select(entry.policy_id).content_sha256 == entry.content_sha256
    management = PolicyManagementService(SimpleNamespace(policy_catalog=catalog))
    assert management.detail(entry.policy_id)["components"][0]["name"] == "weights"
    copied = management.copy(entry.policy_id, "Copied fixture")
    copied_entry = catalog.select(copied["policy_id"])
    assert copied_entry.artifact_path("weights").relative_to(copied_entry.manifest.parent) == Path("weights/actor.bin")
    assert copied_entry.component_sha256 == entry.component_sha256
    assert catalog_from_snapshot(snapshot(copied_entry), tmp_path / "empty").select(copied_entry.policy_id).content_sha256 == entry.content_sha256
    assert json.loads((checkpoint.parent.parent / "export.json").read_text())["policy_id"] == entry.policy_id
