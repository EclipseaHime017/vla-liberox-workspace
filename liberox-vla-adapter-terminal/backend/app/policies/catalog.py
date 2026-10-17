"""Read-only policy overlay registry shared with the training project."""

from __future__ import annotations

import hashlib
import logging
import re
import copy
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import yaml

from .registry import model_module, policy_adapter


ACTION_HORIZON = 8
ACTION_DIM = 7
PROPRIO_DIM = 8
LOGGER = logging.getLogger(__name__)


class _UniqueKeyLoader(yaml.SafeLoader):
    pass


def _unique_mapping(loader: _UniqueKeyLoader, node: yaml.MappingNode, deep: bool = False):
    result: dict[Any, Any] = {}
    for key_node, value_node in node.value:
        key = loader.construct_object(key_node, deep=deep)
        if key in result:
            raise ValueError(f"Duplicate policy manifest key: {key!r}")
        result[key] = loader.construct_object(value_node, deep=deep)
    return result


_UniqueKeyLoader.add_constructor(
    yaml.resolver.BaseResolver.DEFAULT_MAPPING_TAG, _unique_mapping
)


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _stable_hash(value: dict[str, Any]) -> str:
    import json

    return hashlib.sha256(
        json.dumps(value, sort_keys=True, separators=(",", ":")).encode("utf-8")
    ).hexdigest()


@dataclass(frozen=True)
class PolicyEntry:
    policy_id: str
    label: str
    base_checkpoint: str
    stats_key: str
    manifest: Path | None
    action_head: Path | None
    proprio_projector: Path | None
    training_step: int | None
    compatibility_sha256: str | None
    algorithm: str = "iql"
    backbone: Path | None = None
    model_config: dict[str, Any] | None = None
    component_sha256: dict[str, str] = field(default_factory=dict)
    base_revision: str | None = None
    family: str = "vla_adapter"
    actor: Path | None = None
    base_identity: Path | None = None
    parent: dict | None = None
    contract: dict | None = None
    artifacts: dict[str, Path] = field(default_factory=dict)

    @property
    def settings(self) -> dict:
        raw = {**(self.model_config or {}), "family": self.family,
               "base_checkpoint": self.base_checkpoint, "stats_key": self.stats_key}
        settings = model_module("models").model_config({"model": raw})
        if self.contract is not None:
            settings["contract"] = model_module("base_models").validate_contract(self.contract, self.family)
        elif not self.is_base and "contract" not in settings:
            settings["contract"] = model_module("model_artifacts").legacy_contract(settings)
        return settings

    @property
    def content_sha256(self) -> str:
        settings = policy_adapter(self.family).identity_settings(self)
        value = {
            "family": self.family, "base_checkpoint": self.base_checkpoint,
            "stats_key": self.stats_key, "model_config": settings,
            "component_sha256": self.component_sha256,
            "base_revision": self.base_revision,
        }
        if self.parent is not None:
            value["parent"] = self.parent
        orientation = model_module("image_transforms").image_orientation(
            model_module("base_models").model_observation(self.settings))
        if orientation != "rotate_180":
            value["image_orientation"] = orientation
        return _stable_hash(value)

    def artifact_path(self, name):
        return getattr(self, name, None) or self.artifacts.get(name)

    @property
    def component_paths(self):
        return {name: self.artifact_path(name) for name in self.component_sha256}

    @property
    def is_base(self) -> bool:
        return self.manifest is None

    @property
    def inference(self) -> dict:
        return model_module("base_models").execution_config(self.settings)

    def resolve_open_loop_steps(self, value: int | None = None) -> int:
        return model_module("base_models").resolve_open_loop_steps(
            self.settings, value,
        )

    def public(self) -> dict[str, Any]:
        contract = model_module("image_transforms").canonical_contract(
            model_module("base_models").model_contract(self.settings))
        return {
            "policy_id": self.policy_id,
            "label": self.label,
            "base_checkpoint": self.base_checkpoint,
            "stats_key": self.stats_key,
            "kind": "base" if self.is_base else "rynn_iql_overlay",
            "algorithm": None if self.is_base else self.algorithm,
            "model_config": self.settings,
            "model_components": model_module("models").training_ui(self.family).get("components", []),
            "parent_model_id": self.parent["id"] if self.parent else self.settings["base_id"],
            "io": contract["io"],
            "observation": contract["observation"],
            "inference": self.inference,
            "training_step": self.training_step,
            "compatibility_sha256": self.compatibility_sha256,
            "family": self.family,
            "content_sha256": self.content_sha256,
            "base_revision": self.base_revision,
        }


class PolicyCatalog:
    """Read model registrations and validate policy manifests without loading weights."""

    REQUIRED = {
        "schema_version", "policy_id", "label", "base_checkpoint", "stats_key",
        "action_head", "proprio_projector", "action_horizon", "action_dim",
        "proprio_dim", "dataset_sha256", "reward_sha256", "training_step",
        "component_sha256", "compatibility_sha256",
    }

    def __init__(self, registry: Path, *, base_models: dict[str, dict],
                 base_revisions: dict[str, str] | None = None,
                 base_contracts: dict[str, dict] | None = None):
        self.registry = registry.expanduser().resolve()
        self.base_models = copy.deepcopy(base_models)
        self.base_contracts = copy.deepcopy(base_contracts or {})
        self._revisions = {(base_id, self.base_models[base_id]["base_checkpoint"]): revision
                           for base_id, revision in (base_revisions or {}).items()}
        self._entries: dict[str, PolicyEntry] = {}
        self._errors: dict[str, str] = {}
        self._hash_cache: dict[tuple[str, int, int, int, int], str] = {}
        self.refresh()

    def select(self, policy_id: str) -> PolicyEntry:
        """Resolve a base revision only on explicit selection, never list/poll."""
        self.refresh()
        selected = self.entry(policy_id)
        revision = policy_adapter(selected.family).resolve_revision(self, selected)
        pinned = selected.parent["revision"] if selected.parent else selected.settings.get("base_revision")
        if pinned is not None and pinned != revision:
            raise ValueError("Selected model parent weights have changed")
        if selected.parent is None:
            self._revisions[(selected.settings["base_id"], selected.base_checkpoint)] = revision
        self.refresh()
        return self.entry(policy_id)

    def _component_sha256(self, path: Path) -> str:
        stat = path.stat()
        key = (str(path.resolve()), stat.st_ino, stat.st_size, stat.st_mtime_ns, stat.st_ctime_ns)
        cached = self._hash_cache.get(key)
        if cached is not None:
            return cached
        digest = _sha256(path)
        self._hash_cache = {candidate: value for candidate, value in self._hash_cache.items() if candidate[0] != key[0]}
        self._hash_cache[key] = digest
        return digest

    def refresh(self) -> None:
        entries = {}
        errors: dict[str, str] = {}
        for base_id, supplied in self.base_models.items():
            base = model_module("base_models").base_model(base_id)
            settings = model_module("models").model_config({"model": supplied})
            if settings["base_id"] != base_id or settings["family"] != base.family:
                raise ValueError("Base policy ID conflicts with its model configuration")
            entries[base_id] = PolicyEntry(
                policy_id=base_id, label=f"{base.label}（基础模型）",
                base_checkpoint=settings["base_checkpoint"], stats_key=settings["stats_key"],
                manifest=None, action_head=None, proprio_projector=None, training_step=None,
                # Preserve the identity of existing default-base recordings.
                compatibility_sha256=None, model_config=policy_adapter(base.family).base_settings(base_id, settings), family=base.family,
                base_revision=settings.get("base_revision") or self._revisions.get((base_id, settings["base_checkpoint"])),
                contract=model_module("base_models").validate_contract(
                    self.base_contracts.get(base_id, model_module("base_models").model_contract(settings)), base.family),
            )
        if self.registry.is_dir():
            for directory in sorted(self.registry.iterdir()):
                if not directory.is_dir() or directory.is_symlink():
                    continue
                manifest = directory / "policy.yaml"
                if not manifest.is_file() or manifest.is_symlink():
                    continue
                try:
                    entry = self._load(manifest)
                except Exception as exc:
                    errors[directory.name] = str(exc)
                    LOGGER.warning("Ignoring invalid policy overlay %s: %s", directory, exc)
                    continue
                if entry.policy_id in entries:
                    raise ValueError(f"Duplicate policy_id in registry: {entry.policy_id}")
                entries[entry.policy_id] = entry
        self._entries = entries
        self._errors = errors

    def _load(self, manifest: Path) -> PolicyEntry:
        raw = yaml.load(manifest.read_text(encoding="utf-8"), Loader=_UniqueKeyLoader)
        raw, parent = model_module("model_artifacts").decode_child(raw)
        family = model_module("model_artifacts").manifest_family(raw)
        return policy_adapter(family).load_overlay(self, manifest, raw, parent=parent)

    def entry(self, policy_id: str) -> PolicyEntry:
        if policy_id in self._errors:
            raise ValueError(
                f"Invalid policy overlay {policy_id!r}: {self._errors[policy_id]}"
            )
        try:
            return self._entries[policy_id]
        except KeyError as exc:
            raise ValueError(f"Unknown policy_id: {policy_id}") from exc

    def list_policies(self) -> list[dict[str, Any]]:
        self.refresh()
        return [entry.public() for entry in self._entries.values()]
