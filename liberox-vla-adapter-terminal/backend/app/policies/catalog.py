"""Read-only policy overlay registry shared with the training project."""

from __future__ import annotations

import hashlib
import logging
import re
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import yaml

from .registry import model_module


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

    @property
    def settings(self) -> dict:
        raw = {**(self.model_config or {}), "family": self.family,
               "base_checkpoint": self.base_checkpoint, "stats_key": self.stats_key}
        return model_module("models").model_config({"model": raw})

    @property
    def content_sha256(self) -> str:
        settings = self.model_config
        if self.family == "pi05" and settings is not None:
            settings = dict(settings)
            # Legacy LIBERO recordings predate the explicit base selector.
            if settings.get("base_id") == "pi05-libero-base":
                settings.pop("base_id")
        value = {
            "family": self.family, "base_checkpoint": self.base_checkpoint,
            "stats_key": self.stats_key, "model_config": settings,
            "component_sha256": self.component_sha256,
            "base_revision": self.base_revision,
        }
        if self.parent is not None:
            value["parent"] = self.parent
        return _stable_hash(value)

    @property
    def is_base(self) -> bool:
        return self.manifest is None

    def public(self) -> dict[str, Any]:
        return {
            "policy_id": self.policy_id,
            "label": self.label,
            "base_checkpoint": self.base_checkpoint,
            "stats_key": self.stats_key,
            "kind": "base" if self.is_base else "rynn_iql_overlay",
            "algorithm": None if self.is_base else self.algorithm,
            "model_config": self.settings,
            "parent_model_id": self.parent["id"] if self.parent else self.settings["base_id"],
            "io": model_module("base_models").model_contract(self.settings)["io"],
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
                 base_revisions: dict[str, str] | None = None):
        self.registry = registry.expanduser().resolve()
        self.base_models = base_models
        self._revisions = {(base_id, base_models[base_id]["base_checkpoint"]): revision
                           for base_id, revision in (base_revisions or {}).items()}
        self._entries: dict[str, PolicyEntry] = {}
        self._errors: dict[str, str] = {}
        self._hash_cache: dict[tuple[str, int, int, int, int], str] = {}
        self.refresh()

    def select(self, policy_id: str) -> PolicyEntry:
        """Resolve a base revision only on explicit selection, never list/poll."""
        self.refresh()
        selected = self.entry(policy_id)
        if selected.family == "pi05":
            from .pi05_catalog import assets
            identity = assets().checkpoint_identity(Path(selected.base_checkpoint), self._component_sha256,
                base_id=selected.settings["base_id"], parent=selected.parent,
                contract=model_module("base_models").model_contract(selected.settings))
            revision = assets().identity_digest(identity)
            if selected.base_identity is not None:
                import json
                if json.loads(selected.base_identity.read_text()) != identity:
                    raise ValueError("π₀.₅ overlay requires different base weights or normalization assets")
            pinned = selected.parent["revision"] if selected.parent else selected.settings.get("base_revision")
            if pinned is not None and pinned != revision:
                raise ValueError("Selected model parent weights have changed")
            if selected.parent is None:
                self._revisions[(selected.settings["base_id"], selected.base_checkpoint)] = revision
            self.refresh()
            return self.entry(policy_id)
        settings = selected.settings
        pinned = selected.parent["revision"] if selected.parent else settings.get("base_revision")
        if pinned is None and not Path(selected.base_checkpoint).expanduser().is_dir():
            pinned = selected.base_revision
        revision = model_module("checkpoint_assets").resolve_revision(
            selected.base_checkpoint, pinned, self._component_sha256)
        if selected.parent is None:
            self._revisions[(settings["base_id"], selected.base_checkpoint)] = revision
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
                compatibility_sha256=None, model_config=None if base_id == "base" else settings, family=base.family,
                base_revision=settings.get("base_revision") or self._revisions.get((base_id, settings["base_checkpoint"])),
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
        if isinstance(raw, dict) and raw.get("schema_version") == 4:
            from .pi05_catalog import load_overlay
            return load_overlay(self, manifest, raw, parent=parent)
        required = self.REQUIRED | ({"algorithm"} if isinstance(raw, dict) and raw.get("schema_version") in (2, 3) else set())
        if isinstance(raw, dict) and raw.get("schema_version") == 3:
            required |= {"backbone", "model_config"}
        if not isinstance(raw, dict) or set(raw) != required:
            raise ValueError(f"Invalid policy overlay keys: {manifest}")
        if raw["schema_version"] not in (1, 2, 3):
            raise ValueError(f"Unsupported policy overlay schema: {manifest}")
        if raw["schema_version"] == 3:
            config = raw["model_config"]
            if (not isinstance(config, dict) or config.get("family") != "vla_adapter"
                    or config.get("backbone") not in ("frozen", "full", "lora")):
                raise ValueError("Unsupported overlay model configuration")
            if (config["backbone"] == "frozen") != (raw["backbone"] is None):
                raise ValueError("Overlay backbone artifact does not match its model configuration")
            if config["backbone"] != "frozen" and (not isinstance(raw["backbone"], str) or not raw["backbone"].strip()):
                raise ValueError("Adapted overlay backbone must be a non-empty path")
        algorithm = raw.get("algorithm", "iql")
        if algorithm not in ("iql", "bc"):
            raise ValueError("Unknown policy overlay algorithm")
        policy_id = raw["policy_id"]
        label = raw["label"]
        if not isinstance(policy_id, str) or not policy_id or Path(policy_id).name != policy_id:
            raise ValueError(f"Unsafe policy_id in {manifest}")
        if manifest.parent.name != policy_id:
            raise ValueError(f"Policy directory must match policy_id {policy_id!r}")
        if not isinstance(label, str) or not label.strip():
            raise ValueError(f"Policy label must be a non-empty string: {manifest}")
        supplied_settings = raw.get("model_config") or {}
        if (supplied_settings.get("base_checkpoint", raw["base_checkpoint"]) != raw["base_checkpoint"]
                or raw["stats_key"] not in {supplied_settings.get("stats_key", raw["stats_key"]),
                                           f"{supplied_settings.get('stats_key')}_no_noops"}):
            raise ValueError("Overlay model configuration conflicts with its manifest")
        settings = model_module("models").model_config({"model": {**supplied_settings,
            "base_checkpoint": raw["base_checkpoint"], "stats_key": raw["stats_key"]}})
        configured = self.base_models.get(settings["base_id"])
        expected_checkpoint = parent["checkpoint"] if parent else (
            configured["base_checkpoint"] if configured else settings["base_checkpoint"])
        if raw["base_checkpoint"] != expected_checkpoint:
            raise ValueError(
                f"Overlay {policy_id} uses {raw['base_checkpoint']!r}; "
                f"its parent is configured for {expected_checkpoint!r}"
            )
        parent_stats = parent["stats_key"] if parent else (configured["stats_key"] if configured else settings["stats_key"])
        allowed_stats = {parent_stats, f"{parent_stats}_no_noops"}
        if raw["stats_key"] not in allowed_stats:
            raise ValueError(f"Overlay {policy_id} has incompatible stats_key")
        dimensions = (raw["action_horizon"], raw["action_dim"], raw["proprio_dim"])
        if dimensions != (ACTION_HORIZON, ACTION_DIM, PROPRIO_DIM):
            raise ValueError(f"Overlay {policy_id} has incompatible action/proprio dimensions")
        compatibility = {
            "base_checkpoint": raw["base_checkpoint"],
            "stats_key": raw["stats_key"],
            "action_horizon": raw["action_horizon"],
            "action_dim": raw["action_dim"],
            "proprio_dim": raw["proprio_dim"],
        }
        if raw["compatibility_sha256"] != _stable_hash(compatibility):
            raise ValueError(f"Overlay {policy_id} compatibility hash mismatch")
        hashes = raw["component_sha256"]
        names = {"action_head", "proprio_projector"} | ({"backbone"} if raw.get("backbone") else set())
        if not isinstance(hashes, dict) or set(hashes) != names:
            raise ValueError(f"Overlay {policy_id} has invalid component hashes")
        for key in ("dataset_sha256", "reward_sha256", "compatibility_sha256"):
            if key == "reward_sha256" and algorithm == "bc":
                if raw[key] is not None:
                    raise ValueError("BC overlays must not reference rewards")
                continue
            if not isinstance(raw[key], str) or re.fullmatch(r"[0-9a-f]{64}", raw[key]) is None:
                raise ValueError(f"Overlay {policy_id} has invalid {key}")
        if type(raw["training_step"]) is not int or raw["training_step"] < 1:
            raise ValueError(f"Overlay {policy_id} has invalid training_step")
        for key in ("action_horizon", "action_dim", "proprio_dim"):
            if type(raw[key]) is not int:
                raise ValueError(f"Overlay {policy_id} has invalid {key}")

        def component(key: str) -> Path:
            value = raw[key]
            if not isinstance(value, str) or not value:
                raise ValueError(f"Overlay component {key} must be a path")
            path = (manifest.parent / value).resolve()
            try:
                path.relative_to(manifest.parent.resolve())
            except ValueError as exc:
                raise ValueError(f"Overlay component {key} escapes its policy directory") from exc
            if not path.is_file() or path.is_symlink():
                raise ValueError(f"Overlay component {key} is missing or unsafe")
            if (
                not isinstance(hashes[key], str)
                or re.fullmatch(r"[0-9a-f]{64}", hashes[key]) is None
                or self._component_sha256(path) != hashes[key]
            ):
                raise ValueError(f"Overlay component {key} hash mismatch")
            return path

        return PolicyEntry(
            policy_id=policy_id,
            label=label.strip(),
            base_checkpoint=raw["base_checkpoint"],
            stats_key=raw["stats_key"],
            manifest=manifest.resolve(),
            action_head=component("action_head"),
            proprio_projector=component("proprio_projector"),
            training_step=int(raw["training_step"]),
            compatibility_sha256=str(raw["compatibility_sha256"]),
            algorithm=algorithm,
            backbone=component("backbone") if raw.get("backbone") else None,
            model_config=raw.get("model_config"),
            component_sha256=dict(hashes),
            base_revision=parent["revision"] if parent else settings.get("base_revision") or self._revisions.get((settings["base_id"], raw["base_checkpoint"])),
            parent=parent,
        )

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
