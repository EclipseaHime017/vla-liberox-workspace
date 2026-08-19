"""Read-only policy overlay registry shared with the training project."""

from __future__ import annotations

import hashlib
import logging
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import yaml


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
            "training_step": self.training_step,
            "compatibility_sha256": self.compatibility_sha256,
        }


class PolicyCatalog:
    """Validate manifests without importing the independent training system."""

    REQUIRED = {
        "schema_version", "policy_id", "label", "base_checkpoint", "stats_key",
        "action_head", "proprio_projector", "action_horizon", "action_dim",
        "proprio_dim", "dataset_sha256", "reward_sha256", "training_step",
        "component_sha256", "compatibility_sha256",
    }

    def __init__(self, registry: Path, base_checkpoint: str, stats_key: str):
        self.registry = registry.expanduser().resolve()
        self.base_checkpoint = str(base_checkpoint)
        self.stats_key = str(stats_key)
        self._entries: dict[str, PolicyEntry] = {}
        self._errors: dict[str, str] = {}
        self.refresh()

    def refresh(self) -> None:
        entries = {
            "base": PolicyEntry(
                policy_id="base",
                label="VLA-Adapter · Object-Pro（基础模型）",
                base_checkpoint=self.base_checkpoint,
                stats_key=self.stats_key,
                manifest=None,
                action_head=None,
                proprio_projector=None,
                training_step=None,
                compatibility_sha256=None,
            )
        }
        errors: dict[str, str] = {}
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
        if not isinstance(raw, dict) or set(raw) != self.REQUIRED:
            raise ValueError(f"Invalid policy overlay keys: {manifest}")
        if raw["schema_version"] != 1:
            raise ValueError(f"Unsupported policy overlay schema: {manifest}")
        policy_id = raw["policy_id"]
        label = raw["label"]
        if not isinstance(policy_id, str) or not policy_id or Path(policy_id).name != policy_id:
            raise ValueError(f"Unsafe policy_id in {manifest}")
        if manifest.parent.name != policy_id:
            raise ValueError(f"Policy directory must match policy_id {policy_id!r}")
        if not isinstance(label, str) or not label.strip():
            raise ValueError(f"Policy label must be a non-empty string: {manifest}")
        if raw["base_checkpoint"] != self.base_checkpoint:
            raise ValueError(
                f"Overlay {policy_id} uses {raw['base_checkpoint']!r}; "
                f"UI is configured for {self.base_checkpoint!r}"
            )
        allowed_stats = {self.stats_key, f"{self.stats_key}_no_noops"}
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
        if not isinstance(hashes, dict) or set(hashes) != {"action_head", "proprio_projector"}:
            raise ValueError(f"Overlay {policy_id} has invalid component hashes")
        for key in ("dataset_sha256", "reward_sha256", "compatibility_sha256"):
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
                or _sha256(path) != hashes[key]
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
