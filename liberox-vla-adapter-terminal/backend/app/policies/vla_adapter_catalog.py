"""VLA-Adapter policy assets and legacy overlay format."""
from pathlib import Path
import re

from .catalog import PolicyEntry, _stable_hash
from .registry import model_module

PROVIDER = "backend.app.policies.vla_adapter:VLAAdapterPolicyProvider"
ARTIFACTS = ("action_head", "proprio_projector")
SNAPSHOT_FIELDS = ()
FORBIDDEN_FIELDS = ()


def base_settings(base_id, settings):
    return None if base_id == "base" else settings


def identity_settings(entry):
    return entry.model_config


def resolve_revision(catalog, entry):
    settings = entry.settings
    pinned = entry.parent["revision"] if entry.parent else settings.get("base_revision")
    if pinned is None and not Path(entry.base_checkpoint).expanduser().is_dir():
        pinned = entry.base_revision
    return model_module("checkpoint_assets").resolve_revision(
        entry.base_checkpoint, pinned, catalog._component_sha256)


def load_overlay(catalog, manifest, raw, *, parent=None):
    required = catalog.REQUIRED | ({"algorithm"} if isinstance(raw, dict) and raw.get("schema_version") in (2, 3) else set())
    if isinstance(raw, dict) and raw.get("schema_version") == 3:
        required |= {"backbone", "model_config"}
    if not isinstance(raw, dict) or set(raw) != required:
        raise ValueError(f"Invalid policy overlay keys: {manifest}")
    if raw["schema_version"] not in (1, 2, 3):
        raise ValueError(f"Unsupported policy overlay schema: {manifest}")
    if raw["schema_version"] == 3:
        config = raw["model_config"]
        if (not isinstance(config, dict) or model_module("base_models").family_adapter(config.get("family")).POLICY_ADAPTER != __name__
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
    configured = catalog.base_models.get(settings["base_id"])
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
    io = model_module("model_artifacts").legacy_contract(settings)["io"]
    if dimensions != (io["replay_horizon"], io["action_dim"], io["proprio_dim"]):
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
            or catalog._component_sha256(path) != hashes[key]
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
        base_revision=parent["revision"] if parent else settings.get("base_revision") or catalog._revisions.get((settings["base_id"], raw["base_checkpoint"])),
        parent=parent,
        family=settings["family"],
    )
