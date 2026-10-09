"""Model-owned job snapshots; task services treat these as opaque selections."""
from pathlib import Path
import re

from .catalog import PolicyCatalog
from .registry import model_module

COMMON = {"policy_id", "label", "base_checkpoint", "stats_key", "manifest", "action_head",
          "proprio_projector", "training_step", "compatibility_sha256", "family",
          "content_sha256", "base_revision"}
ARTIFACTS = {"vla_adapter": ("action_head", "proprio_projector"), "pi05": ("actor", "base_identity")}


def snapshot(entry):
    value = {key: getattr(entry, key) for key in COMMON}
    value["model_config"] = entry.settings
    if entry.backbone is not None:
        value["backbone"] = entry.backbone
    if entry.family == "pi05":
        value.update(actor=entry.actor, base_identity=entry.base_identity)
    return {key: str(item) if isinstance(item, Path) else item for key, item in value.items()}


def validate_snapshot(value):
    if not isinstance(value, dict) or value.get("family") not in ARTIFACTS:
        raise ValueError("Unsupported policy snapshot family")
    required = COMMON | ({"model_config", "actor", "base_identity"} if value["family"] == "pi05" else set())
    allowed = required | {"model_config", "backbone"}
    if not required <= value.keys() or value.keys() - allowed:
        raise ValueError("policy_snapshot keys mismatch")
    for key in ("policy_id", "label", "base_checkpoint", "stats_key"):
        if not isinstance(value[key], str) or not value[key].strip():
            raise ValueError(f"Invalid policy_snapshot.{key}")
    if Path(value["policy_id"]).name != value["policy_id"] or value["policy_id"] in {".", ".."}:
        raise ValueError("Unsafe policy snapshot ID")
    if re.fullmatch(r"[0-9a-f]{64}", str(value["content_sha256"])) is None:
        raise ValueError("Unverified policy snapshot; register the evaluation again")
    if re.fullmatch(r"(?:[0-9a-f]{40}|[0-9a-f]{64})", str(value["base_revision"])) is None:
        raise ValueError("Base checkpoint revision must be immutable; register the evaluation again")
    settings = None
    if "model_config" in value:
        settings = model_module("models").model_config({"model": value["model_config"]})
        if any(settings[key] != value[key] for key in ("family", "base_checkpoint", "stats_key")):
            raise ValueError("Snapshot model settings conflict with its identity")
    if value["family"] == "pi05" and any(value.get(key) is not None for key in ("action_head", "proprio_projector", "backbone")):
        raise ValueError("π₀.₅ snapshots cannot contain VLA-Adapter artifacts")
    if value["manifest"] is None:
        if value["policy_id"] != (settings["base_id"] if settings else "base"):
            raise ValueError("Base policy ID does not match its model family")
        for key in ("action_head", "proprio_projector", "backbone", "actor", "base_identity", "training_step", "compatibility_sha256"):
            if value.get(key) is not None:
                raise ValueError(f"Base policy snapshot must set {key} to null")
    else:
        for key in ("manifest", *ARTIFACTS[value["family"]], *(("backbone",) if "backbone" in value else ())):
            name = value[key]
            if not isinstance(name, str) or not name.strip():
                raise ValueError(f"Invalid policy_snapshot.{key}")
            path = Path(name).expanduser()
            if not path.is_absolute() or path.is_symlink() or not path.is_file():
                raise FileNotFoundError(f"policy_snapshot.{key} is not a safe existing absolute file")
            value[key] = str(path.resolve())
        if type(value["training_step"]) is not int or value["training_step"] < 1:
            raise ValueError("Invalid policy_snapshot.training_step")
        if re.fullmatch(r"[0-9a-f]{64}", str(value["compatibility_sha256"])) is None:
            raise ValueError("policy_snapshot.compatibility_sha256 must be SHA256")
    return value


def catalog_from_snapshot(value, empty_registry):
    validate_snapshot(value)
    manifest = value["manifest"]
    registry = Path(manifest).parent.parent if manifest else empty_registry
    settings = value.get("model_config", {"family": value["family"],
        "base_checkpoint": value["base_checkpoint"], "stats_key": value["stats_key"]})
    base_id = model_module("models").model_config({"model": settings})["base_id"]
    catalog = PolicyCatalog(registry, base_models={base_id: settings},
                            base_revisions={base_id: value["base_revision"]})
    entry = catalog.select(value["policy_id"])
    if entry.content_sha256 != value["content_sha256"]:
        raise ValueError("Selected policy weights changed after evaluation registration")
    if manifest:
        for key in ("manifest", *ARTIFACTS[entry.family], *(("backbone",) if "backbone" in value else ())):
            if getattr(entry, key) != Path(value[key]).resolve():
                raise ValueError("Policy snapshot does not match its validated registry entry")
        if (entry.training_step, entry.compatibility_sha256) != (value["training_step"], value["compatibility_sha256"]):
            raise ValueError("Policy snapshot metadata differs from the validated model")
    return catalog
