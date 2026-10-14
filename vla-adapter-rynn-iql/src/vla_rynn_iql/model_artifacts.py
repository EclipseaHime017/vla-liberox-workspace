"""Versioned parent identity for trained child models; no model/GPU imports."""
from __future__ import annotations

import copy
import json
import re
from pathlib import Path
import shutil
import tempfile

import yaml

from .base_models import base_model, model_contract, validate_contract
from .io import stable_hash, sha256_file
from .config_sources import UniqueKeyLoader


def parent_snapshot(settings: dict, revision: str, *, contract: dict | None = None) -> dict:
    base = base_model(settings.get("base_id", "base" if settings["family"] == "vla_adapter" else "pi05-libero-base"))
    value = {"id": base.id, "family": base.family, "checkpoint": settings["base_checkpoint"],
             "stats_key": settings["stats_key"], "revision": revision,
             "source": base.source, "source_revision": base.revision,
             "contract": contract or model_contract(settings)}
    value["sha256"] = stable_hash(value)
    validate_parent(value, settings)
    return value


def validate_parent(parent: dict, settings: dict) -> None:
    required = {"id", "family", "checkpoint", "stats_key", "revision", "source", "source_revision", "contract", "sha256"}
    if not isinstance(parent, dict) or set(parent) != required:
        raise ValueError("Child model requires a complete frozen parent snapshot")
    expected_hash = stable_hash({key: value for key, value in parent.items() if key != "sha256"})
    if parent["sha256"] != expected_hash:
        raise ValueError("Parent model snapshot hash mismatch")
    if not isinstance(parent["revision"], str) or re.fullmatch(r"(?:[0-9a-f]{40}|[0-9a-f]{64})", parent["revision"]) is None:
        raise ValueError("Child model requires an immutable parent revision")
    base = base_model(parent["id"])
    if (parent["family"] != base.family or parent["family"] != settings["family"]
            or parent["id"] != settings["base_id"]
            or parent["checkpoint"] != settings["base_checkpoint"]
            or parent["stats_key"] != settings["stats_key"]):
        raise ValueError("Child model conflicts with its parent model")
    validate_contract(parent["contract"], parent["family"])
    if settings.get("base_revision") not in (None, parent["revision"]):
        raise ValueError("Child revision conflicts with parent revision")
    if "contract" in settings and settings["contract"] != parent["contract"]:
        raise ValueError("Child model cannot override inherited input/output contract")


def decode_child(raw: dict) -> tuple[dict, dict | None]:
    """Normalize schema 5 at the boundary; schemas 1–4 retain legacy identity."""
    if not isinstance(raw, dict):
        raise ValueError("Model manifest must be a mapping")
    if raw.get("schema_version") != 5:
        return raw, None
    from .models import model_config
    value = copy.deepcopy(raw)
    parent = value.pop("parent", None)
    supplied = value.get("model_config")
    if not isinstance(supplied, dict) or not isinstance(parent, dict):
        raise ValueError("Child model requires model_config and parent mappings")
    settings = model_config({"model": {"contract": parent.get("contract"), **supplied}})
    validate_parent(parent, settings)
    settings["contract"] = copy.deepcopy(parent["contract"])
    settings["base_revision"] = parent["revision"]
    value["model_config"] = settings
    if value.get("family") != parent["family"]:
        raise ValueError("Child type does not match parent type")
    if value.get("base_checkpoint") != parent["checkpoint"] or value.get("stats_key") != parent["stats_key"]:
        raise ValueError("Child manifest checkpoint conflicts with parent")
    io = parent["contract"]["io"]
    if (value.get("action_horizon"), value.get("action_dim"), value.get("proprio_dim")) != (
            io["replay_horizon"], io["action_dim"], io["proprio_dim"]):
        raise ValueError("Child geometry conflicts with parent I/O")
    if parent["family"] == "vla_adapter":
        value.pop("family")
        value["schema_version"] = 3
    else:
        value["schema_version"] = 4
    return value, parent


def validate_policy(path: Path, payload: dict) -> None:
    """Validate a native child export without loading its model or parent weights."""
    native, parent = decode_child(payload)
    if parent is None:
        raise ValueError("Deployment requires a schema 5 child policy")
    common = {"schema_version", "policy_id", "label", "base_checkpoint", "stats_key",
              "algorithm", "model_config", "action_horizon", "action_dim", "proprio_dim",
              "dataset_sha256", "reward_sha256", "training_step", "component_sha256",
              "compatibility_sha256"}
    settings = native["model_config"]
    if parent["family"] == "vla_adapter":
        required = common | {"action_head", "proprio_projector", "backbone"}
        names = {"action_head", "proprio_projector"}
        if (settings["backbone"] == "frozen") != (native.get("backbone") is None):
            raise ValueError("Policy backbone artifact conflicts with its model configuration")
        if native.get("backbone") is not None:
            names.add("backbone")
    else:
        required = common | {"family", "actor", "base_identity"}
        names = {"actor", "base_identity"}
    if set(native) != required:
        raise ValueError("Invalid child policy manifest fields")
    for key in ("policy_id", "label", "base_checkpoint", "stats_key"):
        if not isinstance(native[key], str) or not native[key].strip():
            raise ValueError(f"Invalid policy field: {key}")
    if (native["policy_id"] in {".", ".."} or Path(native["policy_id"]).name != native["policy_id"]
            or path.parent.name != native["policy_id"]):
        raise ValueError("Policy directory must match its safe policy_id")
    if native["algorithm"] not in {"bc", "iql"}:
        raise ValueError("Unknown policy training algorithm")
    for key in ("action_horizon", "action_dim", "proprio_dim", "training_step"):
        if type(native[key]) is not int or native[key] < 1:
            raise ValueError(f"Invalid policy integer: {key}")
    for key in ("dataset_sha256", "reward_sha256"):
        if key == "reward_sha256" and native["algorithm"] == "bc":
            if native[key] is not None:
                raise ValueError("BC policy must not reference rewards")
        elif not isinstance(native[key], str) or re.fullmatch(r"[0-9a-f]{64}", native[key]) is None:
            raise ValueError(f"Invalid policy hash: {key}")
    compatibility = {key: native[key] for key in
                     ("base_checkpoint", "stats_key", "action_horizon", "action_dim", "proprio_dim")}
    if native["compatibility_sha256"] != stable_hash(compatibility):
        raise ValueError("Policy compatibility hash mismatch")
    hashes = native["component_sha256"]
    if not isinstance(hashes, dict) or set(hashes) != names:
        raise ValueError("Policy component hash set does not match its model configuration")
    for key in names:
        name = native[key]
        if not isinstance(name, str) or name in {".", ".."} or Path(name).name != name:
            raise ValueError(f"Unsafe policy component: {key}")
        if parent["family"] == "pi05" and name != {"actor": "actor.pt", "base_identity": "pi05_identity.json"}[key]:
            raise ValueError(f"Unexpected policy component filename: {key}")
        component = path.parent / name
        if component.is_symlink() or not component.is_file() or sha256_file(component) != hashes[key]:
            raise ValueError(f"Policy component hash mismatch: {key}")
    if parent["family"] == "pi05":
        from .pi05_assets import identity_digest
        identity = json.loads((path.parent / native["base_identity"]).read_text())
        if identity_digest(identity) != parent["revision"]:
            raise ValueError("Policy checkpoint identity conflicts with its parent")


def export_policy(source: Path, destination: Path, base_checkpoint: str) -> Path:
    """Publish a deployment copy; rebind location without changing weight identity."""
    source, destination = source.expanduser().resolve(), destination.expanduser().absolute()
    if not isinstance(base_checkpoint, str) or not base_checkpoint.strip():
        raise ValueError("Deployment requires a base checkpoint path or repository ID")
    if destination.exists():
        raise FileExistsError(f"Deployment destination already exists: {destination}")
    payload = yaml.load(source.read_text(), Loader=UniqueKeyLoader)
    validate_policy(source, payload)
    payload = copy.deepcopy(payload)
    payload["policy_id"] = destination.name
    payload["base_checkpoint"] = base_checkpoint
    payload["model_config"]["base_checkpoint"] = base_checkpoint
    parent = payload.get("parent")
    if parent is not None:
        parent["checkpoint"] = base_checkpoint
        parent["sha256"] = stable_hash({key: value for key, value in parent.items() if key != "sha256"})
    compatibility = {key: payload[key] for key in
                     ("base_checkpoint", "stats_key", "action_horizon", "action_dim", "proprio_dim")}
    payload["compatibility_sha256"] = stable_hash(compatibility)
    decode_child(payload)
    destination.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.TemporaryDirectory(prefix=".policy-export-", dir=destination.parent) as temporary:
        staging = Path(temporary) / destination.name
        staging.mkdir()
        hashes = payload.get("component_sha256")
        if not isinstance(hashes, dict) or not hashes:
            raise ValueError("Policy is missing component hashes")
        for key, digest in hashes.items():
            name = payload.get(key)
            if not isinstance(name, str) or Path(name).name != name:
                raise ValueError(f"Unsafe policy component: {key}")
            path = source.parent / name
            if path.is_symlink() or not path.is_file() or sha256_file(path) != digest:
                raise ValueError(f"Policy component hash mismatch: {key}")
            shutil.copyfile(path, staging / name)
            if sha256_file(staging / name) != digest:
                raise ValueError(f"Policy changed during export: {key}")
        (staging / "policy.yaml").write_text(yaml.safe_dump(payload, sort_keys=False))
        validate_policy(staging / "policy.yaml", payload)
        if destination.exists():
            raise FileExistsError(f"Deployment destination already exists: {destination}")
        staging.rename(destination)
    return destination / "policy.yaml"
