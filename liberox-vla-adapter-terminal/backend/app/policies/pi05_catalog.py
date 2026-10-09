"""π₀.₅ artifact validation, without importing a model or GPU runtime."""
from __future__ import annotations

import json
import re
from pathlib import Path

from ..services.inherited_reward_inputs import offline_module
from .catalog import PolicyEntry, _stable_hash
from .registry import PROJECT


def assets():
    return offline_module(PROJECT, "pi05_assets")


def load_overlay(catalog, manifest: Path, raw: dict, *, parent=None) -> PolicyEntry:
    required = (catalog.REQUIRED - {"action_head", "proprio_projector"}) | {
        "family", "algorithm", "model_config", "actor", "base_identity"}
    if set(raw) != required or raw["family"] != "pi05" or raw["algorithm"] not in {"bc", "iql"}:
        raise ValueError("Invalid π₀.₅ policy manifest")
    settings = offline_module(PROJECT, "models").model_config({"model": raw["model_config"]})
    if settings["family"] != "pi05" or settings["base_checkpoint"] != raw["base_checkpoint"]:
        raise ValueError("π₀.₅ model identity conflicts with its manifest")
    if raw["stats_key"] != settings["stats_key"]:
        raise ValueError("π₀.₅ normalization identity conflicts with its manifest")
    if not isinstance(raw["policy_id"], str) or manifest.parent.name != raw["policy_id"]:
        raise ValueError("π₀.₅ directory must match its policy ID")
    if not isinstance(raw["label"], str) or not raw["label"].strip():
        raise ValueError("π₀.₅ policy label is empty")
    if type(raw["training_step"]) is not int or raw["training_step"] < 1:
        raise ValueError("Invalid π₀.₅ training step")
    if any(type(raw[key]) is not int or raw[key] != value for key, value in
           (("action_horizon", 8), ("action_dim", 7), ("proprio_dim", 8))):
        raise ValueError("π₀.₅ overlay must use platform replay geometry 8/7/8")
    compatibility = {key: raw[key] for key in ("base_checkpoint", "stats_key", "action_horizon", "action_dim", "proprio_dim")}
    if _stable_hash(compatibility) != raw["compatibility_sha256"]:
        raise ValueError("π₀.₅ compatibility hash mismatch")
    for key in ("dataset_sha256", "reward_sha256"):
        if key == "reward_sha256" and raw["algorithm"] == "bc" and raw[key] is None:
            continue
        if re.fullmatch(r"[0-9a-f]{64}", str(raw[key])) is None:
            raise ValueError(f"Invalid π₀.₅ {key}")
    if set(raw["component_sha256"]) != {"actor", "base_identity"}:
        raise ValueError("Invalid π₀.₅ artifact set")
    paths = {}
    for key in ("actor", "base_identity"):
        if raw[key] != {"actor": "actor.pt", "base_identity": "pi05_identity.json"}[key]:
            raise ValueError(f"Unexpected π₀.₅ artifact filename: {key}")
        value = manifest.parent / raw[key]
        if value.is_symlink() or value.resolve().parent != manifest.parent.resolve() or not value.is_file():
            raise ValueError(f"Missing or unsafe π₀.₅ artifact: {key}")
        if catalog._component_sha256(value) != raw["component_sha256"][key]:
            raise ValueError(f"π₀.₅ artifact hash mismatch: {key}")
        paths[key] = value.resolve()
    identity = json.loads(paths["base_identity"].read_text())
    if parent is not None and assets().identity_digest(identity) != parent["revision"]:
        raise ValueError("π₀.₅ child identity conflicts with its parent")
    return PolicyEntry(
        policy_id=raw["policy_id"], label=raw["label"], base_checkpoint=raw["base_checkpoint"],
        stats_key=raw["stats_key"], manifest=manifest.resolve(), action_head=None,
        proprio_projector=None, training_step=raw["training_step"],
        compatibility_sha256=raw["compatibility_sha256"], algorithm=raw["algorithm"],
        model_config=settings, component_sha256=raw["component_sha256"], family="pi05",
        base_revision=assets().identity_digest(identity), parent=parent, **paths)
