"""Versioned parent identity for trained child models; no model/GPU imports."""
from __future__ import annotations

import copy
import re

from .base_models import base_model, model_contract, validate_contract
from .io import stable_hash


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
