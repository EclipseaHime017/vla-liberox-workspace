"""Model capabilities and configuration, independent of training algorithms/GPU imports."""
from __future__ import annotations

import copy
import math
from dataclasses import dataclass
from typing import Any


@dataclass(frozen=True)
class ModelDefinition:
    id: str
    label: str
    backend: str
    backbone_modes: tuple[str, ...]


MODELS = {
    "vla_adapter": ModelDefinition("vla_adapter", "VLA-Adapter · Object-Pro",
                                   "vla_rynn_iql.vla_adapter", ("frozen", "lora", "full")),
}
DEFAULT_MODEL = {
    "family": "vla_adapter", "backbone": "frozen", "action_head": "train",
    "proprio_projector": "train", "lora": {"rank": 32, "alpha": 64, "dropout": 0.0},
    "base_checkpoint": "VLA-Adapter/LIBERO-Object-Pro",
    "stats_key": "libero_object", "use_pro_version": True,
}
MODEL_PARAMETER_PATHS = {
    "model_family": ("family",), "model_backbone": ("backbone",),
    "model_action_head": ("action_head",), "model_proprio_projector": ("proprio_projector",),
    "model_lora_rank": ("lora", "rank"), "model_lora_alpha": ("lora", "alpha"),
    "model_lora_dropout": ("lora", "dropout"),
}


def model_config(raw: dict[str, Any]) -> dict[str, Any]:
    supplied = raw.get("model", {})
    if not isinstance(supplied, dict):
        raise TypeError("model must be a mapping")
    unknown = supplied.keys() - DEFAULT_MODEL.keys()
    if unknown:
        raise ValueError(f"Unknown model keys: {sorted(unknown)}")
    result = copy.deepcopy(DEFAULT_MODEL)
    old = raw.get("vla", {})
    if not isinstance(old, dict) or old.keys() - {"base_checkpoint", "stats_key", "use_pro_version", "freeze_backbone"}:
        raise ValueError("Invalid legacy vla configuration")
    result.update({key: value for key, value in old.items() if key != "freeze_backbone"})
    legacy = raw.get("vla", {}).get("freeze_backbone", True)
    if type(legacy) is not bool:
        raise TypeError("vla.freeze_backbone must be boolean")
    result["backbone"] = "frozen" if legacy else "full"
    result.update(supplied)
    for key in ("base_checkpoint", "stats_key"):
        if not isinstance(result[key], str) or not result[key].strip():
            raise ValueError(f"model.{key} must be a non-empty string")
    if result["use_pro_version"] is not True:
        raise ValueError("The VLA-Adapter backend requires Pro components")
    lora = supplied.get("lora", {})
    if not isinstance(lora, dict):
        raise TypeError("model.lora must be a mapping")
    if lora.keys() - DEFAULT_MODEL["lora"].keys():
        raise ValueError("Unknown model.lora keys")
    result["lora"] = {**DEFAULT_MODEL["lora"], **lora}
    if not isinstance(result["family"], str) or result["family"] not in MODELS:
        raise ValueError(f"Unknown model.family; supported: {', '.join(MODELS)}")
    if result["backbone"] not in MODELS[result["family"]].backbone_modes:
        raise ValueError("model.backbone must be frozen, lora or full")
    for name in ("action_head", "proprio_projector"):
        if result[name] not in ("train", "frozen"):
            raise ValueError(f"model.{name} must be train or frozen")
    if all(result[name] == "frozen" for name in ("backbone", "action_head", "proprio_projector")):
        raise ValueError("At least one model component must be trainable")
    for name in ("rank", "alpha"):
        value = result["lora"][name]
        if type(value) is not int or value < 1:
            raise ValueError(f"model.lora.{name} must be a positive integer")
    dropout = result["lora"]["dropout"]
    if isinstance(dropout, bool) or not isinstance(dropout, (float, int)) or not math.isfinite(dropout) or not 0 <= dropout < 1:
        raise ValueError("model.lora.dropout must be finite and in [0, 1)")
    return result


def model_signature(raw: dict[str, Any]) -> dict[str, Any]:
    """Only active settings affect resume compatibility."""
    settings = model_config(raw)
    # Checkpoint/stats identity is already stored separately in checkpoint metadata.
    for key in ("base_checkpoint", "stats_key", "use_pro_version"):
        settings.pop(key)
    if settings["backbone"] != "lora":
        settings.pop("lora")
    return settings


def model_parameters(raw: dict[str, Any]) -> dict[str, Any]:
    settings = model_config(raw)
    return {name: settings[path[0]] if len(path) == 1 else settings[path[0]][path[1]]
            for name, path in MODEL_PARAMETER_PATHS.items()}


def apply_model_parameters(raw: dict[str, Any], parameters: dict[str, Any]) -> dict[str, Any]:
    settings = model_config(raw)
    for name, path in MODEL_PARAMETER_PATHS.items():
        if name in parameters:
            if len(path) == 1:
                settings[path[0]] = parameters[name]
            else:
                settings[path[0]][path[1]] = parameters[name]
    return model_config({"model": settings})


def model_catalog() -> list[dict[str, Any]]:
    return [{"id": entry.id, "label": entry.label, "backbone_modes": list(entry.backbone_modes)}
            for entry in MODELS.values()]


def model_backend(raw: dict[str, Any]):
    from importlib import import_module
    return import_module(MODELS[model_config(raw)["family"]].backend)
