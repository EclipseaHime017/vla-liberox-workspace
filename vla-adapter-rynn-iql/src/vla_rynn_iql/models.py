"""Model capabilities and configuration, independent of training algorithms/GPU imports."""
from __future__ import annotations

import copy
import math
import re
from dataclasses import dataclass
from pathlib import Path
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
    "pi05": ModelDefinition("pi05", "π₀.₅ · LIBERO", "vla_rynn_iql.pi05", ("frozen", "full")),
}
PI05_DEFAULT_MODEL = {
    "family": "pi05", "backbone": "frozen",
    "base_checkpoint": str(Path(__file__).resolve().parents[3] / "weights/pi05_libero_torch"),
    "stats_key": "physical-intelligence/libero", "environment": "pi05",
    "base_revision": None, "num_inference_steps": 10,
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
    if supplied.get("family") == "pi05":
        unknown = supplied.keys() - PI05_DEFAULT_MODEL.keys()
        if unknown:
            raise ValueError(f"Unknown π₀.₅ model keys: {sorted(unknown)}")
        result = {**PI05_DEFAULT_MODEL, **supplied}
        if result["backbone"] not in MODELS["pi05"].backbone_modes:
            raise ValueError("π₀.₅ supports frozen VLM/action expert training or full fine-tuning; not LoRA")
        for key in ("base_checkpoint", "environment"):
            if not isinstance(result[key], str) or not result[key].strip():
                raise ValueError(f"model.{key} must be a non-empty string")
        if re.fullmatch(r"[A-Za-z0-9_][A-Za-z0-9_.-]*", result["environment"]) is None:
            raise ValueError("model.environment must be a Conda environment name")
        if result["stats_key"] != PI05_DEFAULT_MODEL["stats_key"]:
            raise ValueError("π₀.₅-LIBERO requires its own physical-intelligence/libero normalization assets")
        if result["base_revision"] is not None and re.fullmatch(r"[0-9a-f]{64}", str(result["base_revision"])) is None:
            raise ValueError("model.base_revision must be a π₀.₅ identity SHA256 or null")
        if type(result["num_inference_steps"]) is not int or not 1 <= result["num_inference_steps"] <= 100:
            raise ValueError("model.num_inference_steps must be in 1..100")
        return result
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
        settings.pop(key, None)
    if settings["backbone"] != "lora":
        settings.pop("lora", None)
    settings.pop("environment", None)
    settings.pop("base_revision", None)
    return settings


def model_parameters(raw: dict[str, Any]) -> dict[str, Any]:
    settings = model_config(raw)
    return {name: settings[path[0]] if len(path) == 1 else settings[path[0]][path[1]]
            for name, path in MODEL_PARAMETER_PATHS.items() if path[0] in settings}


def apply_model_parameters(raw: dict[str, Any], parameters: dict[str, Any]) -> dict[str, Any]:
    settings = model_config(raw)
    family = parameters.get("model_family", settings["family"])
    if family != settings["family"]:
        settings = model_config({"model": {"family": family}})
    for name, path in MODEL_PARAMETER_PATHS.items():
        if name in parameters:
            if path[0] not in settings:
                raise ValueError(f"{name} is not supported by {family}")
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
