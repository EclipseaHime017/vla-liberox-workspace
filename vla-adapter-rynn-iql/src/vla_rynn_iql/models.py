"""Model capabilities and configuration, independent of training algorithms/GPU imports."""
from __future__ import annotations

import copy
import math
import re
from pathlib import Path
from typing import Any

from .base_models import BASE_MODELS, MODEL_FAMILIES, base_model, model_contract, registered_defaults


MODELS = MODEL_FAMILIES
PI05_DEFAULT_MODEL = registered_defaults("pi05")
DEFAULT_MODEL = registered_defaults("vla_adapter")
MODEL_PARAMETER_PATHS = {
    "model_family": ("family",), "model_backbone": ("backbone",),
    "model_base_id": ("base_id",),
    "model_action_head": ("action_head",), "model_proprio_projector": ("proprio_projector",),
    "model_lora_rank": ("lora", "rank"), "model_lora_alpha": ("lora", "alpha"),
    "model_lora_dropout": ("lora", "dropout"),
}


def model_config(raw: dict[str, Any]) -> dict[str, Any]:
    if "vla" in raw:
        raise ValueError("Legacy vla settings must be migrated at the configuration boundary")
    supplied = raw.get("model", {})
    if not isinstance(supplied, dict):
        raise TypeError("model must be a mapping")
    if "contract" in supplied:
        supplied = copy.deepcopy(supplied)
        model_contract(supplied)
    if supplied.get("family") == "pi05":
        unknown = supplied.keys() - PI05_DEFAULT_MODEL.keys() - {"contract"}
        if unknown:
            raise ValueError(f"Unknown π₀.₅ model keys: {sorted(unknown)}")
        base = base_model(supplied.get("base_id", PI05_DEFAULT_MODEL["base_id"]))
        if base.family != "pi05":
            raise ValueError("Base model does not belong to π₀.₅")
        result = {**PI05_DEFAULT_MODEL, "base_checkpoint": base.checkpoint,
                  "stats_key": base.stats_key, **supplied}
        if result["backbone"] not in MODELS["pi05"].backbone_modes:
            raise ValueError("π₀.₅ supports frozen VLM/action expert training or full fine-tuning; not LoRA")
        for key in ("base_checkpoint", "environment"):
            if not isinstance(result[key], str) or not result[key].strip():
                raise ValueError(f"model.{key} must be a non-empty string")
        if re.fullmatch(r"[A-Za-z0-9_][A-Za-z0-9_.-]*", result["environment"]) is None:
            raise ValueError("model.environment must be a Conda environment name")
        if result["stats_key"] != base.stats_key and "contract" not in supplied:
            raise ValueError(f"{base.id} requires its own {base.stats_key} normalization assets")
        if result["base_revision"] is not None and re.fullmatch(r"[0-9a-f]{64}", str(result["base_revision"])) is None:
            raise ValueError("model.base_revision must be a π₀.₅ identity SHA256 or null")
        if type(result["num_inference_steps"]) is not int or not 1 <= result["num_inference_steps"] <= 100:
            raise ValueError("model.num_inference_steps must be in 1..100")
        return result
    unknown = supplied.keys() - DEFAULT_MODEL.keys() - {"contract"}
    if unknown:
        raise ValueError(f"Unknown model keys: {sorted(unknown)}")
    family = supplied.get("family", "vla_adapter")
    if family not in MODELS:
        raise ValueError(f"Unknown model.family: {family}")
    result = registered_defaults(family, supplied.get("base_id"))
    result.update(supplied)
    if result["base_revision"] is not None and re.fullmatch(r"(?:[0-9a-f]{40}|[0-9a-f]{64})", str(result["base_revision"])) is None:
        raise ValueError("model.base_revision must be an immutable commit or local SHA256")
    for key in ("base_checkpoint", "stats_key"):
        if not isinstance(result[key], str) or not result[key].strip():
            raise ValueError(f"model.{key} must be a non-empty string")
    if not isinstance(result["environment"], str) or re.fullmatch(r"[A-Za-z0-9_][A-Za-z0-9_.-]*", result["environment"]) is None:
        raise ValueError("model.environment must be a Conda environment name")
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
    for key in ("base_checkpoint", "stats_key", "use_pro_version", "base_id"):
        settings.pop(key, None)
    if settings["backbone"] != "lora":
        settings.pop("lora", None)
    settings.pop("environment", None)
    settings.pop("base_revision", None)
    settings.pop("contract", None)
    return settings


def model_parameters(raw: dict[str, Any]) -> dict[str, Any]:
    settings = model_config(raw)
    return {name: settings[path[0]] if len(path) == 1 else settings[path[0]][path[1]]
            for name, path in MODEL_PARAMETER_PATHS.items() if path[0] in settings}


def base_model_config(base_id: str) -> dict[str, Any]:
    base = base_model(base_id)
    return model_config({"model": {"family": base.family, "base_id": base.id}})


def model_preset(path: Path) -> dict[str, Any]:
    """Project a registered family definition into a training configuration."""
    from .config_sources import read_source
    for family in MODELS.values():
        if path.resolve() == family.config_path:
            io = family.contract["io"]
            return {"model": {"family": family.id, "base_id": family.default_base},
                    "data": {"action_horizon": io["replay_horizon"], "action_dim": io["action_dim"],
                             "proprio_dim": io["proprio_dim"]}}
    return read_source(path)


def apply_model_parameters(raw: dict[str, Any], parameters: dict[str, Any]) -> dict[str, Any]:
    settings = model_config(raw)
    family = parameters.get("model_family", settings["family"])
    if family != settings["family"]:
        settings = model_config({"model": {"family": family}})
    if "model_base_id" in parameters and parameters["model_base_id"] != settings.get("base_id"):
        if base_model(parameters["model_base_id"]).family != family:
            raise ValueError("Base model conflicts with the selected model family")
        settings = base_model_config(parameters["model_base_id"])
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
    return [{"id": entry.id, "label": entry.label, "backbone_modes": list(entry.backbone_modes),
             "bases": [{"id": base.id, "label": base.label, "source": base.source}
                       for base in BASE_MODELS.values() if base.family == entry.id]}
            for entry in MODELS.values()]


def model_backend(raw: dict[str, Any]):
    from importlib import import_module
    return import_module(MODELS[model_config(raw)["family"]].backend)
