"""Model capabilities and configuration, independent of training algorithms/GPU imports."""
from __future__ import annotations

import copy
import re
from pathlib import Path
from typing import Any

from .base_models import BASE_MODELS, MODEL_FAMILIES, DEFAULT_FAMILY, base_model, model_contract, registered_defaults, family_adapter
from .config_sources import merge_config


MODELS = MODEL_FAMILIES
DEFAULT_MODEL = registered_defaults(DEFAULT_FAMILY)
MODEL_CONFIG_KEYS = {"family", "base_id", "base_checkpoint", "stats_key", "contract"} | {
    key for family in MODELS.values() for key in family.defaults}
COMMON_PARAMETER_PATHS = {
    "model_family": ("family",), "model_backbone": ("backbone",),
    "model_base_id": ("base_id",),
}


def parameter_paths(family: str | None = None) -> dict:
    definitions = (MODELS[family],) if family is not None else MODELS.values()
    return {**COMMON_PARAMETER_PATHS,
            **{name: path for definition in definitions for name, path in definition.implementation.PARAMETER_PATHS.items()}}


def model_config(raw: dict[str, Any], *, partial: bool = False) -> dict[str, Any]:
    if "vla" in raw:
        raise ValueError("Legacy vla settings must be migrated at the configuration boundary")
    supplied = raw.get("model", {})
    if not isinstance(supplied, dict):
        raise TypeError("model must be a mapping")
    family = supplied.get("family", DEFAULT_FAMILY)
    implementation = family_adapter(family)
    defaults = registered_defaults(family, supplied.get("base_id"))
    unknown = supplied.keys() - defaults.keys() - {"contract"}
    if unknown:
        raise ValueError(f"Unknown model keys: {sorted(unknown)}")
    result = merge_config(defaults, supplied)
    if "contract" in supplied:
        model_contract(result)
    if result["base_revision"] is not None and re.fullmatch(implementation.REVISION_PATTERN, str(result["base_revision"])) is None:
        raise ValueError("model.base_revision must be an immutable model identity")
    for key in ("base_checkpoint", "stats_key"):
        if not isinstance(result[key], str) or not result[key].strip():
            raise ValueError(f"model.{key} must be a non-empty string")
    if not isinstance(result["environment"], str) or re.fullmatch(r"[A-Za-z0-9_][A-Za-z0-9_.-]*", result["environment"]) is None:
        raise ValueError("model.environment must be a Conda environment name")
    if result["backbone"] not in MODELS[family].backbone_modes:
        raise ValueError(f"model.backbone must be one of {', '.join(MODELS[family].backbone_modes)}")
    return implementation.configure(result, partial=partial)


def model_signature(raw: dict[str, Any]) -> dict[str, Any]:
    """Only active settings affect resume compatibility."""
    settings = model_config(raw)
    # Checkpoint/stats identity is already stored separately in checkpoint metadata.
    for key in ("base_checkpoint", "stats_key", "base_id"):
        settings.pop(key, None)
    settings = family_adapter(settings["family"]).signature(settings)
    settings.pop("environment", None)
    settings.pop("base_revision", None)
    settings.pop("contract", None)
    return settings


def model_parameters(raw: dict[str, Any]) -> dict[str, Any]:
    settings = model_config(raw)
    return {name: settings[path[0]] if len(path) == 1 else settings[path[0]][path[1]]
            for name, path in parameter_paths(settings["family"]).items() if path[0] in settings}


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


def apply_model_parameters(raw: dict[str, Any], parameters: dict[str, Any], *, partial: bool = False) -> dict[str, Any]:
    settings = model_config(raw)
    family = parameters.get("model_family", settings["family"])
    if family != settings["family"]:
        settings = model_config({"model": {"family": family}})
    if "model_base_id" in parameters and parameters["model_base_id"] != settings.get("base_id"):
        if base_model(parameters["model_base_id"]).family != family:
            raise ValueError("Base model conflicts with the selected model family")
        settings = base_model_config(parameters["model_base_id"])
    paths = parameter_paths(family)
    unsupported = (parameters.keys() & parameter_paths().keys()) - paths.keys()
    if unsupported:
        raise ValueError(f"{', '.join(sorted(unsupported))} is not supported by {family}")
    for name, path in paths.items():
        if name in parameters:
            if path[0] not in settings:
                raise ValueError(f"{name} is not supported by {family}")
            if len(path) == 1:
                settings[path[0]] = parameters[name]
            else:
                settings[path[0]][path[1]] = parameters[name]
    return model_config({"model": settings}, partial=partial)


def model_catalog() -> list[dict[str, Any]]:
    return [{"id": entry.id, "label": entry.label, "backbone_modes": list(entry.backbone_modes),
             "training_ui": training_ui(entry.id),
             "bases": [{"id": base.id, "label": base.label, "source": base.source}
                       for base in BASE_MODELS.values() if base.family == entry.id]}
            for entry in MODELS.values()]


def model_backend(raw: dict[str, Any]):
    from importlib import import_module
    return import_module(MODELS[model_config(raw)["family"]].backend)


def replay_stats(settings: dict[str, Any]) -> tuple[dict, dict]:
    """Read a model's replay normalization assets without constructing its actor."""
    import math

    configured = model_config({"model": settings}, partial=True)
    values = family_adapter(configured["family"]).replay_stats(configured)
    io = model_contract(configured)["io"]
    result = []
    for name, stats, size in zip(("action", "proprio"), values, (io["action_dim"], io["proprio_dim"])):
        if not isinstance(stats, dict):
            raise ValueError(f"Invalid {name} normalization statistics")
        stats = copy.deepcopy(stats)
        for key in ("q01", "q99"):
            value = stats.get(key)
            if (not isinstance(value, list) or len(value) < size
                    or any(isinstance(item, bool) or not isinstance(item, (int, float))
                           or not math.isfinite(item) for item in value[:size])):
                raise ValueError(f"Invalid {name} normalization {key}")
            stats[key] = value[:size]
        if any(hi < lo for lo, hi in zip(stats["q01"], stats["q99"])):
            raise ValueError(f"Invalid {name} normalization quantile order")
        if "mask" in stats:
            if not isinstance(stats["mask"], list) or len(stats["mask"]) < size or any(
                type(item) is not bool for item in stats["mask"][:size]
            ):
                raise ValueError(f"Invalid {name} normalization mask")
            stats["mask"] = stats["mask"][:size]
        result.append(stats)
    return tuple(result)


def training_ui(family: str) -> dict:
    definition = MODELS[family]
    value = copy.deepcopy(definition.implementation.TRAINING_UI)
    defaults = model_parameters({"model": {"family": family}})
    for field in value["fields"]:
        field["default"] = defaults.get(field["name"])
        if field["name"] == "model_backbone":
            field["options"] = [item for item in field["options"] if item["value"] in definition.backbone_modes]
    return value
