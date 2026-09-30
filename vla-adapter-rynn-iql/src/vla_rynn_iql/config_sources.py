"""Strict YAML composition with paths anchored to their declaring file."""
from __future__ import annotations

import copy
from pathlib import Path
from typing import Any

import yaml


class UniqueKeyLoader(yaml.SafeLoader):
    pass


def _unique_mapping(loader, node, deep=False):
    result = {}
    for key_node, value_node in node.value:
        key = loader.construct_object(key_node, deep=deep)
        if key in result:
            raise yaml.constructor.ConstructorError(
                "while constructing a mapping", node.start_mark,
                f"duplicate key {key!r}", key_node.start_mark,
            )
        result[key] = loader.construct_object(value_node, deep=deep)
    return result


UniqueKeyLoader.add_constructor(yaml.resolver.BaseResolver.DEFAULT_MAPPING_TAG, _unique_mapping)


def merge_config(base: dict, changes: dict) -> dict:
    result = copy.deepcopy(base)
    for key, value in changes.items():
        result[key] = (merge_config(result[key], value)
                       if isinstance(result.get(key), dict) and isinstance(value, dict)
                       else copy.deepcopy(value))
    return result


def resolve_source_paths(raw: dict, path: Path) -> dict:
    result = copy.deepcopy(raw)

    def resolve(value):
        if not isinstance(value, str) or not value.strip():
            raise TypeError(f"Config path must be a non-empty string in {path}")
        candidate = Path(value).expanduser()
        return str((candidate if candidate.is_absolute() else path.parent / candidate).resolve())

    for section, fields in (
        ("paths", None), ("presets", {"model", "reward"}),
        ("data", {"selection_manifest", "stage_annotations_manifest"}),
        ("reward", {"manifest_path"}), ("training", {"resume_checkpoint"}),
        ("iql", {"resume_checkpoint"}),
    ):
        values = result.get(section, {})
        if not isinstance(values, dict):
            raise TypeError(f"{section} must be a mapping")
        for key, value in values.items():
            if value is not None and (fields is None or key in fields):
                if key == "dataset_sources":
                    if not isinstance(value, list) or not value:
                        raise TypeError("paths.dataset_sources must be a non-empty list")
                    values[key] = [resolve(item) for item in value]
                else:
                    values[key] = resolve(value)
    for section in ("model", "vla"):
        values = result.get(section, {})
        if isinstance(values, dict):
            checkpoint = values.get("base_checkpoint")
            # HF repository IDs are not filesystem paths.
            if isinstance(checkpoint, str) and checkpoint.startswith(("./", "../", "~/", "/")):
                values["base_checkpoint"] = resolve(checkpoint)
    return result


def _source_layers(path: Path, stack: tuple[Path, ...] = (), *,
                   method: str | None = None) -> list[dict[str, Any]]:
    path = path.expanduser().resolve()
    if method is not None:
        from .methods import METHODS
        if not isinstance(method, str) or method not in METHODS:
            raise ValueError(f"Unknown training.method: {method}")
        root = Path(__file__).resolve().parents[2] / "configs" / "training"
        if path in {(root / f"{name}.yaml").resolve() for name in METHODS}:
            path = root / f"{method}.yaml"
    if path in stack:
        raise ValueError(f"Cyclic config inheritance: {' -> '.join(map(str, (*stack, path)))}")
    raw = yaml.load(path.read_text(encoding="utf-8"), Loader=UniqueKeyLoader)
    if not isinstance(raw, dict):
        raise TypeError(f"Config must be a mapping: {path}")
    parent = raw.pop("extends", None)
    layers = []
    if parent is not None:
        if not isinstance(parent, str) or not parent.strip():
            raise TypeError("extends must be one config path")
        training = raw.get("training", {})
        if not isinstance(training, dict):
            raise TypeError("training must be a mapping")
        layers = _source_layers(path.parent / parent, (*stack, path),
                                method=method or training.get("method"))
    return [*layers, resolve_source_paths(raw, path)]


def read_source(path: Path) -> dict[str, Any]:
    result = {}
    for layer in _source_layers(path):
        result = merge_config(result, layer)
    return result


def compose_config(path: Path, *, method: str | None = None,
                   family: str | None = None, overrides: dict | None = None,
                   overrides_path: Path | None = None) -> dict:
    from .methods import METHODS
    from .models import MODELS

    root = Path(__file__).resolve().parents[2] / "configs"
    training_override = (overrides or {}).get("training", {})
    if not isinstance(training_override, dict):
        raise TypeError("overrides.training must be a mapping")
    requested_method = method or training_override.get("method")
    # Choose the method before reading its entry or inherited dependencies.
    layers = [migrate_fields(layer) for layer in _source_layers(path, method=requested_method)]
    if overrides:
        changes = migrate_fields(resolve_source_paths(overrides, overrides_path or path))
        recipe = changes.get("reward", {})
        if ("alpha" in recipe or "fusion_mode" in recipe) and "source" not in recipe:
            recipe["source"] = "final"
        if "source" in recipe and "rynnvalue" not in recipe:
            recipe["rynnvalue"] = recipe["source"] == "rynnvalue"
        elif "rynnvalue" in recipe and "source" not in recipe:
            recipe["source"] = "rynnvalue" if recipe["rynnvalue"] else "sparse"
        layers.append(changes)
    raw = {}
    for layer in layers:
        raw = merge_config(raw, layer)
    presets = raw.pop("presets", {})
    if not isinstance(presets, dict) or presets.keys() - {"model", "reward"}:
        raise ValueError("presets only accepts model and reward paths")
    if not presets and method is None and family is None:
        # A sealed effective config is self-contained, not a live preset reference.
        return raw
    method = requested_method or raw.get("training", {}).get("method", "iql")
    if not isinstance(method, str) or method not in METHODS:
        raise ValueError(f"Unknown training.method: {method}")
    if not METHODS[method].requires_rewards:
        presets.pop("reward", None)
    if family is not None:
        if not isinstance(family, str) or family not in MODELS:
            raise ValueError(f"Unknown model.family: {family}")
        presets["model"] = str(root / "models" / f"{family}.yaml")
    # Inherited runtime defaults < model/reward defaults < run/CLI overrides.
    # The first preset declaration marks the composition boundary.
    pivot = next((index for index, layer in enumerate(layers) if "presets" in layer), 0)
    result = {}
    for layer in layers[:pivot]:
        result = merge_config(result, layer)
    allowed = {"model": {"model", "data"}, "reward": {"reward"}}
    for kind, source in presets.items():
        preset = read_source(Path(source))
        if preset.keys() - allowed[kind]:
            raise ValueError(f"{kind} preset contains unrelated sections: {sorted(preset.keys() - allowed[kind])}")
        result = merge_config(result, preset)
    for layer in layers[pivot:]:
        result = merge_config(result, {key: value for key, value in layer.items() if key != "presets"})
    if method is not None:
        result.setdefault("training", {})["method"] = method
    if family is not None:
        result.setdefault("model", {})["family"] = family
    return result


def migrate_fields(raw: dict) -> dict:
    """Translate old aliases before applying defaults/overrides, never at runtime."""
    from .methods import COMMON_TRAINING_KEYS
    result = copy.deepcopy(raw)
    iql, training = result.get("iql", {}), result.setdefault("training", {})
    if not isinstance(iql, dict) or not isinstance(training, dict):
        raise TypeError("iql and training must be mappings")
    for key in COMMON_TRAINING_KEYS & iql.keys():
        training.setdefault(key, iql.pop(key))
    if result.get("schema_version") == 1 and training.get("method", "iql") == "iql":
        training.setdefault("actor_lr_warmup_steps", None)
    legacy, model = result.get("vla", {}), result.setdefault("model", {})
    if not isinstance(legacy, dict) or not isinstance(model, dict):
        raise TypeError("vla and model must be mappings")
    for key in ("base_checkpoint", "stats_key", "use_pro_version"):
        if key in legacy:
            model.setdefault(key, legacy[key])
    if "freeze_backbone" in legacy:
        if type(legacy["freeze_backbone"]) is not bool:
            raise TypeError("vla.freeze_backbone must be boolean")
        model.setdefault("backbone", "frozen" if legacy["freeze_backbone"] else "full")
    return result
