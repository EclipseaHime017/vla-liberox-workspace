from __future__ import annotations

import hashlib
import json
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import yaml


PROJECT_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_CONFIG = PROJECT_ROOT / "configs" / "robometer_evaluation.yaml"


class UniqueKeyLoader(yaml.SafeLoader):
    pass


def _mapping(loader: UniqueKeyLoader, node: yaml.MappingNode, deep: bool = False):
    result: dict[Any, Any] = {}
    for key_node, value_node in node.value:
        key = loader.construct_object(key_node, deep=deep)
        if key in result:
            raise yaml.constructor.ConstructorError(
                "while constructing a mapping", node.start_mark,
                f"duplicate key {key!r}", key_node.start_mark,
            )
        result[key] = loader.construct_object(value_node, deep=deep)
    return result


UniqueKeyLoader.add_constructor(yaml.resolver.BaseResolver.DEFAULT_MAPPING_TAG, _mapping)

SCHEMA = {
    "schema_version": None,
    "paths": {"selection_manifest": None, "output_dir": None, "robometer_root": None},
    "model": {
        "checkpoint": None, "revision": None, "robometer_commit": None,
        "device": None, "dtype": None,
    },
    "evaluation": {"control_hz": None, "fps": None, "prefix_frames": None, "batch_size": None},
}


def _schema(raw: Any, schema: dict[str, Any], context: str = "config") -> None:
    if not isinstance(raw, dict):
        raise TypeError(f"{context} must be a mapping")
    missing, unknown = sorted(set(schema) - set(raw)), sorted(set(raw) - set(schema))
    if missing:
        raise ValueError(f"Missing {context} keys: {missing}")
    if unknown:
        raise ValueError(f"Unknown {context} keys: {unknown}")
    for key, nested in schema.items():
        if nested is not None:
            _schema(raw[key], nested, f"{context}.{key}")


@dataclass(frozen=True)
class Config:
    path: Path
    raw: dict[str, Any]

    @property
    def digest(self) -> str:
        return hashlib.sha256(json.dumps(self.raw, sort_keys=True).encode()).hexdigest()


def load_config(path: Path = DEFAULT_CONFIG) -> Config:
    path = path.expanduser().resolve()
    raw = yaml.load(path.read_text(encoding="utf-8"), Loader=UniqueKeyLoader)
    _schema(raw, SCHEMA)
    if raw["schema_version"] != 1:
        raise ValueError("Only schema_version=1 is supported")
    for key in ("output_dir", "robometer_root"):
        value = raw["paths"][key]
        if not isinstance(value, str) or not value.strip():
            raise TypeError(f"paths.{key} must be a non-empty string")
        candidate = Path(value).expanduser()
        raw["paths"][key] = str((candidate if candidate.is_absolute() else path.parent / candidate).resolve())
    selection = raw["paths"]["selection_manifest"]
    if selection is not None:
        if not isinstance(selection, str) or not selection.strip():
            raise TypeError("paths.selection_manifest must be null or a non-empty string")
        candidate = Path(selection).expanduser()
        raw["paths"]["selection_manifest"] = str(
            (candidate if candidate.is_absolute() else path.parent / candidate).resolve()
        )
    model, evaluation = raw["model"], raw["evaluation"]
    for key in ("checkpoint", "revision", "robometer_commit", "device", "dtype"):
        if not isinstance(model[key], str) or not model[key].strip():
            raise TypeError(f"model.{key} must be a non-empty string")
    if re.fullmatch(r"cuda:\d+", model["device"]) is None:
        raise ValueError("model.device must identify one CUDA device, for example cuda:0")
    if model["dtype"] != "bfloat16":
        raise ValueError("model.dtype must be bfloat16")
    for key in ("revision", "robometer_commit"):
        if re.fullmatch(r"[0-9a-f]{40}", model[key]) is None:
            raise ValueError(f"model.{key} must be a full 40-character commit hash")
    if type(evaluation["control_hz"]) is not int or evaluation["control_hz"] != 20:
        raise ValueError("evaluation.control_hz must be 20")
    if isinstance(evaluation["fps"], bool) or not isinstance(evaluation["fps"], (int, float)) or not 0 < evaluation["fps"] <= 20:
        raise ValueError("evaluation.fps must be in (0, 20]")
    for key in ("prefix_frames", "batch_size"):
        if type(evaluation[key]) is not int or evaluation[key] < 1:
            raise ValueError(f"evaluation.{key} must be a positive integer")
    if evaluation["prefix_frames"] != 4:
        raise ValueError("Version 1 fixes evaluation.prefix_frames=4 to official use_frame_steps semantics")
    return Config(path, raw)
