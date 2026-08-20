from __future__ import annotations

import hashlib
import json
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import yaml


PROJECT_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_TRAIN_CONFIG = PROJECT_ROOT / "configs" / "liberox_iql.yaml"
DEFAULT_INFERENCE_CONFIG = PROJECT_ROOT / "configs" / "inference.yaml"


class UniqueKeyLoader(yaml.SafeLoader):
    pass


def _unique_mapping(loader: UniqueKeyLoader, node: yaml.MappingNode, deep: bool = False):
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


UniqueKeyLoader.add_constructor(yaml.resolver.BaseResolver.DEFAULT_MAPPING_TAG, _unique_mapping)


TRAIN_SCHEMA = {
    "schema_version": None,
    "paths": {"dataset_sources": None, "work_dir": None, "output_dir": None,
              "vla_adapter_root": None, "libero_x_root": None,
              "rynnvalue_root": None, "policy_registry": None},
    "data": {"project_id": None, "task_ids": None, "action_horizon": None,
             "action_dim": None, "proprio_dim": None, "control_hz": None,
             "success_consecutive_steps": None, "validation_fraction": None,
             "split_seed": None, "allow_no_success": None},
    "reward": {"model": None, "revision": None, "device": None, "dtype": None,
               "max_frames": None, "annotation_batch_size": None,
               "window_overlap": None, "gamma": None,
               "shaping_weight": None, "robot_description": None,
               "camera_description": None},
    "vla": {"base_checkpoint": None, "stats_key": None, "use_pro_version": None,
            "freeze_backbone": None},
    "iql": {"critic_image_size": None, "critic_lr": None, "value_lr": None,
            "policy_peak_lr": None, "policy_final_lr": None, "expectile": None,
            "beta": None, "max_advantage_weight": None, "target_tau": None,
            "critic_warmup_steps": None, "train_steps": None, "micro_batch_size": None,
            "gradient_accumulation_steps": None, "checkpoint_interval": None,
            "resume_checkpoint": None, "seed": None, "device": None, "dtype": None},
}

INFERENCE_SCHEMA = {
    "schema_version": None,
    "paths": {"vla_adapter_root": None, "libero_x_root": None, "output_dir": None},
    "policy": {"overlay": None},
    "evaluation": {"level": None, "task_name": None, "trials": None,
                   "max_steps": None, "open_loop_steps": None, "seed": None,
                   "device": None, "mujoco_gl": None, "compare_base": None},
}


@dataclass(frozen=True)
class LoadedConfig:
    path: Path
    raw: dict[str, Any]

    def section(self, name: str) -> dict[str, Any]:
        return self.raw[name]

    @property
    def digest(self) -> str:
        payload = json.dumps(self.raw, sort_keys=True, default=str).encode()
        return hashlib.sha256(payload).hexdigest()


def _validate_schema(raw: Any, schema: dict[str, Any], context: str = "config") -> None:
    if not isinstance(raw, dict):
        raise TypeError(f"{context} must be a mapping")
    missing = sorted(set(schema) - set(raw))
    unknown = sorted(set(raw) - set(schema))
    if missing:
        raise ValueError(f"Missing {context} keys: {missing}")
    if unknown:
        raise ValueError(f"Unknown {context} keys: {unknown}")
    for key, nested in schema.items():
        if nested is not None:
            _validate_schema(raw[key], nested, f"{context}.{key}")


def _resolve_paths(raw: dict[str, Any], config_path: Path, names: tuple[str, ...]) -> None:
    base = config_path.parent
    paths = raw["paths"]
    for name in names:
        value = paths[name]
        if not isinstance(value, str) or not value.strip():
            raise TypeError(f"paths.{name} must be a non-empty string")
        path = Path(value).expanduser()
        paths[name] = str((path if path.is_absolute() else base / path).resolve())
    if "dataset_sources" in paths:
        sources = paths["dataset_sources"]
        if not isinstance(sources, list) or not sources:
            raise TypeError("paths.dataset_sources must be a non-empty list")
        resolved = []
        for source in sources:
            if not isinstance(source, str) or not source.strip():
                raise TypeError("Every dataset source must be a non-empty string")
            path = Path(source).expanduser()
            resolved.append(str((path if path.is_absolute() else base / path).resolve()))
        paths["dataset_sources"] = resolved


def _number(section: dict[str, Any], name: str, *, low: float | None = None,
            high: float | None = None, integer: bool = False) -> float | int:
    value = section[name]
    valid = type(value) is int if integer else isinstance(value, (int, float)) and not isinstance(value, bool)
    if not valid:
        raise TypeError(f"{name} must be {'an integer' if integer else 'a number'}")
    if low is not None and value < low or high is not None and value > high:
        raise ValueError(f"{name} must be in [{low}, {high}]")
    return value


def _cuda_device(value: Any, name: str) -> None:
    if not isinstance(value, str) or re.fullmatch(r"cuda:\d+", value) is None:
        raise ValueError(f"{name} must identify one CUDA device, for example cuda:0")


def load_train_config(path: Path = DEFAULT_TRAIN_CONFIG) -> LoadedConfig:
    path = path.expanduser().resolve()
    raw = yaml.load(path.read_text(encoding="utf-8"), Loader=UniqueKeyLoader)
    _validate_schema(raw, TRAIN_SCHEMA)
    if raw["schema_version"] != 1:
        raise ValueError("Only schema_version=1 is supported")
    _resolve_paths(
        raw, path,
        ("work_dir", "output_dir", "vla_adapter_root", "libero_x_root",
         "rynnvalue_root", "policy_registry"),
    )
    data, reward, vla, iql = raw["data"], raw["reward"], raw["vla"], raw["iql"]
    if not isinstance(data["project_id"], str) or not data["project_id"].strip():
        raise TypeError("data.project_id must be a non-empty string")
    if not isinstance(data["task_ids"], list) or any(not isinstance(x, str) for x in data["task_ids"]):
        raise TypeError("data.task_ids must be a list of strings")
    for name, expected in (("action_horizon", 8), ("action_dim", 7), ("proprio_dim", 8)):
        _number(data, name, low=1, integer=True)
        if data[name] != expected:
            raise ValueError(f"data.{name} must match the current VLA-Adapter value {expected}")
    _number(data, "control_hz", low=1)
    if float(data["control_hz"]) != 20.0:
        raise ValueError("Version 1 requires data.control_hz=20")
    _number(data, "success_consecutive_steps", low=1, high=100, integer=True)
    _number(data, "validation_fraction", low=0, high=0.9)
    _number(data, "split_seed", integer=True)
    if type(data["allow_no_success"]) is not bool:
        raise TypeError("data.allow_no_success must be boolean")
    for key in ("model", "revision", "device", "dtype", "robot_description", "camera_description"):
        if not isinstance(reward[key], str) or not reward[key].strip():
            raise TypeError(f"reward.{key} must be a non-empty string")
    _number(reward, "max_frames", low=2, integer=True)
    _number(reward, "annotation_batch_size", low=1, integer=True)
    _number(reward, "window_overlap", low=1, integer=True)
    if reward["window_overlap"] >= reward["max_frames"]:
        raise ValueError("reward.window_overlap must be smaller than max_frames")
    _number(reward, "gamma", low=0, high=1)
    _number(reward, "shaping_weight", low=0)
    if reward["dtype"] != "bfloat16":
        raise ValueError(
            "Version 1 requires reward.dtype=bfloat16 to match the pinned RynnValue-4B "
            "checkpoint and the validated 16GB profile"
        )
    _cuda_device(reward["device"], "reward.device")
    for key in ("base_checkpoint", "stats_key"):
        if not isinstance(vla[key], str) or not vla[key].strip():
            raise TypeError(f"vla.{key} must be a non-empty string")
    if vla["use_pro_version"] is not True or vla["freeze_backbone"] is not True:
        raise ValueError("Version 1 requires Pro components and a frozen VLA backbone")
    for key in ("critic_image_size", "critic_warmup_steps", "train_steps", "micro_batch_size",
                "gradient_accumulation_steps", "checkpoint_interval", "seed"):
        _number(iql, key, low=0 if key in {"critic_warmup_steps", "seed"} else 1, integer=True)
    resume = iql["resume_checkpoint"]
    if resume is not None:
        if not isinstance(resume, str) or not resume.strip():
            raise TypeError("iql.resume_checkpoint must be null or a non-empty path")
        path_value = Path(resume).expanduser()
        iql["resume_checkpoint"] = str(
            (path_value if path_value.is_absolute() else path.parent / path_value).resolve()
        )
    if iql["checkpoint_interval"] % iql["gradient_accumulation_steps"] != 0:
        raise ValueError(
            "iql.checkpoint_interval must be divisible by gradient_accumulation_steps "
            "so resumed actor gradients are exact"
        )
    for key in ("critic_lr", "value_lr", "policy_peak_lr", "policy_final_lr", "beta",
                "max_advantage_weight", "target_tau"):
        _number(iql, key, low=0)
    _number(iql, "expectile", low=0, high=1)
    if iql["dtype"] != "bfloat16":
        raise ValueError("Version 1 trains the VLA actor in bfloat16; iql.dtype must be bfloat16")
    _cuda_device(iql["device"], "iql.device")
    if iql["micro_batch_size"] != 1:
        raise ValueError("The validated 16GB profile requires iql.micro_batch_size=1")
    return LoadedConfig(path, raw)


def load_inference_config(path: Path = DEFAULT_INFERENCE_CONFIG) -> LoadedConfig:
    path = path.expanduser().resolve()
    raw = yaml.load(path.read_text(encoding="utf-8"), Loader=UniqueKeyLoader)
    _validate_schema(raw, INFERENCE_SCHEMA)
    if raw["schema_version"] != 1:
        raise ValueError("Only schema_version=1 is supported")
    _resolve_paths(raw, path, ("vla_adapter_root", "libero_x_root", "output_dir"))
    overlay = raw["policy"]["overlay"]
    if not isinstance(overlay, str) or not overlay.strip():
        raise TypeError("policy.overlay must be a non-empty string")
    p = Path(overlay).expanduser()
    raw["policy"]["overlay"] = str((p if p.is_absolute() else path.parent / p).resolve())
    ev = raw["evaluation"]
    if ev["level"] not in {"LEVEL1", "LEVEL2", "LEVEL3", "LEVEL4"}:
        raise ValueError("evaluation.level must be LEVEL1..LEVEL4")
    for key in ("trials", "max_steps", "open_loop_steps", "seed"):
        _number(ev, key, low=0 if key == "seed" else 1, high=8 if key == "open_loop_steps" else None, integer=True)
    if type(ev["compare_base"]) is not bool:
        raise TypeError("evaluation.compare_base must be boolean")
    for key in ("task_name", "device", "mujoco_gl"):
        if not isinstance(ev[key], str) or not ev[key].strip():
            raise TypeError(f"evaluation.{key} must be a non-empty string")
    _cuda_device(ev["device"], "evaluation.device")
    if ev["mujoco_gl"] not in {"egl", "glfw", "osmesa"}:
        raise ValueError("evaluation.mujoco_gl must be egl, glfw, or osmesa")
    return LoadedConfig(path, raw)
