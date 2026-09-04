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
              "annotation_cache": None,
              "vla_adapter_root": None, "libero_x_root": None,
              "rynnvalue_root": None, "policy_registry": None},
    "data": {"project_id": None, "task_ids": None, "selection_manifest": None,
             "action_horizon": None,
             "action_dim": None, "proprio_dim": None, "control_hz": None,
             "success_consecutive_steps": None, "validation_fraction": None,
             "split_seed": None, "allow_no_success": None},
    "reward": {"model": None, "revision": None, "device": None, "dtype": None,
               "max_frames": None, "annotation_batch_size": None,
               "rynnvalue": None, "gamma": None, "shaping_weight": None,
               "robot_description": None,
               "camera_description": None, "accumulate_primitive_steps": None},
    "vla": {"base_checkpoint": None, "stats_key": None, "use_pro_version": None,
            "freeze_backbone": None},
    "iql": {"critic_image_size": None, "critic_lr": None, "value_lr": None,
            "critic_optimizer": None, "critic_weight_decay": None,
            "value_optimizer": None, "value_weight_decay": None,
            "critic_max_grad_norm": None, "value_max_grad_norm": None,
            "policy_peak_lr": None, "policy_final_lr": None, "expectile": None,
            "beta": None, "max_advantage_weight": None, "target_tau": None,
            "critic_warmup_steps": None, "train_steps": None, "micro_batch_size": None,
            "gradient_accumulation_steps": None, "checkpoint_interval": None,
            "resume_checkpoint": None, "seed": None, "device": None, "dtype": None},
    "logging": {
        "tensorboard": None,
        "wandb": {
            "enabled": None,
            "mode": None,
            "project": None,
            "entity": None,
            "run_name": None,
            "group": None,
            "tags": None,
            "log_interval_steps": None,
        },
        "flush_seconds": None,
        "console_interval_steps": None,
    },
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
    # Schema-v1 configurations created before the sparse-only ablation switch
    # preserve their original behavior.
    if (
        isinstance(raw, dict)
        and isinstance(raw.get("reward"), dict)
        and "rynnvalue" not in raw["reward"]
    ):
        raw["reward"]["rynnvalue"] = True
    _validate_schema(raw, TRAIN_SCHEMA)
    if raw["schema_version"] != 1:
        raise ValueError("Only schema_version=1 is supported")
    _resolve_paths(
        raw, path,
        ("work_dir", "output_dir", "annotation_cache", "vla_adapter_root", "libero_x_root",
         "rynnvalue_root", "policy_registry"),
    )
    data, reward, vla, iql, logging_cfg = (
        raw["data"], raw["reward"], raw["vla"], raw["iql"], raw["logging"]
    )
    if not isinstance(data["project_id"], str) or not data["project_id"].strip():
        raise TypeError("data.project_id must be a non-empty string")
    if not isinstance(data["task_ids"], list) or any(not isinstance(x, str) for x in data["task_ids"]):
        raise TypeError("data.task_ids must be a list of strings")
    selection_manifest = data["selection_manifest"]
    if selection_manifest is not None:
        if not isinstance(selection_manifest, str) or not selection_manifest.strip():
            raise TypeError("data.selection_manifest must be null or a non-empty path")
        selection_path = Path(selection_manifest).expanduser()
        data["selection_manifest"] = str(
            (selection_path if selection_path.is_absolute() else path.parent / selection_path).resolve()
        )
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
    if type(reward["rynnvalue"]) is not bool:
        raise TypeError("reward.rynnvalue must be boolean")
    _number(reward, "gamma", low=0, high=1)
    _number(reward, "shaping_weight", low=0)
    if type(reward["accumulate_primitive_steps"]) is not bool:
        raise TypeError("reward.accumulate_primitive_steps must be boolean")
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
                "max_advantage_weight", "target_tau", "critic_weight_decay",
                "value_weight_decay"):
        _number(iql, key, low=0)
    for key in ("critic_max_grad_norm", "value_max_grad_norm"):
        _number(iql, key, low=1e-12)
    for key in ("critic_optimizer", "value_optimizer"):
        if iql[key] not in {"adam", "adamw"}:
            raise ValueError(f"iql.{key} must be adam or adamw")
    _number(iql, "expectile", low=0, high=1)
    if iql["dtype"] != "bfloat16":
        raise ValueError("Version 1 trains the VLA actor in bfloat16; iql.dtype must be bfloat16")
    _cuda_device(iql["device"], "iql.device")
    if type(logging_cfg["tensorboard"]) is not bool:
        raise TypeError("logging.tensorboard must be boolean")
    wandb_cfg = logging_cfg["wandb"]
    if type(wandb_cfg["enabled"]) is not bool:
        raise TypeError("logging.wandb.enabled must be boolean")
    if wandb_cfg["mode"] not in {"online", "offline", "disabled"}:
        raise ValueError("logging.wandb.mode must be online, offline, or disabled")
    if not isinstance(wandb_cfg["project"], str) or not wandb_cfg["project"].strip():
        raise TypeError("logging.wandb.project must be a non-empty string")
    for key in ("entity", "run_name", "group"):
        value = wandb_cfg[key]
        if value is not None and (not isinstance(value, str) or not value.strip()):
            raise TypeError(f"logging.wandb.{key} must be null or a non-empty string")
    tags = wandb_cfg["tags"]
    if (
        not isinstance(tags, list)
        or any(not isinstance(tag, str) or not tag.strip() for tag in tags)
        or len(tags) != len(set(tags))
    ):
        raise TypeError("logging.wandb.tags must be a list of unique non-empty strings")
    _number(
        wandb_cfg, "log_interval_steps", low=1, integer=True,
    )
    _number(logging_cfg, "flush_seconds", low=1, high=3600)
    _number(logging_cfg, "console_interval_steps", low=1, integer=True)
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
