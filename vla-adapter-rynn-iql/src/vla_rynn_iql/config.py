from __future__ import annotations

import hashlib
import json
import math
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import yaml

from .methods import COMMON_TRAINING_KEYS, training_method
from .models import model_config
from .config_sources import UniqueKeyLoader, compose_config, read_source


PROJECT_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_TRAIN_CONFIG = PROJECT_ROOT / "configs" / "training" / "iql.yaml"
DEFAULT_INFERENCE_CONFIG = PROJECT_ROOT / "configs" / "inference.yaml"


TRAIN_SCHEMA = {
    "schema_version": None,
    "training": None,
    "model": None,
    "paths": {"dataset_sources": None, "work_dir": None, "output_dir": None,
              "annotation_cache": None,
              "vla_adapter_root": None, "libero_x_root": None,
              "rynnvalue_root": None, "policy_registry": None},
    "data": {"project_id": None, "task_ids": None, "selection_manifest": None,
             "stage_annotations_manifest": None,
             "action_horizon": None,
             "action_dim": None, "proprio_dim": None, "control_hz": None,
             "success_consecutive_steps": None, "include_post_success": None, "validation_fraction": None,
             "split_seed": None, "allow_no_success": None},
    "reward": {"model": None, "revision": None, "device": None, "dtype": None,
               "max_frames": None, "annotation_batch_size": None,
               "rynnvalue": None, "source": None, "stage_exponent": None,
               "fusion_mode": None, "alpha": None, "final_normalization": None,
               "manifest_path": None, "manifest_sha256": None, "version_id": None,
               "gamma": None, "shaping_weight": None,
               "robot_description": None,
               "camera_description": None, "accumulate_primitive_steps": None},
    "bc": {},
    "iql": {"critic_image_size": None, "critic_lr": None, "value_lr": None,
            "critic_optimizer": None, "critic_weight_decay": None,
            "value_optimizer": None, "value_weight_decay": None,
            "critic_max_grad_norm": None, "value_max_grad_norm": None,
            "expectile": None,
            "beta": None, "max_advantage_weight": None, "target_tau": None,
            "critic_warmup_steps": None},
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


def reward_source(reward: dict[str, Any]) -> str:
    """Resolve legacy sparse ablations without depending on any reward model."""
    legacy = "rynnvalue" if reward.get("rynnvalue", True) else "sparse"
    source = reward.get("source", legacy if "rynnvalue" in reward else
                        "final" if "alpha" in reward or "fusion_mode" in reward else legacy)
    if source not in ("sparse", "rynnvalue", "stage", "final"):
        raise ValueError("reward.source must be sparse, rynnvalue, stage or final")
    return source


def needs_rynnvalue(reward: dict[str, Any]) -> bool:
    source = reward_source(reward)
    return source == "rynnvalue" or (source == "final" and reward["shaping_weight"] > 0)


def needs_stage(reward: dict[str, Any]) -> bool:
    source = reward_source(reward)
    return source == "stage" or (source == "final" and (
        reward.get("fusion_mode", "additive") == "multiplicative" or reward.get("alpha", 0) > 0))


def effective_cumulative(reward: dict[str, Any]) -> bool:
    """Multiplicative Final Reward is defined only on macro-action transitions."""
    return bool(reward["accumulate_primitive_steps"]) and not (
        reward_source(reward) == "final" and reward.get("fusion_mode") == "multiplicative")


def load_train_config(path: Path = DEFAULT_TRAIN_CONFIG, *, method: str | None = None,
                      family: str | None = None, overrides: dict | None = None,
                      overrides_path: Path | None = None) -> LoadedConfig:
    path = path.expanduser().resolve()
    raw = compose_config(path, method=method, family=family, overrides=overrides,
                         overrides_path=overrides_path)
    return validate_train_config(raw, path)


def validate_train_config(raw: dict[str, Any], path: Path) -> LoadedConfig:
    """Normalize legacy keys once; consumers see only their own canonical section."""
    import copy
    raw = copy.deepcopy(raw)
    path = path.expanduser().resolve()
    if raw.get("schema_version") not in (1, 2):
        raise ValueError("Only schema_version=1 or 2 is supported")
    raw.setdefault("training", {})
    training = raw["training"]
    if not isinstance(training, dict):
        raise TypeError("training must be a mapping")
    unknown = set(training) - COMMON_TRAINING_KEYS - {"method", "actor_lr_warmup_steps"}
    if unknown:
        raise ValueError(f"Unknown training keys: {sorted(unknown)}")
    training.setdefault("method", "iql")
    method = training_method(raw)
    legacy_iql = raw.setdefault("iql", {})
    if not isinstance(legacy_iql, dict):
        raise TypeError("iql must be a mapping")
    unknown = set(legacy_iql) - set(TRAIN_SCHEMA["iql"]) - COMMON_TRAINING_KEYS
    if unknown:
        raise ValueError(f"Unknown config.iql keys: {sorted(unknown)}")
    common_defaults = (read_source(PROJECT_ROOT / "configs" / "runtime.yaml")["training"]
                       if COMMON_TRAINING_KEYS - training.keys() - legacy_iql.keys() else {})
    for key in COMMON_TRAINING_KEYS:
        if key not in training:
            training[key] = legacy_iql[key] if key in legacy_iql else common_defaults[key]
        legacy_iql.pop(key, None)
    if training.get("actor_lr_warmup_steps") is None:
        # Compatibility only: old IQL configs tied actor LR warmup to critic warmup.
        training["actor_lr_warmup_steps"] = (legacy_iql.get("critic_warmup_steps", 1000)
                                            if method.name == "iql" else 1000)
    _number(training, "actor_lr_warmup_steps", low=0, integer=True)
    raw.setdefault("bc", {})
    if method.name == "bc":
        raw["iql"] = {}
        raw["reward"] = {}
    raw.setdefault("reward", {})
    raw["model"] = model_config(raw)
    raw.pop("vla", None)
    raw["schema_version"] = 2
    # Canonicalize old boolean configs. Explicit contradictory choices fail fast.
    if method.requires_rewards and isinstance(raw.get("reward"), dict):
        reward = raw["reward"]
        source = reward_source(reward)
        if "rynnvalue" in reward:
            if type(reward["rynnvalue"]) is not bool:
                raise TypeError("reward.rynnvalue must be boolean")
            if "source" in reward and source != "final" and reward["rynnvalue"] != (source == "rynnvalue"):
                raise ValueError("reward.source conflicts with legacy reward.rynnvalue")
        reward.update(source=source, rynnvalue=needs_rynnvalue({**reward, "source": source}))
        reward.setdefault("stage_exponent", 2.0)
        reward.setdefault("fusion_mode", "additive")
        reward.setdefault("alpha", 0.0)
        # Absence denotes historical, unscaled Final Reward snapshots.
        reward.setdefault("final_normalization", "none")
        for name in ("manifest_path", "manifest_sha256", "version_id"):
            reward.setdefault(name, None)
    if isinstance(raw, dict) and isinstance(raw.get("data"), dict):
        raw["data"].setdefault("stage_annotations_manifest", None)
        raw["data"].setdefault("include_post_success", True)
    schema = {**TRAIN_SCHEMA}
    if not method.requires_rewards:
        schema.update(reward=None, iql={})
    _validate_schema(raw, schema)
    _resolve_paths(
        raw, path,
        ("work_dir", "output_dir", "annotation_cache", "vla_adapter_root", "libero_x_root",
         "rynnvalue_root", "policy_registry"),
    )
    data, reward, iql, logging_cfg = raw["data"], raw["reward"], raw["iql"], raw["logging"]
    if method.requires_rewards:
        pinned = [reward[name] for name in ("manifest_path", "manifest_sha256", "version_id")]
        if any(value is not None for value in pinned):
            if any(not isinstance(value, str) or not value.strip() for value in pinned):
                raise ValueError("Pinned rewards require manifest_path, manifest_sha256 and version_id together")
            if re.fullmatch(r"[0-9a-f]{64}", reward["manifest_sha256"]) is None:
                raise ValueError("reward.manifest_sha256 must be a SHA256 digest")
            manifest_path = Path(reward["manifest_path"]).expanduser()
            reward["manifest_path"] = str(
                (manifest_path if manifest_path.is_absolute() else path.parent / manifest_path).resolve()
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
    stage_manifest = data["stage_annotations_manifest"]
    if stage_manifest is not None:
        if not isinstance(stage_manifest, str) or not stage_manifest.strip():
            raise TypeError("data.stage_annotations_manifest must be null or a non-empty path")
        stage_path = Path(stage_manifest).expanduser()
        data["stage_annotations_manifest"] = str(
            (stage_path if stage_path.is_absolute() else path.parent / stage_path).resolve()
        )
    for name, expected in (("action_horizon", 8), ("action_dim", 7), ("proprio_dim", 8)):
        _number(data, name, low=1, integer=True)
        if data[name] != expected:
            raise ValueError(f"data.{name} must match the current VLA-Adapter value {expected}")
    _number(data, "control_hz", low=1)
    if float(data["control_hz"]) != 20.0:
        raise ValueError("Version 1 requires data.control_hz=20")
    _number(data, "success_consecutive_steps", low=1, high=100, integer=True)
    if type(data["include_post_success"]) is not bool:
        raise TypeError("data.include_post_success must be boolean")
    _number(data, "validation_fraction", low=0, high=0.9)
    _number(data, "split_seed", integer=True)
    if type(data["allow_no_success"]) is not bool:
        raise TypeError("data.allow_no_success must be boolean")
    if method.requires_rewards:
        for key in ("model", "revision", "device", "dtype", "robot_description", "camera_description"):
            if not isinstance(reward[key], str) or not reward[key].strip():
                raise TypeError(f"reward.{key} must be a non-empty string")
        _number(reward, "max_frames", low=2, integer=True)
        _number(reward, "annotation_batch_size", low=1, integer=True)
        if type(reward["rynnvalue"]) is not bool:
            raise TypeError("reward.rynnvalue must be boolean")
        _number(reward, "gamma", low=0, high=1)
        _number(reward, "stage_exponent", low=1)
        if not math.isfinite(reward["stage_exponent"]):
            raise ValueError("reward.stage_exponent must be finite")
        _number(reward, "shaping_weight", low=0)
        _number(reward, "alpha", low=0, high=1)
        if any(not math.isfinite(reward[key]) for key in ("gamma", "shaping_weight", "alpha")):
            raise ValueError("Reward parameters must be finite")
        if reward["fusion_mode"] not in {"additive", "multiplicative"}:
            raise ValueError("reward.fusion_mode must be additive or multiplicative")
        if reward["final_normalization"] not in {"none", "initial_chunk_v1"}:
            raise ValueError("Unsupported reward.final_normalization")
        if type(reward["accumulate_primitive_steps"]) is not bool:
            raise TypeError("reward.accumulate_primitive_steps must be boolean")
        reward["accumulate_primitive_steps"] = effective_cumulative(reward)
        if reward["dtype"] != "bfloat16":
            raise ValueError(
                "Version 1 requires reward.dtype=bfloat16 to match the pinned RynnValue-4B "
                "checkpoint and the validated 16GB profile"
            )
        _cuda_device(reward["device"], "reward.device")
    for key in ("train_steps", "micro_batch_size", "gradient_accumulation_steps", "checkpoint_interval", "seed"):
        _number(training, key, low=0 if key == "seed" else 1, integer=True)
    resume = training["resume_checkpoint"]
    if resume is not None:
        if not isinstance(resume, str) or not resume.strip():
            raise TypeError("training.resume_checkpoint must be null or a non-empty path")
        candidate = Path(resume).expanduser()
        training["resume_checkpoint"] = str(
            (candidate if candidate.is_absolute() else path.parent / candidate).resolve())
    if training["checkpoint_interval"] % training["gradient_accumulation_steps"]:
        raise ValueError("training.checkpoint_interval must be divisible by gradient_accumulation_steps so resumed actor gradients are exact")
    for key in ("policy_peak_lr", "policy_final_lr"):
        _number(training, key, low=0)
    if training["dtype"] != "bfloat16":
        raise ValueError("training.dtype must be bfloat16")
    _cuda_device(training["device"], "training.device")
    if method.name == "iql":
        for key in ("critic_image_size", "critic_warmup_steps"):
            _number(iql, key, low=0 if key == "critic_warmup_steps" else 1, integer=True)
        for key in ("critic_lr", "value_lr", "beta", "max_advantage_weight",
                    "target_tau", "critic_weight_decay", "value_weight_decay"):
            _number(iql, key, low=0)
        for key in ("critic_max_grad_norm", "value_max_grad_norm"):
            _number(iql, key, low=1e-12)
        for key in ("critic_optimizer", "value_optimizer"):
            if iql[key] not in {"adam", "adamw"}:
                raise ValueError(f"iql.{key} must be adam or adamw")
        _number(iql, "expectile", low=0, high=1)
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
