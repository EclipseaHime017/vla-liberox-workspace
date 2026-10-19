"""Server execution settings; model and algorithm settings remain in training/*.yaml."""
from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import yaml

from .config import UniqueKeyLoader
from .config_sources import strict_keys, resolve_path, validate_overrides


@dataclass(frozen=True)
class DistributedConfig:
    gpu_ids: tuple[int, ...]
    backend: str = "nccl"
    zero_stage: int = 1
    timeout_seconds: int = 1800
    data_workers_per_rank: int = 2
    prefetch_factor: int = 2
    pin_memory: bool = True
    persistent_workers: bool = True

    @property
    def world_size(self) -> int:
        return len(self.gpu_ids)


@dataclass(frozen=True)
class ServerConfig:
    path: Path
    training_config: Path
    runs_root: Path
    task_id: str | None
    output_root: Path
    cache_root: Path
    reward_source: str
    distributed: DistributedConfig
    overrides: dict[str, Any]
    model_profiles: dict[str, dict] = field(default_factory=dict)


def configure_server(path, *, task=None, runs_root=None, model=None, method=None):
    """Resolve common CLI overrides for direct and queued training."""
    from dataclasses import replace
    from .base_models import base_model, model_contract
    from .models import base_model_config
    from .config_sources import merge_config
    server = load_server_config(path)
    overrides = server.overrides
    if model:
        base = base_model(model)
        settings = overrides["model"]
        family = settings.get("family")
        if family is None:
            from .config import load_train_config
            family = load_train_config(server.training_config, overrides=overrides,
                                       overrides_path=server.path).section("model")["family"]
        identity = {"family", "base_id", "base_checkpoint", "base_revision", "stats_key", "contract"}
        retained = {key: value for key, value in settings.items() if key not in identity} if family == base.family else {}
        selected = {key: value for key, value in base_model_config(base.id).items() if key in identity}
        selected["contract"] = model_contract(selected)
        overrides = {**overrides, "model": {**retained, **selected}}
    if method:
        overrides = merge_config(overrides, {"training": {"method": method}})
    return replace(server, overrides=overrides, task_id=task or server.task_id,
                   runs_root=Path(runs_root).expanduser().resolve() if runs_root else server.runs_root)


def model_profiles(raw) -> dict[str, dict]:
    """Server-only batch and adaptation defaults, below explicit run overrides."""
    from .models import MODELS, model_config
    if not isinstance(raw, dict):
        raise TypeError("model_profiles must be a mapping")
    result = {}
    for family, profile in raw.items():
        if family not in MODELS:
            raise ValueError(f"Unknown model_profiles family: {family}")
        values = validate_overrides(profile)
        allowed = {"training": {"micro_batch_size", "gradient_accumulation_steps"},
                   "model": {"backbone", "lora", "action_head", "proprio_projector"}}
        for section, settings in values.items():
            if settings.keys() - allowed.get(section, set()):
                raise ValueError(f"Invalid model_profiles.{family}.{section} keys")
        for key, value in values["training"].items():
            if type(value) is not int or value < 1:
                raise ValueError(f"model_profiles.{family}.training.{key} must be a positive integer")
        model_config({"model": {"family": family, **values["model"]}})
        result[family] = {section: values[section] for section in allowed}
    return result


def distributed_config(raw: dict) -> DistributedConfig:
    defaults = DistributedConfig((0,)).__dict__
    if not isinstance(raw, dict) or raw.keys() - defaults.keys():
        raise ValueError("Unknown distributed settings")
    values = {**defaults, **raw}
    ids = values["gpu_ids"]
    if (not isinstance(ids, (list, tuple)) or not 1 <= len(ids) <= 8
            or any(type(item) is not int or item < 0 for item in ids) or len(set(ids)) != len(ids)):
        raise ValueError("gpu_ids must specify 1–8 distinct non-negative GPU indices")
    if values["backend"] != "nccl" or type(values["zero_stage"]) is not int or values["zero_stage"] != 1:
        raise ValueError("Server training requires NCCL and ZeRO stage 1")
    for key, minimum in (("timeout_seconds", 1), ("data_workers_per_rank", 0), ("prefetch_factor", 1)):
        if type(values[key]) is not int or values[key] < minimum:
            raise ValueError(f"distributed.{key} must be an integer >= {minimum}")
    for key in ("pin_memory", "persistent_workers"):
        if type(values[key]) is not bool:
            raise TypeError(f"distributed.{key} must be boolean")
    if not values["data_workers_per_rank"] and values["persistent_workers"]:
        raise ValueError("persistent_workers requires data_workers_per_rank > 0")
    return DistributedConfig(**{**values, "gpu_ids": tuple(ids)})


def load_server_config(path: Path) -> ServerConfig:
    import re
    path = path.expanduser().resolve()
    raw = yaml.load(path.read_text(), Loader=UniqueKeyLoader)
    profiles = model_profiles(raw.pop("model_profiles", {}) if isinstance(raw, dict) else {})
    if isinstance(raw, dict) and raw.get("schema_version") == 3:
        environment = raw.pop("environment", None)
        if environment is not None and (not isinstance(environment, str) or not re.fullmatch(r"[\w.-]+", environment)):
            raise ValueError("Legacy environment must be a Conda environment name")
    strict_keys(raw, {"schema_version", "training_config", "runs_root", "task_id",
                      "output_root", "cache_root", "reward_source",
                      "distributed", "overrides"}, "server")
    if raw["schema_version"] not in (3, 4):
        raise ValueError("Use server schema_version=4; model YAML owns the training environment")
    if raw["reward_source"] not in {"final", "rynnvalue", "stage", "sparse"}:
        raise ValueError("reward_source must be final, rynnvalue, stage or sparse")
    paths = {key: resolve_path(raw[key], path.parent, key) for key in
             ("training_config", "runs_root", "output_root", "cache_root")}
    task_id = raw["task_id"]
    if task_id is not None and (not isinstance(task_id, str) or not task_id.strip()):
        raise ValueError("task_id must be null or a non-empty canonical task ID")
    overrides = validate_overrides(raw["overrides"])
    # Semantic evaluator parameters belong to each copied global evaluation.
    if set(overrides["reward"]) - {"gamma", "accumulate_primitive_steps"}:
        raise ValueError("Server reward overrides only accept gamma and accumulate_primitive_steps")
    if set(overrides["data"]) - {"include_post_success", "validation_fraction", "split_seed"}:
        raise ValueError("Server data overrides accept include_post_success, validation_fraction and split_seed")
    if "device" in overrides["training"]:
        raise ValueError("Select server devices through distributed.gpu_ids, not training.device")
    return ServerConfig(path, **paths, task_id=task_id,
                        reward_source=raw["reward_source"], distributed=distributed_config(raw["distributed"]),
                        overrides=overrides, model_profiles=profiles)


def validate_global_batch(raw: dict, distributed: DistributedConfig) -> tuple[int, int]:
    size = raw["training"]["micro_batch_size"]
    if type(size) is not int or size < distributed.world_size or size % distributed.world_size:
        raise ValueError(f"Global training.micro_batch_size must be divisible by {distributed.world_size} GPUs")
    if raw["training"]["checkpoint_interval"] % raw["training"]["gradient_accumulation_steps"]:
        raise ValueError("checkpoint_interval must be divisible by gradient_accumulation_steps")
    return size, size // distributed.world_size
