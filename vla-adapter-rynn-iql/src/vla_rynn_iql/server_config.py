from __future__ import annotations

import copy
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import yaml

from .config import TRAIN_SCHEMA, UniqueKeyLoader
from .terminal_pipeline import (
    MODES,
    ORDERS,
    OUTCOMES,
    SOURCE_TYPES,
    TerminalPipelineConfig,
)


@dataclass(frozen=True)
class DistributedConfig:
    gpu_ids: tuple[int, ...]
    backend: str
    zero_stage: int
    timeout_seconds: int
    data_workers_per_rank: int
    prefetch_factor: int
    pin_memory: bool
    persistent_workers: bool

    @property
    def world_size(self) -> int:
        return len(self.gpu_ids)


@dataclass(frozen=True)
class ReplayCacheConfig:
    enabled: bool
    root: Path
    rebuild: bool


@dataclass(frozen=True)
class ServerPipelineConfig:
    path: Path
    terminal: TerminalPipelineConfig
    distributed: DistributedConfig
    replay_cache: ReplayCacheConfig


def _mapping(value: Any, expected: set[str], context: str) -> dict[str, Any]:
    if not isinstance(value, dict):
        raise TypeError(f"{context} must be a mapping")
    missing = sorted(expected - set(value))
    unknown = sorted(set(value) - expected)
    if missing:
        raise ValueError(f"Missing {context} keys: {missing}")
    if unknown:
        raise ValueError(f"Unknown {context} keys: {unknown}")
    return value


def _resolve(value: Any, base: Path, context: str) -> Path:
    if not isinstance(value, str) or not value.strip():
        raise TypeError(f"{context} must be a non-empty path")
    path = Path(value).expanduser()
    return (path if path.is_absolute() else base / path).resolve()


def _validate_overrides(value: Any) -> dict[str, dict[str, Any]]:
    sections = set(TRAIN_SCHEMA) - {"schema_version"}
    raw = _mapping(value, sections, "overrides")
    managed = {"paths": {"work_dir"}, "data": {"task_ids", "selection_manifest"}}
    for section, values in raw.items():
        if not isinstance(values, dict):
            raise TypeError(f"overrides.{section} must be a mapping")
        unknown = sorted(set(values) - set(TRAIN_SCHEMA[section]))
        if unknown:
            raise ValueError(f"Unknown overrides.{section} keys: {unknown}")
        forbidden = sorted(set(values) & managed.get(section, set()))
        if forbidden:
            raise ValueError(f"Pipeline-managed overrides.{section} keys: {forbidden}")
    return copy.deepcopy(raw)


def load_server_config(path: Path) -> ServerPipelineConfig:
    path = path.expanduser().resolve()
    raw = yaml.load(path.read_text(encoding="utf-8"), Loader=UniqueKeyLoader)
    root = _mapping(
        raw,
        {
            "schema_version", "base_config", "pipeline_root", "environments",
            "selection", "overrides", "distributed", "replay_cache",
        },
        "server config",
    )
    if root["schema_version"] != 1:
        raise ValueError("Only server pipeline schema_version=1 is supported")

    environments = _mapping(
        root["environments"], {"prepare", "annotate", "train"}, "environments"
    )
    for name, environment in environments.items():
        if not isinstance(environment, str) or re.fullmatch(
            r"[A-Za-z0-9_.-]+", environment
        ) is None:
            raise ValueError(f"environments.{name} is not a valid Conda environment name")

    selection = _mapping(
        root["selection"],
        {"task_id", "mode", "seed", "source_types", "outcomes", "size", "quotas"},
        "selection",
    )
    # Reuse the already-audited terminal selection validation without writing a
    # temporary file by constructing its value only after checking the same
    # externally visible shape here. Detailed selection values are validated by
    # select_candidates/build_selection_manifest before execution.
    if not isinstance(selection["task_id"], str) or not selection["task_id"].strip():
        raise TypeError("selection.task_id must be a non-empty string")
    if selection["mode"] not in MODES:
        raise ValueError("selection.mode must be quota, random, or all")
    if type(selection["seed"]) is not int:
        raise TypeError("selection.seed must be an integer")
    for key, allowed in (("source_types", SOURCE_TYPES), ("outcomes", OUTCOMES)):
        values = selection[key]
        if (
            not isinstance(values, list)
            or not values
            or any(value not in allowed for value in values)
            or len(values) != len(set(values))
        ):
            raise ValueError(
                f"selection.{key} must contain unique values from {sorted(allowed)}"
            )
    quotas = selection["quotas"]
    if not isinstance(quotas, list):
        raise TypeError("selection.quotas must be a list")
    pairs: set[tuple[str, str]] = set()
    for index, quota in enumerate(quotas):
        quota = _mapping(
            quota,
            {"source_type", "outcome", "count", "order"},
            f"selection.quotas[{index}]",
        )
        pair = (quota["source_type"], quota["outcome"])
        if pair[0] not in SOURCE_TYPES or pair[1] not in OUTCOMES or pair in pairs:
            raise ValueError("Quota source/outcome pairs must be unique and supported")
        if type(quota["count"]) is not int or quota["count"] < 1:
            raise ValueError("Every quota count must be a positive integer")
        if quota["order"] not in ORDERS:
            raise ValueError(f"Quota order must be one of {sorted(ORDERS)}")
        pairs.add(pair)
    if selection["mode"] == "quota":
        if not quotas or selection["size"] is not None:
            raise ValueError("quota mode requires quotas and size: null")
    elif selection["mode"] == "random":
        if type(selection["size"]) is not int or selection["size"] < 1 or quotas:
            raise ValueError("random mode requires a positive size and quotas: []")
    elif selection["size"] is not None or quotas:
        raise ValueError("all mode requires size: null and quotas: []")

    distributed_raw = _mapping(
        root["distributed"],
        {
            "gpu_ids", "backend", "zero_stage", "timeout_seconds",
            "data_workers_per_rank", "prefetch_factor", "pin_memory",
            "persistent_workers",
        },
        "distributed",
    )
    gpu_ids = distributed_raw["gpu_ids"]
    if (
        not isinstance(gpu_ids, list)
        or not 1 <= len(gpu_ids) <= 8
        or any(type(value) is not int or value < 0 for value in gpu_ids)
        or len(gpu_ids) != len(set(gpu_ids))
    ):
        raise ValueError("distributed.gpu_ids must contain 1-8 unique non-negative integers")
    if distributed_raw["backend"] != "nccl":
        raise ValueError("Server GPU training currently requires distributed.backend=nccl")
    if distributed_raw["zero_stage"] != 1:
        raise ValueError("Server GPU training currently requires distributed.zero_stage=1")
    for key in ("timeout_seconds", "prefetch_factor"):
        if type(distributed_raw[key]) is not int or distributed_raw[key] < 1:
            raise ValueError(f"distributed.{key} must be a positive integer")
    workers = distributed_raw["data_workers_per_rank"]
    if type(workers) is not int or workers < 0:
        raise ValueError("distributed.data_workers_per_rank must be a non-negative integer")
    for key in ("pin_memory", "persistent_workers"):
        if type(distributed_raw[key]) is not bool:
            raise TypeError(f"distributed.{key} must be boolean")
    if workers == 0 and distributed_raw["persistent_workers"]:
        raise ValueError("persistent_workers requires data_workers_per_rank > 0")

    cache_raw = _mapping(root["replay_cache"], {"enabled", "root", "rebuild"}, "replay_cache")
    for key in ("enabled", "rebuild"):
        if type(cache_raw[key]) is not bool:
            raise TypeError(f"replay_cache.{key} must be boolean")
    if not cache_raw["enabled"]:
        raise ValueError(
            "The server trainer requires replay_cache.enabled=true to avoid repeated NPZ decompression"
        )

    terminal = TerminalPipelineConfig(
        path=path,
        base_config=_resolve(root["base_config"], path.parent, "base_config"),
        pipeline_root=_resolve(root["pipeline_root"], path.parent, "pipeline_root"),
        environments=dict(environments),
        selection=copy.deepcopy(selection),
        overrides=_validate_overrides(root["overrides"]),
    )
    distributed = DistributedConfig(
        gpu_ids=tuple(gpu_ids),
        backend=str(distributed_raw["backend"]),
        zero_stage=int(distributed_raw["zero_stage"]),
        timeout_seconds=int(distributed_raw["timeout_seconds"]),
        data_workers_per_rank=workers,
        prefetch_factor=int(distributed_raw["prefetch_factor"]),
        pin_memory=bool(distributed_raw["pin_memory"]),
        persistent_workers=bool(distributed_raw["persistent_workers"]),
    )
    replay_cache = ReplayCacheConfig(
        enabled=True,
        root=_resolve(cache_raw["root"], path.parent, "replay_cache.root"),
        rebuild=bool(cache_raw["rebuild"]),
    )
    return ServerPipelineConfig(path, terminal, distributed, replay_cache)


def validate_global_batch(raw: dict[str, Any], distributed: DistributedConfig) -> tuple[int, int]:
    global_micro_batch = int(raw["iql"]["micro_batch_size"])
    if global_micro_batch % distributed.world_size:
        raise ValueError(
            "iql.micro_batch_size is the global micro batch in server mode and must be "
            f"divisible by {distributed.world_size} GPUs; got {global_micro_batch}"
        )
    local_micro_batch = global_micro_batch // distributed.world_size
    if local_micro_batch < 1:
        raise ValueError("Server local micro batch must be at least one sample per GPU")
    return global_micro_batch, local_micro_batch
