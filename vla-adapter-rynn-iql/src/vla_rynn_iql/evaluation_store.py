from __future__ import annotations

import json
import copy
import os
import shutil
import tempfile
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import numpy as np

from .io import atomic_json, sha256_file
from .rewards import (
    ANNOTATION_SCHEMA_VERSION,
    REWARD_SCHEMA_VERSION,
    OFFICIAL_OUTPUT_KEYS,
)


SIDECAR_NAME = "rynnvalue_evaluation.json"
VALUES_NAME = "rynnvalue_evaluation.npz"
PBRS_ARRAY_KEYS = frozenset({"pbrs_shaping_reward", "pbrs_chunk_reward"})
REQUIRED_ARRAY_KEYS = frozenset(OFFICIAL_OUTPUT_KEYS) | PBRS_ARRAY_KEYS | {"boundary_steps"}
COMPATIBLE_SIDECAR_SCHEMA_VERSIONS = frozenset({5, ANNOTATION_SCHEMA_VERSION})


def _utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def _load_json(path: Path) -> dict[str, Any] | None:
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return None
    return value if isinstance(value, dict) else None


def valid_bound_evaluation(episode: dict[str, Any]) -> dict[str, Any] | None:
    """Return a hash-checked durable trajectory evaluation, if present."""
    trajectory = Path(episode["trajectory_path"])
    sidecar = trajectory.parent / SIDECAR_NAME
    values = trajectory.parent / VALUES_NAME
    if (
        trajectory.is_symlink()
        or not trajectory.is_file()
        or sidecar.is_symlink()
        or not sidecar.is_file()
        or values.is_symlink()
        or not values.is_file()
    ):
        return None
    payload = _load_json(sidecar)
    if payload is None:
        return None
    if (
        payload.get("schema_version") not in COMPATIBLE_SIDECAR_SCHEMA_VERSIONS
        or payload.get("run_id") != episode.get("run_id")
        or payload.get("trajectory_sha256") != sha256_file(trajectory)
        or payload.get("observations_sha256") != episode.get("observations_sha256")
        or payload.get("values_sha256") != sha256_file(values)
    ):
        return None
    try:
        with np.load(values, allow_pickle=False) as arrays:
            if not REQUIRED_ARRAY_KEYS.issubset(arrays.files):
                return None
            boundaries = np.asarray(arrays["boundary_steps"], dtype=np.int64)
            if (
                len(boundaries) < 2
                or int(boundaries[0]) != 0
                or int(boundaries[-1]) != int(episode["recorded_action_count"])
            ):
                return None
    except (KeyError, OSError, ValueError):
        return None
    return payload


def _atomic_copy(source: Path, destination: Path) -> None:
    descriptor, temporary_name = tempfile.mkstemp(
        prefix=f".{destination.name}.", suffix=".tmp", dir=destination.parent,
    )
    os.close(descriptor)
    temporary = Path(temporary_name)
    try:
        shutil.copyfile(source, temporary)
        os.replace(temporary, destination)
    finally:
        temporary.unlink(missing_ok=True)


def bind_config_evaluations(config) -> dict[str, Any]:
    """Bind original model rewards, never a Stage/fused reward as model output."""
    from .config import LoadedConfig, reward_source
    from .rewards import materialize_reward_manifest

    work = Path(config.section("paths")["work_dir"])
    if reward_source(config.section("reward")) == "rynnvalue":
        return bind_reward_manifest(work / "dataset_manifest.json", work / "rewards" / "reward_manifest.json")
    raw = copy.deepcopy(config.raw)
    binding_work = work / "model_binding"
    (binding_work / "annotations").mkdir(parents=True, exist_ok=True)
    for relative in ("dataset_manifest.json", "annotations/annotation_manifest.json"):
        shutil.copyfile(work / relative, binding_work / relative)
    raw["paths"]["work_dir"] = str(binding_work)
    raw["reward"].update(source="rynnvalue", rynnvalue=True, manifest_path=None, manifest_sha256=None, version_id=None)
    reward_path = materialize_reward_manifest(LoadedConfig(config.path, raw))
    return bind_reward_manifest(binding_work / "dataset_manifest.json", reward_path)


def bind_reward_manifest(
    prepared_path: Path,
    reward_path: Path,
) -> dict[str, Any]:
    """Atomically bind cached RynnValue results beside their source trajectories."""
    prepared = _load_json(prepared_path)
    rewards = _load_json(reward_path)
    if prepared is None:
        raise ValueError(f"Prepared manifest is invalid: {prepared_path}")
    if rewards is None or rewards.get("complete") is not True:
        raise ValueError(f"Reward manifest is incomplete: {reward_path}")
    if rewards.get("reward_config", {}).get("source", "rynnvalue") != "rynnvalue":
        raise ValueError("Only original RynnValue rewards may be bound as RynnValue evaluation")
    if (
        rewards.get("schema_version") != REWARD_SCHEMA_VERSION
        or rewards.get("kind") != "derived_iql_reward"
    ):
        raise ValueError(
            f"Derived reward manifest must use schema v{REWARD_SCHEMA_VERSION}"
        )
    if rewards.get("dataset_sha256") != prepared.get("dataset_sha256"):
        raise ValueError("Reward manifest does not match the prepared dataset")

    episodes = {item["run_id"]: item for item in prepared.get("episodes", [])}
    bound: list[str] = []
    skipped: list[str] = []
    for reward in rewards.get("episodes", []):
        run_id = str(reward.get("run_id") or "")
        episode = episodes.get(run_id)
        if episode is None:
            raise ValueError(f"Reward entry has no prepared episode: {run_id}")
        existing = valid_bound_evaluation(episode)
        if (
            existing is not None
            and existing.get("source_key") == reward.get("source_key")
            and existing.get("values_sha256") == reward.get("annotation_sha256")
        ):
            skipped.append(run_id)
            continue

        trajectory = Path(episode["trajectory_path"])
        if trajectory.is_symlink() or not trajectory.is_file():
            raise FileNotFoundError(f"Unsafe or missing trajectory for {run_id}: {trajectory}")
        source = Path(str(reward.get("annotation_path") or ""))
        if source.is_symlink() or not source.is_file():
            raise FileNotFoundError(f"Missing annotation for {run_id}: {source}")
        if sha256_file(source) != reward.get("annotation_sha256"):
            raise ValueError(f"Annotation hash mismatch for {run_id}")
        with np.load(source, allow_pickle=False) as arrays:
            missing = sorted(REQUIRED_ARRAY_KEYS - set(arrays.files))
            if missing:
                raise ValueError(f"Annotation for {run_id} is missing arrays: {missing}")
            boundaries = np.asarray(arrays["boundary_steps"], dtype=np.int64)
            if (
                len(boundaries) < 2
                or int(boundaries[0]) != 0
                or int(boundaries[-1]) != int(episode["recorded_action_count"])
            ):
                raise ValueError(f"Annotation boundaries do not cover {run_id}")

        episode_dir = trajectory.parent
        destination = episode_dir / VALUES_NAME
        _atomic_copy(source, destination)
        payload = {
            "schema_version": ANNOTATION_SCHEMA_VERSION,
            "run_id": run_id,
            "evaluated_at": _utc_now(),
            "trajectory_sha256": sha256_file(trajectory),
            "observations_sha256": episode.get("observations_sha256"),
            "source_key": reward.get("source_key"),
            "values_file": VALUES_NAME,
            "values_sha256": sha256_file(destination),
            "boundary_count": int(len(boundaries)),
            "annotator": reward.get("annotator") or rewards.get("annotator") or {},
            "annotation_config": rewards.get("annotation_config") or {},
            "reward_config": rewards.get("reward_config") or {},
            "official_outputs": reward.get("official_outputs") or {},
            "pbrs_reward": reward.get("pbrs_reward") or {},
            "environment_success": reward.get("environment_success"),
        }
        atomic_json(episode_dir / SIDECAR_NAME, payload)
        bound.append(run_id)
    return {
        "bound": bound,
        "skipped": skipped,
        "bound_count": len(bound),
        "skipped_count": len(skipped),
    }
