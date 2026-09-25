from __future__ import annotations

import copy
import hashlib
import json
import math
import os
import random
import re
import tempfile
import zipfile
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable

import numpy as np
import yaml

from .config import TRAIN_SCHEMA, UniqueKeyLoader, load_train_config, reward_source
from .data import MANIFEST_NAME, MANIFEST_SCHEMA_VERSION, confirmed_terminal_step, replay_chunks
from .evaluation_store import valid_bound_evaluation
from .io import atomic_json, sha256_file, stable_hash
from .methods import COMMON_TRAINING_KEYS
from .models import DEFAULT_MODEL
from .rewards import (
    ANNOTATION_SCHEMA_VERSION,
    REWARD_SCHEMA_VERSION,
    official_inference_config,
    reward_derivation_config,
    reward_implementation_fingerprint,
)


SOURCE_TYPES = frozenset({"inference", "manual", "policy_requery"})
OUTCOMES = frozenset({"success", "failure"})
ORDERS = frozenset({"random", "oldest", "newest"})
MODES = frozenset({"quota", "random", "all"})
MANAGED_OVERRIDES = {
    "paths": {"work_dir"},
    "data": {"task_ids", "selection_manifest"},
}


@dataclass(frozen=True)
class TerminalPipelineConfig:
    path: Path
    base_config: Path
    pipeline_root: Path
    environments: dict[str, str]
    selection: dict[str, Any]
    overrides: dict[str, dict[str, Any]]


@dataclass(frozen=True)
class Candidate:
    run_id: str
    task_id: str
    root_run_id: str
    parent_run_id: str | None
    source_type: str
    outcome: str
    created_at: str
    resume_step: int
    action_count: int
    run_path: Path
    trajectory_path: Path
    observations_path: Path


def _strict_keys(value: Any, expected: set[str], context: str) -> dict[str, Any]:
    if not isinstance(value, dict):
        raise TypeError(f"{context} must be a mapping")
    missing = sorted(expected - set(value))
    unknown = sorted(set(value) - expected)
    if missing:
        raise ValueError(f"Missing {context} keys: {missing}")
    if unknown:
        raise ValueError(f"Unknown {context} keys: {unknown}")
    return value


def _resolve(path: str, base: Path, context: str) -> Path:
    if not isinstance(path, str) or not path.strip():
        raise TypeError(f"{context} must be a non-empty path")
    value = Path(path).expanduser()
    return (value if value.is_absolute() else base / value).resolve()


def _validate_overrides(overrides: Any) -> dict[str, dict[str, Any]]:
    if isinstance(overrides, dict):
        overrides.setdefault("training", {})
        overrides.setdefault("model", {})
        for section in TRAIN_SCHEMA:
            if section != "schema_version":
                overrides.setdefault(section, {})
        overrides.setdefault("vla", {})
    sections = (set(TRAIN_SCHEMA) - {"schema_version"}) | {"vla"}
    result = _strict_keys(overrides, sections, "overrides")
    for section, values in result.items():
        if not isinstance(values, dict):
            raise TypeError(f"overrides.{section} must be a mapping")
        allowed = (COMMON_TRAINING_KEYS | {"method", "actor_lr_warmup_steps"}
                   if section == "training" else set(DEFAULT_MODEL) if section == "model"
                   else {"base_checkpoint", "stats_key", "use_pro_version", "freeze_backbone"} if section == "vla"
                   else set(TRAIN_SCHEMA[section]) | (COMMON_TRAINING_KEYS if section == "iql" else set()))
        unknown = sorted(set(values) - allowed)
        if unknown:
            raise ValueError(f"Unknown overrides.{section} keys: {unknown}")
        managed = sorted(set(values) & MANAGED_OVERRIDES.get(section, set()))
        if managed:
            raise ValueError(f"Pipeline-managed overrides.{section} keys: {managed}")
    return result


def load_terminal_config(path: Path) -> TerminalPipelineConfig:
    path = path.expanduser().resolve()
    raw = yaml.load(path.read_text(encoding="utf-8"), Loader=UniqueKeyLoader)
    root = _strict_keys(
        raw,
        {"schema_version", "base_config", "pipeline_root", "environments", "selection", "overrides"},
        "terminal config",
    )
    if root["schema_version"] != 1:
        raise ValueError("Only terminal pipeline schema_version=1 is supported")
    environments = _strict_keys(
        root["environments"], {"prepare", "annotate", "train"}, "environments"
    )
    for name, value in environments.items():
        if not isinstance(value, str) or re.fullmatch(r"[A-Za-z0-9_.-]+", value) is None:
            raise ValueError(f"environments.{name} is not a valid Conda environment name")

    selection = _strict_keys(
        root["selection"],
        {"task_id", "mode", "seed", "source_types", "outcomes", "size", "quotas"},
        "selection",
    )
    if not isinstance(selection["task_id"], str) or not selection["task_id"].strip():
        raise TypeError("selection.task_id must be a non-empty string")
    if selection["mode"] not in MODES:
        raise ValueError(f"selection.mode must be one of {sorted(MODES)}")
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
            raise ValueError(f"selection.{key} must contain unique values from {sorted(allowed)}")
    quotas = selection["quotas"]
    if not isinstance(quotas, list):
        raise TypeError("selection.quotas must be a list")
    seen: set[tuple[str, str]] = set()
    for index, quota in enumerate(quotas):
        quota = _strict_keys(
            quota, {"source_type", "outcome", "count", "order"},
            f"selection.quotas[{index}]",
        )
        key = (quota["source_type"], quota["outcome"])
        if key[0] not in SOURCE_TYPES or key[1] not in OUTCOMES or key in seen:
            raise ValueError("Quota source/outcome pairs must be unique and supported")
        if type(quota["count"]) is not int or quota["count"] < 1:
            raise ValueError("Every quota count must be a positive integer")
        if quota["order"] not in ORDERS:
            raise ValueError(f"Quota order must be one of {sorted(ORDERS)}")
        seen.add(key)
    if selection["mode"] == "quota":
        if not quotas or selection["size"] is not None:
            raise ValueError("quota mode requires quotas and size: null")
    elif selection["mode"] == "random":
        if type(selection["size"]) is not int or selection["size"] < 1 or quotas:
            raise ValueError("random mode requires a positive size and quotas: []")
    elif selection["size"] is not None or quotas:
        raise ValueError("all mode requires size: null and quotas: []")

    return TerminalPipelineConfig(
        path=path,
        base_config=_resolve(root["base_config"], path.parent, "base_config"),
        pipeline_root=_resolve(root["pipeline_root"], path.parent, "pipeline_root"),
        environments=dict(environments),
        selection=copy.deepcopy(selection),
        overrides=copy.deepcopy(_validate_overrides(root["overrides"])),
    )


def merged_training_config(config: TerminalPipelineConfig) -> dict[str, Any]:
    raw = load_train_config(config.base_config, overrides=config.overrides,
                            overrides_path=config.path).raw
    raw["data"]["task_ids"] = [config.selection["task_id"]]
    raw["data"]["selection_manifest"] = None
    return raw


def _safe_extract(source: Path, target: Path) -> None:
    target.mkdir(parents=True, exist_ok=True)
    root = target.resolve()
    with zipfile.ZipFile(source) as archive:
        for info in archive.infolist():
            destination = (target / info.filename).resolve()
            if destination != root and root not in destination.parents:
                raise ValueError(f"ZIP contains an unsafe path: {info.filename}")
        archive.extractall(target)


def dataset_roots(raw: dict[str, Any], import_root: Path) -> list[Path]:
    roots: list[Path] = []
    for source_value in raw["paths"]["dataset_sources"]:
        source = Path(source_value)
        if source.is_dir():
            roots.append(source)
        elif source.is_file() and source.suffix.lower() == ".zip":
            fingerprint = sha256_file(source)[:16]
            target = import_root / fingerprint
            marker = target / ".complete"
            if not marker.is_file():
                if target.exists():
                    raise ValueError(f"Incomplete import cache must be removed: {target}")
                _safe_extract(source, target)
                marker.touch()
            roots.append(target)
        else:
            raise FileNotFoundError(f"Unsupported dataset source: {source}")
    return roots


def _artifact_path(run_path: Path, value: Any, fallback: Path) -> Path:
    if isinstance(value, str) and value.strip():
        path = Path(value).expanduser()
        return (path if path.is_absolute() else run_path.parent / path).resolve()
    return fallback.resolve()


def discover_candidates(
    roots: Iterable[Path], project_id: str, success_consecutive_steps: int = 5,
) -> tuple[list[Candidate], list[dict[str, str]]]:
    candidates: list[Candidate] = []
    rejected: list[dict[str, str]] = []
    seen: set[str] = set()
    for root in roots:
        for run_path in sorted(root.rglob("run.json")):
            try:
                run = json.loads(run_path.read_text(encoding="utf-8"))
                if run.get("project_id") not in (None, project_id):
                    continue
                run_id = str(run.get("id") or "")
                task_id = str(run.get("task_id") or "")
                if not run_id or not task_id:
                    raise ValueError("missing id or task_id")
                if run_id in seen:
                    raise RuntimeError(f"Conflicting duplicate run id across dataset sources: {run_id}")
                if run.get("status") != "COMPLETED" or run.get("error"):
                    raise ValueError("run is not successfully completed")
                trajectory = _artifact_path(
                    run_path, run.get("trajectory"),
                    run_path.parent / "episodes" / "episode_000" / "trajectory.npz",
                )
                observations = trajectory.with_name("trajectory_observations.npz")
                if run_path.is_symlink() or trajectory.is_symlink() or observations.is_symlink():
                    raise ValueError("symlink artifacts are not supported")
                if not trajectory.is_file() or not observations.is_file():
                    raise FileNotFoundError("trajectory.npz or trajectory_observations.npz is missing")
                with np.load(trajectory, allow_pickle=False) as arrays:
                    if "env_action" not in arrays.files or "done" not in arrays.files:
                        raise ValueError("trajectory has no env_action or done")
                    action_count = int(len(arrays["env_action"]))
                    done = np.asarray(arrays["done"], dtype=bool)
                if action_count < 1 or int(run.get("action_count") or action_count) != action_count:
                    raise ValueError("action_count does not match trajectory")
                if len(done) != action_count:
                    raise ValueError("done length does not match trajectory")
                kind = str(run.get("kind") or "")
                source_type = (
                    "manual" if kind == "branch" and run.get("control_mode") == "manual"
                    else "policy_requery" if kind == "branch"
                    else "inference"
                )
                root_id = str(
                    run.get("root_session_id") or run.get("parent_session_id") or run_id
                )
                candidates.append(Candidate(
                    run_id=run_id,
                    task_id=task_id,
                    root_run_id=root_id,
                    parent_run_id=run.get("parent_session_id"),
                    source_type=source_type,
                    outcome=(
                        "success"
                        if confirmed_terminal_step(done, success_consecutive_steps) is not None
                        else "failure"
                    ),
                    created_at=str(run.get("created_at") or ""),
                    resume_step=int(run.get("resume_step") or 0) if kind == "branch" else 0,
                    action_count=action_count,
                    run_path=run_path.resolve(),
                    trajectory_path=trajectory,
                    observations_path=observations,
                ))
                seen.add(run_id)
            except (OSError, ValueError, json.JSONDecodeError) as exc:
                rejected.append({"path": str(run_path), "reason": f"{type(exc).__name__}: {exc}"})
    return candidates, rejected


def resolve_task_id(requested: str, candidates: Iterable[Candidate]) -> str:
    available = sorted({item.task_id for item in candidates})
    if requested in available:
        return requested
    matches = [value for value in available if value.endswith(f"::{requested}")]
    if len(matches) == 1:
        return matches[0]
    if not matches:
        raise ValueError(f"Task {requested!r} was not found; available tasks: {available}")
    raise ValueError(f"Task {requested!r} is ambiguous: {matches}")


def _ordered(items: list[Candidate], order: str, seed: int) -> list[Candidate]:
    stable = sorted(items, key=lambda item: item.run_id)
    if order == "random":
        return random.Random(seed).sample(stable, len(stable))
    return sorted(
        stable, key=lambda item: (item.created_at, item.run_id), reverse=order == "newest"
    )


def select_candidates(
    candidates: Iterable[Candidate], selection: dict[str, Any], canonical_task_id: str,
) -> list[Candidate]:
    filtered = [
        item for item in candidates
        if item.task_id == canonical_task_id
        and item.source_type in set(selection["source_types"])
        and item.outcome in set(selection["outcomes"])
    ]
    mode = selection["mode"]
    seed = int(selection["seed"])
    if mode == "all":
        selected = sorted(filtered, key=lambda item: item.run_id)
    elif mode == "random":
        count = int(selection["size"])
        if count > len(filtered):
            raise ValueError(f"Requested {count} trajectories but only {len(filtered)} are eligible")
        selected = random.Random(seed).sample(sorted(filtered, key=lambda item: item.run_id), count)
    else:
        selected = []
        for index, quota in enumerate(selection["quotas"]):
            group = [
                item for item in filtered
                if item.source_type == quota["source_type"] and item.outcome == quota["outcome"]
            ]
            count = int(quota["count"])
            if count > len(group):
                raise ValueError(
                    f"Quota {quota['source_type']}/{quota['outcome']} requests {count}, "
                    f"but only {len(group)} are eligible"
                )
            selected.extend(_ordered(group, quota["order"], seed + index)[:count])
    if not selected:
        raise ValueError("Selection resolved to zero trajectories")
    if len({item.run_id for item in selected}) != len(selected):
        raise ValueError("Selection resolved to duplicate trajectories")
    return selected


def _split(root_id: str, seed: int, fraction: float) -> str:
    digest = hashlib.sha256(f"{seed}:{root_id}".encode()).digest()
    value = int.from_bytes(digest[:8], "big") / float(2 ** 64)
    return "validation" if value < fraction else "train"


def _artifact(path: Path) -> dict[str, Any]:
    if path.is_symlink() or not path.is_file():
        raise FileNotFoundError(f"Unsafe or missing dataset artifact: {path}")
    return {"path": str(path.resolve()), "size": path.stat().st_size, "sha256": sha256_file(path)}


def build_selection_manifest(
    selected: list[Candidate], raw: dict[str, Any], selection: dict[str, Any],
    canonical_task_id: str,
) -> dict[str, Any]:
    data = raw["data"]
    members = []
    for item in selected:
        members.append({
            "run_id": item.run_id,
            "root_run_id": item.root_run_id,
            "parent_run_id": item.parent_run_id,
            "source_type": item.source_type,
            "outcome": item.outcome,
            "resume_step": item.resume_step,
            "end_step": item.action_count,
            "action_count": item.action_count,
            "chunk_count": math.ceil(item.resume_step / 8)
            + math.ceil(max(0, item.action_count - item.resume_step) / 8),
            "split": _split(
                item.root_run_id, int(data["split_seed"]), float(data["validation_fraction"])
            ),
            "artifacts": {
                "manifest": _artifact(item.run_path),
                "trajectory": _artifact(item.trajectory_path),
                "observations": _artifact(item.observations_path),
            },
        })
    if members and all(item["split"] == "validation" for item in members):
        first_root = members[0]["root_run_id"]
        for member in members:
            if member["root_run_id"] == first_root:
                member["split"] = "train"
    selection_record = copy.deepcopy(selection)
    selection_record["task_id"] = canonical_task_id
    immutable = {
        "task_id": canonical_task_id,
        "selection": selection_record,
        "validation_fraction": data["validation_fraction"],
        "split_seed": data["split_seed"],
        "success_consecutive_steps": data["success_consecutive_steps"],
        "members": members,
    }
    digest = stable_hash(immutable)
    return {
        "schema_version": 1,
        "id": f"terminal-ds-{digest[:12]}",
        "project_id": data["project_id"],
        **immutable,
        "dataset_sha256": digest,
    }


def prepare_fingerprint(selection_manifest: dict[str, Any], raw: dict[str, Any]) -> str:
    # Schema-4 evaluation tails already describe full replay and have reusable
    # annotations. Keep their prepare identity stable across the replay policy change.
    data = raw["data"]
    return stable_hash({
        "selection_sha256": selection_manifest["dataset_sha256"],
        "manifest_schema_version": MANIFEST_SCHEMA_VERSION,
        "data": {key: data[key] for key in (
            "project_id", "action_horizon", "action_dim", "proprio_dim", "control_hz",
            "success_consecutive_steps", "validation_fraction", "split_seed", "allow_no_success",
        )},
    })


def prepare_cache_valid(work_dir: Path, fingerprint: str) -> bool:
    state_path = work_dir / "prepare_state.json"
    manifest_path = work_dir / MANIFEST_NAME
    try:
        state = json.loads(state_path.read_text(encoding="utf-8"))
        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return False
    try:
        for episode in manifest["episodes"]:
            replay_chunks(episode)
    except (KeyError, TypeError, ValueError):
        return False
    return bool(
        state.get("fingerprint") == fingerprint
        and state.get("manifest_sha256") == sha256_file(manifest_path)
        and manifest.get("schema_version") == MANIFEST_SCHEMA_VERSION
        and manifest.get("source_dataset_sha256") == state.get("selection_sha256")
    )


def mark_prepare_cache(work_dir: Path, fingerprint: str, selection_sha256: str) -> None:
    manifest = work_dir / MANIFEST_NAME
    if not manifest.is_file():
        raise FileNotFoundError(f"Prepare did not create {manifest}")
    atomic_json(work_dir / "prepare_state.json", {
        "schema_version": 1,
        "fingerprint": fingerprint,
        "selection_sha256": selection_sha256,
        "manifest_sha256": sha256_file(manifest),
    })


def annotation_cache_valid(work_dir: Path, reward_config: dict[str, Any]) -> bool:
    prepared_path = work_dir / MANIFEST_NAME
    annotation_path = work_dir / "annotations" / "annotation_manifest.json"
    try:
        prepared = json.loads(prepared_path.read_text(encoding="utf-8"))
        annotations = json.loads(annotation_path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return False
    if (
        annotations.get("schema_version") != ANNOTATION_SCHEMA_VERSION
        or annotations.get("kind") != "rynnvalue_annotation"
        or annotations.get("complete") is not True
        or annotations.get("dataset_sha256") != prepared.get("dataset_sha256")
        or annotations.get("annotation_config") != official_inference_config(reward_config)
    ):
        return False
    for episode in annotations.get("episodes", []):
        path = Path(str(episode.get("annotation_path") or ""))
        if (
            path.is_symlink()
            or not path.is_file()
            or episode.get("annotation_sha256") != sha256_file(path)
        ):
            return False
    return len(annotations.get("episodes", [])) == len(prepared.get("episodes", []))


def reward_cache_valid(work_dir: Path, reward_config: dict[str, Any]) -> bool:
    # Direct sources must validate current Stage sidecars/frozen annotations in
    # materialize_reward_manifest, which cheaply reuses matching reward arrays.
    if reward_source(reward_config) != "rynnvalue":
        return False
    prepared_path = work_dir / MANIFEST_NAME
    annotation_path = work_dir / "annotations" / "annotation_manifest.json"
    reward_path = work_dir / "rewards" / "reward_manifest.json"
    try:
        prepared = json.loads(prepared_path.read_text(encoding="utf-8"))
        annotations = json.loads(annotation_path.read_text(encoding="utf-8"))
        rewards = json.loads(reward_path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return False
    if (
        rewards.get("schema_version") != REWARD_SCHEMA_VERSION
        or rewards.get("kind") != "derived_iql_reward"
        or rewards.get("complete") is not True
        or rewards.get("dataset_sha256") != prepared.get("dataset_sha256")
        or rewards.get("annotation_manifest_sha256") != stable_hash(annotations)
        or rewards.get("reward_config") != reward_derivation_config(reward_config)
        or rewards.get("derivation_implementation_sha256") != reward_implementation_fingerprint("rynnvalue")
    ):
        return False
    for episode in rewards.get("episodes", []):
        path = Path(str(episode.get("reward_path") or ""))
        if (
            path.is_symlink()
            or not path.is_file()
            or episode.get("reward_sha256") != sha256_file(path)
        ):
            return False
    return len(rewards.get("episodes", [])) == len(prepared.get("episodes", []))


def bound_evaluation_count(selection_manifest: dict[str, Any]) -> int:
    count = 0
    for member in selection_manifest["members"]:
        episode = {
            "run_id": member["run_id"],
            "trajectory_path": member["artifacts"]["trajectory"]["path"],
            "trajectory_sha256": member["artifacts"]["trajectory"]["sha256"],
            "observations_sha256": member["artifacts"]["observations"]["sha256"],
            "recorded_action_count": member["end_step"],
        }
        if valid_bound_evaluation(episode) is not None:
            count += 1
    return count


def validate_effective_config(raw: dict[str, Any]) -> None:
    """Run the ordinary strict train-config validation without persistent output."""
    descriptor, name = tempfile.mkstemp(prefix="terminal-effective-", suffix=".yaml")
    os.close(descriptor)
    path = Path(name)
    try:
        path.write_text(yaml.safe_dump(raw, sort_keys=False), encoding="utf-8")
        load_train_config(path)
    finally:
        path.unlink(missing_ok=True)
