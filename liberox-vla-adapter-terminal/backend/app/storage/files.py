"""Atomic persistence and read-only indexing of UI and legacy trajectories."""

from __future__ import annotations

import csv
import hashlib
import json
import os
import tempfile
from pathlib import Path
from typing import Any, Iterable

from trajectory_utils import load_trajectory
import yaml


def atomic_write_csv(
    path: Path,
    fieldnames: list[str],
    rows: Iterable[dict[str, Any]],
) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    descriptor, temporary_name = tempfile.mkstemp(
        prefix=f".{path.name}.",
        suffix=".tmp",
        dir=path.parent,
    )
    try:
        with os.fdopen(descriptor, "w", encoding="utf-8", newline="") as handle:
            writer = csv.DictWriter(handle, fieldnames=fieldnames)
            writer.writeheader()
            writer.writerows(rows)
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temporary_name, path)
    except BaseException:
        Path(temporary_name).unlink(missing_ok=True)
        raise


def atomic_write_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    descriptor, temporary_name = tempfile.mkstemp(
        prefix=f".{path.name}.",
        suffix=".tmp",
        dir=path.parent,
    )
    try:
        with os.fdopen(descriptor, "w", encoding="utf-8") as handle:
            json.dump(value, handle, ensure_ascii=False, indent=2)
            handle.write("\n")
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temporary_name, path)
    except BaseException:
        Path(temporary_name).unlink(missing_ok=True)
        raise


def atomic_write_bytes(path: Path, value: bytes) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    descriptor, temporary_name = tempfile.mkstemp(
        prefix=f".{path.name}.", suffix=".tmp", dir=path.parent,
    )
    try:
        with os.fdopen(descriptor, "wb") as handle:
            handle.write(value)
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temporary_name, path)
    except BaseException:
        Path(temporary_name).unlink(missing_ok=True)
        raise


def atomic_write_yaml(path: Path, value: Any) -> None:
    payload = yaml.safe_dump(value, allow_unicode=True, sort_keys=False).encode("utf-8")
    atomic_write_bytes(path, payload)


def safe_artifacts(directory: Path) -> dict[str, str]:
    if not directory.is_dir():
        return {}
    artifacts: dict[str, str] = {}
    for path in sorted(directory.rglob("*")):
        if path.is_file() and not path.name.startswith("."):
            name = path.relative_to(directory).as_posix()
            artifacts[name] = str(path.resolve())
    return artifacts


def legacy_session_id(path: Path) -> str:
    digest = hashlib.sha1(str(path.resolve()).encode("utf-8")).hexdigest()[:16]
    return f"legacy-{digest}"


def scan_trajectories(roots: Iterable[Path]) -> list[dict[str, Any]]:
    """Return valid trajectories without changing legacy directories."""
    found: dict[Path, dict[str, Any]] = {}
    for root in roots:
        if not root.is_dir():
            continue
        for path in root.rglob("*.npz"):
            if path.name.endswith("_observations.npz") or path.name == "source_trajectory.npz":
                continue
            resolved = path.resolve()
            if resolved in found:
                continue
            try:
                trajectory, metadata = load_trajectory(resolved)
            except Exception:
                continue
            parent_id = metadata.get("parent_session_id")
            source = metadata.get("source_trajectory")
            is_branch = bool(parent_id or source)
            session_id = str(metadata.get("session_id") or legacy_session_id(resolved))
            run_directory = resolved.parent
            for candidate in resolved.parents:
                if (candidate / "run.json").is_file() or (candidate / "session.json").is_file():
                    run_directory = candidate
                    break
            found[resolved] = {
                "id": session_id,
                "kind": "branch" if is_branch else "original",
                "task_id": metadata.get("task_id"),
                "parent_session_id": parent_id,
                "root_session_id": metadata.get("root_session_id"),
                "source_trajectory": source,
                "resume_step": metadata.get("source_resume_step"),
                "control_mode": metadata.get("control_mode", "policy"),
                "manual_source": metadata.get("manual_source"),
                "manual_translation_gain": metadata.get("manual_translation_gain"),
                "manual_rotation_gain": metadata.get("manual_rotation_gain"),
                "spacemouse_connected": None,
                "spacemouse_stale": None,
                "spacemouse_latency_ms": None,
                "spacemouse_status": None,
                "spacemouse_deadman_ms": metadata.get("spacemouse_deadman_ms"),
                "status": "ERROR" if metadata.get("error") else "COMPLETED",
                "created_at": metadata.get("created_at"),
                "completed_at": metadata.get("completed_at"),
                "task": metadata.get("task"),
                "task_name": metadata.get("task_name"),
                "level": metadata.get("level"),
                "checkpoint": metadata.get("checkpoint"),
                "max_steps": int(metadata.get("target_total_steps", len(trajectory["env_action"]))),
                "open_loop_steps": metadata.get("open_loop_steps"),
                "current_step": len(trajectory["env_action"]),
                "state_count": len(trajectory["sim_state"]),
                "action_count": len(trajectory["env_action"]),
                "policy_queries": int(metadata.get("policy_queries", len(trajectory.get("inference_query_step", [])))),
                "success": bool(metadata.get("success", bool(trajectory["done"].any()))),
                "error": metadata.get("error"),
                "stopped_reason": metadata.get("stopped_reason", "legacy"),
                "measured_control_hz": metadata.get("measured_control_hz"),
                "simulated_duration_seconds": metadata.get(
                    "simulated_duration_seconds",
                    len(trajectory["env_action"]) / float(metadata.get("control_hz", 20)),
                ),
                "output_dir": str(run_directory),
                "trajectory": str(resolved),
                "artifacts": safe_artifacts(run_directory),
                "branchable": not is_branch,
                "legacy": True,
                "managed": False,
            }
    return sorted(
        found.values(),
        key=lambda item: (item.get("created_at") or "", item["id"]),
        reverse=True,
    )
