"""Continuation roles and immutable action provenance for saved trajectories."""
from __future__ import annotations

import json
from pathlib import Path

from vla_rynn_iql.execution import validate_task_prompt

from ..storage.files import atomic_write_json


def continuation_public(run: dict) -> dict:
    promoted = False
    path = Path(run["output_dir"]) / "continuation.json"
    if path.is_file():
        value = json.loads(path.read_text(encoding="utf-8"))
        if value.get("schema_version") != 1 or value.get("run_id") != run["id"]:
            raise ValueError(f"Invalid continuation metadata: {path}")
        promoted = value.get("role") == "original"
    complete = run.get("status") == "COMPLETED" and not run.get("error")
    valid = complete and int(run.get("action_count") or 0) > 0 and bool(run.get("trajectory"))
    role = "original" if promoted or run.get("kind") == "original" else "branch"
    return {**run, "role": role, "can_promote": valid and role == "branch",
            "branchable": valid and role == "original"}


def promote_run(run: dict) -> dict:
    current = continuation_public(run)
    if current["role"] == "original":
        return current
    if not current["can_promote"] or not Path(run["trajectory"]).is_file():
        raise ValueError("Only a completed branch with a saved trajectory can become a mainline")
    atomic_write_json(Path(run["output_dir"]) / "continuation.json",
                      {"schema_version": 1, "run_id": run["id"], "role": "original"})
    return continuation_public(run)


def trajectory_segments(metadata: dict, count: int) -> list[dict]:
    """Partition actions by their original producer without changing saved recordings."""
    validate_task_prompt(metadata, metadata.get("task"), context="Trajectory")
    segments = metadata.get("trajectory_segments")
    if segments is None:
        run_id = str(metadata.get("session_id") or metadata.get("id") or "")
        resume = int(metadata.get("source_resume_step") or metadata.get("resume_step") or 0)
        root = str(metadata.get("root_session_id") or metadata.get("parent_session_id") or run_id)
        segments = []
        if resume:
            segments.append(dict(start=0, end=min(resume, count), origin_run_id=root))
        if resume < count:
            segments.append(dict(start=resume, end=count, origin_run_id=run_id))
    cursor = 0
    for segment in segments:
        if (type(segment.get("start")) is not int or type(segment.get("end")) is not int
                or segment["start"] != cursor or not cursor < segment["end"] <= count
                or not isinstance(segment.get("origin_run_id"), str) or not segment["origin_run_id"]):
            raise ValueError("Invalid trajectory origin segments")
        cursor = segment["end"]
    if cursor != count:
        raise ValueError("Trajectory segments do not cover all actions")
    result = [{key: segment[key] for key in ("start", "end", "origin_run_id", "open_loop_steps")
               if key in segment} for segment in segments]
    if metadata.get("open_loop_steps") is not None:
        for segment in result:
            if segment["origin_run_id"] == str(metadata.get("session_id") or metadata.get("id") or ""):
                segment.setdefault("open_loop_steps", metadata["open_loop_steps"])
    return result


def prefix_segments(metadata: dict, count: int, end: int) -> list[dict]:
    return [{**segment, "end": min(end, segment["end"])}
            for segment in trajectory_segments(metadata, count) if segment["start"] < end]
