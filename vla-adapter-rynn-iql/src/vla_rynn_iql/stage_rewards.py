"""Human keyframe scores. Pure data/math: no model, simulator or UI imports."""
from __future__ import annotations

import hashlib
import json
import math
from typing import Any, Sequence

import numpy as np

STAGE_SCHEMA_VERSION = 1
STAGE_FILENAME = "stage_annotation.json"


def _digest(value: Any) -> str:
    return hashlib.sha256(json.dumps(value, sort_keys=True, separators=(",", ":"),
                                     allow_nan=False).encode()).hexdigest()


def _exponent(value: float) -> float:
    if isinstance(value, bool) or not isinstance(value, (float, int)):
        raise TypeError("Stage exponent must be a finite number >= 1")
    if not math.isfinite(value) or value < 1:
        raise ValueError("Stage exponent must be a finite number >= 1")
    return float(value)


def confirmed_success_step(done: Sequence[bool], consecutive_steps: int) -> int | None:
    """Observation index AFTER the action completing the first success streak."""
    if type(consecutive_steps) is not int or consecutive_steps < 1:
        raise ValueError("success_consecutive_steps must be a positive integer")
    streak = 0
    for index, value in enumerate(done):
        if not isinstance(value, (bool, np.bool_)):
            raise TypeError("done must contain booleans")
        streak = streak + 1 if value else 0
        if streak >= consecutive_steps:
            return index + 1
    return None


def stage_anchors(annotation: dict[str, Any]) -> list[dict[str, Any]]:
    frames = annotation["keyframes"]
    positives = sum(frame["kind"] == "positive" for frame in frames)
    negatives = len(frames) - positives
    success = annotation["success_step"]
    denominator = positives - negatives + 1 if success is not None else positives + 1
    if denominator <= 0:
        raise ValueError("Stage normalization requires a positive denominator")
    weight = 1.0 / denominator
    score = -1.0
    anchors = [{"step": 0, "kind": "start", "score": score}]
    for frame in frames:
        score += weight if frame["kind"] == "positive" else -weight
        anchors.append({**frame, "score": score})
    if success is not None:
        if not math.isclose(score + weight, 0.0, abs_tol=1e-9):
            raise ValueError("Stage success normalization does not end at zero")
        anchors.append({"step": success, "kind": "success", "score": 0.0})
    return anchors


def build_stage_annotation(*, run_id: str, trajectory_sha256: str,
                           done: Sequence[bool], keyframes: list[dict[str, Any]],
                           success_consecutive_steps: int = 5,
                           exponent: float = 2.0) -> dict[str, Any]:
    if not isinstance(run_id, str) or not run_id:
        raise ValueError("Stage run_id must be nonempty")
    if (not isinstance(trajectory_sha256, str) or len(trajectory_sha256) != 64
            or any(c not in "0123456789abcdef" for c in trajectory_sha256)):
        raise ValueError("Stage trajectory_sha256 must be a SHA256 digest")
    if len(done) < 1:
        raise ValueError("Stage annotation requires at least one executed action")
    success_step = confirmed_success_step(done, success_consecutive_steps)
    if not isinstance(keyframes, list):
        raise TypeError("Stage keyframes must be a list")
    ordered = []
    seen = set()
    for frame in keyframes:
        if not isinstance(frame, dict) or set(frame) != {"step", "kind"}:
            raise ValueError("Each keyframe must contain only step and kind")
        step, kind = frame["step"], frame["kind"]
        if type(step) is not int or not 0 < step <= len(done):
            raise ValueError("Keyframes must reference observation steps in (0, action_count]")
        if success_step is not None and step >= success_step:
            raise ValueError("Manual keyframes must precede the automatic success anchor")
        if step in seen:
            raise ValueError(f"Duplicate keyframe at observation {step}")
        if kind not in ("positive", "negative"):
            raise ValueError("Keyframe kind must be positive or negative")
        seen.add(step)
        ordered.append({"step": step, "kind": kind})
    payload = {
        "schema_version": STAGE_SCHEMA_VERSION, "run_id": run_id,
        "trajectory_sha256": trajectory_sha256, "action_count": len(done),
        "success_consecutive_steps": success_consecutive_steps,
        "success_step": success_step,
        "keyframes": sorted(ordered, key=lambda item: item["step"]),
        "exponent": _exponent(exponent),
    }
    stage_anchors(payload)
    payload["annotation_sha256"] = _digest(payload)
    return payload


def validate_stage_annotation(payload: dict[str, Any], *, run_id: str,
                              trajectory_sha256: str, done: Sequence[bool],
                              success_consecutive_steps: int) -> dict[str, Any]:
    if not isinstance(payload, dict):
        raise ValueError("Stage annotation must be an object")
    expected = build_stage_annotation(
        run_id=run_id, trajectory_sha256=trajectory_sha256, done=done,
        keyframes=payload.get("keyframes"),
        success_consecutive_steps=success_consecutive_steps,
        exponent=payload.get("exponent"),
    )
    if _digest(payload) != _digest(expected):
        raise ValueError("Stage annotation is stale, corrupted or uses a different success threshold")
    return expected


def stage_scores(annotation: dict[str, Any], exponent: float | None = None) -> np.ndarray:
    power = _exponent(annotation["exponent"] if exponent is None else exponent)
    anchors = stage_anchors(annotation)
    scores = np.full(int(annotation["action_count"]) + 1, anchors[-1]["score"], dtype=np.float64)
    scores[0] = -1.0
    for left, right in zip(anchors, anchors[1:]):
        start, end = left["step"], right["step"]
        x = np.arange(end - start + 1, dtype=np.float64) / (end - start)
        scores[start:end + 1] = left["score"] + (right["score"] - left["score"]) * x ** power
    return scores.astype(np.float32)


def stage_chunk_reward(scores: np.ndarray, start: int, length: int,
                       gamma: float, accumulate_primitive_steps: bool) -> float:
    if length < 1 or start < 0 or start + length >= len(scores):
        raise ValueError("Invalid stage reward chunk bounds")
    if accumulate_primitive_steps:
        return float(np.dot(np.power(gamma, np.arange(length)), scores[start + 1:start + length + 1]))
    return float(scores[start + length])
