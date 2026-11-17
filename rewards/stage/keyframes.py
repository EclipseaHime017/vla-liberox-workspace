"""Keyframe (Stage) annotations: schema, construction, validation and loading."""
from __future__ import annotations

import hashlib
import json
import math
from pathlib import Path
from typing import Any, Sequence

import numpy as np

from core.config import LoadedConfig
from core.io import sha256_file
from core.storage_paths import storage_path


STAGE_SCHEMA_VERSION = 2


STAGE_FILENAME = "stage_annotation.json"


def _digest(value: Any) -> str:
    return hashlib.sha256(json.dumps(value, sort_keys=True, separators=(",", ":"),
                                     allow_nan=False).encode()).hexdigest()


def checked_exponent(value: float) -> float:
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
    if "success_step" not in annotation:
        raise ValueError("Stage scores require an evaluation context, not bare keyframe labels")
    success = annotation["success_step"]
    if success is not None and any(frame["step"] >= success for frame in frames):
        raise ValueError("Manual keyframes must precede the automatic success anchor for this dataset")
    denominator = positives - negatives + 1 if success is not None else positives + 1
    if denominator <= 0:
        raise ValueError("Stage normalization requires a positive denominator")
    weight = 1.0 / denominator
    score = -1.0
    anchors = [{"step": 0, "kind": "start", "score": score}]
    net_stages = 0
    for frame in frames:
        net_stages += 1 if frame["kind"] == "positive" else -1
        # The same normalization without accumulated rounding at a zero anchor.
        score = (net_stages - denominator) / denominator
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
    # Retain these arguments for old callers, but never persist recipe settings
    # in the primary human-label identity.
    confirmed_success_step(done, success_consecutive_steps)
    checked_exponent(exponent)
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
        if step in seen:
            raise ValueError(f"Duplicate keyframe at observation {step}")
        if kind not in ("positive", "negative"):
            raise ValueError("Keyframe kind must be positive or negative")
        seen.add(step)
        ordered.append({"step": step, "kind": kind})
    payload = {
        "schema_version": STAGE_SCHEMA_VERSION, "run_id": run_id,
        "trajectory_sha256": trajectory_sha256, "action_count": len(done),
        "keyframes": sorted(ordered, key=lambda item: item["step"]),
    }
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
    )
    if payload.get("schema_version") == 1:
        # Check the original v1 byte-independent canonical identity, not the
        # current reward formula. Legacy labels remain usable with a new p or
        # dataset success threshold and do not need to be rewritten.
        expected.pop("annotation_sha256")
        expected.update(
            schema_version=1,
            success_consecutive_steps=payload.get("success_consecutive_steps"),
            success_step=confirmed_success_step(done, payload.get("success_consecutive_steps")),
            exponent=checked_exponent(payload.get("exponent")),
        )
        expected["annotation_sha256"] = _digest(expected)
    if _digest(payload) != _digest(expected):
        raise ValueError("Stage annotation is stale or corrupted")
    return expected


def load_stage_annotations(config: LoadedConfig, manifest: dict[str, Any]) -> dict[str, Any]:
    """Validate every selected member, including validation and deduplicated prefixes."""
    frozen_path = config.section("data").get("stage_annotations_manifest")
    frozen = None
    if frozen_path is not None:
        frozen = json.loads(Path(frozen_path).read_text(encoding="utf-8"))
        if (not isinstance(frozen, dict) or set(frozen) != {"schema_version", "annotations"}
                or frozen["schema_version"] != 1 or not isinstance(frozen["annotations"], dict)):
            raise ValueError("Invalid frozen stage annotations manifest")
    selected: dict[str, Any] = {}
    invalid: list[str] = []
    for episode in manifest["episodes"]:
        run_id = str(episode["run_id"])
        trajectory = storage_path(episode["trajectory_path"])
        try:
            if frozen is None and episode.get("observation_orientation") == "vla_policy":
                raise ValueError(
                    "Stage cannot reuse labels for a CSV/video-reconstructed trajectory; "
                    "provide the original trajectory.npz, observations and bound stage_annotation.json. "
                    "Annotation hashes are never silently rebound."
                )
            payload = (frozen["annotations"][run_id] if frozen is not None else
                       json.loads((trajectory.parent / STAGE_FILENAME).read_text(encoding="utf-8")))
            actual_sha = sha256_file(trajectory)
            if actual_sha != episode["trajectory_sha256"]:
                raise ValueError("Trajectory changed after Prepare; run Prepare again")
            with np.load(trajectory, allow_pickle=False) as arrays:
                done = np.asarray(arrays["done"])
            selected[run_id] = validate_stage_annotation(
                payload, run_id=run_id, trajectory_sha256=actual_sha, done=done,
                success_consecutive_steps=int(config.section("data")["success_consecutive_steps"]),
            )
        except (OSError, KeyError, TypeError, ValueError) as exc:
            invalid.append(f"{run_id}: {exc}")
    if invalid:
        raise ValueError("Stage annotations missing or invalid; training stopped:\n" + "\n".join(invalid))
    return {"schema_version": 1, "annotations": selected}
