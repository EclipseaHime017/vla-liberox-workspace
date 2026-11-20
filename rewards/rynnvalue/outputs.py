"""Saved official RynnValue outputs: inference contract, validation and safe reuse."""
from __future__ import annotations

import json
from typing import Any

import numpy as np

from core.io import sha256_file, stable_hash
from core.storage_paths import storage_path


# Outputs have one row per observation step, each conditioned on a time-uniform prefix
# of all frames up to that step (the official inference program).
ANNOTATION_SCHEMA_VERSION = 7


OFFICIAL_INFERENCE_CONFIG_KEYS = frozenset({
    "model", "revision", "dtype", "max_frames",
    "robot_description", "camera_description",
})


OFFICIAL_OUTPUT_KEYS = (
    "absolute_temporal_distance_seconds",
    "absolute_value_entropy_nats",
    "absolute_value_logits",
    "relative_temporal_distance_seconds",
    "relative_value_logits",
)


def official_inference_config(reward_config: dict[str, Any]) -> dict[str, Any]:
    """Return only settings that can change frozen RynnValue outputs."""
    return {
        key: reward_config[key]
        for key in sorted(OFFICIAL_INFERENCE_CONFIG_KEYS)
    }


def validate_official_outputs(
    outputs: dict[str, np.ndarray], observation_count: int
) -> dict[str, np.ndarray]:
    """Official RynnValue outputs with one row per observation step of the recording."""
    required = {
        "absolute_temporal_distance_seconds": 2,
        "absolute_value_entropy_nats": 2,
        "absolute_value_logits": 3,
        "relative_temporal_distance_seconds": 1,
        "relative_value_logits": 2,
    }
    if not isinstance(outputs, dict):
        raise TypeError("RynnValue annotator must return the complete official output mapping")
    if set(outputs) != set(required):
        raise ValueError(
            "RynnValue official output keys do not match the recorded contract: "
            f"{sorted(outputs)}"
        )
    normalized: dict[str, np.ndarray] = {}
    for name, dimensions in required.items():
        value = np.asarray(outputs[name], dtype=np.float32)
        if value.ndim != dimensions or value.shape[0] != observation_count:
            raise ValueError(
                f"RynnValue {name} shape {value.shape} does not match "
                f"{observation_count} observations and {dimensions} dimensions"
            )
        if not np.isfinite(value).all():
            raise ValueError(f"RynnValue {name} contains NaN or Inf")
        normalized[name] = value
    if normalized["absolute_temporal_distance_seconds"].shape != normalized[
        "absolute_value_entropy_nats"
    ].shape:
        raise ValueError("RynnValue absolute distance/entropy head shapes do not match")
    if normalized["absolute_value_logits"].shape[:2] != normalized[
        "absolute_temporal_distance_seconds"
    ].shape:
        raise ValueError("RynnValue absolute logits do not match decoded head outputs")
    return normalized


def _official_inference_config_matches(
    previous: dict[str, Any], requested: dict[str, Any],
) -> bool:
    return all(previous.get(key) == requested.get(key) for key in OFFICIAL_INFERENCE_CONFIG_KEYS)


def reusable_manifest_entry(
    episode: dict[str, Any], reward_cfg: dict[str, Any],
    annotation_manifest: dict[str, Any] | None,
) -> tuple[dict[str, np.ndarray], dict[str, Any], dict[str, Any]] | None:
    """Reuse official heads from a previous annotation manifest in this work dir."""
    if not isinstance(annotation_manifest, dict):
        return None
    previous_cfg = annotation_manifest.get("annotation_config")
    if not isinstance(previous_cfg, dict):
        previous_cfg = annotation_manifest.get("reward_config")
    if not isinstance(previous_cfg, dict) or not _official_inference_config_matches(
        previous_cfg, reward_cfg,
    ):
        return None
    entry = next(
        (item for item in annotation_manifest.get("episodes", [])
         if item.get("run_id") == episode["run_id"]),
        None,
    )
    if not isinstance(entry, dict):
        return None
    path = storage_path(str(entry.get("annotation_path") or ""))
    if (
        path.is_symlink() or not path.is_file()
        or entry.get("annotation_sha256") != sha256_file(path)
    ):
        return None
    official_metadata = entry.get("official_outputs")
    annotator_metadata = entry.get("annotator") or annotation_manifest.get("annotator")
    if not isinstance(official_metadata, dict) or not isinstance(annotator_metadata, dict):
        return None
    expected_key = stable_hash({
        "annotation_schema_version": ANNOTATION_SCHEMA_VERSION,
        "run": episode["run_id"], "trajectory": episode["trajectory_sha256"],
        "observations": episode["observations_sha256"], "prompt": episode["prompt"],
        "annotation_config": official_inference_config(reward_cfg), "model": annotator_metadata,
    })
    if entry.get("source_key") != expected_key:
        # Shared versions may have equal run IDs but different source files.
        return None
    if official_metadata.get("prefix_image_slots") != int(reward_cfg["max_frames"]):
        return None
    try:
        with np.load(path, allow_pickle=False) as arrays:
            official_outputs = validate_official_outputs(
                {name: arrays[name] for name in OFFICIAL_OUTPUT_KEYS},
                int(episode["recorded_action_count"]) + 1,
            )
    except (KeyError, OSError, ValueError):
        return None
    analysis = official_metadata.get("analysis")
    if not isinstance(analysis, dict):
        return None
    return official_outputs, analysis, annotator_metadata


def reusable_official_sidecar(
    episode: dict[str, Any], reward_cfg: dict[str, Any]
) -> tuple[dict[str, np.ndarray], dict[str, Any], dict[str, Any]] | None:
    """Load hash-checked official outputs bound beside the trajectory without a forward."""
    trajectory = storage_path(episode["trajectory_path"])
    sidecar = trajectory.parent / "rynnvalue_evaluation.json"
    values = trajectory.parent / "rynnvalue_evaluation.npz"
    if not sidecar.is_file() or sidecar.is_symlink() or not values.is_file() or values.is_symlink():
        return None
    try:
        payload = json.loads(sidecar.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return None
    if (
        payload.get("schema_version") != ANNOTATION_SCHEMA_VERSION
        or payload.get("run_id") != episode["run_id"]
        or payload.get("trajectory_sha256") != episode["trajectory_sha256"]
        or payload.get("observations_sha256") != episode["observations_sha256"]
        or payload.get("values_sha256") != sha256_file(values)
    ):
        return None
    saved = payload.get("episode") or {}
    saved_prompt = payload.get("prompt", saved.get("prompt"))
    saved_manifest = payload.get("source_manifest_sha256", saved.get("source_manifest_sha256"))
    if saved_prompt is None and saved_manifest is None:
        return None
    if saved_prompt is not None and saved_prompt != episode.get("prompt"):
        return None
    if saved_manifest is not None and saved_manifest != episode.get("source_manifest_sha256"):
        return None
    annotator_metadata = payload.get("annotator")
    official_metadata = payload.get("official_outputs")
    if not isinstance(annotator_metadata, dict) or not isinstance(official_metadata, dict):
        return None
    requested_revision = annotator_metadata.get("requested_revision")
    resolved_revision = annotator_metadata.get("resolved_revision")
    previous_inference_cfg = payload.get("annotation_config")
    if not isinstance(previous_inference_cfg, dict):
        previous_inference_cfg = payload.get("reward_config")
    if not isinstance(previous_inference_cfg, dict):
        return None
    if (
        annotator_metadata.get("model") != reward_cfg.get("model")
        or reward_cfg.get("revision") not in {requested_revision, resolved_revision}
        or official_metadata.get("prefix_image_slots") != int(reward_cfg["max_frames"])
        or previous_inference_cfg.get("robot_description") != reward_cfg.get("robot_description")
        or previous_inference_cfg.get("camera_description") != reward_cfg.get("camera_description")
    ):
        return None
    try:
        with np.load(values, allow_pickle=False) as arrays:
            official_outputs = validate_official_outputs(
                {name: arrays[name] for name in OFFICIAL_OUTPUT_KEYS},
                int(episode["recorded_action_count"]) + 1,
            )
    except (KeyError, OSError, ValueError):
        return None
    analysis = official_metadata.get("analysis")
    if not isinstance(analysis, dict):
        return None
    return official_outputs, analysis, annotator_metadata
