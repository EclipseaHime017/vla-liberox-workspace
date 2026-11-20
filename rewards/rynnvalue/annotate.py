"""Run RynnValue over a prepared dataset, reusing compatible saved outputs.

Like the official inference program, every observation step of a recording is evaluated
and each prediction conditions on ``max_frames`` frames resampled uniformly in time from
the recording's prefix up to that step.
"""
from __future__ import annotations

import json
import logging
import os
import uuid
from pathlib import Path
from typing import Any

import numpy as np

from core.config import LoadedConfig
from core.data import load_manifest
from core.io import atomic_json, sha256_file, stable_hash
from core.storage_paths import storage_path, with_dataset_lease
from rewards.rynnvalue.model import reward_view, RynnValueAnnotator, TemporalValueAnnotator
from rewards.rynnvalue.outputs import (
    ANNOTATION_SCHEMA_VERSION,
    official_inference_config,
    reusable_manifest_entry,
    reusable_official_sidecar,
    validate_official_outputs,
)


LOG = logging.getLogger(__name__)


@with_dataset_lease


def annotate_manifest(
    config: LoadedConfig,
    annotator: TemporalValueAnnotator | None = None,
    *,
    overwrite: bool = False,
    reuse_only: bool = False,
) -> Path:
    manifest = load_manifest(config)
    reward_cfg = config.section("reward")
    annotation_cfg = official_inference_config(reward_cfg)
    active_annotator = annotator
    manifest_annotator_metadata: dict[str, Any] | None = None

    def live_annotator() -> TemporalValueAnnotator:
        nonlocal active_annotator
        if active_annotator is None:
            if reuse_only:
                raise ValueError("Required RynnValue evaluation is missing or incompatible; "
                                 "evaluate RynnValue or use All before generating Final Reward")
            active_annotator = RynnValueAnnotator(config)
        return active_annotator

    work_dir = Path(config.section("paths")["work_dir"])
    annotation_dir = work_dir / "annotations"
    annotation_dir.mkdir(parents=True, exist_ok=True)
    previous_manifests = []
    for previous_manifest_path in (
        annotation_dir / "annotation_manifest.json",
        work_dir / "rewards" / "reward_manifest.json",
    ):
        try:
            candidate = json.loads(previous_manifest_path.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError):
            continue
        if isinstance(candidate, dict):
            previous_manifests.append(candidate)
    cache_dir = Path(config.section("paths")["annotation_cache"])
    cache_dir.mkdir(parents=True, exist_ok=True)
    index: list[dict[str, Any]] = []
    for episode in manifest["episodes"]:
        reusable = None
        if annotator is None and not overwrite:
            for candidate in previous_manifests:
                reusable = reusable_manifest_entry(episode, reward_cfg, candidate)
                if reusable is not None:
                    break
            reusable = reusable or reusable_official_sidecar(episode, reward_cfg)
        episode_annotator_metadata = (
            reusable[2] if reusable is not None else live_annotator().metadata
        )
        if manifest_annotator_metadata is None:
            manifest_annotator_metadata = episode_annotator_metadata
        elif stable_hash(manifest_annotator_metadata) != stable_hash(episode_annotator_metadata):
            # Every reusable episode has already passed the requested model /
            # revision / inference-contract checks.  Do not discard valuable
            # evaluations merely because provenance such as snapshot paths or
            # package metadata differs across machines.  Per-episode metadata
            # remains authoritative when the dataset contains mixed provenance.
            manifest_annotator_metadata = {}
        source_key = stable_hash({
            "annotation_schema_version": ANNOTATION_SCHEMA_VERSION,
            "run": episode["run_id"],
            "trajectory": episode["trajectory_sha256"],
            "observations": episode["observations_sha256"],
            "prompt": episode["prompt"],
            "annotation_config": annotation_cfg,
            "model": episode_annotator_metadata,
        })
        output = cache_dir / f"{source_key}.npz"
        meta_path = cache_dir / f"{source_key}.json"
        if not overwrite and meta_path.is_file():
            try:
                current = json.loads(meta_path.read_text(encoding="utf-8"))
            except (OSError, json.JSONDecodeError):
                current = None
            cached_output = storage_path(str(current.get("annotation_path") or output)) if isinstance(current, dict) else output
            if (
                isinstance(current, dict)
                and current.get("source_key") == source_key
                and not cached_output.is_symlink() and cached_output.is_file()
                and current.get("annotation_sha256") == sha256_file(cached_output)
            ):
                index.append(current)
                continue
        if reusable is not None:
            official_outputs, analysis, _ = reusable
            LOG.info("Reusing official RynnValue outputs for %s", episode["run_id"])
        else:
            current_annotator = live_annotator()
            with np.load(storage_path(episode["observations_path"]), allow_pickle=False) as images:
                raw = images["agentview_image"]
                frames = [reward_view(frame, episode["observation_orientation"]) for frame in raw]
            if len(frames) != int(episode["recorded_action_count"]) + 1:
                raise ValueError(f"Observations do not cover every step of {episode['run_id']}")
            official_outputs = validate_official_outputs(
                current_annotator.predict(episode["prompt"], frames), len(frames)
            )
            analysis = current_annotator.analyze(episode["prompt"], frames)
        # Keep previously pinned official outputs intact on forced evaluation.
        output = cache_dir / f"{source_key}-{uuid.uuid4().hex}.npz"
        temporary = cache_dir / f".{output.name}.{os.getpid()}.npz"
        try:
            np.savez_compressed(temporary, **official_outputs)
            os.replace(temporary, output)
        finally:
            temporary.unlink(missing_ok=True)
        metadata = {
            "schema_version": ANNOTATION_SCHEMA_VERSION,
            "run_id": episode["run_id"], "source_key": source_key,
            "annotation_path": str(output.resolve()), "annotation_sha256": sha256_file(output),
            "environment_success": episode["success"],
            "annotator": episode_annotator_metadata,
            "official_outputs": {
                "inference_method": "prefix_uniform_last_slot",
                "evaluated_steps": "every_observation",
                "prefix_image_slots": int(reward_cfg["max_frames"]),
                "absolute_slot": "last_value_slot_of_each_uniformly_resampled_prefix",
                "relative_slot": "between_last_two_slots_of_each_uniformly_resampled_prefix",
                "observation_count": len(official_outputs["absolute_temporal_distance_seconds"]),
                "array_keys": sorted(official_outputs),
                "analysis": analysis,
            },
        }
        atomic_json(meta_path, metadata)
        index.append(metadata)
        LOG.info("Annotated %s (%d observations)", episode["run_id"],
                 len(official_outputs["absolute_temporal_distance_seconds"]))
        atomic_json(annotation_dir / "annotation_manifest.json", {
            "schema_version": ANNOTATION_SCHEMA_VERSION,
            "kind": "rynnvalue_annotation",
            "dataset_sha256": manifest["dataset_sha256"],
            "annotation_config": annotation_cfg,
            "annotator": manifest_annotator_metadata or {},
            "complete": False, "episodes": index,
        })
    index_path = annotation_dir / "annotation_manifest.json"
    atomic_json(index_path, {
        "schema_version": ANNOTATION_SCHEMA_VERSION,
        "kind": "rynnvalue_annotation",
        "dataset_sha256": manifest["dataset_sha256"],
        "annotation_config": annotation_cfg,
        "annotator": manifest_annotator_metadata or {},
        "complete": True, "episodes": index,
    })
    return index_path


def load_annotation_index(config: LoadedConfig) -> dict[str, Any]:
    """Load official RynnValue outputs, reusing compatible saved ones without a new forward."""
    manifest = load_manifest(config)
    path = (
        Path(config.section("paths")["work_dir"])
        / "annotations" / "annotation_manifest.json"
    )
    if not path.is_file():
        LOG.info("No separated annotation manifest; migrating reusable RynnValue outputs")
        path = annotate_manifest(config)
    payload = json.loads(path.read_text(encoding="utf-8"))
    expected_config = official_inference_config(config.section("reward"))
    if (
        payload.get("schema_version") != ANNOTATION_SCHEMA_VERSION
        or payload.get("kind") != "rynnvalue_annotation"
        or payload.get("complete") is not True
        or payload.get("dataset_sha256") != manifest["dataset_sha256"]
        or payload.get("annotation_config") != expected_config
    ):
        raise ValueError(
            "RynnValue annotation cache does not match the prepared data or inference "
            "configuration; run `python -m rewards rynnvalue-annotate`. Reward-only settings do not require "
            "another model evaluation."
        )
    if len(payload.get("episodes", [])) != len(manifest.get("episodes", [])):
        raise ValueError("RynnValue annotation manifest does not cover every episode")
    for episode in payload["episodes"]:
        annotation_path = storage_path(str(episode.get("annotation_path") or ""))
        if (
            annotation_path.is_symlink()
            or not annotation_path.is_file()
            or episode.get("annotation_sha256") != sha256_file(annotation_path)
        ):
            raise ValueError(
                f"RynnValue annotation cache is missing or corrupted for {episode.get('run_id')}"
            )
    return payload
