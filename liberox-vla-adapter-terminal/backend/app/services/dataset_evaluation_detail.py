"""Read pinned dataset plots, without evaluating models or recomputing rewards."""
from __future__ import annotations

import hashlib
import json
from functools import lru_cache
from pathlib import Path
from typing import Any

import numpy as np

from .trajectory_reward_snapshot import SIDECAR, needs_snapshot_validation, read_reward_snapshot


def _signature(path: Path) -> tuple[int, ...]:
    stat = path.stat()
    return stat.st_ino, stat.st_size, stat.st_mtime_ns, stat.st_ctime_ns


@lru_cache(maxsize=64)
def _read_checked(path_name: str, digest: str, signature: tuple[int, ...], arrays: bool):
    path = Path(path_name)
    if path.is_symlink() or not path.is_file():
        raise ValueError("Evaluation artifact is missing or is a symlink")
    actual = hashlib.sha256(path.read_bytes()).hexdigest()
    if actual != digest:
        raise ValueError(f"Evaluation artifact hash mismatch: {path.name}")
    if arrays:
        with np.load(path, allow_pickle=False) as data:
            result = {key: data[key] for key in data.files}
    else:
        result = json.loads(path.read_text(encoding="utf-8"))
    if signature != _signature(path):
        raise ValueError("Evaluation artifact changed during read")
    return result


def _artifact(path: str, digest: str, *, arrays: bool = False):
    candidate = Path(path)
    if not path or not digest:
        raise ValueError("Evaluation artifact is not sealed; generate a new version")
    return _read_checked(str(candidate), digest, _signature(candidate), arrays)


def _episode(manifest: dict, run_id: str) -> dict:
    matches = [item for item in manifest.get("episodes", []) if item.get("run_id") == run_id]
    if len(matches) != 1:
        raise ValueError(f"Evaluation version must contain exactly one entry for {run_id}")
    return matches[0]


def _reward_detail(version: dict, run_id: str, times: list[float],
                   trajectory_path: str | None = None) -> tuple[dict, dict | None]:
    if version.get("legacy") and not version.get("reward_manifest_sha256"):
        # Read existing legacy arrays as history; this is not a training pin or
        # an automatic migration. New evaluation is required to seal a version.
        path = Path(version["reward_manifest_path"])
        if path.is_symlink() or not path.is_file():
            raise ValueError("Legacy reward artifacts are unavailable; generate a new version")
        version = {**version, "reward_manifest_sha256": hashlib.sha256(path.read_bytes()).hexdigest()}
    manifest = _artifact(version.get("reward_manifest_path", ""),
                         version.get("reward_manifest_sha256", ""))
    if manifest.get("complete") is not True:
        raise ValueError("Reward manifest is incomplete")
    item = _episode(manifest, run_id)
    if trajectory_path and item.get("trajectory_sha256"):
        _artifact(trajectory_path, item["trajectory_sha256"], arrays=True)
    arrays = _artifact(item["reward_path"], item["reward_sha256"], arrays=True)
    return _reward_arrays_detail(version, manifest, item, arrays, times)


def _reward_arrays_detail(version: dict, manifest: dict, item: dict,
                          arrays: dict, times: list[float]) -> tuple[dict, dict | None]:
    """Format saved signals for plots, shared by dataset and global results."""
    boundaries = np.asarray(arrays["boundary_steps"], dtype=int)
    final = np.asarray(arrays.get("final_reward", arrays.get("pbrs_chunk_reward")))
    if (boundaries.ndim != 1 or final.ndim != 1 or len(boundaries) != len(final) + 1
            or len(final) < 1 or boundaries[0] != 0 or boundaries[-1] != len(times) - 1
            or np.any(np.diff(boundaries) <= 0) or not np.isfinite(final).all()):
        raise ValueError("Reward version does not cover the complete recorded trajectory")
    recipe = manifest["reward_config"]
    source = version["evaluator"]
    result = {
        "status": "READY", "source": source, "version_id": version["id"],
        "reward_config": recipe, "boundary_steps": boundaries.tolist(),
        "chunk_start_steps": boundaries[:-1].tolist(),
        "chunk_end_steps": boundaries[1:].tolist(),
        "chunk_lengths": np.diff(boundaries).tolist(), "final_reward": final.tolist(),
    }
    for stored, public in (("sparse_reward", "sparse_reward"), ("dense_reward", "dense_reward"),
                           ("pbrs_shaping_reward", "shape_reward")):
        if stored in arrays:
            values = arrays[stored]
            if values.shape != final.shape or not np.isfinite(values).all():
                raise ValueError(f"Invalid reward component: {stored}")
            result[public] = values.tolist()
    if source == "rynnvalue" and "shape_reward" in result:
        # Old combined files stored Shape/Final only. Recover the displayed
        # components from those stored values, never from today's formula.
        dense = np.asarray(result.get("dense_reward", float(recipe.get("shaping_weight", .1))
                                      * np.asarray(result["shape_reward"])))
        result.setdefault("dense_reward", dense.tolist())
        result.setdefault("sparse_reward", (final - dense).tolist())
    if source == "stage":
        scores = np.asarray(arrays["stage_score"])
        stored_times = arrays.get("time_seconds", np.asarray(times))
        if scores.shape != (len(times),) or not np.isfinite(scores).all():
            raise ValueError("Stage score length does not match the recorded trajectory")
        if not np.array_equal(stored_times, np.asarray(times)):
            raise ValueError("Stage version time axis differs from the recorded trajectory")
        result.update(observation_steps=list(range(len(times))), time_seconds=times,
                      stage_scores=scores.tolist())
    rynn = None
    if source == "rynnvalue":
        keys = ("absolute_temporal_distance_seconds", "absolute_value_entropy_nats",
                "absolute_value_logits", "relative_temporal_distance_seconds", "relative_value_logits")
        outputs = {key: arrays[key].astype(float).tolist() for key in keys}
        if any(len(outputs[key]) != len(boundaries) for key in keys):
            raise ValueError("RynnValue output boundary length mismatch")
        metadata = item.get("official_outputs") or {}
        # The pure annotation metadata retains analysis and official slot semantics.
        if version.get("annotation_manifest_path") and version.get("annotation_manifest_sha256"):
            annotation = _artifact(version["annotation_manifest_path"],
                                   version["annotation_manifest_sha256"])
            metadata = _episode(annotation, item["run_id"]).get("official_outputs") or metadata
        rynn = {
            "status": "READY", "evaluated_at": version.get("completed_at"),
            "model": (item.get("annotator") or manifest.get("annotator") or {}).get("model")
                     or version.get("parameters", {}).get("checkpoint"),
            "revision": (item.get("annotator") or manifest.get("annotator") or {}).get("resolved_revision")
                        or version.get("parameters", {}).get("revision"),
            "boundary_steps": result["boundary_steps"],
            "official_outputs": {**metadata, **outputs}, "reward_config": recipe,
            "pbrs_reward": {key: result[key] for key in (
                "sparse_reward", "dense_reward", "shape_reward", "final_reward",
                "chunk_start_steps", "chunk_end_steps", "chunk_lengths") if key in result},
        }
        rynn["pbrs_reward"]["accumulate_primitive_steps"] = bool(recipe.get("accumulate_primitive_steps"))
    return result, rynn


def _robometer_detail(version: dict, run_id: str) -> dict:
    manifest = _artifact(version.get("robometer_manifest_path", ""),
                         version.get("robometer_manifest_sha256", ""))
    if manifest.get("complete") is not True:
        raise ValueError("Robometer manifest is incomplete")
    item = _episode(manifest, run_id)
    arrays = _artifact(item["annotation_path"], item["values_sha256"], arrays=True)
    keys = ("observation_steps", "time_seconds", "progress_pred", "success_probs")
    lengths = {len(arrays[key]) for key in keys}
    if len(lengths) != 1 or 0 in lengths or any(not np.isfinite(arrays[key]).all() for key in keys):
        raise ValueError("Invalid Robometer output arrays")
    return {"status": "READY", "evaluated_at": version.get("completed_at"),
            "model": manifest.get("annotator", {}).get("model"),
            "revision": manifest.get("annotator", {}).get("revision"),
            "version_id": version["id"], "evaluation_config": manifest.get("evaluation_config", {}),
            **{key: arrays[key].tolist() for key in keys}}


def attach_dataset_context(result: dict, datasets: Any, dataset_id: str | None,
                           version_id: str | None) -> dict:
    """Annotate a detail response with explicit, dataset-scoped immutable curves."""
    run_id = result["run"]["id"]
    contexts = []
    global_candidates = []
    for dataset in datasets.list(result["run"].get("task_id")):
        if any(member["run_id"] == run_id for member in dataset.get("members", [])):
            contexts.append({"dataset_id": dataset["id"], "dataset_name": dataset["name"],
                             "reward_version_id": dataset.get("reward_version_id"),
                             "robometer_version_id": dataset.get("robometer_version_id"),
                             "versions": dataset.get("evaluation_versions", [])})
            for saved in dataset.get("evaluation_versions", []):
                if saved.get("status") == "READY" and saved.get("evaluator") != "robometer":
                    global_candidates.append((saved.get("completed_at") or saved.get("created_at") or "",
                                              dataset["id"], saved["id"]))
    result.update(available_dataset_contexts=contexts, dataset_context=None,
                  global_evaluation=None, reward_evaluation=None, global_evaluation_pending=False,
                  global_evaluation_error=None)
    if dataset_id is None:
        if version_id is not None:
            raise ValueError("version_id requires dataset_id")
        snapshot = read_reward_snapshot(result["run"])
        if snapshot:
            metadata, arrays = snapshot["metadata"], snapshot["arrays"]
            saved = {"id": metadata.get("evaluation_id") or "global", "evaluator": metadata["source"],
                     "completed_at": metadata.get("evaluated_at"),
                     "parameters": metadata["reward_config"]}
            reward, rynn = _reward_arrays_detail(
                saved, {"reward_config": metadata["reward_config"]}, metadata["entry"],
                arrays, result["series"]["time_seconds"],
            )
            result.update(reward_evaluation=reward, global_evaluation={
                "source": metadata["source"], "config": metadata["reward_config"],
                "evaluated_at": metadata.get("evaluated_at"), "origin": metadata.get("origin"),
            })
            if rynn:
                result.update(evaluation=rynn, rynnvalue_evaluation=rynn)
        elif result["run"].get("trajectory") and Path(result["run"]["trajectory"]).with_name(SIDECAR).exists():
            # A recorded global choice must not silently fall back to a different
            # dataset/evaluator while its source identity is being revalidated.
            pending = needs_snapshot_validation(result["run"])
            result.update(global_evaluation_pending=pending, evaluation=None, rynnvalue_evaluation=None,
                          global_evaluation_error=None if pending else "全局评价的源数据已变化或文件损坏，请手动重新评价覆盖。")
        elif result.get("rynnvalue_evaluation"):
            existing = result["rynnvalue_evaluation"]
            result["global_evaluation"] = {"source": "rynnvalue", "origin": "trajectory",
                "config": existing.get("reward_config", {}), "evaluated_at": existing.get("evaluated_at")}
        else:
            # Show the first saved result immediately. Publication of the small
            # standalone copy is queued by the API; no observation hash here.
            for _, owner, identifier in sorted(global_candidates):
                try:
                    saved = datasets.get_version(owner, identifier)
                    reward, rynn = _reward_detail(saved, run_id, result["series"]["time_seconds"],
                                                  result["run"].get("trajectory"))
                except (OSError, KeyError, TypeError, ValueError):
                    continue
                result.update(reward_evaluation=reward, global_evaluation_pending=True,
                              global_evaluation={"source": saved["evaluator"],
                                  "config": reward["reward_config"], "origin": "dataset",
                                  "evaluated_at": saved.get("completed_at")})
                if rynn:
                    result.update(evaluation=rynn, rynnvalue_evaluation=rynn)
                break
        return result
    dataset = datasets.get(dataset_id, quick_verify=False)
    if not any(member["run_id"] == run_id for member in dataset.get("members", [])):
        raise ValueError("Trajectory is not a member of this frozen dataset")
    selected = version_id or dataset.get("reward_version_id") or dataset.get("robometer_version_id")
    result.update(evaluation=None, rynnvalue_evaluation=None, robometer_evaluation=None)
    context = {"dataset_id": dataset_id, "dataset_name": dataset["name"],
               "version_id": selected, "source": None, "config": {}, "status": "NOT_EVALUATED"}
    result["dataset_context"] = context
    if selected is None:
        return result
    version = datasets.get_version(dataset_id, selected)
    if version.get("status") != "READY":
        raise ValueError("Selected evaluation version is not ready")
    context.update(source=version["evaluator"], config=version.get("parameters", {}), status="READY")
    reward_id = selected if version["evaluator"] != "robometer" else dataset.get("reward_version_id")
    robo_id = selected if version["evaluator"] == "robometer" else dataset.get("robometer_version_id")
    if reward_id:
        reward_version = version if reward_id == selected else datasets.get_version(dataset_id, reward_id)
        reward, rynn = _reward_detail(reward_version, run_id, result["series"]["time_seconds"],
                                     result["run"].get("trajectory"))
        result.update(reward_evaluation=reward, evaluation=rynn, rynnvalue_evaluation=rynn)
    if robo_id:
        robo_version = version if robo_id == selected else datasets.get_version(dataset_id, robo_id)
        result["robometer_evaluation"] = _robometer_detail(robo_version, run_id)
    return result
