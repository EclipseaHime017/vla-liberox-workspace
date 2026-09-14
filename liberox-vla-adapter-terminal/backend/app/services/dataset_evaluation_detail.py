"""Read pinned dataset plots, without evaluating models or recomputing rewards."""
from __future__ import annotations

import hashlib
import json
from functools import lru_cache
from pathlib import Path
from typing import Any

import numpy as np

from .trajectory_reward_snapshot import SOURCES, snapshot_path, needs_snapshot_validation, read_reward_snapshot


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
    """Resolve each evaluator separately: dataset result, otherwise global."""
    run = result["run"]
    run_id = run["id"]
    contexts, candidates = [], []
    for dataset in datasets.list(run.get("task_id")):
        if not any(member["run_id"] == run_id for member in dataset.get("members", [])):
            continue
        contexts.append({"dataset_id": dataset["id"], "dataset_name": dataset["name"],
                         "reward_version_id": dataset.get("reward_version_id"),
                         "robometer_version_id": dataset.get("robometer_version_id"),
                         "evaluation_version_ids": dataset.get("evaluation_version_ids", {}),
                         "versions": dataset.get("evaluation_versions", [])})
        for saved in dataset.get("evaluation_versions", []):
            if saved.get("status") == "READY":
                candidates.append((saved.get("completed_at") or saved.get("created_at") or "",
                                   dataset["id"], saved["id"], saved["evaluator"]))
    result.update(available_dataset_contexts=contexts, dataset_context=None,
                  global_evaluation=None, reward_evaluation=None, reward_evaluations={},
                  evaluation_sources={}, global_evaluation_pending=bool(result.get("native_evaluation_pending")), global_evaluation_error=None)
    times = result["series"]["time_seconds"]

    def publish(source: str, reward: dict | None, output: dict | None, context: dict) -> None:
        result["evaluation_sources"][source] = {"status": "READY", **context}
        if reward is not None:
            result["reward_evaluations"][source] = reward
        if source == "rynnvalue":
            result.update(evaluation=output, rynnvalue_evaluation=output)
        elif source == "robometer":
            result["robometer_evaluation"] = output

    for source in (*SOURCES, "robometer"):
        snapshot = read_reward_snapshot(run, source) if source in SOURCES else None
        path = snapshot_path(run, source) if source in SOURCES else None
        if snapshot:
            metadata = snapshot["metadata"]
            saved = {"id": metadata.get("evaluation_id") or "global", "evaluator": source,
                     "completed_at": metadata.get("evaluated_at"), "parameters": metadata["reward_config"]}
            try:
                reward, output = _reward_arrays_detail(saved, {"reward_config": metadata["reward_config"]},
                    metadata["entry"], snapshot["arrays"], times)
                publish(source, reward, output, {"origin": "global", "config": metadata["reward_config"],
                        "evaluated_at": metadata.get("evaluated_at")})
            except (KeyError, ValueError, TypeError) as exc:
                result["evaluation_sources"][source] = {"status": "ERROR", "origin": "global", "error": str(exc)}
            continue
        if path is not None and path.exists():
            pending = needs_snapshot_validation(run, source)
            result["global_evaluation_pending"] |= pending
            result["evaluation_sources"][source] = {"origin": "global", "status": "PENDING" if pending else "ERROR",
                "error": None if pending else "源数据已变化或评价损坏，请重新评价该类型。"}
            if source == "rynnvalue":
                result.update(evaluation=None, rynnvalue_evaluation=None)
            continue
        native = result.get(f"{source}_evaluation")
        if native:
            publish(source, None, native, {"origin": "global",
                "config": native.get("reward_config", native.get("evaluation_config", {})),
                "evaluated_at": native.get("evaluated_at")})
            continue
        for _, owner, identifier, evaluator in sorted(candidates):
            if evaluator != source:
                continue
            try:
                saved = datasets.get_version(owner, identifier)
                if source == "robometer":
                    reward, output = None, _robometer_detail(saved, run_id)
                else:
                    reward, output = _reward_detail(saved, run_id, times, run.get("trajectory"))
                publish(source, reward, output, {"origin": "global", "config": saved.get("parameters", {}),
                        "evaluated_at": saved.get("completed_at")})
                result["global_evaluation_pending"] |= source != "robometer"
                break
            except (OSError, KeyError, TypeError, ValueError):
                continue

    selected = None
    if dataset_id is not None:
        dataset = datasets.get(dataset_id, quick_verify=False)
        if not any(member["run_id"] == run_id for member in dataset.get("members", [])):
            raise ValueError("Trajectory is not a member of this frozen dataset")
        ids = dict(dataset.get("evaluation_version_ids", {}))
        # Compatibility with lightweight callers and pre-map datasets.
        if not ids:
            from .training_dataset_service import TrainingDatasetService
            ids = TrainingDatasetService.current_evaluation_ids(dataset)
        selected = version_id or dataset.get("reward_version_id")
        if version_id:
            explicit = datasets.get_version(dataset_id, version_id)
            ids[explicit["evaluator"]] = version_id
        result["dataset_context"] = {"dataset_id": dataset_id, "dataset_name": dataset["name"],
            "version_id": selected, "status": "READY" if ids else "GLOBAL_FALLBACK", "source": None, "config": {}}
        for source, identifier in ids.items():
            # Invalid local data is reported, never silently replaced by global.
            result["reward_evaluations"].pop(source, None)
            if source == "rynnvalue":
                result.update(evaluation=None, rynnvalue_evaluation=None)
            elif source == "robometer":
                result["robometer_evaluation"] = None
            try:
                saved = datasets.get_version(dataset_id, identifier)
                if saved.get("status") != "READY":
                    raise ValueError("Evaluation is not ready")
                if source == "robometer":
                    reward, output = None, _robometer_detail(saved, run_id)
                else:
                    reward, output = _reward_detail(saved, run_id, times, run.get("trajectory"))
                publish(source, reward, output, {"origin": "dataset", "config": saved.get("parameters", {}),
                        "evaluated_at": saved.get("completed_at"), "version_id": identifier})
            except (OSError, KeyError, TypeError, ValueError) as exc:
                result["evaluation_sources"][source] = {"origin": "dataset", "status": "ERROR", "error": str(exc)}
    elif version_id is not None:
        raise ValueError("version_id requires dataset_id")
    rewards = result["reward_evaluations"]
    result["reward_evaluation"] = next((value for value in rewards.values() if value.get("version_id") == selected),
                                       next(iter(rewards.values()), None))
    chosen = result["reward_evaluation"]
    if chosen:
        context = result["evaluation_sources"][chosen["source"]]
        result["global_evaluation"] = {"source": chosen["source"], **context} if dataset_id is None else None
        if result["dataset_context"]:
            result["dataset_context"].update(source=chosen["source"], config=chosen["reward_config"])
    errors = [item["error"] for item in result["evaluation_sources"].values() if item.get("error")]
    result["global_evaluation_error"] = "; ".join(errors) or None
    return result
