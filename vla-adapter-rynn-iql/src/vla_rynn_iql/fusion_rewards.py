"""Final Reward: Stage baseline plus RynnValue shaping, independent of inference."""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np

from .config import LoadedConfig, effective_cumulative, needs_rynnvalue, needs_stage
from .data import load_manifest
from .io import atomic_json, sha256_file, stable_hash
from .stage_rewards import stage_annotation_context, stage_chunk_reward, stage_scores


def import_saved_model_inputs(config: LoadedConfig, references: Path) -> None:
    """Freeze already evaluated signals, allowing per-trajectory inference recipes."""
    from .rewards import ANNOTATION_SCHEMA_VERSION
    prepared = load_manifest(config)
    source = json.loads(references.read_text())
    if source.get("source_dataset_sha256") != prepared.get("source_dataset_sha256"):
        raise ValueError("Saved evaluations belong to another frozen dataset")
    payload = {"schema_version": ANNOTATION_SCHEMA_VERSION, "kind": "rynnvalue_annotation",
               "input_kind": "saved_evaluations", "complete": True,
               "dataset_sha256": prepared["dataset_sha256"], "annotation_config": {},
               "episodes": source["episodes"]}
    _validate_saved_model_inputs(payload, prepared)
    atomic_json(Path(config.section("paths")["work_dir"]) / "annotations" / "annotation_manifest.json", payload)


def _validate_saved_model_inputs(payload: dict, prepared: dict) -> None:
    from .rewards import ANNOTATION_SCHEMA_VERSION, OFFICIAL_OUTPUT_KEYS, validate_official_outputs
    if (payload.get("schema_version") != ANNOTATION_SCHEMA_VERSION or not payload.get("complete")
            or payload.get("dataset_sha256") != prepared["dataset_sha256"]):
        raise ValueError("Saved RynnValue input manifest is incomplete or mismatched")
    entries = {entry["run_id"]: entry for entry in payload["episodes"]}
    if len(entries) != len(payload["episodes"]) or set(entries) != {e["run_id"] for e in prepared["episodes"]}:
        raise ValueError("Saved RynnValue input membership mismatch")
    for episode in prepared["episodes"]:
        entry = entries[episode["run_id"]]
        for key in ("trajectory_sha256", "observations_sha256", "source_manifest_sha256", "prompt"):
            if entry.get(key) != episode.get(key):
                raise ValueError(f"Saved RynnValue {key} mismatch: {episode['run_id']}")
        path = Path(entry["annotation_path"])
        if path.is_symlink() or sha256_file(path) != entry["annotation_sha256"]:
            raise ValueError(f"Saved RynnValue values changed: {episode['run_id']}")
        with np.load(path, allow_pickle=False) as arrays:
            if not np.array_equal(arrays["boundary_steps"], episode["reward_boundaries"]):
                raise ValueError(f"Saved RynnValue boundary mismatch: {episode['run_id']}")
            validate_official_outputs({k: arrays[k] for k in OFFICIAL_OUTPUT_KEYS}, len(episode["reward_boundaries"]))


def fused_reward_arrays(episode: dict, recipe: dict, signals: dict) -> dict:
    """Pure reduction; source model outputs and manual keyframes are never changed."""
    from .rewards import chunk_reward_components, sparse_macro_reward, sparse_primitive_return

    count = int(episode["recorded_action_count"])
    chunks = episode.get("evaluation_chunks", episode["chunks"])
    boundaries = np.asarray(episode["reward_boundaries"], dtype=np.int64)
    done = np.zeros(count, dtype=bool)
    if episode["terminal_step"] is not None:
        done[int(episode["terminal_step"]):] = True
    gamma = float(recipe["gamma"])
    cumulative = effective_cumulative(recipe)
    kappa = float(recipe["shaping_weight"])
    alpha = float(recipe.get("alpha", 0))
    mode = recipe.get("fusion_mode", "additive")
    if mode not in {"additive", "multiplicative"}:
        raise ValueError("Unknown Final Reward fusion mode")
    if not all(np.isfinite(x) for x in (gamma, alpha, kappa)) or not 0 <= alpha <= 1:
        raise ValueError("Invalid Final Reward parameters")
    scores = signals.get("stage_score") if needs_stage(recipe) else None
    if needs_stage(recipe):
        if scores is None or scores.shape != (count + 1,) or not np.isfinite(scores).all():
            raise ValueError(f"Required Stage timeline is missing or invalid: {episode['run_id']}")
        if np.any(scores > 0):
            raise ValueError(f"Stage score exceeds zero; check keyframes: {episode['run_id']}")
    lookup = None
    if needs_rynnvalue(recipe):
        distances = signals.get("absolute_temporal_distance_seconds")
        if (distances is None or distances.shape != (len(boundaries), 1)
                or not np.isfinite(distances).all()):
            raise ValueError(f"Required RynnValue outputs are missing: {episode['run_id']}")
        lookup = dict(zip(boundaries.tolist(), distances[:, 0].tolist()))
    sparse, shape, stage, final = [], [], [], []
    for chunk in chunks:
        start, length, end = int(chunk["start"]), int(chunk["length"]), int(chunk["end"])
        if lookup is not None:
            b, f, original = chunk_reward_components(
                done, start, length, lookup[start], lookup[end], gamma, kappa, cumulative, True)
        else:
            b = (sparse_primitive_return(done, start, length, gamma) if cumulative
                 else sparse_macro_reward(done, start, length))
            f, original = 0., b
        s = stage_chunk_reward(scores, start, length, gamma, cumulative) if scores is not None else 0.
        if mode == "additive":
            value = (1 - alpha) * b + alpha * s + kappa * f
        else:
            value = -float(scores[end]) * original
        sparse.append(b)
        shape.append(f)
        stage.append(s)
        final.append(value)
    b, f = np.asarray(sparse, dtype=np.float32), np.asarray(shape, dtype=np.float32)
    raw_final = np.asarray(final, dtype=np.float64)
    if not np.isfinite(raw_final).all():
        raise ValueError(f"Non-finite Final Reward: {episode['run_id']}")
    normalization = recipe.get("final_normalization", "none")
    normalized = raw_final
    normalization_arrays = {}
    if normalization == "initial_chunk_v1":
        if not chunks or int(chunks[0]["start"]) != 0:
            raise ValueError(f"Final Reward rescale requires the full trajectory from step 0: {episode['run_id']}")
        reference = float(raw_final[0])
        if reference >= 0:
            raise ValueError(f"Initial Final Reward must be negative for rescale: {episode['run_id']} (R0={reference})")
        # Use the full recorded first chunk, not the deduplicated replay suffix.
        # Never reuse a previous scale after changing gamma or cumulative mode.
        with np.errstate(over="ignore", invalid="ignore"):
            scale = np.divide(1., -reference)
            normalized = raw_final / -reference
        normalization_arrays = {
            "raw_final_reward": raw_final,
            "final_reward_reference": np.asarray(reference, dtype=np.float64),
            "final_reward_scale": np.asarray(scale, dtype=np.float64),
        }
    elif normalization != "none":
        raise ValueError(f"Unknown Final Reward normalization: {normalization}")
    result = dict(sparse_reward=b, pbrs_shaping_reward=f, dense_reward=kappa * f,
                  original_final_reward=b + kappa * f,
                  final_reward=np.asarray(normalized, dtype=np.float32), **normalization_arrays)
    result["pbrs_chunk_reward"] = result["final_reward"]
    if scores is not None:
        result.update(stage_score=scores, stage_chunk_reward=np.asarray(stage, dtype=np.float32))
    if not all(np.isfinite(values).all() for values in result.values()):
        raise ValueError("Non-finite Final Reward")
    return result


def _valid_cached_result(payload: dict, prepared: dict, snapshot: dict | None) -> bool:
    from .rewards import _episode_reward_metadata

    expected = {episode["run_id"]: episode for episode in prepared["episodes"]}
    entries = payload["episodes"]
    if (not payload.get("complete") or len(entries) != len(expected)
            or {entry["run_id"] for entry in entries} != set(expected)):
        return False
    if snapshot is not None:
        path = Path(payload["stage_annotations_path"])
        if path.is_symlink() or json.loads(path.read_text()) != snapshot:
            return False
    for entry in entries:
        episode = expected[entry["run_id"]]
        if any(entry.get(key) != value for key, value in _episode_reward_metadata(episode).items()):
            return False
        path = Path(entry["reward_path"])
        if (path.is_symlink() or sha256_file(path) != entry["reward_sha256"]
                or entry["annotation_path"] != str(path)):
            return False
        with np.load(path, allow_pickle=False) as arrays:
            size = len(episode.get("evaluation_chunks", episode["chunks"]))
            if (arrays["final_reward"].shape != (size,)
                    or not np.isfinite(arrays["final_reward"]).all()
                    or not np.array_equal(arrays["boundary_steps"], episode["reward_boundaries"])):
                return False
    return True


def materialize_final_reward(config: LoadedConfig, *, force: bool = False) -> Path:
    """Recompute from saved signals only; missing inputs never instantiate a model."""
    from .rewards import (
        REWARD_SCHEMA_VERSION, OFFICIAL_OUTPUT_KEYS, _episode_reward_metadata,
        _episode_timeline_arrays, _reward_generation_dir, annotate_manifest,
        load_annotation_index, load_stage_annotations, reward_derivation_config,
        reward_implementation_fingerprint, validate_official_outputs,
    )

    manifest = load_manifest(config)
    reward = config.section("reward")
    recipe = reward_derivation_config(reward)
    snapshot = load_stage_annotations(config, manifest) if needs_stage(reward) else None
    annotation = None
    if needs_rynnvalue(reward):
        index_path = Path(config.section("paths")["work_dir"]) / "annotations" / "annotation_manifest.json"
        if index_path.is_file():
            candidate = json.loads(index_path.read_text())
            if candidate.get("input_kind") == "saved_evaluations":
                _validate_saved_model_inputs(candidate, manifest)
                annotation = candidate
            else:
                try:
                    annotation = load_annotation_index(config)
                except ValueError:
                    pass
        if annotation is None:
            annotate_manifest(config, reuse_only=True)
            annotation = load_annotation_index(config)
    by_run = {item["run_id"]: item for item in annotation["episodes"]} if annotation else {}
    identity = dict(schema_version=REWARD_SCHEMA_VERSION, kind="derived_iql_reward",
                    dataset_sha256=manifest["dataset_sha256"], reward_config=recipe,
                    stage_annotations_sha256=stable_hash(snapshot) if snapshot else None,
                    annotation_manifest_sha256=stable_hash(annotation) if annotation else None,
                    annotation_config=annotation["annotation_config"] if annotation else None,
                    derivation_implementation_sha256=reward_implementation_fingerprint("final"))
    root = Path(config.section("paths")["work_dir"]) / "rewards"
    path = root / "reward_manifest.json"
    if path.is_file() and not force:
        try:
            previous = json.loads(path.read_text())
            if (all(previous.get(k) == v for k, v in identity.items())
                    and _valid_cached_result(previous, manifest, snapshot)):
                return path
        except (OSError, ValueError, KeyError, TypeError):
            pass
    pending = []
    for episode in manifest["episodes"]:
        signals = {"boundary_steps": np.asarray(episode["reward_boundaries"], dtype=np.int64),
                   **_episode_timeline_arrays(episode)}
        annotation_hash = None
        if snapshot:
            labels = snapshot["annotations"][episode["run_id"]]
            context = stage_annotation_context(labels, done=signals["environment_done"],
                success_consecutive_steps=config.section("data")["success_consecutive_steps"],
                exponent=recipe["stage_exponent"])
            signals["stage_score"] = stage_scores(context)
            annotation_hash = labels["annotation_sha256"]
        if annotation:
            entry = by_run[episode["run_id"]]
            with np.load(entry["annotation_path"], allow_pickle=False) as data:
                if not np.array_equal(data["boundary_steps"], episode["reward_boundaries"]):
                    raise ValueError(f"RynnValue boundaries mismatch: {episode['run_id']}")
                signals.update(validate_official_outputs({k: data[k] for k in OFFICIAL_OUTPUT_KEYS},
                                                        len(episode["reward_boundaries"])))
        signals.update(fused_reward_arrays(episode, recipe, signals))
        pending.append((episode, signals, annotation_hash))
    directory = _reward_generation_dir(root)
    snapshot_path = directory / "stage_annotations.json" if snapshot else None
    if snapshot_path:
        atomic_json(snapshot_path, snapshot)
    entries = []
    for episode, arrays, annotation_hash in pending:
        output = directory / f"{stable_hash(episode['run_id'])}.npz"
        np.savez_compressed(output, **arrays)
        digest = sha256_file(output)
        original = by_run.get(episode["run_id"], {})
        entries.append({**_episode_reward_metadata(episode), "run_id": episode["run_id"],
            "source": "final", "reward_path": str(output.resolve()), "reward_sha256": digest,
            "annotation_path": str(output.resolve()), "annotation_sha256": digest,
            "stage_annotation_sha256": annotation_hash, "environment_success": episode["success"],
            **({"official_annotation_path": original["annotation_path"],
                "official_annotation_sha256": original["annotation_sha256"],
                "annotation_config": original.get("annotation_config", annotation["annotation_config"]),
                "official_outputs": original.get("official_outputs", {})} if original else {})})
    payload = {**identity, "complete": True, "episodes": entries,
               "stage_annotations_path": str(snapshot_path.resolve()) if snapshot_path else None}
    atomic_json(directory / "reward_manifest.json", payload)
    atomic_json(path, payload)
    return path
