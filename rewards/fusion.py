"""Per-action training rewards from saved signals; Final Reward fuses Stage and RynnValue."""
from __future__ import annotations

import json
from pathlib import Path

from core.storage_paths import with_dataset_lease, storage_path

import numpy as np

from core.config import LoadedConfig, needs_rynnvalue, needs_stage
from core.data import load_manifest
from core.io import atomic_json, sha256_file, stable_hash
from rewards.stage.reward import stage_annotation_context, stage_rewards, stage_scores


def import_saved_model_inputs(config: LoadedConfig, references: Path) -> None:
    """Freeze already evaluated signals, allowing per-trajectory inference recipes."""
    from rewards.rynnvalue.outputs import ANNOTATION_SCHEMA_VERSION
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
    from rewards.rynnvalue.outputs import ANNOTATION_SCHEMA_VERSION, OFFICIAL_OUTPUT_KEYS, validate_official_outputs
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
        path = storage_path(entry["annotation_path"])
        if path.is_symlink() or sha256_file(path) != entry["annotation_sha256"]:
            raise ValueError(f"Saved RynnValue values changed: {episode['run_id']}")
        with np.load(path, allow_pickle=False) as arrays:
            validate_official_outputs({k: arrays[k] for k in OFFICIAL_OUTPUT_KEYS},
                                      int(episode["recorded_action_count"]) + 1)


def step_reward_arrays(episode: dict, recipe: dict, signals: dict) -> dict:
    """Per-action rewards of one recording from its saved signals; never runs an evaluator.

    ``signals`` hold one value per observation: ``stage_score`` and the RynnValue outputs
    (``absolute_temporal_distance_seconds``) as the source requires. Every returned reward
    array has one value per action.
    """
    from core.config import reward_source
    from rewards.pbrs import shaping_rewards, sparse_rewards, success_mask

    count = int(episode["recorded_action_count"])
    done = success_mask(episode)
    gamma = float(recipe["gamma"])
    source = reward_source(recipe)
    if source == "final":
        return fused_reward_arrays(episode, recipe, signals)
    if source == "stage":
        scores = _stage_signal(episode, signals)
        final = stage_rewards(scores)
        return {"stage_score": scores, "stage_reward": final, "final_reward": final.astype(np.float32)}
    sparse = sparse_rewards(done)
    if source == "sparse":
        return {"sparse_reward": sparse.astype(np.float32), "final_reward": sparse.astype(np.float32)}
    kappa = float(recipe["shaping_weight"])
    shape = shaping_rewards(-_rynnvalue_signal(episode, signals), done, gamma,
                            bool(recipe.get("zero_success_potential", False)))
    if count != len(shape):
        raise ValueError(f"RynnValue outputs do not cover {episode['run_id']}")
    return {"sparse_reward": sparse.astype(np.float32), "pbrs_shaping_reward": shape.astype(np.float32),
            "dense_reward": (kappa * shape).astype(np.float32),
            "final_reward": (sparse + kappa * shape).astype(np.float32)}


def _stage_signal(episode: dict, signals: dict) -> np.ndarray:
    scores = signals.get("stage_score")
    count = int(episode["recorded_action_count"])
    if scores is None or scores.shape != (count + 1,) or not np.isfinite(scores).all():
        raise ValueError(f"Required Stage timeline is missing or invalid: {episode['run_id']}")
    if np.any(scores > 0):
        raise ValueError(f"Stage score exceeds zero; check keyframes: {episode['run_id']}")
    return np.asarray(scores, dtype=np.float64)


def _rynnvalue_signal(episode: dict, signals: dict) -> np.ndarray:
    """Remaining time per observation from the official absolute head."""
    distances = signals.get("absolute_temporal_distance_seconds")
    count = int(episode["recorded_action_count"])
    if (distances is None or distances.shape != (count + 1, 1)
            or not np.isfinite(distances).all()):
        raise ValueError(f"Required RynnValue outputs are missing: {episode['run_id']}")
    return np.asarray(distances[:, 0], dtype=np.float64)


def fused_reward_arrays(episode: dict, recipe: dict, signals: dict) -> dict:
    """Final Reward per action; source model outputs and manual keyframes are never changed.

    additive: base * sparse + alpha * stage + kappa * shaping; multiplicative:
    -stage(t+1) * (sparse + kappa * shaping).
    """
    from rewards.pbrs import shaping_rewards, sparse_rewards, success_mask

    done = success_mask(episode)
    gamma = float(recipe["gamma"])
    kappa = float(recipe["shaping_weight"])
    alpha = float(recipe.get("alpha", 0))
    mode = recipe.get("fusion_mode", "additive")
    base_coefficient = recipe.get("base_coefficient", 1 - alpha)
    if (type(base_coefficient) not in (int, float) or not np.isfinite(base_coefficient)
            or base_coefficient < 0):
        raise ValueError("Invalid Final Reward base_coefficient")
    if "base_coefficient" in recipe and mode != "additive":
        raise ValueError("base_coefficient requires additive Final Reward")
    if mode not in {"additive", "multiplicative"}:
        raise ValueError("Unknown Final Reward fusion mode")
    if not all(np.isfinite(x) for x in (gamma, alpha, kappa)) or not 0 <= alpha <= 1:
        raise ValueError("Invalid Final Reward parameters")
    scores = _stage_signal(episode, signals) if needs_stage(recipe) else None
    b = sparse_rewards(done)
    f = (shaping_rewards(-_rynnvalue_signal(episode, signals), done, gamma,
                         bool(recipe.get("zero_success_potential", False)))
         if needs_rynnvalue(recipe) else np.zeros_like(b))
    s = stage_rewards(scores) if scores is not None else np.zeros_like(b)
    original = b + kappa * f
    final = base_coefficient * b + alpha * s + kappa * f if mode == "additive" else -s * original
    result = dict(sparse_reward=b.astype(np.float32), pbrs_shaping_reward=f.astype(np.float32),
                  dense_reward=(kappa * f).astype(np.float32),
                  original_final_reward=original.astype(np.float32),
                  final_reward=final.astype(np.float32))
    if scores is not None:
        result.update(stage_score=scores, stage_reward=s.astype(np.float32))
    if not all(np.isfinite(values).all() for values in result.values()):
        raise ValueError("Non-finite Final Reward")
    return result


def _valid_cached_result(payload: dict, prepared: dict, snapshot: dict | None) -> bool:
    from rewards.manifest import _episode_reward_metadata

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
        path = storage_path(entry["reward_path"])
        if (path.is_symlink() or sha256_file(path) != entry["reward_sha256"]
                or entry["annotation_path"] != entry["reward_path"]):
            return False
        with np.load(path, allow_pickle=False) as arrays:
            if (arrays["final_reward"].shape != (int(episode["recorded_action_count"]),)
                    or not np.isfinite(arrays["final_reward"]).all()):
                return False
    return True


@with_dataset_lease
def materialize_final_reward(config: LoadedConfig, *, force: bool = False) -> Path:
    """Recompute from saved signals only; missing inputs never instantiate a model."""
    from rewards.manifest import REWARD_SCHEMA_VERSION, _episode_reward_metadata, _episode_timeline_arrays, _reward_generation_dir, reward_derivation_config, reward_implementation_fingerprint
    from rewards.stage.keyframes import load_stage_annotations
    from rewards.rynnvalue.outputs import OFFICIAL_OUTPUT_KEYS, validate_official_outputs
    from rewards.rynnvalue.annotate import annotate_manifest, load_annotation_index

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
        signals = dict(_episode_timeline_arrays(episode))
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
            with np.load(storage_path(entry["annotation_path"]), allow_pickle=False) as data:
                signals.update(validate_official_outputs({k: data[k] for k in OFFICIAL_OUTPUT_KEYS},
                                                        int(episode["recorded_action_count"]) + 1))
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
