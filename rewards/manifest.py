"""Reward versions: derive, cache and pin per-chunk training rewards.

Sources are sparse success, RynnValue potential shaping (rewards.rynnvalue outputs),
keyframe Stage progress (rewards.stage) and their fusion (rewards.fusion). Derived caches are keyed
by dataset, configuration and formula code; raw model outputs are never recomputed here.
"""
from __future__ import annotations

import json
import logging
import os
import uuid
from pathlib import Path
from typing import Any

import numpy as np

from core.config import (LEGACY_REWARD_KEYS, LoadedConfig, current_reward_settings, needs_rynnvalue,
                         reward_source)
from core.data import load_manifest
from core.io import atomic_json, sha256_file, stable_hash
from core.storage_paths import storage_path, with_dataset_lease
from rewards.pbrs import chunk_reward_components, sparse_primitive_return
from rewards.rynnvalue.annotate import load_annotation_index
from rewards.rynnvalue.outputs import (
    official_inference_config,
    OFFICIAL_OUTPUT_KEYS,
    validate_official_outputs,
)
from rewards.stage.keyframes import load_stage_annotations


LOG = logging.getLogger(__name__)


REWARD_SCHEMA_VERSION = 1


REWARD_DERIVATION_CONFIG_KEYS = frozenset({"rynnvalue", "gamma", "shaping_weight"})


# Saved versions recording LEGACY_REWARD_KEYS are always re-derived from their signals.
TRAINING_REDUCTION_CONFIG_KEYS = frozenset({"gamma", "zero_success_potential"})


def reward_derivation_config(reward_config: dict[str, Any]) -> dict[str, Any]:
    """Return cheap reward-reduction settings, independent of VLM inference."""
    result = {
        key: reward_config[key]
        for key in sorted(REWARD_DERIVATION_CONFIG_KEYS)
    }
    source = reward_source(reward_config)
    if reward_config.get("zero_success_potential", False):
        result["zero_success_potential"] = True
    result["rynnvalue"] = needs_rynnvalue(reward_config)
    # Preserve RynnValue's existing manifest/checkpoint identity exactly.
    if source != "rynnvalue":
        result["source"] = source
    if source in {"stage", "final"}:
        result["stage_exponent"] = float(reward_config.get("stage_exponent", 2.0))
    if source == "final":
        result.update(fusion_mode=reward_config.get("fusion_mode", "additive"),
                      alpha=float(reward_config.get("alpha", 0.0)))
        if reward_config.get("base_coefficient") is not None:
            result["base_coefficient"] = float(reward_config["base_coefficient"])
        normalization = reward_config.get("final_normalization", "none")
        if normalization != "none":
            result["final_normalization"] = normalization
    return result


def reward_implementation_fingerprint(source: str) -> str:
    """Invalidate cheap derived caches when formula code changes, never raw VLM outputs."""
    root = Path(__file__).parent
    files = ["manifest.py", "pbrs.py"]
    if source in {"stage", "final"}:
        files += ["stage/keyframes.py", "stage/reward.py"]
    if source == "final":
        files.append("fusion.py")
    return stable_hash({name: sha256_file(root / name) for name in files})


def _reward_generation_dir(root: Path) -> Path:
    """Every rebuild gets new files; old training references are never overwritten."""
    directory = root / "versions" / uuid.uuid4().hex
    directory.mkdir(parents=True, exist_ok=False)
    return directory


def _episode_reward_metadata(episode: dict[str, Any]) -> dict[str, Any]:
    return {
        "trajectory_sha256": episode["trajectory_sha256"],
        "observations_sha256": episode["observations_sha256"],
        "recorded_action_count": episode["recorded_action_count"],
        "reward_boundaries": episode["reward_boundaries"],
        "evaluation_chunks": episode.get("evaluation_chunks", episode["chunks"]),
    }


def _episode_timeline_arrays(episode: dict[str, Any]) -> dict[str, np.ndarray]:
    with np.load(storage_path(episode["trajectory_path"]), allow_pickle=False) as trajectory:
        times = np.asarray(trajectory["time_seconds"], dtype=np.float64)
        done = np.asarray(trajectory["done"], dtype=bool)
    return {"observation_steps": np.arange(len(times), dtype=np.int64),
            "time_seconds": times, "environment_done": done}


def reward_manifest_digest(index: dict[str, Any]) -> str:
    """Content identity for direct rewards; legacy RynnValue hashes stay unchanged."""
    if index.get("portable_source_sha256"):
        return stable_hash({"source": index["portable_source_sha256"],
            "dataset_sha256": index["dataset_sha256"], "reward_config": index["reward_config"],
            "implementation": index.get("derivation_implementation_sha256"),
            "episodes": [{key: entry.get(key) for key in ("run_id", "reward_sha256",
                "saved_reward_config", "training_reward_config", "saved_annotation_config")}
                for entry in sorted(index["episodes"], key=lambda entry: entry["run_id"])]})
    if index.get("binding_kind") == "global_trajectory_snapshots":
        return stable_hash({"dataset_sha256": index["dataset_sha256"], "episodes": [
            {key: entry.get(key) for key in ("run_id", "reward_sha256", "saved_reward_config",
                "saved_annotation_config", "training_reward_config", "stage_annotation_sha256")}
            for entry in sorted(index["episodes"], key=lambda item: item["run_id"])]})
    if (index.get("reward_config", {}).get("source", "rynnvalue") == "rynnvalue"
            and not index.get("source_reward_manifest_sha256")):
        return stable_hash(index)
    identity = {
        "schema_version": index["schema_version"], "kind": index["kind"],
        "dataset_sha256": index["dataset_sha256"], "reward_config": index["reward_config"],
        "stage_annotations_sha256": index.get("stage_annotations_sha256"),
        "episodes": [{"run_id": item["run_id"], "reward_sha256": item["reward_sha256"],
                      "stage_annotation_sha256": item.get("stage_annotation_sha256")}
                     for item in index["episodes"]],
    }
    if "derivation_implementation_sha256" in index:
        identity["derivation_implementation_sha256"] = index["derivation_implementation_sha256"]
    if index.get("source_reward_manifest_sha256"):
        identity.update(source_reward_manifest_sha256=index["source_reward_manifest_sha256"],
                        source_reward_version_id=index["source_reward_version_id"])
    return stable_hash(identity)


@with_dataset_lease


def materialize_reward_manifest(config: LoadedConfig, *, force: bool = False) -> Path:
    """Build the cheap reward cache from immutable official RynnValue outputs."""
    if config.section("reward").get("manifest_path"):
        # Explicit version consumers must never mutate or regenerate their input.
        index = load_pinned_reward_index(config)
        return Path(index.get("training_adaptation_manifest_path") or config.section("reward")["manifest_path"])
    if reward_source(config.section("reward")) == "final":
        from rewards.fusion import materialize_final_reward
        return materialize_final_reward(config, force=force)
    if reward_source(config.section("reward")) != "rynnvalue":
        return _materialize_direct_rewards(config, force=force)
    manifest = load_manifest(config)
    annotation_index = load_annotation_index(config)
    reward_cfg = config.section("reward")
    derivation_cfg = reward_derivation_config(reward_cfg)
    implementation = reward_implementation_fingerprint("rynnvalue")
    reward_dir = Path(config.section("paths")["work_dir"]) / "rewards"
    reward_dir.mkdir(parents=True, exist_ok=True)
    index_path = reward_dir / "reward_manifest.json"

    if not force and index_path.is_file():
        try:
            current = json.loads(index_path.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError):
            current = None
        if (
            isinstance(current, dict)
            and current.get("schema_version") == REWARD_SCHEMA_VERSION
            and current.get("kind") == "derived_iql_reward"
            and current.get("complete") is True
            and current.get("dataset_sha256") == manifest["dataset_sha256"]
            and current.get("annotation_manifest_sha256") == stable_hash(annotation_index)
            and current.get("reward_config") == derivation_cfg
            and current.get("derivation_implementation_sha256") == implementation
            and len(current.get("episodes", [])) == len(manifest.get("episodes", []))
        ):
            valid = True
            for episode in current["episodes"]:
                reward_path = storage_path(str(episode.get("reward_path") or ""))
                if (
                    reward_path.is_symlink()
                    or not reward_path.is_file()
                    or episode.get("reward_sha256") != sha256_file(reward_path)
                ):
                    valid = False
                    break
            if valid:
                return index_path

    generation_dir = _reward_generation_dir(reward_dir)
    annotations = {
        str(item["run_id"]): item for item in annotation_index["episodes"]
    }
    index: list[dict[str, Any]] = []
    for episode in manifest["episodes"]:
        annotation = annotations.get(str(episode["run_id"]))
        if annotation is None:
            raise KeyError(f"RynnValue annotation is missing for {episode['run_id']}")
        annotation_path = storage_path(annotation["annotation_path"])
        expected_boundaries = np.asarray(episode["reward_boundaries"], dtype=np.int64)
        with np.load(annotation_path, allow_pickle=False) as source:
            boundaries = np.asarray(source["boundary_steps"], dtype=np.int64)
            if not np.array_equal(boundaries, expected_boundaries):
                raise ValueError(
                    f"RynnValue annotation boundaries do not match {episode['run_id']}"
                )
            official_outputs = validate_official_outputs(
                {name: source[name] for name in OFFICIAL_OUTPUT_KEYS}, len(boundaries)
            )
        absolute = official_outputs["absolute_temporal_distance_seconds"]
        if absolute.shape[1] != 1:
            raise ValueError(
                "The pinned RynnValue reward path expects one official absolute value head; "
                f"got {absolute.shape[1]}. Raw heads were not averaged."
            )
        value_lookup = {
            int(boundary): float(value)
            for boundary, value in zip(boundaries, absolute[:, 0])
        }
        done = np.zeros(int(episode["recorded_action_count"]), dtype=bool)
        if episode["terminal_step"] is not None:
            done[int(episode["terminal_step"]):] = True
        evaluation_chunks = episode.get("evaluation_chunks", episode["chunks"])
        components = [
            chunk_reward_components(
                done,
                int(chunk["start"]),
                int(chunk["length"]),
                value_lookup[int(chunk["start"])],
                value_lookup[int(chunk["end"])],
                float(derivation_cfg["gamma"]),
                float(derivation_cfg["shaping_weight"]),
                bool(derivation_cfg["rynnvalue"]),
                bool(derivation_cfg.get("zero_success_potential", False)),
            )
            for chunk in evaluation_chunks
        ]
        sparse = np.asarray([item[0] for item in components], dtype=np.float32)
        shaping = np.asarray([item[1] for item in components], dtype=np.float32)
        final = np.asarray([item[2] for item in components], dtype=np.float32)
        reward_key = stable_hash({
            "reward_schema_version": REWARD_SCHEMA_VERSION,
            "derivation_implementation_sha256": implementation,
            "run_id": episode["run_id"],
            "annotation_sha256": annotation["annotation_sha256"],
            "chunks": evaluation_chunks,
            "terminal_step": episode["terminal_step"],
            "reward_config": derivation_cfg,
        })
        output = generation_dir / f"{reward_key}.npz"
        temporary = generation_dir / f".{reward_key}.{os.getpid()}.npz"
        try:
            # The combined file preserves the existing UI/binding contract while
            # the canonical annotation cache remains reward-agnostic.
            np.savez_compressed(
                temporary,
                boundary_steps=boundaries,
                **_episode_timeline_arrays(episode),
                **official_outputs,
                sparse_reward=sparse,
                pbrs_shaping_reward=shaping,
                dense_reward=(
                    float(derivation_cfg["shaping_weight"]) * shaping
                    if bool(derivation_cfg["rynnvalue"])
                    else np.zeros_like(shaping)
                ),
                pbrs_chunk_reward=final,
                final_reward=final,
            )
            os.replace(temporary, output)
        finally:
            temporary.unlink(missing_ok=True)
        metadata = {
            "schema_version": REWARD_SCHEMA_VERSION,
            "run_id": episode["run_id"],
            "source_key": reward_key,
            **_episode_reward_metadata(episode),
            # Compatibility: older consumers use annotation_path for the file that
            # contains final rewards and official outputs together.
            "annotation_path": str(output.resolve()),
            "annotation_sha256": sha256_file(output),
            "reward_path": str(output.resolve()),
            "reward_sha256": sha256_file(output),
            "official_annotation_path": str(annotation_path.resolve()),
            "official_annotation_sha256": annotation["annotation_sha256"],
            "environment_success": episode["success"],
            "annotator": annotation.get("annotator") or annotation_index.get("annotator") or {},
            "official_outputs": annotation.get("official_outputs") or {},
            "pbrs_reward": {
                "array_keys": [
                    "sparse_reward", "pbrs_shaping_reward", "dense_reward",
                    "pbrs_chunk_reward",
                ],
                "description": "Deterministic reward cache derived without RynnValue forward",
                **derivation_cfg,
            },
        }
        index.append(metadata)
        atomic_json(index_path, {
            "schema_version": REWARD_SCHEMA_VERSION,
            "kind": "derived_iql_reward",
            "dataset_sha256": manifest["dataset_sha256"],
            "annotation_manifest_sha256": stable_hash(annotation_index),
            "annotation_config": annotation_index["annotation_config"],
            "reward_config": derivation_cfg,
            "derivation_implementation_sha256": implementation,
            "annotator": annotation_index.get("annotator") or {},
            "complete": False,
            "episodes": index,
        })
    atomic_json(index_path, {
        "schema_version": REWARD_SCHEMA_VERSION,
        "kind": "derived_iql_reward",
        "dataset_sha256": manifest["dataset_sha256"],
        "annotation_manifest_sha256": stable_hash(annotation_index),
        "annotation_config": annotation_index["annotation_config"],
        "reward_config": derivation_cfg,
        "derivation_implementation_sha256": implementation,
        "annotator": annotation_index.get("annotator") or {},
        "complete": True,
        "episodes": index,
    })
    # The top-level manifest is the CLI's mutable cache pointer. Each generation
    # also has an immutable copy suitable for explicit training version pinning.
    atomic_json(generation_dir / "reward_manifest.json", json.loads(index_path.read_text()))
    LOG.info(
        "Materialized %d reward arrays from cached RynnValue outputs (%s)",
        len(index), derivation_cfg,
    )
    return index_path


def load_reward_index(config: LoadedConfig) -> dict[str, Any]:
    """Load or cheaply rebuild rewards for the active training semantics."""
    if config.section("reward").get("manifest_path"):
        return load_pinned_reward_index(config)
    path = materialize_reward_manifest(config)
    payload = json.loads(path.read_text(encoding="utf-8"))
    if payload.get("complete") is not True:
        raise ValueError("Derived reward manifest is incomplete")
    return payload


def load_pinned_reward_index(config: LoadedConfig) -> dict[str, Any]:
    """Read fixed semantic rewards; adapt only training discount/reduction privately.

    We intentionally do not compare the current formula implementation: an old
    version denotes its saved numbers, even after code or live keyframes change.
    """
    reward = config.section("reward")
    path = Path(reward["manifest_path"])
    if path.is_symlink() or not path.is_file() or sha256_file(path) != reward["manifest_sha256"]:
        raise ValueError("Pinned reward manifest is missing or its hash does not match")
    payload = json.loads(path.read_text(encoding="utf-8"))
    manifest = load_manifest(config)
    if (payload.get("schema_version") != REWARD_SCHEMA_VERSION
            or payload.get("kind") != "derived_iql_reward" or payload.get("complete") is not True):
        raise ValueError("Pinned reward version is incomplete or unsupported")
    if payload.get("dataset_sha256") != manifest["dataset_sha256"]:
        raise ValueError("Pinned reward version belongs to a different prepared dataset")
    requested_recipe = reward_derivation_config(reward)
    stored_recipe = payload.get("reward_config")
    composed = payload.get("binding_kind") == "global_trajectory_snapshots"
    if not isinstance(stored_recipe, dict) or (not composed and {
        key: value for key, value in current_reward_settings(stored_recipe).items()
        if key not in TRAINING_REDUCTION_CONFIG_KEYS
    } != {
        key: value for key, value in requested_recipe.items() if key not in TRAINING_REDUCTION_CONFIG_KEYS
    }):
        raise ValueError("Training reward parameters conflict with the pinned reward version")
    if (not composed and reward_source(reward) == "rynnvalue"
            and payload.get("annotation_config") != official_inference_config(reward)):
        raise ValueError("Training model-evaluation parameters conflict with the pinned reward version")
    if payload.get("version_id", reward["version_id"]) != reward["version_id"]:
        raise ValueError("Pinned reward version ID does not match")
    expected = {str(item["run_id"]): item for item in manifest["episodes"]}
    entries = payload.get("episodes", [])
    if (not isinstance(entries, list) or len(entries) != len(expected)
            or {str(item.get("run_id")) for item in entries} != set(expected)):
        raise ValueError("Pinned reward members do not match the prepared dataset")
    for entry in entries:
        if composed and (not isinstance(entry.get("saved_reward_config"), dict)
                         or reward_source(entry["saved_reward_config"]) != reward_source(reward)):
            raise ValueError("Global reward binding contains a different reward source")
        episode = expected[str(entry["run_id"])]
        for path_key, hash_key in (("trajectory_path", "trajectory_sha256"),
                                   ("observations_path", "observations_sha256")):
            source_path = storage_path(episode[path_key])
            if (not source_path.is_file() or sha256_file(source_path) != episode[hash_key]):
                raise ValueError(f"Pinned reward source data changed: {episode['run_id']} ({path_key})")
        values_path = storage_path(entry["reward_path"])
        if (values_path.is_symlink() or not values_path.is_file()
                or sha256_file(values_path) != entry["reward_sha256"]):
            raise ValueError(f"Pinned reward arrays are missing or corrupted: {episode['run_id']}")
        # Legacy manifest adapters may lack this metadata, but current versions
        # record all indices so even equal-size, wrongly ordered chunks fail.
        for name, value in _episode_reward_metadata(episode).items():
            if name in entry and entry[name] != value:
                raise ValueError(f"Pinned reward {name} mismatch: {episode['run_id']}")
        with np.load(values_path, allow_pickle=False) as arrays:
            final = arrays["final_reward"] if "final_reward" in arrays else arrays["pbrs_chunk_reward"]
            chunks = episode.get("evaluation_chunks", episode["chunks"])
            if final.shape != (len(chunks),) or not np.isfinite(final).all():
                raise ValueError(f"Pinned reward chunk values are invalid: {episode['run_id']}")
            if not np.array_equal(arrays["boundary_steps"], episode["reward_boundaries"]):
                raise ValueError(f"Pinned reward boundary mismatch: {episode['run_id']}")
        if entry.get("annotation_path") != entry["reward_path"]:
            raise ValueError("Pinned reward replay alias must reference the validated reward arrays")
    snapshot_path = payload.get("stage_annotations_path")
    if snapshot_path:
        snapshot = Path(snapshot_path)
        if (snapshot.is_symlink() or not snapshot.is_file()
                or stable_hash(json.loads(snapshot.read_text())) != payload.get("stage_annotations_sha256")):
            raise ValueError("Pinned Stage keyframe snapshot is missing or corrupted")
    if stored_recipe != requested_recipe or (composed and any(
        LEGACY_REWARD_KEYS & set(entry["saved_reward_config"])
        or any(entry["saved_reward_config"].get(key) != requested_recipe.get(key)
               for key in TRAINING_REDUCTION_CONFIG_KEYS) for entry in entries)):
        return _adapt_pinned_training_rewards(config, payload, manifest, requested_recipe)
    return payload


def _adapt_pinned_training_rewards(config: LoadedConfig, pinned: dict[str, Any],
                                  prepared: dict[str, Any], recipe: dict[str, Any]) -> dict[str, Any]:
    """Change gamma or terminal potentials using saved signals, never rerun a semantic evaluator."""
    from rewards.stage.reward import stage_chunk_reward

    work = Path(config.section("paths")["work_dir"]).resolve()
    target_root = (Path(config.section("paths")["output_dir"]) / "reward_adaptations").resolve()
    protected = [work, Path(config.section("reward")["manifest_path"]).resolve().parent]
    if (work.parent / "version.json").is_file():
        protected.append(work.parent)
    if any(target_root.is_relative_to(path) or path.is_relative_to(target_root) for path in protected):
        raise ValueError("Training reward adaptation output must be outside the sealed evaluation directory")
    source = reward_source(config.section("reward"))
    implementation = reward_implementation_fingerprint(source)
    entries = {str(item["run_id"]): item for item in pinned["episodes"]}
    generated = []
    # Validate all saved semantic signals before creating an output directory.
    for episode in prepared["episodes"]:
        entry = entries[str(episode["run_id"])]
        effective = recipe
        if pinned.get("binding_kind") == "global_trajectory_snapshots":
            effective = {**current_reward_settings(entry["saved_reward_config"]), **{
                key: recipe.get(key, False) for key in TRAINING_REDUCTION_CONFIG_KEYS}}
            if not effective.get("zero_success_potential", False):
                effective.pop("zero_success_potential", None)
            if not LEGACY_REWARD_KEYS & set(entry["saved_reward_config"]) and all(
                    effective.get(key) == entry["saved_reward_config"].get(key)
                    for key in TRAINING_REDUCTION_CONFIG_KEYS):
                generated.append((entry, episode, None))
                continue
        gamma = float(effective["gamma"])
        with np.load(storage_path(entry["reward_path"]), allow_pickle=False) as values:
            arrays = {name: values[name] for name in values.files}
        chunks = episode.get("evaluation_chunks", episode["chunks"])
        done = np.zeros(int(episode["recorded_action_count"]), dtype=bool)
        if episode["terminal_step"] is not None:
            done[int(episode["terminal_step"]):] = True
        if source == "final":
            from rewards.fusion import fused_reward_arrays
            arrays.update(fused_reward_arrays(episode, effective, arrays))
            final = arrays["final_reward"]
        elif source == "stage":
            scores = arrays.get("stage_score")
            if scores is None or scores.shape != (len(done) + 1,) or not np.isfinite(scores).all():
                raise ValueError(f"Pinned Stage score timeline is unavailable: {episode['run_id']}")
            final = np.asarray([
                stage_chunk_reward(scores, int(chunk["start"]), int(chunk["length"]), gamma)
                for chunk in chunks
            ], dtype=np.float32)
            arrays["stage_chunk_reward"] = final
        elif source == "rynnvalue":
            distances = arrays.get("absolute_temporal_distance_seconds")
            boundaries = np.asarray(episode["reward_boundaries"], dtype=np.int64)
            if (distances is None or distances.shape != (len(boundaries), 1)
                    or not np.isfinite(distances).all()):
                raise ValueError(f"Pinned RynnValue temporal-distance timeline is unavailable: {episode['run_id']}")
            lookup = dict(zip(boundaries.tolist(), distances[:, 0].tolist()))
            components = np.asarray([
                chunk_reward_components(done, int(chunk["start"]), int(chunk["length"]),
                    lookup[int(chunk["start"])], lookup[int(chunk["end"])], gamma,
                    float(effective["shaping_weight"]), True,
                    bool(effective.get("zero_success_potential", False)))
                for chunk in chunks
            ], dtype=np.float32)
            arrays["sparse_reward"], arrays["pbrs_shaping_reward"], final = components.T
            arrays["dense_reward"] = float(effective["shaping_weight"]) * arrays["pbrs_shaping_reward"]
        else:
            final = np.asarray([
                sparse_primitive_return(done, int(chunk["start"]), int(chunk["length"]), gamma)
                for chunk in chunks
            ], dtype=np.float32)
            arrays["sparse_reward"] = final
        arrays.update(final_reward=final, pbrs_chunk_reward=final)
        if not np.isfinite(final).all():
            raise ValueError(f"Non-finite adapted reward: {episode['run_id']}")
        generated.append(({**entry, "training_reward_config": effective}, episode, arrays))

    directory = target_root / uuid.uuid4().hex
    directory.mkdir(parents=True, exist_ok=False)
    result = {**pinned, "reward_config": recipe, "episodes": [],
              "source_reward_manifest_sha256": config.section("reward")["manifest_sha256"],
              "source_reward_version_id": config.section("reward")["version_id"],
              "source_derivation_implementation_sha256": pinned.get("derivation_implementation_sha256"),
              "derivation_implementation_sha256": implementation,
              "training_adaptation_manifest_path": str(directory / "reward_manifest.json")}
    for entry, episode, arrays in generated:
        if arrays is None:
            result["episodes"].append(entry)
            continue
        effective = entry.get("training_reward_config", recipe)
        key = stable_hash({"source_reward_sha256": entry["reward_sha256"],
                           "run_id": episode["run_id"], "reward_config": effective,
                           "derivation_implementation_sha256": implementation})
        path = directory / f"{key}.npz"
        np.savez_compressed(path, **arrays)
        digest = sha256_file(path)
        adapted = {**entry, "source_key": key, "reward_path": str(path), "annotation_path": str(path),
                   "reward_sha256": digest, "annotation_sha256": digest}
        if "pbrs_reward" in adapted:
            adapted["pbrs_reward"] = {**adapted["pbrs_reward"], **effective,
                                     "description": "Training-local reduction of pinned model outputs"}
        result["episodes"].append(adapted)
    atomic_json(directory / "reward_manifest.json", result)
    LOG.info("Adapted pinned %s rewards for training: gamma=%s; source version %s",
             source, recipe["gamma"], result["source_reward_version_id"])
    return result


def _materialize_direct_rewards(config: LoadedConfig, *, force: bool) -> Path:
    """Sparse/Stage are independent reward sources and never need VLM evaluation."""
    from rewards.stage.reward import stage_annotation_context, stage_chunk_reward, stage_scores

    manifest = load_manifest(config)
    derivation = reward_derivation_config(config.section("reward"))
    source = derivation["source"]
    snapshot = load_stage_annotations(config, manifest) if source == "stage" else None
    snapshot_hash = stable_hash(snapshot) if snapshot is not None else None
    directory = Path(config.section("paths")["work_dir"]) / "rewards"
    directory.mkdir(parents=True, exist_ok=True)
    index_path = directory / "reward_manifest.json"
    identity = {
        "schema_version": REWARD_SCHEMA_VERSION, "kind": "derived_iql_reward",
        "dataset_sha256": manifest["dataset_sha256"], "reward_config": derivation,
        "stage_annotations_sha256": snapshot_hash,
        "derivation_implementation_sha256": reward_implementation_fingerprint(source),
    }
    if index_path.exists() and not force:
        try:
            previous = json.loads(index_path.read_text(encoding="utf-8"))
            if (all(previous.get(key) == value for key, value in identity.items())
                    and previous.get("complete") is True
                    and len(previous["episodes"]) == len(manifest["episodes"])
                    and all(not storage_path(item["reward_path"]).is_symlink()
                            and storage_path(item["reward_path"]).is_file()
                            and sha256_file(storage_path(item["reward_path"])) == item["reward_sha256"]
                            for item in previous["episodes"])):
                if snapshot is None or (
                    Path(previous["stage_annotations_path"]).is_file()
                    and json.loads(Path(previous["stage_annotations_path"]).read_text()) == snapshot
                ):
                    return index_path
        except (OSError, ValueError, KeyError, TypeError):
            pass
    generation_dir = _reward_generation_dir(directory)
    snapshot_path = generation_dir / f"stage_annotations_{snapshot_hash}.json" if snapshot else None
    if snapshot_path is not None:
        atomic_json(snapshot_path, snapshot)
    index = []
    gamma = float(derivation["gamma"])
    for episode in manifest["episodes"]:
        chunks = episode.get("evaluation_chunks", episode["chunks"])
        done = np.zeros(int(episode["recorded_action_count"]), dtype=bool)
        if episode["terminal_step"] is not None:
            done[int(episode["terminal_step"]):] = True
        if source == "stage":
            annotation = snapshot["annotations"][str(episode["run_id"])]
            with np.load(storage_path(episode["trajectory_path"]), allow_pickle=False) as trajectory:
                context = stage_annotation_context(annotation, done=trajectory["done"],
                    success_consecutive_steps=int(config.section("data")["success_consecutive_steps"]),
                    exponent=derivation["stage_exponent"])
            scores = stage_scores(context)
            final = np.asarray([stage_chunk_reward(scores, int(chunk["start"]),
                int(chunk["length"]), gamma) for chunk in chunks], dtype=np.float32)
            arrays = {"stage_score": scores, "stage_chunk_reward": final}
            annotation_hash = annotation["annotation_sha256"]
        else:
            final = np.asarray([
                sparse_primitive_return(done, int(chunk["start"]), int(chunk["length"]), gamma)
                for chunk in chunks
            ], dtype=np.float32)
            arrays = {"sparse_reward": final}
            annotation_hash = None
        key = stable_hash({**identity, "run_id": episode["run_id"], "chunks": chunks,
                           "trajectory_sha256": episode["trajectory_sha256"],
                           "annotation_sha256": annotation_hash})
        output = generation_dir / f"{key}.npz"
        temporary = generation_dir / f".{key}.{os.getpid()}.npz"
        try:
            np.savez_compressed(temporary, boundary_steps=np.asarray(episode["reward_boundaries"]),
                                **_episode_timeline_arrays(episode),
                                final_reward=final, pbrs_chunk_reward=final, **arrays)
            os.replace(temporary, output)
        finally:
            temporary.unlink(missing_ok=True)
        digest = sha256_file(output)
        index.append({"run_id": episode["run_id"], "reward_path": str(output.resolve()),
                      **_episode_reward_metadata(episode),
                      "reward_sha256": digest, "annotation_path": str(output.resolve()),
                      "annotation_sha256": digest, "stage_annotation_sha256": annotation_hash,
                      "source": source, "environment_success": episode["success"]})
    atomic_json(index_path, {**identity, "complete": True, "episodes": index,
                            "stage_annotations_path": str(snapshot_path.resolve()) if snapshot_path else None})
    atomic_json(generation_dir / "reward_manifest.json", json.loads(index_path.read_text()))
    LOG.info("Materialized %d %s reward arrays without reward-model evaluation", len(index), source)
    return index_path
