"""Register P7 Rynn/W5 semantics in a new dataset; never modify P4 or source versions."""
from __future__ import annotations

import copy
import json
from datetime import datetime, timezone
from pathlib import Path
import shutil
import uuid

import yaml

from .training_dataset_service import TrainingDatasetService, _stable_hash


def derive_p7_dataset(dataset_manifest: Path, version_id: str, name: str) -> dict:
    """Publish only a fresh sibling dataset; original files/active pointers stay read-only.

    Dataset discovery uses */dataset.json, so publishing that file last makes this
    independent CLI visible to the existing backend without a restart or new API.
    No evaluator is constructed. Missing cached signals fail instead of a VLM fallback.
    """
    gamma = .99
    from vla_rynn_iql.config import validate_train_config
    from vla_rynn_iql.config_sources import read_source
    from vla_rynn_iql.io import atomic_json, sha256_file
    from vla_rynn_iql.rewards import load_pinned_reward_index, reward_derivation_config

    path = Path(dataset_manifest).absolute()
    if path.is_symlink() or path.parent.is_symlink() or path.name != "dataset.json":
        raise ValueError("Expected a regular frozen dataset.json")
    if not isinstance(name, str) or not 1 <= len(name.strip()) <= 100:
        raise ValueError("name must contain 1..100 characters")
    original = json.loads(path.read_text())
    if original.get("id") != path.parent.name or TrainingDatasetService._immutable_error(original):
        raise ValueError("Frozen dataset identity mismatch")
    # These public read paths use root only and never touch the database or launch jobs.
    reader = object.__new__(TrainingDatasetService)
    reader.root = path.parent.parent
    version = reader.validate_version(original["id"], version_id)
    if version.get("legacy") or version["evaluator"] != "rynnvalue":
        raise ValueError("P7 registration requires a sealed RynnValue version")
    source = version["evaluator"]
    source_paths = [path, path.parent / "annotations" / version_id / "version.json",
                    *(Path(version[key]) for key in ("config_path", "prepared_manifest_path", "reward_manifest_path"))]
    before = {str(p): sha256_file(p) for p in source_paths}
    prepared = json.loads(Path(version["prepared_manifest_path"]).read_text())
    raw = read_source(Path(version["config_path"]))
    if (prepared.get("source_dataset_sha256") != original["dataset_sha256"]
            or raw["reward"]["source"] != source):
        raise ValueError("Saved version source/dataset identity mismatch")
    members = {item["run_id"]: item for item in original["members"]}
    if (len(members) != len(original["members"]) or len(prepared["episodes"]) != len(members)
            or {e["run_id"] for e in prepared["episodes"]} != set(members)
            or any(e["split"] != members[e["run_id"]]["split"] for e in prepared["episodes"])):
        raise ValueError("Saved prepared membership/splits differ from frozen dataset")
    if (len(members) != 55
            or sum(m["split"] == "train" for m in members.values()) != 44
            or sum(m["split"] == "validation" for m in members.values()) != 11):
        raise ValueError("P7 requires the unchanged 55 members with 44 train / 11 validation")
    original_recipe = reward_derivation_config(raw["reward"])
    if original_recipe.get("shaping_weight") != .1 or original_recipe["accumulate_primitive_steps"]:
        raise ValueError("P7 requires original Rynn kappa=.1 and macro-chunk discounting")
    timestamp = datetime.now(timezone.utc).isoformat()
    dataset_id, new_version = "ds_" + uuid.uuid4().hex, "ann_" + uuid.uuid4().hex
    directory = path.parent.parent / dataset_id
    directory.mkdir(exist_ok=False)
    version_dir = directory / "annotations" / new_version
    work = version_dir / "work"
    work.mkdir(parents=True)
    child = copy.deepcopy(original)
    child.update(id=dataset_id, name=name.strip(), parent_dataset_id=original["id"],
        created_at=timestamp, updated_at=timestamp, include_post_success=False,
        evaluation_versions=[], evaluation_version_ids={}, annotation_history=[],
        pending_annotation_id=None, pending_annotation_config=None,
        annotation_id=None, reward_version_id=None, robometer_version_id=None,
        annotation_status="NOT_STARTED", annotation_config=None)
    prepared["source_dataset_id"] = dataset_id
    atomic_json(work / "dataset_manifest.json", prepared)
    raw["paths"].update(work_dir=str(work), output_dir=str(directory / "derivation"),
                        annotation_cache=str(directory / "cache"))
    raw["data"].update(selection_manifest=str(directory / "dataset.json"), include_post_success=False,
                       bootstrap_failed_end=True)
    raw["reward"].update(gamma=gamma, zero_success_potential=True, manifest_path=version["reward_manifest_path"],
                        manifest_sha256=version["reward_manifest_sha256"], version_id=version_id)
    config_path = version_dir / "effective_config.yaml"
    config = validate_train_config(raw, config_path)
    # Full source array/hash/boundary verification; only deterministic CPU reward reduction.
    adapted = load_pinned_reward_index(config)
    recipe = adapted["reward_config"]
    if recipe != {**original_recipe, "gamma": gamma, "zero_success_potential": True}:
        raise ValueError("Derivation changed parameters beyond P7 gamma / success potential")
    rewards = work / "rewards"
    rewards.mkdir()
    adapted = copy.deepcopy(adapted)
    # Freeze every array into the new version, independent of old outputs.
    for index, entry in enumerate(adapted["episodes"]):
        origin = Path(entry["reward_path"])
        target = rewards / f"{index:06d}.npz"
        shutil.copyfile(origin, target)
        if sha256_file(target) != entry["reward_sha256"]:
            raise ValueError("Derived arrays changed while copying")
        entry.update(reward_path=str(target), annotation_path=str(target),
                     annotation_sha256=entry["reward_sha256"])
    if adapted.get("stage_annotations_path"):
        source_stage = Path(adapted["stage_annotations_path"])
        stage = json.loads(source_stage.read_text())
        if _stable_hash(stage) != adapted["stage_annotations_sha256"]:
            raise ValueError("Saved Stage identity mismatch")
        target = work / "stage_annotations.json"
        atomic_json(target, stage)
        adapted["stage_annotations_path"] = str(target)
        raw["data"]["stage_annotations_manifest"] = str(target)
    adapted.update(version_id=new_version)
    adapted.pop("training_adaptation_manifest_path", None)
    reward_path = rewards / "reward_manifest.json"
    atomic_json(reward_path, adapted)
    raw["reward"].update(manifest_path=str(reward_path), manifest_sha256=sha256_file(reward_path), version_id=new_version)
    config_path.write_text(yaml.safe_dump(raw, sort_keys=False))
    parameters = {**version.get("parameters", {}), "gamma": gamma, "source": source,
                  "zero_success_potential": True, "bootstrap_failed_end": True,
                  "include_post_success": False}
    published = {"id": new_version, "dataset_id": dataset_id, "dataset_sha256": child["dataset_sha256"],
        "evaluator": source, "status": "READY", "complete": True, "parameters": parameters,
        "created_at": timestamp, "completed_at": timestamp, "error": None,
        "run_ids": list(members), "work_dir": str(work), "config_path": str(config_path),
        "config_sha256": sha256_file(config_path), "prepared_manifest_path": str(work / "dataset_manifest.json"),
        "prepared_manifest_sha256": sha256_file(work / "dataset_manifest.json"),
        "reward_manifest_path": str(reward_path), "reward_manifest_sha256": sha256_file(reward_path),
        "reward_config": recipe, "derivation_implementation_sha256": adapted.get("derivation_implementation_sha256"),
        "derivation": {"kind": "p7_rynn_w5", "source_dataset_id": original["id"],
                       "source_version_id": version_id, "source_sha256": before,
                       "source_recipe": original_recipe, "gamma": gamma,
                       "training_data": {"include_post_success": False, "bootstrap_failed_end": True},
                       "critic_budget": {"train_steps": 10000, "batch_size": 64, "seed": 7}}}
    atomic_json(version_dir / "version.json", published)
    child.update(evaluation_versions=[{key: published.get(key) for key in
        ("id", "evaluator", "status", "parameters", "created_at", "completed_at", "error")}],
        evaluation_version_ids={source: new_version}, reward_version_id=new_version,
        annotation_id=new_version, annotation_status="READY", annotation_config=parameters)
    if (TrainingDatasetService._immutable_error(child)
            or before != {str(p): sha256_file(p) for p in source_paths}):
        raise ValueError("Source dataset/version changed during derivation")
    reader.validate_version(original["id"], version_id)
    # Last write is the publication point. No old dataset/active pointer is changed.
    with (directory / "dataset.json").open("x") as stream:
        json.dump(child, stream, indent=2, ensure_ascii=False)
        stream.write("\n")
    reader.validate_version(dataset_id, new_version)
    return {"dataset_id": dataset_id, "version_id": new_version, "source": source,
            "dataset_manifest": str(directory / "dataset.json"), "version": published,
            "critic_request": {"dataset_id": dataset_id, "task_id": original["task_id"],
                "reward_source": "rynnvalue", "train_steps": 10000, "batch_size": 64,
                "seed": 7, "init_state_index": 0}}
