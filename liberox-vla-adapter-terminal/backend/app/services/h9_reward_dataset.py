"""Seal precomputed H9 arrays as NEW versions; never evaluate or rewrite sources."""
from __future__ import annotations

import copy
import json
from datetime import datetime, timezone
from pathlib import Path
import shutil
import uuid

import yaml

from .dataset_export_rewards import checked
from .training_dataset_service import TrainingDatasetService, _stable_hash

ARMS = {"T3-2": (.7, 0.), "T3-3": (.7, .1), "T3-4": (1., .1)}


def register_h9_dataset(dataset_manifest: Path, source_version_id: str, config_path: Path,
                        config_sha256: str, prepared_sha256: str, reward_sha256: str,
                        arm: str, name: str) -> dict:
    from vla_rynn_iql.config import load_train_config, validate_train_config
    from vla_rynn_iql.io import atomic_json, sha256_file
    from vla_rynn_iql.rewards import load_pinned_reward_index, reward_derivation_config

    if arm not in ARMS or not isinstance(name, str) or not 1 <= len(name.strip()) <= 100:
        raise ValueError("Expected an H9 arm and a name of 1..100 characters")
    path = Path(dataset_manifest).absolute()
    if path.name != "dataset.json" or path.is_symlink() or path.parent.is_symlink():
        raise ValueError("Expected a regular frozen dataset.json")
    original = json.loads(path.read_text())
    if original.get("id") != path.parent.name or TrainingDatasetService._immutable_error(original):
        raise ValueError("Frozen dataset identity mismatch")
    reader = object.__new__(TrainingDatasetService)
    reader.root = path.parent.parent
    source_version = reader.validate_version(original["id"], source_version_id)
    if source_version.get("legacy") or source_version["evaluator"] != "final":
        raise ValueError("H9 requires a sealed source Final version")
    config_path = checked(config_path, config_sha256)
    config = load_train_config(config_path)
    raw = copy.deepcopy(config.raw)
    work = Path(raw["paths"]["work_dir"])
    prepared_path = checked(work / "dataset_manifest.json", prepared_sha256)
    reward_path = checked(work / "rewards/reward_manifest.json", reward_sha256)
    prepared, rewards = (json.loads(p.read_text()) for p in (prepared_path, reward_path))
    recipe = reward_derivation_config(raw["reward"])
    a, kappa = ARMS[arm]
    required = {"source": "final", "fusion_mode": "additive", "gamma": .99,
                "alpha": .3, "base_coefficient": a, "shaping_weight": kappa,
                "stage_exponent": 2, "zero_success_potential": True,
                "accumulate_primitive_steps": False}
    if (any(recipe.get(k) != v for k, v in required.items())
            or recipe.get("final_normalization", "none") != "none"
            or raw["data"].get("include_post_success", True)
            or raw["data"].get("bootstrap_failed_end", False) is not True
            or raw["iql"]["beta"] != .3):
        raise ValueError("Configuration is not the frozen H9 arm/W5 semantics")
    if rewards.get("reward_config") != recipe or rewards.get("binding_kind"):
        raise ValueError("Materialized reward recipe must exactly match H9 config")
    members = {m["run_id"]: m for m in original["members"]}
    episodes = prepared["episodes"]
    if (prepared.get("source_dataset_sha256") != original["dataset_sha256"]
            or len(members) != len(original["members"]) or len(episodes) != len(members)
            or {e["run_id"] for e in episodes} != set(members)):
        raise ValueError("Prepared source dataset/membership mismatch")
    for ep in episodes:
        member = members[ep["run_id"]]
        if ep["split"] != member["split"]:
            raise ValueError("Prepared split mismatch")
        for artifact, field in (("manifest", "source_manifest_sha256"),
                                ("trajectory", "trajectory_sha256"),
                                ("observations", "observations_sha256")):
            item = member["artifacts"][artifact]
            checked(item["path"], item["sha256"])
            if ep[field] != item["sha256"]:
                raise ValueError("Prepared recording identity mismatch")
    # Existing strict reader validates full arrays, chunk order, source artifacts and
    # Stage snapshot. Exact recipe equality prevents its adaptation/writing branch.
    raw["reward"].update(manifest_path=str(reward_path), manifest_sha256=reward_sha256,
                         version_id=rewards.get("version_id", "h9-numeric-input"))
    validated = load_pinned_reward_index(validate_train_config(raw, config_path))
    if validated != rewards or any(e.get("saved_reward_config", recipe) != recipe for e in rewards["episodes"]):
        raise ValueError("H9 numeric inputs must not request adaptation or mixed recipes")
    originals = [path, path.parent / "annotations" / source_version_id / "version.json",
                 config_path, prepared_path, reward_path,
                 *(Path(e["reward_path"]) for e in rewards["episodes"])]
    if rewards.get("stage_annotations_path"):
        originals.append(Path(rewards["stage_annotations_path"]))
    before = {str(p): sha256_file(p) for p in originals}
    if (before[str(config_path)] != config_sha256 or before[str(prepared_path)] != prepared_sha256
            or before[str(reward_path)] != reward_sha256):
        raise ValueError("Numeric inputs changed during validation")
    dataset_id, version_id = "ds_" + uuid.uuid4().hex, "ann_" + uuid.uuid4().hex
    directory = path.parent.parent / dataset_id
    directory.mkdir(exist_ok=False)
    version_dir = directory / "annotations" / version_id
    target_work = version_dir / "work"
    target_rewards = target_work / "rewards"
    target_rewards.mkdir(parents=True)
    timestamp = datetime.now(timezone.utc).isoformat()
    child = copy.deepcopy(original)
    child.update(id=dataset_id, name=name.strip(), parent_dataset_id=original["id"],
        created_at=timestamp, updated_at=timestamp, include_post_success=False,
        evaluation_versions=[], evaluation_version_ids={}, annotation_history=[],
        pending_annotation_id=None, pending_annotation_config=None,
        annotation_id=version_id, reward_version_id=version_id, robometer_version_id=None,
        annotation_status="READY")
    prepared["source_dataset_id"] = dataset_id
    atomic_json(target_work / "dataset_manifest.json", prepared)
    for number, entry in enumerate(rewards["episodes"]):
        target = target_rewards / f"{number:06d}.npz"
        shutil.copyfile(entry["reward_path"], target)
        checked(target, entry["reward_sha256"])
        entry.update(reward_path=str(target), annotation_path=str(target),
                     annotation_sha256=entry["reward_sha256"])
    if rewards.get("stage_annotations_path"):
        target = target_work / "stage_annotations.json"
        shutil.copyfile(rewards["stage_annotations_path"], target)
        if _stable_hash(json.loads(target.read_text())) != rewards["stage_annotations_sha256"]:
            raise ValueError("Stage snapshot changed")
        rewards["stage_annotations_path"] = str(target)
        raw["data"]["stage_annotations_manifest"] = str(target)
    rewards["version_id"] = version_id
    # A kappa=0 materialization legitimately has no Rynn inference config. The
    # portable snapshot contract expects a mapping, not JSON null. Normalize only
    # the new copy; source bytes and every numerical array remain unchanged.
    if rewards.get("annotation_config") is None:
        rewards["annotation_config"] = {}
    reward_path = target_rewards / "reward_manifest.json"
    atomic_json(reward_path, rewards)
    raw["paths"].update(work_dir=str(target_work), output_dir=str(directory / "training-output"),
                         annotation_cache=str(directory / "cache"))
    raw["data"]["selection_manifest"] = str(directory / "dataset.json")
    raw["reward"].update(manifest_path=str(reward_path), manifest_sha256=sha256_file(reward_path),
                         version_id=version_id)
    saved_config = version_dir / "effective_config.yaml"
    saved_config.write_text(yaml.safe_dump(raw, sort_keys=False))
    version = {"id": version_id, "dataset_id": dataset_id, "dataset_sha256": child["dataset_sha256"],
        "evaluator": "final", "status": "READY", "complete": True, "parameters": recipe,
        "created_at": timestamp, "completed_at": timestamp, "error": None,
        "run_ids": list(members), "work_dir": str(target_work),
        "reward_config": recipe, "config_path": str(saved_config), "config_sha256": sha256_file(saved_config),
        "prepared_manifest_path": str(target_work / "dataset_manifest.json"),
        "prepared_manifest_sha256": sha256_file(target_work / "dataset_manifest.json"),
        "reward_manifest_path": str(reward_path), "reward_manifest_sha256": sha256_file(reward_path),
        "derivation_implementation_sha256": rewards.get("derivation_implementation_sha256"),
        "derivation": {"kind": "h9_materialized", "arm": arm, "source_dataset_id": original["id"],
            "source_version_id": source_version_id, "source_sha256": before,
            "training_data": {"include_post_success": False, "bootstrap_failed_end": True}}}
    atomic_json(version_dir / "version.json", version)
    child.update(evaluation_versions=[{k: version.get(k) for k in
        ("id", "evaluator", "status", "parameters", "created_at", "completed_at", "error")}],
        evaluation_version_ids={"final": version_id}, annotation_config=recipe)
    if TrainingDatasetService._immutable_error(child) or before != {str(p): sha256_file(p) for p in originals}:
        raise ValueError("Source inputs changed during registration")
    reader.validate_version(original["id"], source_version_id)
    with (directory / "dataset.json").open("x") as stream:
        json.dump(child, stream, indent=2, ensure_ascii=False)
        stream.write("\n")
    reader.validate_version(dataset_id, version_id)
    return {"dataset_id": dataset_id, "version_id": version_id,
            "dataset_manifest": str(directory / "dataset.json"), "version": version}
