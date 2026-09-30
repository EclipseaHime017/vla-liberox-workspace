import copy
import json
import shutil
from pathlib import Path

import numpy as np
import pytest
import yaml

from vla_rynn_iql.config import LoadedConfig
from vla_rynn_iql.data import load_manifest, prepare_dataset
from vla_rynn_iql.io import atomic_json, sha256_file, stable_hash
from vla_rynn_iql.portable_dataset import export_bundle, materialize_bundle, read_bundle
from vla_rynn_iql.rewards import annotate_manifest, load_reward_index
from vla_rynn_iql.replay import ReplayDataset
from test_replay import FakeAnnotator, _stats


def make_bundle(configured, tmp_path):
    config = copy.deepcopy(configured)
    config.raw["reward"].update(source="final", alpha=0.0, final_normalization="initial_chunk_v1")
    prepared = prepare_dataset(config).manifest
    annotate_manifest(config, FakeAnnotator())
    rewards = load_reward_index(config)
    index = tmp_path / "source-reward.json"
    atomic_json(index, rewards)
    recipe = tmp_path / "source-config.yaml"
    recipe.write_text(yaml.safe_dump(config.raw))
    members = []
    for episode in load_manifest(config)["episodes"]:
        artifacts = {name: {"path": episode[field], "sha256": sha256_file(Path(episode[field])),
                            "size": Path(episode[field]).stat().st_size} for name, field in (
                            ("manifest", "source_manifest"), ("trajectory", "trajectory_path"),
                            ("observations", "observations_path"))}
        members.append({"run_id": episode["run_id"], "artifacts": artifacts, "split": episode["split"]})
    frozen = {"id": "ds_test", "name": "test", "task_id": "LEVEL1::task", "members": members,
              "validation_fraction": .2, "split_seed": 7, "success_consecutive_steps": 5,
              "include_post_success": False, "dataset_sha256": "original-frozen-identity"}
    selection = tmp_path / "frozen/dataset.json"
    atomic_json(selection, frozen)
    version = {"config_path": str(recipe), "prepared_manifest_path": str(prepared),
               "reward_manifest_path": str(index), "id": "evaluation-1",
               "config_sha256": sha256_file(recipe), "prepared_manifest_sha256": sha256_file(prepared),
               "reward_manifest_sha256": sha256_file(index)}
    target = export_bundle(tmp_path / "export/task/ds_test", frozen, selection, prepared, {"final": version})
    return target, config, rewards


def test_moved_bundle_trains_without_originals_and_keeps_arrays(configured, tmp_path):
    bundle, original_config, original_rewards = make_bundle(configured, tmp_path)
    arrays = {entry["run_id"]: Path(entry["reward_path"]).read_bytes() for entry in original_rewards["episodes"]}
    moved = tmp_path / "server/copied"
    moved.parent.mkdir()
    shutil.copytree(bundle, moved)
    # Original source files become inaccessible, emulating a different server.
    Path(configured.section("paths")["dataset_sources"][0]).rename(tmp_path / "offline-recordings")
    Path(configured.section("paths")["work_dir"]).rename(tmp_path / "offline-work")
    before = {p: sha256_file(p) for p in moved.rglob("*") if p.is_file()}
    raw = materialize_bundle(moved, tmp_path / "runtime", original_config.raw, "final")
    config = LoadedConfig(tmp_path / "effective.yaml", raw)
    rewards = load_reward_index(config)
    for entry in rewards["episodes"]:
        assert Path(entry["reward_path"]).read_bytes() == arrays[entry["run_id"]]
    assert len(ReplayDataset(config, _stats(7), _stats(8), reward_index=rewards)) > 0
    # Training can change gamma from saved semantic signals without evaluator access.
    raw["reward"]["gamma"] = .92
    adapted = load_reward_index(config)
    assert adapted["reward_config"]["gamma"] == .92
    assert before == {p: sha256_file(p) for p in moved.rglob("*") if p.is_file()}


def test_bundle_rejects_corruption_and_traversal(configured, tmp_path):
    bundle, _, _ = make_bundle(configured, tmp_path)
    descriptor = bundle / "training_bundle.json"
    original = json.loads(descriptor.read_text())
    edited = copy.deepcopy(original)
    edited["files"]["../escape"] = {"size": 0, "sha256": "x"}
    edited["bundle_sha256"] = stable_hash({key: value for key, value in edited.items() if key != "bundle_sha256"})
    atomic_json(descriptor, edited)
    with pytest.raises(ValueError, match="Unsafe"):
        read_bundle(bundle, verify=True)
    atomic_json(descriptor, original)
    values = next(bundle.rglob("trajectory.npz"))
    with values.open("r+b") as stream:
        stream.seek(-1, 2)
        stream.write(b"x")
    with pytest.raises(ValueError, match="hash mismatch"):
        read_bundle(bundle, verify=True)


def test_execution_cannot_write_into_bundle(configured, tmp_path):
    bundle, config, _ = make_bundle(configured, tmp_path)
    with pytest.raises(ValueError, match="outside"):
        materialize_bundle(bundle, bundle / "inputs", config.raw, "final")
    assert not (bundle / "inputs").exists()
