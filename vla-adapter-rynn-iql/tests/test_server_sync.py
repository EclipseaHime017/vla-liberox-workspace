import copy
import json
from types import SimpleNamespace

import numpy as np
import pytest
import torch

from test_execution_replay import recording
from test_server_inputs import global_runs
from test_replay import _stats
from vla_rynn_iql.config import LoadedConfig
from vla_rynn_iql.data import _load_run, load_manifest, prepare_dataset, validate_training_manifest
from vla_rynn_iql.io import atomic_json, sha256_file
from vla_rynn_iql.models import base_model_config
from vla_rynn_iql.replay import ReplayDataset
from vla_rynn_iql.rewards import load_reward_index
from vla_rynn_iql.server_inputs import _validate_reward_layout, prepare_global_inputs
from vla_rynn_iql.server_replay import build_replay_cache, CachedReplayDataset


@pytest.mark.parametrize("method", ["bc", "iql"])
@pytest.mark.parametrize("entrypoint", ["single", "distributed"])
@pytest.mark.parametrize("changed", ["legacy", "chunking", "action_dim", "proprio_dim"])
def test_training_entries_reject_incompatible_prepared_replay(configured, tmp_path, monkeypatch,
                                                              method, entrypoint, changed):
    from vla_rynn_iql import training, distributed_training
    config = copy.deepcopy(configured)
    config.raw["training"].update(method=method, device="cpu")
    path = prepare_dataset(config).manifest
    manifest = load_manifest(config)
    if changed == "legacy":
        manifest["schema_version"] = 4
        manifest.pop("chunking")
    elif changed == "chunking":
        manifest["chunking"] = "fixed_eight_steps"
    else:
        manifest[changed] += 1
    atomic_json(path, manifest)
    assert load_manifest(config) == manifest  # Historical manifests remain readable outside training.

    def unexpected(*args, **kwargs):
        pytest.fail("incompatible replay must be rejected before reward, model or cache loading")

    module = training if entrypoint == "single" else distributed_training
    monkeypatch.setattr(module, "load_reward_index", unexpected)
    monkeypatch.setattr(module, "model_backend", unexpected)
    monkeypatch.setattr(torch.cuda, "is_available", lambda: False)
    with pytest.raises(ValueError, match="legacy chunking|does not match the selected model"):
        if entrypoint == "single":
            training.train(config)
        else:
            monkeypatch.setattr(distributed_training.dist, "get_rank", lambda: 0)
            monkeypatch.setattr(distributed_training.dist, "get_world_size", lambda: 1)
            distributed_training._train(config, SimpleNamespace(world_size=1), tmp_path / "cache",
                                         tmp_path / "run", torch.device("cpu"), None)


def test_current_recorded_layout_is_valid_for_training(configured):
    prepare_dataset(configured)
    validate_training_manifest(load_manifest(configured), configured)


@pytest.mark.parametrize("steps", [5, 8])
def test_saved_rewards_reuse_recorded_layout_with_pi05_ten_step_actor(configured, tmp_path, steps):
    config, _, _ = recording(configured, tmp_path, family="vla_adapter", steps=steps)
    task, source, _ = global_runs(config)
    original = {path: sha256_file(path) for path in task.runs[0].path.parent.rglob("*") if path.is_file()}
    source.raw["model"] = base_model_config("pi05-libero-base")
    source.raw["data"]["action_horizon"] = 10
    raw = prepare_global_inputs(task, tmp_path / "server-inputs", source.raw, "final")
    loaded = LoadedConfig(tmp_path / "effective.yaml", raw)
    manifest = load_manifest(loaded)
    assert manifest["action_horizon"] == steps
    assert raw["data"]["action_horizon"] == 10
    assert original == {path: sha256_file(path) for path in original}
    stats, proprio = _stats(7), _stats(8)
    stats["codec"] = proprio["codec"] = "openpi_quantile"
    reward = load_reward_index(loaded)
    shared = ReplayDataset(loaded, stats, proprio, reward_index=reward)
    cached, _ = build_replay_cache(loaded, tmp_path / "cache", stats, proprio, reward)
    sample = CachedReplayDataset(cached)[0]
    assert sample["actions"].shape == (steps, 7)
    assert sample["actor_actions"].shape == (10, 7)
    assert sample["actor_action_mask"].sum() == 10
    assert sample["reward"] == shared[0]["reward"]


def test_legacy_reward_chunks_allow_only_absent_optional_provenance():
    chunk = dict(start=0, end=5, length=5, action_source="policy", interrupted=False,
                 origin_run_id="origin", policy_instruction="pick")
    episode = dict(run_id="run", reward_boundaries=[0, 5], chunks=[chunk], evaluation_chunks=[chunk])
    saved = {key: value for key, value in chunk.items() if key not in {"origin_run_id", "policy_instruction"}}
    metadata = {"entry": {"evaluation_chunks": [saved]}}
    arrays = {"boundary_steps": np.array([0, 5]), "final_reward": np.array([-1.])}
    _validate_reward_layout(metadata, arrays, episode)
    for key, value in (("end", 4), ("action_source", "human"), ("policy_instruction", "place")):
        bad = copy.deepcopy(metadata)
        bad["entry"]["evaluation_chunks"][0][key] = value
        with pytest.raises(ValueError, match="chunk semantics"):
            _validate_reward_layout(bad, arrays, episode)
    with pytest.raises(ValueError, match="Regenerate rewards on main"):
        _validate_reward_layout({}, {"boundary_steps": np.array([0, 6])}, episode)


@pytest.mark.parametrize("location", ["run", "trajectory"])
def test_camera_ablation_is_rejected_before_training(configured, tmp_path, location):
    config, path, trajectory = recording(configured, tmp_path)
    if location == "run":
        raw = json.loads(path.read_text())
        raw["disabled_policy_cameras"] = ["wrist"]
        atomic_json(path, raw)
    else:
        with np.load(trajectory, allow_pickle=False) as source:
            arrays = dict(source)
        arrays["metadata_json"] = np.asarray(json.dumps({"disabled_policy_cameras": ["wrist"]}))
        np.savez(trajectory, **arrays)
    assert _load_run(path, config) is None


def test_prepare_checks_only_image_headers(configured, tmp_path, monkeypatch):
    config, path, _ = recording(configured, tmp_path)
    original = np.load

    def guard(source, *args, **kwargs):
        assert not str(source).endswith("trajectory_observations.npz")
        return original(source, *args, **kwargs)

    monkeypatch.setattr(np, "load", guard)
    assert _load_run(path, config)["recorded_action_count"] == 17


@pytest.mark.parametrize("location", ["run", "trajectory"])
def test_handoff_without_instruction_segments_still_bounds_actions(configured, tmp_path, location):
    from vla_rynn_iql.data import prepare_dataset
    from vla_rynn_iql.replay import ActionDataset
    config, path, trajectory = recording(configured, tmp_path, steps=8)
    if location == "run":
        run = json.loads(path.read_text())
        atomic_json(path, {**run, "handoff_steps": [5]})
    else:
        with np.load(trajectory) as source:
            arrays = dict(source)
        arrays["metadata_json"] = np.asarray(json.dumps({"handoff_steps": [5]}))
        np.savez(trajectory, **arrays)
    prepare_dataset(config)
    manifest = load_manifest(config)
    assert manifest["episodes"][0]["handoff_steps"] == [5]
    assert manifest["episodes"][0]["chunks"][0]["end"] == 5
    assert ActionDataset(config, _stats(7), _stats(8))[0]["actor_action_mask"].sum() == 5
