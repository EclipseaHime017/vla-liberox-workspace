"""Samples start at recorded re-planning points; each is the model's planned chunk, shared by
actor and critic."""
from __future__ import annotations

import json
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest
import torch

from conftest import _episode
from test_replay import _stats
from train.models import vla_adapter
from core.registry import base_model, model_contract
from core.config import load_train_config
from core.data import _load_run, add_episode_chunks, prepare_dataset
from core.execution import CHUNKING_RULE, execution_steps, query_steps
from train.replay import ActionDataset
from train.pipeline import prepare_fingerprint


def recording(configured, tmp_path, *, family="pi05", steps=5, queries=(), count=17):
    config = load_train_config(configured.path, family=family)
    root = tmp_path / "recorded"
    _episode(root, "run", action_count=count)
    path = next(root.rglob("run.json"))
    run = json.loads(path.read_text())
    run["open_loop_steps"] = steps
    path.write_text(json.dumps(run))
    trajectory = path.parent / "episodes/episode_000/trajectory.npz"
    with np.load(trajectory) as source:
        arrays = dict(source)
    arrays["inference_query_step"] = np.array(queries, dtype=np.int64)
    arrays["env_action"][:, 0] = np.arange(count) / count
    arrays["raw_action"][:, 0] = arrays["env_action"][:, 0]
    np.savez(trajectory, **arrays)
    config.raw["paths"]["dataset_sources"] = [str(root)]
    config.raw["data"]["allow_no_success"] = True
    return config, path, trajectory


def prepared(config):
    return json.loads(prepare_dataset(config).manifest.read_text())


def intervals(episode):
    return [(c["start"], c["end"]) for c in episode["chunks"]]


def test_planned_chunk_crosses_queries_and_is_shared_by_actor_and_critic(configured, tmp_path):
    config, _, _ = recording(configured, tmp_path, steps=5, queries=[0, 3, 8, 13, 17])
    manifest = prepared(config)
    assert manifest["chunking"] == CHUNKING_RULE
    assert "action_horizon" not in manifest
    assert intervals(manifest["episodes"][0]) == [(0, 3), (3, 8), (8, 13), (13, 17)]
    dataset = ActionDataset(config, _stats(7), _stats(8))
    first = dataset[0]
    assert first["actions"].shape == (10, 7)
    assert first["action_mask"].all()
    assert first["chunk_length"] == 10
    np.testing.assert_allclose(first["raw_actions"][:, 0], np.arange(10) / 17)
    assert dataset[-1]["action_mask"].sum() == 4


@pytest.mark.parametrize("supervision", [4, 8])
def test_planned_chunk_follows_the_model_horizon_not_the_recording(configured, tmp_path, supervision):
    config, _, _ = recording(configured, tmp_path, family="vla_adapter", steps=10, count=23)
    contract = model_contract(config.raw["model"])
    contract["io"]["replay_horizon"] = supervision
    config.raw["model"]["contract"] = contract
    manifest = prepared(config)
    assert intervals(manifest["episodes"][0]) == [(0, 10), (10, 20), (20, 23)]
    sample = ActionDataset(config, _stats(7), _stats(8))[0]
    assert sample["action_mask"].sum() == supervision
    assert sample["chunk_length"] == supervision


def test_layout_and_prepare_cache_do_not_depend_on_target_supervision(configured, tmp_path):
    config, _, _ = recording(configured, tmp_path, steps=7, count=23)
    first = prepared(config)
    selection = {"dataset_sha256": "frozen"}
    fingerprint = prepare_fingerprint(selection, config.raw)
    config.raw["model"]["contract"] = model_contract(config.raw["model"])
    config.raw["model"]["contract"]["io"]["native_action_horizon"] = 50
    config.raw["model"]["contract"]["io"]["replay_horizon"] = 50
    config.raw["data"]["action_horizon"] = 50
    second = prepared(config)
    assert second["dataset_sha256"] == first["dataset_sha256"]
    assert prepare_fingerprint(selection, config.raw) == fingerprint
    assert ActionDataset(config, _stats(7), _stats(8))[0]["actions"].shape == (50, 7)


def test_planned_chunk_stops_at_producer_boundary_not_at_query(configured, tmp_path):
    config, path, trajectory = recording(configured, tmp_path, steps=5, queries=[0, 5, 10, 15])
    with np.load(trajectory) as source:
        arrays = dict(source)
    arrays["metadata_json"] = np.asarray(json.dumps({"trajectory_segments": [
        dict(start=0, end=7, origin_run_id="first", open_loop_steps=5),
        dict(start=7, end=17, origin_run_id="second", open_loop_steps=5),
    ]}))
    np.savez(trajectory, **arrays)
    prepared(config)
    dataset = ActionDataset(config, _stats(7), _stats(8))
    assert dataset[0]["action_mask"].sum() == 7
    assert dataset[1]["action_mask"].sum() == 2
    assert dataset[2]["prompt"] == "do the task"
    assert dataset[2]["action_mask"].sum() == 10


def test_success_boundary_does_not_restart_policy_queries(configured, tmp_path):
    config, _, trajectory = recording(configured, tmp_path, steps=5, queries=[0, 5, 10, 15])
    with np.load(trajectory) as source:
        arrays = dict(source)
    arrays["done"][8:] = True  # confirmation after action 12
    np.savez(trajectory, **arrays)
    path = trajectory.parents[2] / "run.json"
    run = json.loads(path.read_text())
    run["success"] = True
    path.write_text(json.dumps(run))
    manifest = prepared(config)
    assert intervals(manifest["episodes"][0]) == [(0, 5), (5, 10), (10, 13), (13, 15), (15, 17)]
    dataset = ActionDataset(config, _stats(7), _stats(8))
    assert dataset[1]["action_mask"].sum() == 10
    config.raw["data"]["include_post_success"] = False
    assert ActionDataset(config, _stats(7), _stats(8))[1]["action_mask"].sum() == 8


def test_legacy_uses_source_model_default_and_rejects_unknown_source():
    base = base_model("pi05-liberox-base")
    for run in ({"policy_id": base.id}, {"base_checkpoint": base.source}):
        assert execution_steps(run, {}) == (base.inference["open_loop_steps"], "source_model_default")
    assert execution_steps({"open_loop_steps": 7}, {"open_loop_steps": 5}) == (7, "run")
    assert execution_steps({"policy_snapshot": {"inference": {"open_loop_steps": 5}}},
                           {"open_loop_steps": 7}) == (7, "trajectory")
    with pytest.raises(ValueError, match="source model"):
        execution_steps({"policy_id": "removed-model"}, {})


def test_training_geometry_uses_selected_variant_not_family_default(configured, monkeypatch):
    import copy
    from dataclasses import replace
    from core.registry import BASE_MODELS

    original = base_model("pi05-libero-base")
    contract = copy.deepcopy(original.contract)
    contract["io"].update(native_action_horizon=12, replay_horizon=12)
    monkeypatch.setitem(BASE_MODELS, "another-variant", replace(original, id="another-variant", contract=contract))
    config = load_train_config(Path(__file__).parents[2] / "configs/methods/bc.yaml",
                               overrides={"model": {"family": "pi05", "base_id": "another-variant"}})
    assert config.raw["data"]["action_horizon"] == 12


def test_legacy_saved_simulation_config(configured, tmp_path):
    config, path, _ = recording(configured, tmp_path)
    run = json.loads(path.read_text())
    run.pop("open_loop_steps")
    path.write_text(json.dumps(run))
    (path.parent / "config.yaml").write_text("simulation:\n  open_loop_steps: 7\n")
    episode = _load_run(path, config)
    assert episode["open_loop_steps"] == 7
    assert episode["execution_steps_source"] == "saved_simulation_config"


def test_legacy_child_uses_its_own_default_not_updated_base_default(tmp_path):
    import yaml
    from core.model_config import base_model_config

    settings = base_model_config("pi05-liberox-base")
    overlay = tmp_path / "policy.yaml"
    overlay.write_text(yaml.safe_dump({"schema_version": 4, "family": "pi05", "model_config": settings}))
    assert execution_steps({"policy_id": "old-child", "policy_overlay": str(overlay)}, {}) == (8, "source_child_default")


def test_old_branch_recovers_parent_cadence(configured, tmp_path):
    config, path, trajectory = recording(configured, tmp_path, steps=7, count=23)
    parent = tmp_path / "parent.npz"
    np.savez(parent, env_action=np.zeros((20, 7)), metadata_json=np.asarray(json.dumps({
        "open_loop_steps": 5, "session_id": "parent", "task": "pick"})))
    with np.load(trajectory) as source:
        arrays = dict(source)
    arrays["action_source"][13:] = "human"
    arrays["metadata_json"] = np.asarray(json.dumps({"source_trajectory": str(parent)}))
    np.savez(trajectory, **arrays)
    run = json.loads(path.read_text())
    run.update(kind="branch", resume_step=13, parent_session_id="parent")
    path.write_text(json.dumps(run))
    episode = _load_run(path, config)
    add_episode_chunks(episode)
    assert intervals(episode) == [(0, 5), (5, 10), (10, 13), (13, 20), (20, 23)]
    assert [chunk["copied_prefix"] for chunk in episode["chunks"]] == [True, True, True, False, False]


def test_query_validation_allows_unexecuted_last_proposal():
    assert query_steps(np.array([0, 5, 10]), 10) == [0, 5]
    for values in ([0., 5.], [0, 0], [0, 11], [-1], [True]):
        with pytest.raises(ValueError):
            query_steps(values, 10)


def test_old_multigeneration_prefix_keeps_each_producers_execution_steps(tmp_path):
    from core.data import _execution_segments

    root = tmp_path / "root.npz"
    np.savez(root, env_action=np.zeros((12, 7)), metadata_json=np.asarray(json.dumps({
        "open_loop_steps": 5, "session_id": "root", "task": "pick"})))
    parent = tmp_path / "parent.npz"
    np.savez(parent, env_action=np.zeros((20, 7)), metadata_json=np.asarray(json.dumps({
        "open_loop_steps": 7, "session_id": "parent", "task": "pick",
        "source_trajectory": str(root), "source_resume_step": 8})))
    segments = _execution_segments({"id": "child", "resume_step": 17, "task": "pick"},
                                  {"source_trajectory": str(parent)}, 23, 10)
    assert [(s["start"], s["end"], s["open_loop_steps"]) for s in segments] == [
        (0, 8, 5), (8, 17, 7), (17, 23, 10)]


def test_moved_legacy_prefix_uses_local_saved_source_and_never_child_cadence(tmp_path):
    from core.data import _execution_segments

    parent = tmp_path / "source_trajectory.npz"
    np.savez(parent, env_action=np.zeros((20, 7)), metadata_json=np.asarray(json.dumps({
        "open_loop_steps": 5, "session_id": "root", "task": "pick"})))
    run = {"id": "child", "task": "pick", "resume_step": 13}
    metadata = {"source_trajectory": str(tmp_path / "old-device/source.npz")}
    result = _execution_segments(run, metadata, 23, 7, trajectory_path=tmp_path / "trajectory.npz")
    assert [s["open_loop_steps"] for s in result] == [5, 7]
    parent.unlink()
    with pytest.raises(ValueError, match="source trajectory"):
        _execution_segments(run, metadata, 23, 7, trajectory_path=tmp_path / "trajectory.npz")


def test_vla_loss_uses_only_the_sample_window(monkeypatch):
    prediction = torch.ones(1, 8, 7, requires_grad=True)
    monkeypatch.setattr(vla_adapter, "predict_batch", lambda *args: prediction)
    batch = {"actions": torch.zeros(1, 4, 7), "action_mask": torch.ones(1, 4, dtype=torch.bool)}
    losses, _ = vla_adapter.actor_losses(SimpleNamespace(), batch, "cpu")
    losses.mean().backward()
    assert losses.item() == 1
    assert prediction.grad[:, :4].count_nonzero() == 28
    assert not prediction.grad[:, 4:].any()


def test_critic_and_actor_use_the_same_planned_chunk(configured):
    from train.methods.bc import BehaviorCloning
    from train.methods.iql import ImplicitQLearning

    algorithm = ImplicitQLearning(configured, torch.device("cpu"))
    horizon = configured.section("data")["action_horizon"]
    pixels = torch.zeros(1, 6, 32, 32)
    batch = {"actions": torch.ones(1, horizon, 7), "action_mask": torch.ones(1, horizon, dtype=torch.bool)}
    assert algorithm.agent.q1(pixels, torch.zeros(1, 8), batch["actions"], batch["action_mask"]).shape == (1,)
    prediction = torch.zeros(1, horizon, 7, requires_grad=True)
    assert BehaviorCloning().actor_loss(prediction, batch, None).item() == 1
    assert algorithm.actor_loss(prediction, batch, torch.ones(1)).item() == 1


def test_wide_execution_intervals_warn_without_changing_recorded_chunks(caplog):
    from core.data import transition_layout_summary

    episodes = [{"run_id": "missing-query", "chunks": [{"start": 0, "end": 200, "length": 200}]}]
    assert transition_layout_summary(episodes) == {"max_steps": 200, "p95_steps": 200.0}
    assert "Wide execution intervals" in caplog.text
    assert "missing-query" in caplog.text
    assert episodes[0]["chunks"][0]["length"] == 200
