import copy
from dataclasses import replace
from pathlib import Path

import numpy as np
import pytest
import torch

from vla_rynn_iql import base_models
from vla_rynn_iql.data import prepare_dataset
from vla_rynn_iql.image_transforms import canonical_contract
from vla_rynn_iql.io import stable_hash
from vla_rynn_iql.model_artifacts import decode_child, parent_identity, parent_snapshot
from vla_rynn_iql.models import base_model_config
from vla_rynn_iql.replay import ActionDataset, ReplayDataset
from vla_rynn_iql.rewards import annotate_manifest, load_reward_index
from vla_rynn_iql.server_replay import build_replay_cache, CachedReplayDataset
from test_replay import FakeAnnotator, _stats


@pytest.mark.parametrize("family", ["vla_adapter", "pi05"])
@pytest.mark.parametrize("method", ["bc", "iql"])
@pytest.mark.parametrize("vertical,horizontal", [(False, False), (True, False), (False, True), (True, True)])
def test_actor_critic_and_mmap_share_variant_pixels(configured, tmp_path, family, method, vertical, horizontal):
    config = copy.deepcopy(configured)
    settings = base_model_config("base" if family == "vla_adapter" else "pi05-liberox-base")
    contract = base_models.model_contract(settings)
    contract["observation"] = {"image_flip_vertical": vertical, "image_flip_horizontal": horizontal}
    config.raw["model"] = {**settings, "contract": contract}
    config.raw["training"]["method"] = method
    config.raw["iql"]["critic_image_size"] = 16
    frames = {}
    for path in Path(config.raw["paths"]["dataset_sources"][0]).rglob("trajectory_observations.npz"):
        with np.load(path) as source:
            count = len(source["agentview_image"])
        agent = (np.arange(count, dtype=np.uint8)[:, None, None, None]
                 + np.arange(768, dtype=np.uint8).reshape(1, 16, 16, 3))
        wrist = agent + np.uint8(71)
        np.savez_compressed(path, agentview_image=agent, wrist_image=wrist)
        frames[str(path)] = agent, wrist
    prepare_dataset(config)
    action_stats, proprio_stats = _stats(7), _stats(8)
    if family == "pi05":
        action_stats["codec"] = proprio_stats["codec"] = "openpi_quantile"
    reward = None
    if method == "iql":
        annotate_manifest(config, FakeAnnotator())
        reward = load_reward_index(config)
        replay = ReplayDataset(config, action_stats, proprio_stats, reward_index=reward)
    else:
        replay = ActionDataset(config, action_stats, proprio_stats)
    path, _ = build_replay_cache(config, tmp_path / "cache", action_stats, proprio_stats, reward)
    cached = CachedReplayDataset(path)
    v, h = -1 if vertical else 1, -1 if horizontal else 1
    for index in range(len(replay)):
        sample, item = replay[index], cached[index]
        episode, chunk_index, _ = replay.items[index]
        chunk = replay.chunks[episode["run_id"]][chunk_index]
        agent, wrist = frames[episode["observations_path"]]
        expected = [array[chunk["start"], ::v, ::h].copy() for array in (agent, wrist)]
        np.testing.assert_array_equal(sample["agent_image"], expected[0])
        np.testing.assert_array_equal(sample["wrist_image"], expected[1])
        if method == "iql":
            pixels = torch.from_numpy(np.concatenate(expected, axis=-1)).permute(2, 0, 1)
            next_pixels = torch.from_numpy(np.concatenate(
                [array[chunk["end"], ::v, ::h] for array in (agent, wrist)], axis=-1).copy()).permute(2, 0, 1)
            assert torch.equal(sample["pixels"], pixels)
            assert torch.equal(sample["next_pixels"], next_pixels)
        for key, value in sample.items():
            assert torch.equal(value, item[key]) if isinstance(value, torch.Tensor) else value == item[key]
    assert "image_transforms.py" in cached.metadata["identity"]["implementation"]
    config.raw["model"]["contract"]["observation"]["image_flip_vertical"] = not vertical
    changed, hit = build_replay_cache(config, tmp_path / "cache", action_stats, proprio_stats, reward)
    assert changed != path and not hit


@pytest.mark.parametrize("base_id", ["base", "pi05-libero-base"])
def test_legacy_parent_encoding_is_equivalent_but_changed_view_is_not(base_id):
    settings = base_model_config(base_id)
    current = parent_snapshot(settings, "a" * 64)
    legacy = copy.deepcopy(current)
    legacy["contract"].pop("observation")
    legacy["contract"]["io"]["image_orientation"] = "rotate_180"
    legacy["sha256"] = stable_hash({k: v for k, v in legacy.items() if k != "sha256"})
    assert parent_identity(current) == parent_identity(legacy)
    changed = copy.deepcopy(current)
    changed["contract"]["observation"]["image_flip_horizontal"] = False
    changed["sha256"] = stable_hash({k: v for k, v in changed.items() if k != "sha256"})
    assert parent_identity(current) != parent_identity(changed)
    raw = {"schema_version": 5, "family": settings["family"], "model_config": settings,
           "parent": legacy, "base_checkpoint": settings["base_checkpoint"], "stats_key": settings["stats_key"],
           "action_horizon": current["contract"]["io"]["replay_horizon"], "action_dim": 7, "proprio_dim": 8}
    decoded, saved_parent = decode_child(raw)
    original = base_models.BASE_MODELS[base_id]
    replacement = copy.deepcopy(original.contract)
    replacement["observation"]["image_flip_vertical"] = False
    base_models.BASE_MODELS[base_id] = replace(original, contract=replacement)
    try:
        assert base_models.model_observation(decoded["model_config"]) == {
            "image_flip_vertical": True, "image_flip_horizontal": True}
        assert saved_parent == legacy
        assert canonical_contract(legacy["contract"]) == canonical_contract(current["contract"])
    finally:
        base_models.BASE_MODELS[base_id] = original


@pytest.mark.parametrize("base_id", ["pi05-liberox-base", "vla-adapter-pro-liberox-base"])
def test_liberox_variants_only_flip_vertically(base_id):
    assert base_models.model_observation(base_model_config(base_id)) == {
        "image_flip_vertical": True, "image_flip_horizontal": False}


def test_resume_rejects_changed_view_before_restoring_weights(configured, tmp_path):
    import json
    from types import SimpleNamespace
    from vla_rynn_iql.models import model_config
    from vla_rynn_iql.training import _restore_checkpoint
    settings = model_config(configured.raw)
    original = parent_snapshot(settings, "a" * 64)
    changed = copy.deepcopy(original["contract"])
    changed["observation"]["image_flip_vertical"] = False
    configured.raw["model"]["contract"] = changed
    for parent in (original, None):
        (tmp_path / "checkpoint.json").write_text(json.dumps({"algorithm": "iql", "model_config": settings,
            "model_parent": parent, "model_identity": None}))
        components = SimpleNamespace(identity=None, parent=parent_snapshot(configured.raw["model"], "a" * 64))
        with pytest.raises(ValueError, match="parent model|input contract"):
            _restore_checkpoint(tmp_path, components, None, None, None, torch.device("cpu"), configured, {}, None)
