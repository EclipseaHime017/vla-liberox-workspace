import copy

import pytest
import torch

from vla_rynn_iql.data import prepare_dataset
from vla_rynn_iql.replay import ActionDataset, ReplayDataset
from vla_rynn_iql.rewards import annotate_manifest, load_reward_index
from vla_rynn_iql.server_replay import build_replay_cache, CachedReplayDataset, DistributedBatchSampler
from test_replay import FakeAnnotator, _stats


@pytest.mark.parametrize("method", ["bc", "iql"])
@pytest.mark.parametrize("keep_tail", [False, True])
def test_cache_matches_shared_replay(configured, tmp_path, method, keep_tail):
    config = copy.deepcopy(configured)
    config.raw["training"]["method"] = method
    config.raw["data"]["include_post_success"] = keep_tail
    prepare_dataset(config)
    reward = None
    if method == "iql":
        annotate_manifest(config, FakeAnnotator())
        reward = load_reward_index(config)
        original = ReplayDataset(config, _stats(7), _stats(8), reward_index=reward)
    else:
        original = ActionDataset(config, _stats(7), _stats(8))
    path, hit = build_replay_cache(config, tmp_path / "cache", _stats(7), _stats(8), reward)
    assert not hit
    cached = CachedReplayDataset(path)
    assert len(cached) == len(original)
    for i in range(len(original)):
        a, b = original[i], cached[i]
        assert set(a) == set(b)
        for key, value in a.items():
            if isinstance(value, torch.Tensor):
                assert torch.equal(value, b[key]), key
            else:
                assert value == b[key], key
    assert build_replay_cache(config, tmp_path / "cache", _stats(7), _stats(8), reward)[1]
    damaged = path / "actions.npy"
    with damaged.open("r+b") as stream:
        stream.seek(-1, 2)
        stream.write(b"x")
    assert not build_replay_cache(config, tmp_path / "cache", _stats(7), _stats(8), reward)[1]


def test_distributed_sampling_same_global_order_and_resume():
    reference = list(DistributedBatchSampler(31, 8, 0, 1, 8, 0, 11))
    ranks = [list(DistributedBatchSampler(31, 8, rank, 2, 8, 0, 11)) for rank in range(2)]
    assert [left + right for left, right in zip(*ranks)] == reference
    assert list(DistributedBatchSampler(31, 8, 0, 1, 8, 7, 11)) == reference[7:]
