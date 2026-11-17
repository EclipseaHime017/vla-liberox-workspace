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
@pytest.mark.parametrize("family", ["vla_adapter", "pi05"])
def test_cache_matches_shared_replay(configured, tmp_path, method, keep_tail, family):
    from vla_rynn_iql.models import model_config
    config = copy.deepcopy(configured)
    config.raw["model"] = model_config({"model": {"family": family}})
    action_stats, proprio_stats = _stats(7), _stats(8)
    if family == "pi05":
        for stats in (action_stats, proprio_stats):
            stats["codec"] = "openpi_quantile"
    config.raw["training"]["method"] = method
    config.raw["data"]["include_post_success"] = keep_tail
    prepare_dataset(config)
    reward = None
    if method == "iql":
        annotate_manifest(config, FakeAnnotator())
        reward = load_reward_index(config)
        original = ReplayDataset(config, action_stats, proprio_stats, reward_index=reward)
    else:
        original = ActionDataset(config, action_stats, proprio_stats)
    path, hit = build_replay_cache(config, tmp_path / "cache", action_stats, proprio_stats, reward)
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
    assert build_replay_cache(config, tmp_path / "cache", action_stats, proprio_stats, reward)[1]
    damaged = path / "actions.npy"
    with damaged.open("r+b") as stream:
        stream.seek(-1, 2)
        stream.write(b"x")
    assert not build_replay_cache(config, tmp_path / "cache", action_stats, proprio_stats, reward)[1]


def test_distributed_sampling_same_global_order_and_resume():
    reference = list(DistributedBatchSampler(31, 8, 0, 1, 8, 0, 11))
    ranks = [list(DistributedBatchSampler(31, 8, rank, 2, 8, 0, 11)) for rank in range(2)]
    assert [left + right for left, right in zip(*ranks)] == reference
    assert list(DistributedBatchSampler(31, 8, 0, 1, 8, 7, 11)) == reference[7:]


def test_instruction_grouped_sampling_is_global_and_resume_stable():
    groups = {"pick": [0, 2, 4], "place": [1, 3, 5, 6]}
    reference = list(DistributedBatchSampler(7, 8, 0, 1, 8, 0, 11, groups))
    ranks = [list(DistributedBatchSampler(7, 8, rank, 2, 8, 0, 11, groups)) for rank in range(2)]
    assert [left + right for left, right in zip(*ranks)] == reference
    assert list(DistributedBatchSampler(7, 8, 0, 1, 8, 7, 11, groups)) == reference[7:]
    assert all(any(set(batch) <= set(group) for group in groups.values()) for batch in reference)
    sampler = DistributedBatchSampler(7, 8, 0, 1, 8, 0, 11, groups)
    generator = torch.Generator().manual_seed(8)
    assert [sampler.draw_batch(generator).tolist() for _ in range(11)] == reference
    plain = list(DistributedBatchSampler(7, 1, 0, 1, 8, 0, 11))
    assert list(DistributedBatchSampler(7, 1, 0, 1, 8, 0, 11, groups)) == plain
