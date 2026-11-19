"""Value Flows: flow critic over returns, next chunks from the actor, best-of-N inference."""
from __future__ import annotations

import copy
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest
import torch

from core.config import LoadedConfig, load_train_config
from train.methods.base import build_algorithm, save_critic
from train.methods.critics import ValueFlowCritic, ValueFlowScorer, ValueVectorField
from train.methods.selectors import CRITIC_FILE, load_critic
from train.methods.vf import ValueFlows

QUANTILE = {"q01": [-1.0] * 8, "q99": [1.0] * 8, "codec": "openpi_quantile"}


def vf_config(tmp_path=None) -> LoadedConfig:
    return LoadedConfig(Path(tmp_path or ".") / "vf.yaml", load_train_config(method="vf", family="pi05").raw)


def transition_batch(size=3, horizon=10, bootstrap=(1.0, 0.0, 1.0)):
    torch.manual_seed(0)
    return {"pixels": torch.randint(0, 255, (size, 6, 32, 32), dtype=torch.uint8),
            "next_pixels": torch.randint(0, 255, (size, 6, 32, 32), dtype=torch.uint8),
            "proprio": torch.randn(size, 8), "next_proprio": torch.randn(size, 8),
            "actions": torch.randn(size, horizon, 7).clamp(-1, 1), "action_mask": torch.ones(size, horizon),
            "reward": -torch.rand(size), "bootstrap_mask": torch.tensor(bootstrap),
            "chunk_length": torch.full((size,), horizon), "prompt": ["task"] * size}


class FakeSampler:
    """Stands in for the π₀.₅ sampler: environment chunks per next observation."""
    action_stats = QUANTILE
    process_action = staticmethod(lambda action: np.clip(action, -1, 1))

    def __init__(self):
        self.calls = []

    def __call__(self, batch, rows, count):
        self.calls.append((list(rows), count))
        return np.random.default_rng(len(self.calls)).uniform(-1.2, 1.2, (len(rows), count, 10, 7)).astype(np.float32)


class LinearField:
    def __init__(self, slope):
        self.slope = slope

    def field(self, returns, times, condition):
        return self.slope * returns


def test_euler_integration_and_jvp_spread_follow_the_field():
    critic = ValueFlowCritic(num_flow_steps=4, clip_flow_returns=False)
    noise = torch.tensor([[0.5], [-2.0]])
    with torch.no_grad():
        returns, spread = critic.integrate(LinearField(0.8), noise, None, tangent=True)
    growth = (1 + 0.8 / 4) ** 4
    torch.testing.assert_close(returns, noise * growth)
    torch.testing.assert_close(spread, torch.full_like(noise, growth))
    half, _ = critic.integrate(LinearField(0.8), noise, None, end=torch.full_like(noise, 0.5))
    torch.testing.assert_close(half, noise * (1 + 0.8 * 0.5 / 4) ** 4)


def test_jvp_spread_matches_finite_differences_on_a_real_field():
    torch.manual_seed(1)
    critic = ValueFlowCritic(horizon=10, num_flow_steps=6, clip_flow_returns=False)
    batch = transition_batch()
    flow = critic.target_flow1
    with torch.no_grad():
        condition = flow.condition(batch["pixels"], batch["proprio"], batch["actions"], batch["action_mask"])
        noise = torch.randn(3, 1)
        _, spread = critic.integrate(flow, noise, condition, tangent=True)
        step = 1e-3
        plus, _ = critic.integrate(flow, noise + step, condition)
        minus, _ = critic.integrate(flow, noise - step, condition)
    torch.testing.assert_close(spread, (plus - minus) / (2 * step), atol=2e-3, rtol=2e-2)


def test_returns_are_clipped_to_the_discounted_reward_range():
    critic = ValueFlowCritic(discount=0.9, num_flow_steps=2)
    critic.set_reward_range(-1.0, 0.0)
    assert critic.return_bounds == pytest.approx((-10.0, 0.0))
    returns, _ = critic.integrate(LinearField(100.0), torch.tensor([[1.0], [-1.0]]), None)
    assert returns.tolist() == [[0.0], [-10.0]]
    with pytest.raises(RuntimeError, match="reward range"):
        ValueFlowCritic(horizon=10).update(transition_batch(), torch.zeros(3, 10, 7), torch.ones(3, 10, dtype=torch.bool))


def test_update_samples_next_chunks_only_where_returns_bootstrap():
    config = vf_config()
    algorithm = build_algorithm(config, torch.device("cpu"), transition_horizon=10)
    assert isinstance(algorithm, ValueFlows) and algorithm.executed == 5
    sampler = FakeSampler()
    algorithm.bind_actor_sampler(sampler)
    algorithm.bind_reward_range(-1.0, 0.0)
    before = copy.deepcopy(algorithm.agent.state_dict())
    context, metrics = algorithm.update(transition_batch(), step=0)
    assert context is None and sampler.calls == [([0, 2], 16)]
    assert {"critic_loss", "bcfm_loss", "dcfm_loss", "q_mean", "q_std", "confidence_weight_mean"} == metrics.keys()
    assert all(np.isfinite(value) for value in metrics.values())
    assert 0.5 <= metrics["confidence_weight_mean"] <= 1.5
    after = algorithm.agent.state_dict()
    assert not torch.equal(before["flow1.head.0.weight"], after["flow1.head.0.weight"])
    # Targets move by tau toward the updated online flow.
    expected = torch.lerp(before["target_flow1.head.0.weight"], after["flow1.head.0.weight"], 0.005)
    torch.testing.assert_close(after["target_flow1.head.0.weight"], expected)
    actions, mask = algorithm.next_actions(transition_batch(bootstrap=(0.0, 0.0, 0.0)))
    # Terminal transitions never bootstrap: no actor sampling at all.
    assert not actions.any() and mask.sum(1).tolist() == [5, 5, 5] and len(sampler.calls) == 1


def test_next_chunk_is_the_best_candidate_under_the_online_flows(monkeypatch):
    algorithm = build_algorithm(vf_config(), torch.device("cpu"), transition_horizon=10)
    algorithm.bind_actor_sampler(FakeSampler())
    # Score each candidate by its first executed action's x (after clipping and encoding).
    monkeypatch.setattr(algorithm.agent, "one_step_q",
                        lambda flows, conditions, noise: torch.cat([c[:, -140] for c in conditions[:1]]))
    batch = transition_batch()
    actions, _ = algorithm.next_actions(batch)
    chunks = FakeSampler()(batch, [0, 2], 16)
    for row, chunk in zip((0, 2), chunks):
        assert actions[row, 0, 0].item() == pytest.approx(np.clip(chunk[:, 0, 0], -1, 1).max(), abs=1e-5)


def test_exported_flow_critic_scores_like_the_online_flows(tmp_path):
    config = vf_config(tmp_path)
    algorithm = build_algorithm(config, torch.device("cpu"), transition_horizon=10)
    algorithm.bind_reward_range(-1.0, 0.0)
    save_critic(algorithm, config, tmp_path)
    scorer = load_critic(tmp_path / CRITIC_FILE, "cpu")
    assert isinstance(scorer, ValueFlowScorer)
    assert scorer.settings["return_bounds"] == pytest.approx([-1.0 / (1 - config.section("reward")["gamma"]), 0.0])
    assert ValueFlows.selector(config) == {"kind": "argmax", "num_samples": 16}
    batch = transition_batch()
    inputs = (batch["pixels"], batch["proprio"], batch["actions"], batch["action_mask"])
    torch.manual_seed(5)
    scores = scorer(batch)
    torch.manual_seed(5)
    flows = (algorithm.agent.flow1, algorithm.agent.flow2)
    with torch.no_grad():
        expected = algorithm.agent.one_step_q(flows, [flow.condition(*inputs) for flow in flows], torch.randn(3, 1))
    torch.testing.assert_close(scores, expected.squeeze(-1))


def test_replay_carries_next_actor_inputs_and_reward_range(configured):
    from core.data import prepare_dataset
    from rewards.rynnvalue.annotate import annotate_manifest
    from train.replay import ReplayDataset
    from test_replay import FakeAnnotator, _stats
    configured.raw["training"]["method"] = "vf"
    configured.raw["vf"] = load_train_config(method="vf", family="pi05").raw["vf"]
    prepare_dataset(configured)
    annotate_manifest(configured, FakeAnnotator())
    replay = ReplayDataset(configured, _stats(7), _stats(8), split="train")
    sample = replay[0]
    assert sample["next_agent_image"].shape == sample["agent_image"].shape
    assert sample["next_raw_proprio"].dtype == torch.float32 and sample["next_raw_proprio"].shape == (8,)
    rewards = [replay[index]["reward"].item() for index in range(len(replay))]
    assert replay.reward_range() == pytest.approx((min(rewards), max(rewards)))


def test_distributed_value_flows_match_single_device(tmp_path):
    import torch.distributed as dist
    from train.methods.distributed import DistributedAlgorithm
    dist.init_process_group("gloo", init_method=f"file://{tmp_path / 'rendezvous'}", rank=0, world_size=1)
    try:
        config = vf_config(tmp_path)
        torch.manual_seed(3)
        single = build_algorithm(config, torch.device("cpu"), transition_horizon=10)
        torch.manual_seed(3)
        parallel = DistributedAlgorithm(config, torch.device("cpu"), transition_horizon=10)
        assert isinstance(parallel.inner.agent.flow1, torch.nn.parallel.DistributedDataParallel)
        for algorithm in (single, parallel):
            algorithm.bind_actor_sampler(FakeSampler())
            algorithm.bind_reward_range(-1.0, 0.0)
        for step in range(2):
            torch.manual_seed(10 + step)
            single.update(transition_batch(), step)
            torch.manual_seed(10 + step)
            parallel.update(transition_batch(), step)
        exported, reference = parallel.export_critic(), single.export_critic()
        assert exported["state"].keys() == reference["state"].keys()
        for key, value in reference["state"].items():
            torch.testing.assert_close(exported["state"][key], value, atol=2e-5, rtol=2e-4)
        for optimizer in parallel.optimizers:
            optimizer.consolidate_state_dict(to=0)
        assert parallel.checkpoint()["model"].keys() == single.checkpoint()["model"].keys()
    finally:
        dist.destroy_process_group()
