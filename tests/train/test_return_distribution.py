"""Value Flows return distributions: exported semantics, sampling, encoding and realized returns."""
from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest
import torch

from core.config import LoadedConfig, load_train_config
from core.image_transforms import policy_view
from rewards.manifest import reward_derivation_config
from train.encoding import candidate_batch
from train.methods.base import build_algorithm, save_critic
from train.methods.critics import _aggregate
from train.methods.selectors import CRITIC_FILE, load_critic
from train.return_distribution import candidate_returns, histogram_edges, realized_returns

QUANTILE = {"q01": [-1.0] * 8, "q99": [1.0] * 8, "codec": "openpi_quantile"}


def exported_flow_critic(tmp_path: Path, **reward):
    raw = load_train_config(method="vf", family="pi05").raw
    raw["reward"].update(reward)
    config = LoadedConfig(tmp_path / "vf.yaml", raw)
    algorithm = build_algorithm(config, torch.device("cpu"))
    algorithm.bind_reward_range(-1.0, 0.0)
    save_critic(algorithm, config, tmp_path)
    return config, algorithm, load_critic(tmp_path / CRITIC_FILE, "cpu")


def test_exported_flow_critic_declares_the_returns_it_models(tmp_path):
    config, _, scorer = exported_flow_critic(tmp_path, source="sparse")
    settings = scorer.settings
    assert settings["discount"] == config.section("reward")["gamma"]
    assert settings["num_flow_steps"] == config.section("vf")["num_flow_steps"]
    assert settings["ret_agg"] == config.section("vf")["ret_agg"]
    assert settings["reward_config"] == reward_derivation_config(config.section("reward"))
    assert settings["success_consecutive_steps"] == config.section("data")["success_consecutive_steps"]


def test_sampled_returns_integrate_both_online_flows_like_training(tmp_path):
    _, algorithm, scorer = exported_flow_critic(tmp_path)
    torch.manual_seed(0)
    batch = {"pixels": torch.randint(0, 255, (3, 6, 32, 32), dtype=torch.uint8), "proprio": torch.randn(3, 8),
             "actions": torch.randn(3, 10, 7).clamp(-1, 1), "action_mask": torch.ones(3, 10, dtype=torch.bool)}
    samples = scorer.sample_returns(batch, 64, torch.Generator().manual_seed(4))
    assert samples.shape == (3, 64)
    noise = torch.randn(3, 64, generator=torch.Generator().manual_seed(4))
    agent = algorithm.agent
    inputs = (batch["pixels"], batch["proprio"], batch["actions"], batch["action_mask"])
    with torch.no_grad():
        expected = _aggregate(*[agent.integrate(flow, noise, flow.condition(*inputs))[0]
                                for flow in (agent.flow1, agent.flow2)], agent.ret_agg)
    torch.testing.assert_close(samples, expected, atol=1e-5, rtol=1e-5)
    low, high = agent.return_bounds
    assert samples.min() >= low and samples.max() <= high


class RecordingScorer(torch.nn.Module):
    """Returns the first candidate action's x as every sample; records what it was shown."""

    def __init__(self):
        super().__init__()
        self.anchor = torch.nn.Parameter(torch.zeros(()))
        self.settings = {"horizon": 10, "image_size": 16}
        self.seen = []

    @torch.no_grad()
    def sample_returns(self, batch, count, generator=None):
        self.seen.append(batch)
        return batch["actions"][:, 0, :1].expand(-1, count) + self.anchor


def test_candidates_are_encoded_like_inference_at_each_query():
    rng = np.random.default_rng(0)
    trajectory = {"eef_position": rng.normal(size=(13, 3)), "eef_axis_angle": rng.normal(size=(13, 3)),
                  "gripper_qpos": rng.normal(size=(13, 2))}
    observations = {key: rng.integers(0, 255, (13, 24, 24, 3), dtype=np.uint8)
                    for key in ("agentview_image", "wrist_image")}
    candidates = rng.uniform(-1.5, 1.5, (2, 3, 10, 7)).astype(np.float32)
    selection = {"query_step": np.asarray([0, 10]), "candidates": candidates}
    scorer = RecordingScorer()
    clip = lambda action: np.clip(action, -1, 1)
    returns = candidate_returns(scorer, trajectory, observations, selection, stored_orientation="libero_raw",
                                image_orientation="rotate_180", action_stats=QUANTILE, proprio_stats=QUANTILE,
                                process_action=clip, samples=5)
    assert returns.shape == (2, 3, 5) and returns.dtype == np.float32
    np.testing.assert_allclose(returns[1, :, 0], np.clip(candidates[1, :, 0, 0], -1, 1), atol=1e-6)
    expected = candidate_batch(policy_view(observations["agentview_image"][10], "libero_raw", "rotate_180"),
                               policy_view(observations["wrist_image"][10], "libero_raw", "rotate_180"),
                               np.concatenate([trajectory[key][10] for key in ("eef_position", "eef_axis_angle",
                                                                                "gripper_qpos")]).astype(np.float32),
                               candidates[1], image_size=16, horizon=10, action_stats=QUANTILE,
                               proprio_stats=QUANTILE, process_action=clip)
    for key, value in expected.items():
        torch.testing.assert_close(scorer.seen[1][key], value)


def sparse_settings(**changes):
    recipe = reward_derivation_config({**load_train_config(method="vf", family="pi05").raw["reward"],
                                       "source": "sparse", "gamma": 0.5})
    return {"reward_config": recipe, "discount": 0.5, "success_consecutive_steps": 2, **changes}


def test_realized_sparse_return_counts_steps_to_confirmed_success():
    done = np.asarray([False, False, True, True, True, False])
    values, reason = realized_returns(done, sparse_settings(), np.asarray([0, 2, 3, 5]))
    assert reason is None
    # Success is confirmed at action 3: rewards -1, -1, -1, 0, 0, 0.
    np.testing.assert_allclose(values, [-1 - 0.5 - 0.25, -1.0, 0.0, 0.0])


def test_realized_return_is_unavailable_when_rewards_need_a_reward_model():
    recipe = reward_derivation_config({**load_train_config(method="vf", family="pi05").raw["reward"],
                                       "source": "rynnvalue"})
    values, reason = realized_returns(np.zeros(4, dtype=bool), sparse_settings(reward_config=recipe), np.asarray([0]))
    assert values is None and "评价" in reason


def test_histogram_edges_are_shared_and_never_degenerate():
    edges = histogram_edges(np.full((2, 3, 4), -2.0, dtype=np.float32), bins=8)
    assert len(edges) == 9 and edges[0] < -2.0 < edges[-1]
    edges = histogram_edges(np.zeros((1, 1, 2)), np.asarray([-5.0]), bins=4)
    assert edges[0] == -5.0 and edges[-1] == 0.0
    with pytest.raises(ValueError):
        histogram_edges(np.zeros((1, 1, 0)))
