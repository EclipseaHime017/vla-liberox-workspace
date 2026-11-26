"""Return distributions of a Value Flows policy along one recorded rollout.

At every policy query the rollout records each sampled candidate chunk, its selection score
(one-step flow Q) and the executed index. The policy's exported flow critic then samples each
candidate's return distribution Z(s, a) at that observation, with the same encoding as
inference (``train.encoding.candidate_batch``). When the critic's reward recipe needs no
reward-model signals (for example sparse rewards), the recording's realized discounted
return from each query serves as a single-sample reference.
"""
from __future__ import annotations

from typing import Any, Callable

import numpy as np
import torch

from core.data import confirmed_terminal_step
from core.image_transforms import policy_view
from train.encoding import candidate_batch
from train.models.vla_adapter import proprio_from_trajectory


def candidate_returns(scorer: torch.nn.Module, trajectory: dict[str, np.ndarray],
                      observations: dict[str, np.ndarray], selection: dict[str, np.ndarray], *,
                      stored_orientation: str, image_orientation: str, action_stats: dict[str, Any],
                      proprio_stats: dict[str, Any], process_action: Callable[[np.ndarray], np.ndarray],
                      samples: int, seed: int = 0) -> np.ndarray:
    """Return samples of every candidate at every query, shape (queries, candidates, samples)."""
    settings = scorer.settings
    device = next(scorer.parameters()).device
    generator = torch.Generator().manual_seed(seed)
    raw_proprio = proprio_from_trajectory(trajectory)
    result = []
    for step, candidates in zip(selection["query_step"], selection["candidates"]):
        step = int(step)
        agent, wrist = (policy_view(observations[key][step], stored_orientation, image_orientation)
                        for key in ("agentview_image", "wrist_image"))
        batch = candidate_batch(agent, wrist, raw_proprio[step], np.asarray(candidates),
                                image_size=settings["image_size"], horizon=settings["horizon"],
                                action_stats=action_stats, proprio_stats=proprio_stats,
                                process_action=process_action)
        batch = {key: value.to(device) for key, value in batch.items()}
        result.append(scorer.sample_returns(batch, samples, generator).float().cpu().numpy())
    return np.stack(result).astype(np.float32)


def realized_returns(done: np.ndarray, settings: dict[str, Any],
                     query_steps: np.ndarray) -> tuple[np.ndarray | None, str | None]:
    """Discounted return of the recording from each query under the critic's reward recipe.

    The recording's end is terminal, as in training replay. Returns ``(None, reason)`` when
    the recipe needs reward-model signals (RynnValue outputs or Stage keyframes) that a new
    rollout does not have.
    """
    from rewards.fusion import step_reward_arrays

    done = np.asarray(done, dtype=bool)
    episode = {"run_id": "rollout", "recorded_action_count": len(done),
               "terminal_step": confirmed_terminal_step(done, settings["success_consecutive_steps"])}
    try:
        rewards = np.asarray(step_reward_arrays(episode, settings["reward_config"], {})["final_reward"],
                             dtype=np.float64)
    except ValueError as exc:
        return None, f"该奖励配方需要奖励模型的评价，新 rollout 没有：{exc}"
    returns = np.zeros(len(rewards) + 1)
    for index in range(len(rewards) - 1, -1, -1):
        returns[index] = rewards[index] + settings["discount"] * returns[index + 1]
    return returns[np.asarray(query_steps, dtype=np.int64)].astype(np.float32), None


def histogram_edges(samples: np.ndarray, references: np.ndarray | None = None, bins: int = 48) -> np.ndarray:
    """Shared bin edges over every sample and reference, so histograms compare across time."""
    values = samples.reshape(-1)
    if references is not None:
        values = np.concatenate((values, references.reshape(-1)))
    low, high = float(values.min()), float(values.max())
    if high - low < 1e-6:
        low, high = low - 0.5, high + 0.5
    return np.linspace(low, high, bins + 1, dtype=np.float32)
