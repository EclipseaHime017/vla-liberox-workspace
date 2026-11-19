"""Critic inputs, shared by training replay and inference-time candidate scoring.

Both sides must encode observations and action chunks identically: the two policy-view
camera images stacked on channels and resized to ``critic_image_size``, proprio and actions
normalized with the model's own statistics, and chunks padded to the transition horizon
with a mask over the executed steps.
"""
from __future__ import annotations

from typing import Any, Callable

import numpy as np
import torch
from torch.nn import functional as F


def critic_pixels(agent: np.ndarray, wrist: np.ndarray, size: int) -> torch.Tensor:
    """(H, W, 3) policy-view images -> uint8 (6, size, size)."""
    value = torch.from_numpy(np.concatenate((agent, wrist), axis=-1).copy()).permute(2, 0, 1)
    value = F.interpolate(value.unsqueeze(0).float(), size=(size, size), mode="bilinear", align_corners=False)
    return value.squeeze(0).clamp(0, 255).to(torch.uint8)


def _quantile(action_stats: dict[str, Any]) -> bool:
    return action_stats.get("codec") == "openpi_quantile"


def encode_proprio(raw: np.ndarray, action_stats: dict[str, Any], proprio_stats: dict[str, Any]) -> np.ndarray:
    if _quantile(action_stats):
        from train.models.pi05 import normalize
        return normalize(raw, proprio_stats)
    from train.models.vla_adapter import normalize_with_stats
    return normalize_with_stats(raw, proprio_stats)


def encode_actions(raw: np.ndarray, action_stats: dict[str, Any]) -> np.ndarray:
    if _quantile(action_stats):
        from train.models.pi05 import normalize
        return normalize(raw, action_stats)
    from train.models.vla_adapter import env_to_dataset_actions
    return env_to_dataset_actions(raw, action_stats)


def padded_chunk(encoded: np.ndarray, horizon: int, action_dim: int) -> tuple[np.ndarray, np.ndarray]:
    length = len(encoded)
    if not 1 <= length <= horizon:
        raise ValueError(f"Critic chunk length {length} must be in [1, {horizon}]")
    actions = np.zeros((horizon, action_dim), dtype=np.float32)
    mask = np.zeros(horizon, dtype=bool)
    actions[:length] = encoded
    mask[:length] = True
    return actions, mask


def encode_chunks(chunks: np.ndarray, executed: int, horizon: int, action_stats: dict[str, Any],
                  process_action: Callable[[np.ndarray], np.ndarray]) -> tuple[torch.Tensor, torch.Tensor]:
    """Sampled environment-space chunks (M, native_horizon, action_dim) as the critic sees them.

    Only the ``executed`` prefix counts, after the same post-processing the environment applies.
    """
    if chunks.ndim != 3 or not 1 <= executed <= chunks.shape[1]:
        raise ValueError("Chunks must be (M, horizon, action_dim) with a valid executed length")
    action_dim = chunks.shape[2]
    encoded = [padded_chunk(encode_actions(np.stack([process_action(step) for step in chunk[:executed]]),
                                           action_stats), horizon, action_dim) for chunk in chunks]
    return (torch.from_numpy(np.stack([actions for actions, _ in encoded])),
            torch.from_numpy(np.stack([mask for _, mask in encoded])))


def candidate_batch(agent: np.ndarray, wrist: np.ndarray, raw_proprio: np.ndarray,
                    candidates: np.ndarray, executed: int, *, image_size: int, horizon: int,
                    action_stats: dict[str, Any], proprio_stats: dict[str, Any],
                    process_action: Callable[[np.ndarray], np.ndarray]) -> dict[str, torch.Tensor]:
    """Score inputs for N candidate chunks (N, native_horizon, action_dim) of one observation."""
    actions, mask = encode_chunks(candidates, executed, horizon, action_stats, process_action)
    count = len(candidates)
    pixels = critic_pixels(agent, wrist, image_size)
    proprio = torch.as_tensor(np.asarray(encode_proprio(raw_proprio[None], action_stats, proprio_stats))[0])
    return {"pixels": pixels.unsqueeze(0).expand(count, -1, -1, -1),
            "proprio": proprio.unsqueeze(0).expand(count, -1), "actions": actions, "action_mask": mask}
