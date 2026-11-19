"""Critic inputs, shared by training replay and inference-time candidate scoring.

Both sides must encode observations and action chunks identically: the two policy-view
camera images stacked on channels and resized to ``critic_image_size``, proprio and actions
normalized with the model's own statistics, and chunks padded to the transition horizon
with a mask over the executed steps.
"""
from __future__ import annotations

from typing import Any

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


def candidate_batch(agent: np.ndarray, wrist: np.ndarray, raw_proprio: np.ndarray,
                    candidates: np.ndarray, executed: int, *, image_size: int, horizon: int,
                    action_stats: dict[str, Any], proprio_stats: dict[str, Any]) -> dict[str, torch.Tensor]:
    """Score inputs for N candidate chunks of one observation: only the executed prefix counts.

    ``candidates`` are environment-space actions (N, native_horizon, action_dim), already
    passed through the same action post-processing the environment receives.
    """
    if candidates.ndim != 3 or not 1 <= executed <= candidates.shape[1]:
        raise ValueError("Candidates must be (N, horizon, action_dim) with a valid executed length")
    count, _, action_dim = candidates.shape
    pixels = critic_pixels(agent, wrist, image_size)
    proprio = torch.as_tensor(np.asarray(encode_proprio(raw_proprio[None], action_stats, proprio_stats))[0])
    chunks = [padded_chunk(encode_actions(candidate[:executed], action_stats), horizon, action_dim)
              for candidate in candidates]
    return {
        "pixels": pixels.unsqueeze(0).expand(count, -1, -1, -1),
        "proprio": proprio.unsqueeze(0).expand(count, -1),
        "actions": torch.from_numpy(np.stack([actions for actions, _ in chunks])),
        "action_mask": torch.from_numpy(np.stack([mask for _, mask in chunks])),
    }
