"""Behaviour cloning: equal-weight actor regression on the dataset actions."""
from __future__ import annotations

from typing import Any

import torch

from core.config import LoadedConfig
from train.methods.base import weighted_masked_l1


class BehaviorCloning:
    def __init__(self, config: LoadedConfig | None = None, device: torch.device | None = None):
        pass

    def update(self, batch: dict, step: int) -> tuple[None, dict]:
        return None, {}

    def actor_loss(self, prediction: torch.Tensor, batch: dict, context: Any) -> torch.Tensor:
        weights = torch.ones(prediction.shape[0], device=prediction.device)
        return weighted_masked_l1(prediction, batch["actions"], batch["action_mask"], weights)

    def checkpoint(self) -> dict:
        return {}

    def weight_actor_losses(self, losses: torch.Tensor, context: Any) -> torch.Tensor:
        return losses.mean()

    def restore(self, state: dict) -> None:
        if state:
            raise ValueError("BC checkpoints must not contain critic state")
