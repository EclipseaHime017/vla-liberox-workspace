"""Training-method interface and registry; the trainer owns actor optimization and persistence.

A method declares itself in ``core.methods.METHODS`` (torch-free: flags, critic kind,
actor capabilities, extra platform parameters) and implements ``TrainingAlgorithm`` here.
Per step the trainer calls ``update`` (critic/value learning), computes per-sample actor
losses through the model backend, and reduces them with ``weight_actor_losses``.

Optional hooks, used only when present:

- ``critic_batch_keys``: replay fields ``update`` reads (default ``CRITIC_BATCH_KEYS``);
- ``distribute(wrap, zero_optimizer)``: wrap critic modules for DDP and replace their
  optimizers with ZeRO ones; returns those optimizers.
"""
from __future__ import annotations

import importlib
from typing import Any, Protocol

import torch

from core.config import LoadedConfig
from core.methods import training_method

# Replay fields of one (s, a, r, s') chunk transition.
CRITIC_BATCH_KEYS = frozenset({"pixels", "next_pixels", "proprio", "next_proprio", "actions",
                               "action_mask", "reward", "bootstrap_mask", "chunk_length"})


class TrainingAlgorithm(Protocol):
    def update(self, batch: dict, step: int) -> tuple[Any, dict[str, float]]: ...
    def actor_loss(self, prediction: torch.Tensor, batch: dict, context: Any) -> torch.Tensor: ...
    def weight_actor_losses(self, losses: torch.Tensor, context: Any) -> torch.Tensor: ...
    def checkpoint(self) -> dict: ...
    def restore(self, state: dict) -> None: ...


def critic_batch(algorithm: Any, batch: dict, device: torch.device, extra: frozenset[str] = frozenset()) -> dict:
    """Move the fields the algorithm's critic reads to the device; non-tensors pass through."""
    keys = set(getattr(algorithm, "critic_batch_keys", CRITIC_BATCH_KEYS)) | set(extra)
    return {key: value.to(device, non_blocking=True) if isinstance(value, torch.Tensor) else value
            for key, value in batch.items() if key in keys}


# method name -> "module:Class"; imported only when selected.
IMPLEMENTATIONS = {
    "bc": "train.methods.bc:BehaviorCloning",
    "iql": "train.methods.iql:ImplicitQLearning",
}


def implementation(name: str) -> type:
    module, attribute = IMPLEMENTATIONS[name].split(":")
    return getattr(importlib.import_module(module), attribute)


def build_algorithm(config: LoadedConfig, device: torch.device, *,
                    transition_horizon: int | None = None) -> TrainingAlgorithm:
    method = training_method(config.raw)
    cls = implementation(method.name)
    if method.requires_transitions:
        return cls(config, device, transition_horizon=transition_horizon)
    return cls(config, device)


def weighted_masked_l1(
    prediction: torch.Tensor,
    target: torch.Tensor,
    mask: torch.Tensor,
    weights: torch.Tensor,
) -> torch.Tensor:
    per_element = (prediction.float() - target.float()).abs()
    valid = mask.float().unsqueeze(-1).expand_as(per_element)
    per_sample = (per_element * valid).sum(dim=(1, 2)) / valid.sum(dim=(1, 2)).clamp_min(1.0)
    return (per_sample * weights.float()).mean()
