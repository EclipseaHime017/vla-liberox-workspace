"""Training-method interface and registry; the trainer owns actor optimization and persistence.

A method declares itself in ``core.methods.METHODS`` (torch-free: flags, critic kind,
actor capabilities, extra platform parameters) and implements ``TrainingAlgorithm`` here.
Per step the trainer calls ``update`` (critic/value learning), computes per-sample actor
losses through the model backend, and reduces them with ``weight_actor_losses``.

Optional hooks, used only when present:

- ``critic_batch_keys``: replay fields ``update`` reads (default ``CRITIC_BATCH_KEYS``);
- ``distribute(wrap, zero_optimizer)``: wrap critic modules for DDP and replace their
  optimizers with ZeRO ones; returns those optimizers;
- ``export_critic()``: inference-only critic payload saved as ``critic.pt`` beside each
  checkpoint, and ``selector(config)`` (static): the exported policy's ``selector`` section
  (methods declaring ``exports_critic``);
- ``bind_actor_sampler(sampler)``: the backend's sampler of action chunks at replay next
  observations, and ``bind_reward_range(low, high)``: the training replay's reward range
  (methods declaring ``actor_samples_in_training``).
"""
from __future__ import annotations

import importlib
import shutil
from pathlib import Path
from typing import Any, Protocol

import torch

from core.config import LoadedConfig
from core.methods import training_method

# Replay fields of one (s, a, r, s') chunk transition.
CRITIC_BATCH_KEYS = frozenset({"pixels", "next_pixels", "proprio", "next_proprio", "actions",
                               "action_mask", "reward", "bootstrap_mask", "chunk_length"})
# Actor inputs at s' (replay adds them for methods declaring actor_samples_in_training).
NEXT_ACTOR_KEYS = frozenset({"next_agent_image", "next_wrist_image", "next_raw_proprio", "prompt"})


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
    "ifql": "train.methods.ifql:ImplicitFlowQLearning",
    "vf": "train.methods.vf:ValueFlows",
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


def bind_training_inputs(algorithm: Any, config: LoadedConfig, backend: Any, components: Any, dataset: Any) -> None:
    """Give methods that sample the actor during training its sampler and the replay reward range."""
    if training_method(config.raw).actor_samples_in_training:
        algorithm.bind_actor_sampler(backend.next_action_sampler(components))
        algorithm.bind_reward_range(*dataset.reward_range())


def save_critic(algorithm: Any, config: LoadedConfig, directory: Path) -> None:
    """Write the inference critic beside a checkpoint's actor (methods that export one)."""
    if training_method(config.raw).exports_critic:
        from train.methods.selectors import CRITIC_FILE
        torch.save(algorithm.export_critic(), directory / CRITIC_FILE)


def critic_artifacts(config: LoadedConfig, checkpoint: Path, target: Path) -> dict[str, Any]:
    """Policy manifest fields for the checkpoint's critic and the method's selector."""
    method = training_method(config.raw)
    if not method.exports_critic:
        return {}
    from core.io import sha256_file
    from train.methods.selectors import CRITIC_FILE
    source = checkpoint / CRITIC_FILE
    if source.is_symlink() or not source.is_file():
        raise ValueError(f"{method.label} checkpoint lacks its exported critic: {source}")
    shutil.copyfile(source, target / CRITIC_FILE)
    digest = sha256_file(target / CRITIC_FILE)
    if digest != sha256_file(source):
        raise ValueError("Critic changed while exporting")
    return {"critic": CRITIC_FILE, "selector": implementation(method.name).selector(config),
            "component_sha256": {"critic": digest}}


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
