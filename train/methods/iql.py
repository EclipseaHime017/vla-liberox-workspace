"""Implicit Q-learning: BC weighted by exp(β·advantage) from an expectile critic."""
from __future__ import annotations

from dataclasses import asdict
from typing import Any

import torch

from core.config import LoadedConfig
from train.methods.base import weighted_masked_l1
from train.methods.critics import ExpectileCritic, advantage_weights


class ImplicitQLearning:
    def __init__(self, config: LoadedConfig, device: torch.device, *, transition_horizon: int | None = None):
        cfg, reward, data = config.section("iql"), config.section("reward"), config.section("data")
        self.cfg = cfg
        self.agent = ExpectileCritic(
            horizon=transition_horizon or data["action_horizon"], action_dim=data["action_dim"],
            proprio_dim=data["proprio_dim"], discount=reward["gamma"],
            expectile=cfg["expectile"], tau=cfg["target_tau"],
            critic_lr=cfg["critic_lr"], value_lr=cfg["value_lr"],
            **{key: cfg[key] for key in (
                "critic_optimizer", "critic_weight_decay", "value_optimizer", "value_weight_decay",
                "critic_max_grad_norm", "value_max_grad_norm",
            )},
            accumulate_primitive_steps=reward["accumulate_primitive_steps"],
        ).to(device)

    def update(self, batch: dict, step: int) -> tuple[torch.Tensor, dict[str, float]]:
        metrics = asdict(self.agent.update(batch))
        if step < self.cfg["critic_warmup_steps"]:
            weights = torch.ones(batch["actions"].shape[0], device=batch["actions"].device)
        else:
            weights = advantage_weights(
                self.agent.last_advantage, self.cfg["beta"], self.cfg["max_advantage_weight"],
            )
        metrics["advantage_weight_mean"] = float(weights.mean())
        return weights, metrics

    def actor_loss(self, prediction: torch.Tensor, batch: dict, context: Any) -> torch.Tensor:
        return weighted_masked_l1(prediction, batch.get("actor_actions", batch["actions"]),
                                  batch.get("actor_action_mask", batch["action_mask"]), context)

    def distribute(self, wrap, zero_optimizer) -> list:
        return self.agent.distribute(wrap, zero_optimizer)

    def checkpoint(self) -> dict:
        return self.agent.checkpoint()

    def weight_actor_losses(self, losses: torch.Tensor, context: Any) -> torch.Tensor:
        if context.shape != losses.shape:
            raise ValueError("Actor losses and IQL weights must have one value per sample")
        return (losses * context.detach().float()).mean()

    def restore(self, state: dict) -> None:
        self.agent.restore(state)
