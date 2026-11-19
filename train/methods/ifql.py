"""Implicit flow Q-learning (IFQL): the flow-policy variant of IDQL.

Reference: Park, Li, Levine, "Flow Q-Learning" (2025), IFQL baseline,
https://github.com/seohongpark/fql/blob/e8cd16eb490332924dfa2492097219f181765933/agents/ifql.py
(MIT; the copy in https://github.com/chongyi-zheng/value-flows at 01833354 is the same).
IDQL: Hansen-Estruch et al., arXiv:2304.10573.

- Critic: IQL expectile twin Q and V on dataset actions only. V fits the expectile of the
  target min-Q; Q fits r + γ·V(s'); target Q is Polyak-averaged.
- Actor: the behaviour flow policy with an equal-weight flow-matching (BC) loss.
- Inference: sample ``num_samples`` chunks from the actor and execute the one with the
  highest online min(Q1, Q2) (``train.methods.selectors``).

Differences from the reference: action chunks with one discount per chunk; the critic's
pixel encoder; V and Q take sequential steps with separate optimizers (the repository's
IQL critic order) instead of one joint gradient step.
"""
from __future__ import annotations

from dataclasses import asdict
from typing import Any

import torch

from core.config import LoadedConfig
from train.methods.base import weighted_masked_l1
from train.methods.critics import ExpectileCritic
from train.methods.selectors import CRITIC_SCHEMA_VERSION


class ImplicitFlowQLearning:
    def __init__(self, config: LoadedConfig, device: torch.device, *, transition_horizon: int | None = None):
        cfg, reward, data = config.section("ifql"), config.section("reward"), config.section("data")
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

    def update(self, batch: dict, step: int) -> tuple[None, dict[str, float]]:
        # The critic never weights the actor: it only ranks sampled chunks at inference.
        return None, asdict(self.agent.update(batch))

    def actor_loss(self, prediction: torch.Tensor, batch: dict, context: Any) -> torch.Tensor:
        weights = torch.ones(prediction.shape[0], device=prediction.device)
        return weighted_masked_l1(prediction, batch.get("actor_actions", batch["actions"]),
                                  batch.get("actor_action_mask", batch["action_mask"]), weights)

    def weight_actor_losses(self, losses: torch.Tensor, context: Any) -> torch.Tensor:
        return losses.mean()

    def distribute(self, wrap, zero_optimizer) -> list:
        return self.agent.distribute(wrap, zero_optimizer)

    def checkpoint(self) -> dict:
        return self.agent.checkpoint()

    def restore(self, state: dict) -> None:
        self.agent.restore(state)

    def export_critic(self) -> dict[str, Any]:
        return {"schema_version": CRITIC_SCHEMA_VERSION,
                **self.agent.inference_payload(self.cfg["critic_image_size"])}

    @staticmethod
    def selector(config: LoadedConfig) -> dict[str, Any]:
        return {"kind": "argmax", "num_samples": int(config.section("ifql")["num_samples"])}
