"""Value Flows: a flow-matching critic over the full return distribution.

Reference: Zheng et al., "Value Flows" (arXiv:2510.07650),
https://github.com/chongyi-zheng/value-flows/blob/01833354f547ad842bdaa21d2b761a16069f9724/agents/value_flows.py
(MIT). Offline setting with rejection-sampling policy extraction (``'rs'``):

- Critic (``ValueFlowCritic``): twin flows trained with the BCFM and DCFM losses, weighted by
  the flow's return uncertainty; next chunks at s' are sampled from the current actor and the
  best of ``train_num_samples`` by the online flows is used (rejection sampling).
- Actor: the behaviour flow policy with an equal-weight flow-matching (BC) loss.
- Inference: sample ``num_samples`` chunks and execute the one with the highest one-step
  flow Q (``train.methods.selectors``).

Differences from the reference: action chunks with one discount per chunk (γ^L when rewards
accumulate primitive steps); the critic's pixel encoder; the one-step distilled actor and its
Q loss serve only the reference's online ``rpg`` extraction and are not ported; next chunks
are scored over the variant's default executed steps.
"""
from __future__ import annotations

from typing import Any

import torch

from core.config import LoadedConfig
from core.registry import execution_config
from train.methods.base import CRITIC_BATCH_KEYS, NEXT_ACTOR_KEYS, weighted_masked_l1
from train.encoding import encode_chunks
from train.methods.critics import ValueFlowCritic
from train.methods.selectors import CRITIC_SCHEMA_VERSION


class ValueFlows:
    critic_batch_keys = CRITIC_BATCH_KEYS | NEXT_ACTOR_KEYS

    def __init__(self, config: LoadedConfig, device: torch.device, *, transition_horizon: int | None = None):
        cfg, reward, data = config.section("vf"), config.section("reward"), config.section("data")
        self.cfg = cfg
        self.horizon, self.action_dim = transition_horizon or data["action_horizon"], data["action_dim"]
        self.agent = ValueFlowCritic(
            horizon=self.horizon, action_dim=self.action_dim, proprio_dim=data["proprio_dim"],
            discount=reward["gamma"], tau=cfg["target_tau"], num_flow_steps=cfg["num_flow_steps"],
            bcfm_lambda=cfg["bcfm_lambda"], dcfm_lambda=cfg["dcfm_lambda"],
            confidence_weight_temp=cfg["confidence_weight_temp"], ret_agg=cfg["ret_agg"], q_agg=cfg["q_agg"],
            clip_flow_returns=cfg["clip_flow_returns"], critic_lr=cfg["critic_lr"],
            critic_optimizer=cfg["critic_optimizer"], critic_weight_decay=cfg["critic_weight_decay"],
            critic_max_grad_norm=cfg["critic_max_grad_norm"],
            accumulate_primitive_steps=reward["accumulate_primitive_steps"],
        ).to(device)
        # Sampled next chunks execute as the trained child would by default.
        self.executed = min(execution_config(config.section("model"), trained=True)["open_loop_steps"],
                            self.horizon)
        self.sampler = None

    def bind_actor_sampler(self, sampler) -> None:
        """``sampler(batch, rows, count)`` -> (rows, count, native_horizon, action_dim) environment
        chunks at the next observations; it also provides ``action_stats`` and ``process_action``."""
        self.sampler = sampler

    def bind_reward_range(self, low: float, high: float) -> None:
        self.agent.set_reward_range(low, high)

    def next_actions(self, batch: dict) -> tuple[torch.Tensor, torch.Tensor]:
        """Best of ``train_num_samples`` actor chunks at each bootstrapped next observation."""
        size, device = len(batch["reward"]), batch["actions"].device
        actions = torch.zeros(size, self.horizon, self.action_dim, device=device)
        mask = torch.zeros(size, self.horizon, dtype=torch.bool, device=device)
        mask[:, :self.executed] = True
        rows = torch.nonzero(batch["bootstrap_mask"] > 0).flatten()
        if len(rows) == 0:  # terminal transitions never bootstrap
            return actions, mask
        if self.sampler is None:
            raise RuntimeError("Value Flows needs the actor sampler before training")
        count = int(self.cfg["train_num_samples"])
        chunks = self.sampler(batch, rows.tolist(), count)
        encoded, masks = encode_chunks(chunks.reshape(-1, *chunks.shape[2:]), self.executed, self.horizon,
                                       self.sampler.action_stats, self.sampler.process_action)
        encoded = encoded.to(device).view(len(rows), count, self.horizon, self.action_dim)
        masks = masks.to(device).view(len(rows), count, self.horizon)
        best = self.agent.select_next(batch["next_pixels"][rows], batch["next_proprio"][rows], encoded, masks)
        actions[rows] = encoded[torch.arange(len(rows), device=device), best]
        return actions, mask

    def update(self, batch: dict, step: int) -> tuple[None, dict[str, float]]:
        # The critic never weights the actor: it only ranks sampled chunks at inference.
        next_actions, next_mask = self.next_actions(batch)
        return None, self.agent.update(batch, next_actions, next_mask)

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
        return {"kind": "argmax", "num_samples": int(config.section("vf")["num_samples"])}
