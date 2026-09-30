"""Method-specific objectives; the runner owns actor optimization and persistence."""
from __future__ import annotations

from dataclasses import asdict
from typing import Any, Protocol

import torch

from .config import LoadedConfig
from .iql import PixelIQL, advantage_weights, weighted_masked_l1
from .methods import training_method


class TrainingAlgorithm(Protocol):
    def update(self, batch: dict, step: int) -> tuple[Any, dict[str, float]]: ...
    def actor_loss(self, prediction: torch.Tensor, batch: dict, context: Any) -> torch.Tensor: ...
    def checkpoint(self) -> dict: ...
    def restore(self, state: dict) -> None: ...


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

    def restore(self, state: dict) -> None:
        if state:
            raise ValueError("BC checkpoints must not contain critic state")


class ImplicitQLearning:
    def __init__(self, config: LoadedConfig, device: torch.device):
        cfg, reward, data = config.section("iql"), config.section("reward"), config.section("data")
        self.cfg = cfg
        self.agent = PixelIQL(
            horizon=data["action_horizon"], action_dim=data["action_dim"],
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
                self.agent.advantage(batch), self.cfg["beta"], self.cfg["max_advantage_weight"],
            )
        metrics["advantage_weight_mean"] = float(weights.mean())
        return weights, metrics

    def actor_loss(self, prediction: torch.Tensor, batch: dict, context: Any) -> torch.Tensor:
        return weighted_masked_l1(prediction, batch["actions"], batch["action_mask"], context)

    def checkpoint(self) -> dict:
        return self.agent.checkpoint()

    def restore(self, state: dict) -> None:
        self.agent.restore(state)


ALGORITHMS = {"bc": BehaviorCloning, "iql": ImplicitQLearning}


def build_algorithm(config: LoadedConfig, device: torch.device) -> TrainingAlgorithm:
    name = training_method(config.raw).name
    return ALGORITHMS[name](config, device)
