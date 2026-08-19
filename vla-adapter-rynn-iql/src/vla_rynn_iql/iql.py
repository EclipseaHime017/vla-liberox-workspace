from __future__ import annotations

import copy
from dataclasses import dataclass
from typing import Any

import torch
from torch import nn
from torch.nn import functional as F


class PixelEncoder(nn.Module):
    def __init__(self, proprio_dim: int = 8, output_dim: int = 256):
        super().__init__()
        channels = (6, 32, 64, 128, 256)
        layers: list[nn.Module] = []
        for input_channels, output_channels in zip(channels, channels[1:]):
            layers.extend((
                nn.Conv2d(input_channels, output_channels, 3, stride=2, padding=1),
                nn.GroupNorm(8, output_channels), nn.SiLU(),
            ))
        self.vision = nn.Sequential(*layers, nn.AdaptiveAvgPool2d(1))
        self.proprio = nn.Sequential(nn.Linear(proprio_dim, 64), nn.LayerNorm(64), nn.SiLU())
        self.output = nn.Sequential(nn.Linear(256 + 64, output_dim), nn.LayerNorm(output_dim), nn.SiLU())

    def forward(self, pixels: torch.Tensor, proprio: torch.Tensor) -> torch.Tensor:
        pixels = pixels.float() / 255.0
        visual = self.vision(pixels).flatten(1)
        return self.output(torch.cat((visual, self.proprio(proprio.float())), dim=-1))


class QNetwork(nn.Module):
    def __init__(self, horizon: int = 8, action_dim: int = 7, proprio_dim: int = 8):
        super().__init__()
        self.encoder = PixelEncoder(proprio_dim)
        action_size = horizon * action_dim
        self.head = nn.Sequential(
            nn.Linear(256 + action_size * 2, 512), nn.LayerNorm(512), nn.SiLU(),
            nn.Linear(512, 256), nn.SiLU(), nn.Linear(256, 1),
        )

    def forward(self, pixels: torch.Tensor, proprio: torch.Tensor,
                actions: torch.Tensor, action_mask: torch.Tensor) -> torch.Tensor:
        masked = actions.float() * action_mask.float().unsqueeze(-1)
        action_features = torch.cat((masked.flatten(1), action_mask.float().unsqueeze(-1).expand_as(actions).flatten(1)), dim=-1)
        return self.head(torch.cat((self.encoder(pixels, proprio), action_features), dim=-1)).squeeze(-1)


class ValueNetwork(nn.Module):
    def __init__(self, proprio_dim: int = 8):
        super().__init__()
        self.encoder = PixelEncoder(proprio_dim)
        self.head = nn.Sequential(nn.Linear(256, 256), nn.SiLU(), nn.Linear(256, 1))

    def forward(self, pixels: torch.Tensor, proprio: torch.Tensor) -> torch.Tensor:
        return self.head(self.encoder(pixels, proprio)).squeeze(-1)


def expectile_loss(diff: torch.Tensor, expectile: float) -> torch.Tensor:
    weight = torch.where(diff > 0, expectile, 1.0 - expectile)
    return (weight * diff.square()).mean()


def chunk_bellman_target(
    reward: torch.Tensor,
    next_value: torch.Tensor,
    chunk_length: torch.Tensor,
    bootstrap_mask: torch.Tensor,
    discount: float,
) -> torch.Tensor:
    chunk_discount = discount ** chunk_length.float()
    return reward.float() + chunk_discount * bootstrap_mask.float() * next_value


@dataclass
class IQLMetrics:
    q_loss: float
    value_loss: float
    q_mean: float
    value_mean: float
    advantage_mean: float


class PixelIQL(nn.Module):
    def __init__(self, *, horizon: int = 8, action_dim: int = 7, proprio_dim: int = 8,
                 discount: float = 0.99, expectile: float = 0.8, tau: float = 0.005,
                 critic_lr: float = 3e-4, value_lr: float = 3e-4):
        super().__init__()
        self.q1 = QNetwork(horizon, action_dim, proprio_dim)
        self.q2 = QNetwork(horizon, action_dim, proprio_dim)
        self.target_q1 = copy.deepcopy(self.q1).requires_grad_(False)
        self.target_q2 = copy.deepcopy(self.q2).requires_grad_(False)
        self.value = ValueNetwork(proprio_dim)
        self.discount, self.expectile, self.tau = discount, expectile, tau
        self.q_optimizer = torch.optim.AdamW(
            list(self.q1.parameters()) + list(self.q2.parameters()), lr=critic_lr
        )
        self.value_optimizer = torch.optim.AdamW(self.value.parameters(), lr=value_lr)

    @torch.no_grad()
    def advantage(self, batch: dict[str, torch.Tensor], target: bool = True) -> torch.Tensor:
        q1 = (self.target_q1 if target else self.q1)(
            batch["pixels"], batch["proprio"], batch["actions"], batch["action_mask"]
        )
        q2 = (self.target_q2 if target else self.q2)(
            batch["pixels"], batch["proprio"], batch["actions"], batch["action_mask"]
        )
        return torch.minimum(q1, q2) - self.value(batch["pixels"], batch["proprio"])

    def update(self, batch: dict[str, torch.Tensor]) -> IQLMetrics:
        with torch.no_grad():
            next_value = self.value(batch["next_pixels"], batch["next_proprio"])
            target = chunk_bellman_target(
                batch["reward"], next_value, batch["chunk_length"],
                batch["bootstrap_mask"], self.discount,
            )
        q1 = self.q1(batch["pixels"], batch["proprio"], batch["actions"], batch["action_mask"])
        q2 = self.q2(batch["pixels"], batch["proprio"], batch["actions"], batch["action_mask"])
        q_loss = F.mse_loss(q1, target) + F.mse_loss(q2, target)
        self.q_optimizer.zero_grad(set_to_none=True)
        q_loss.backward()
        nn.utils.clip_grad_norm_(list(self.q1.parameters()) + list(self.q2.parameters()), 10.0)
        self.q_optimizer.step()

        with torch.no_grad():
            target_q = torch.minimum(
                self.target_q1(batch["pixels"], batch["proprio"], batch["actions"], batch["action_mask"]),
                self.target_q2(batch["pixels"], batch["proprio"], batch["actions"], batch["action_mask"]),
            )
        value = self.value(batch["pixels"], batch["proprio"])
        value_loss = expectile_loss(target_q - value, self.expectile)
        self.value_optimizer.zero_grad(set_to_none=True)
        value_loss.backward()
        nn.utils.clip_grad_norm_(self.value.parameters(), 10.0)
        self.value_optimizer.step()
        self.soft_update()
        advantage = target_q - value.detach()
        return IQLMetrics(
            float(q_loss.detach()), float(value_loss.detach()), float(target_q.mean()),
            float(value.detach().mean()), float(advantage.mean()),
        )

    @torch.no_grad()
    def soft_update(self) -> None:
        for source, target in ((self.q1, self.target_q1), (self.q2, self.target_q2)):
            for source_parameter, target_parameter in zip(source.parameters(), target.parameters()):
                target_parameter.lerp_(source_parameter, self.tau)

    def checkpoint(self) -> dict[str, Any]:
        return {
            "model": self.state_dict(),
            "q_optimizer": self.q_optimizer.state_dict(),
            "value_optimizer": self.value_optimizer.state_dict(),
        }

    def restore(self, state: dict[str, Any]) -> None:
        self.load_state_dict(state["model"])
        self.q_optimizer.load_state_dict(state["q_optimizer"])
        self.value_optimizer.load_state_dict(state["value_optimizer"])


def advantage_weights(advantage: torch.Tensor, beta: float, maximum: float) -> torch.Tensor:
    return torch.exp(beta * advantage).clamp(max=maximum)


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
