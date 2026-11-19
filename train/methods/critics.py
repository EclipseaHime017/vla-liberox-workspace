"""Critics: expectile Q/V (IQL-style) on a replaceable observation encoder."""
from __future__ import annotations

import copy
from dataclasses import dataclass
from typing import Any, Callable

import torch
from torch import nn
from torch.nn import functional as F
from torch.nn.parallel import DistributedDataParallel


def unwrap(module: nn.Module) -> nn.Module:
    return module.module if isinstance(module, DistributedDataParallel) else module


_OPTIMIZERS = {"adam": torch.optim.Adam, "adamw": torch.optim.AdamW}


def module_state(owner: nn.Module, names: tuple[str, ...]) -> dict[str, Any]:
    """``owner.state_dict()`` with DDP wrappers removed, so 1 GPU and N GPUs save alike."""
    return {f"{name}.{key}": value for name in names
            for key, value in unwrap(getattr(owner, name)).state_dict().items()}


def load_module_state(owner: nn.Module, names: tuple[str, ...], state: dict[str, Any]) -> None:
    prefixes = tuple(f"{name}." for name in names)
    if any(not key.startswith(prefixes) for key in state):
        raise ValueError("Critic checkpoint contains unknown modules")
    for name in names:
        prefix = f"{name}."
        unwrap(getattr(owner, name)).load_state_dict(
            {key[len(prefix):]: value for key, value in state.items() if key.startswith(prefix)}, strict=True)


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
        self.output_dim = output_dim

    def forward(self, pixels: torch.Tensor, proprio: torch.Tensor) -> torch.Tensor:
        pixels = pixels.float() / 255.0
        visual = self.vision(pixels).flatten(1)
        return self.output(torch.cat((visual, self.proprio(proprio.float())), dim=-1))


class QNetwork(nn.Module):
    def __init__(self, horizon: int = 8, action_dim: int = 7, proprio_dim: int = 8,
                 encoder_factory: Callable[[int], nn.Module] = PixelEncoder):
        super().__init__()
        self.encoder = encoder_factory(proprio_dim)
        action_size = horizon * action_dim
        self.head = nn.Sequential(
            nn.Linear(self.encoder.output_dim + action_size * 2, 512), nn.LayerNorm(512), nn.SiLU(),
            nn.Linear(512, 256), nn.SiLU(), nn.Linear(256, 1),
        )

    def forward(self, pixels: torch.Tensor, proprio: torch.Tensor,
                actions: torch.Tensor, action_mask: torch.Tensor) -> torch.Tensor:
        masked = actions.float() * action_mask.float().unsqueeze(-1)
        action_features = torch.cat((masked.flatten(1), action_mask.float().unsqueeze(-1).expand_as(actions).flatten(1)), dim=-1)
        return self.head(torch.cat((self.encoder(pixels, proprio), action_features), dim=-1)).squeeze(-1)


class ValueNetwork(nn.Module):
    def __init__(self, proprio_dim: int = 8, encoder_factory: Callable[[int], nn.Module] = PixelEncoder):
        super().__init__()
        self.encoder = encoder_factory(proprio_dim)
        self.head = nn.Sequential(nn.Linear(self.encoder.output_dim, 256), nn.SiLU(), nn.Linear(256, 1))

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
    accumulate_primitive_steps: bool = False,
) -> torch.Tensor:
    """Bellman target matching the configured chunk reward semantics."""
    if accumulate_primitive_steps:
        bootstrap_discount = discount ** chunk_length.float()
    else:
        bootstrap_discount = torch.full_like(next_value, discount, dtype=torch.float32)
    return reward.float() + bootstrap_discount * bootstrap_mask.float() * next_value


@dataclass
class CriticMetrics:
    q_loss: float
    value_loss: float
    q_mean: float
    value_mean: float
    advantage_mean: float


class ExpectileCritic(nn.Module):
    """Twin Q and an expectile V trained from dataset actions only (independent of the actor).

    ``advantage(batch)`` = min(Q1, Q2) − V. ``encoder_factory(proprio_dim)`` builds the
    observation encoder; it must expose ``output_dim``.
    """

    def __init__(self, *, horizon: int = 8, action_dim: int = 7, proprio_dim: int = 8,
                 discount: float = 0.99, expectile: float = 0.8, tau: float = 0.005,
                 critic_lr: float = 3e-4, value_lr: float = 3e-4,
                 critic_optimizer: str = "adamw", critic_weight_decay: float = 0.01,
                 value_optimizer: str = "adamw", value_weight_decay: float = 0.01,
                 critic_max_grad_norm: float = 10.0, value_max_grad_norm: float = 10.0,
                 accumulate_primitive_steps: bool = False,
                 encoder_factory: Callable[[int], nn.Module] = PixelEncoder):
        super().__init__()
        self.q1 = QNetwork(horizon, action_dim, proprio_dim, encoder_factory)
        self.q2 = QNetwork(horizon, action_dim, proprio_dim, encoder_factory)
        self.target_q1 = copy.deepcopy(self.q1).requires_grad_(False)
        self.target_q2 = copy.deepcopy(self.q2).requires_grad_(False)
        self.value = ValueNetwork(proprio_dim, encoder_factory)
        self.discount, self.expectile, self.tau = discount, expectile, tau
        self.critic_max_grad_norm = float(critic_max_grad_norm)
        self.value_max_grad_norm = float(value_max_grad_norm)
        self.accumulate_primitive_steps = bool(accumulate_primitive_steps)
        if critic_optimizer not in _OPTIMIZERS or value_optimizer not in _OPTIMIZERS:
            raise ValueError("critic_optimizer and value_optimizer must be adam or adamw")
        self.optimizer_settings = {
            "q": (_OPTIMIZERS[critic_optimizer], {"lr": critic_lr, "weight_decay": critic_weight_decay}),
            "value": (_OPTIMIZERS[value_optimizer], {"lr": value_lr, "weight_decay": value_weight_decay}),
        }
        q_class, q_kwargs = self.optimizer_settings["q"]
        value_class, value_kwargs = self.optimizer_settings["value"]
        self.q_optimizer = q_class(list(self.q1.parameters()) + list(self.q2.parameters()), **q_kwargs)
        self.value_optimizer = value_class(self.value.parameters(), **value_kwargs)

    MODULES = ("q1", "q2", "target_q1", "target_q2", "value")

    def distribute(self, wrap: Callable[[nn.Module], nn.Module], zero_optimizer: Callable) -> list:
        """Wrap the trained networks for DDP; targets stay local copies of rank 0's online Q."""
        for name in ("q1", "q2", "value"):
            setattr(self, name, wrap(getattr(self, name)))
        for online, target in ((self.q1, self.target_q1), (self.q2, self.target_q2)):
            target.load_state_dict(unwrap(online).state_dict())
        q_class, q_kwargs = self.optimizer_settings["q"]
        value_class, value_kwargs = self.optimizer_settings["value"]
        self.q_optimizer = zero_optimizer(list(self.q1.parameters()) + list(self.q2.parameters()),
                                          q_class, **q_kwargs)
        self.value_optimizer = zero_optimizer(self.value.parameters(), value_class, **value_kwargs)
        return [self.q_optimizer, self.value_optimizer]

    @torch.no_grad()
    def advantage(self, batch: dict[str, torch.Tensor], target: bool = False) -> torch.Tensor:
        """Return the policy weight advantage from the online critic by default.

        RynnValue's official ``PiIQLLearner.compute_advantages`` evaluates the
        just-updated online critic and value function.  Target critics are used
        to fit V, not to weight the policy demonstrations.
        """
        q1 = (self.target_q1 if target else self.q1)(
            batch["pixels"], batch["proprio"], batch["actions"], batch["action_mask"]
        )
        q2 = (self.target_q2 if target else self.q2)(
            batch["pixels"], batch["proprio"], batch["actions"], batch["action_mask"]
        )
        return torch.minimum(q1, q2) - self.value(batch["pixels"], batch["proprio"])

    def update(self, batch: dict[str, torch.Tensor]) -> CriticMetrics:
        # Official IQL order: (1) fit V to the frozen target-Q expectile,
        # (2) fit online Q to a Bellman target built with the newly updated V,
        # then (3) Polyak-update target Q.  Reversing the first two stages adds
        # an unintended extra lag to every Bellman update.
        with torch.no_grad():
            target_q_for_value = torch.minimum(
                self.target_q1(
                    batch["pixels"], batch["proprio"], batch["actions"],
                    batch["action_mask"],
                ),
                self.target_q2(
                    batch["pixels"], batch["proprio"], batch["actions"],
                    batch["action_mask"],
                ),
            )
        value = self.value(batch["pixels"], batch["proprio"])
        value_loss = expectile_loss(target_q_for_value - value, self.expectile)
        self.value_optimizer.zero_grad(set_to_none=True)
        value_loss.backward()
        nn.utils.clip_grad_norm_(self.value.parameters(), self.value_max_grad_norm)
        self.value_optimizer.step()

        with torch.no_grad():
            next_value = self.value(batch["next_pixels"], batch["next_proprio"])
            target = chunk_bellman_target(
                batch["reward"], next_value, batch["chunk_length"],
                batch["bootstrap_mask"], self.discount,
                self.accumulate_primitive_steps,
            )
        q1 = self.q1(batch["pixels"], batch["proprio"], batch["actions"], batch["action_mask"])
        q2 = self.q2(batch["pixels"], batch["proprio"], batch["actions"], batch["action_mask"])
        # The official ensemble loss averages over both Q heads.
        q_loss = 0.5 * (F.mse_loss(q1, target) + F.mse_loss(q2, target))
        self.q_optimizer.zero_grad(set_to_none=True)
        q_loss.backward()
        nn.utils.clip_grad_norm_(
            list(self.q1.parameters()) + list(self.q2.parameters()),
            self.critic_max_grad_norm,
        )
        self.q_optimizer.step()
        self.soft_update()
        with torch.no_grad():
            online_q = torch.minimum(
                self.q1(batch["pixels"], batch["proprio"], batch["actions"], batch["action_mask"]),
                self.q2(batch["pixels"], batch["proprio"], batch["actions"], batch["action_mask"]),
            )
            updated_value = self.value(batch["pixels"], batch["proprio"])
            advantage = online_q - updated_value
        # Same online-critic advantage as advantage(batch), without a second forward.
        self.last_advantage = advantage
        return CriticMetrics(
            float(q_loss.detach()), float(value_loss.detach()), float(online_q.mean()),
            float(updated_value.mean()), float(advantage.mean()),
        )

    @torch.no_grad()
    def soft_update(self) -> None:
        for source, target in ((self.q1, self.target_q1), (self.q2, self.target_q2)):
            for source_parameter, target_parameter in zip(source.parameters(), target.parameters()):
                target_parameter.lerp_(source_parameter, self.tau)

    def checkpoint(self) -> dict[str, Any]:
        return {
            "model": module_state(self, self.MODULES),
            "q_optimizer": self.q_optimizer.state_dict(),
            "value_optimizer": self.value_optimizer.state_dict(),
        }

    def restore(self, state: dict[str, Any]) -> None:
        load_module_state(self, self.MODULES, state["model"])
        self.q_optimizer.load_state_dict(state["q_optimizer"])
        self.value_optimizer.load_state_dict(state["value_optimizer"])


def advantage_weights(advantage: torch.Tensor, beta: float, maximum: float) -> torch.Tensor:
    return torch.exp(beta * advantage).clamp(max=maximum)
