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
        return self.head(torch.cat((self.encoder(pixels, proprio), action_features(actions, action_mask)), dim=-1)).squeeze(-1)


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
) -> torch.Tensor:
    """r + gamma^L * V(s'): rewards accumulate the transition's L primitive steps."""
    return reward.float() + discount ** chunk_length.float() * bootstrap_mask.float() * next_value


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
                 encoder_factory: Callable[[int], nn.Module] = PixelEncoder):
        super().__init__()
        self.q1 = QNetwork(horizon, action_dim, proprio_dim, encoder_factory)
        self.q2 = QNetwork(horizon, action_dim, proprio_dim, encoder_factory)
        self.target_q1 = copy.deepcopy(self.q1).requires_grad_(False)
        self.target_q2 = copy.deepcopy(self.q2).requires_grad_(False)
        self.value = ValueNetwork(proprio_dim, encoder_factory)
        self.discount, self.expectile, self.tau = discount, expectile, tau
        self.geometry = {"horizon": int(horizon), "action_dim": int(action_dim), "proprio_dim": int(proprio_dim)}
        self.critic_max_grad_norm = float(critic_max_grad_norm)
        self.value_max_grad_norm = float(value_max_grad_norm)
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

    def inference_payload(self, image_size: int) -> dict[str, Any]:
        """Online Q1/Q2 (scores) and V (diagnostics) for scoring chunks at inference."""
        return {"kind": "expectile", "settings": {**self.geometry, "image_size": int(image_size)},
                "state": module_state(self, ("q1", "q2", "value"))}

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


class ExpectileScorer(nn.Module):
    """Inference view of an exported expectile critic: score = min(Q1, Q2) of the online heads."""

    def __init__(self, settings: dict[str, Any]):
        super().__init__()
        self.settings = dict(settings)
        geometry = (settings["horizon"], settings["action_dim"], settings["proprio_dim"])
        self.q1, self.q2 = QNetwork(*geometry), QNetwork(*geometry)
        self.value = ValueNetwork(settings["proprio_dim"])

    @torch.no_grad()
    def forward(self, batch: dict[str, torch.Tensor]) -> torch.Tensor:
        inputs = (batch["pixels"], batch["proprio"], batch["actions"], batch["action_mask"])
        return torch.minimum(self.q1(*inputs), self.q2(*inputs))


def action_features(actions: torch.Tensor, action_mask: torch.Tensor) -> torch.Tensor:
    """Masked chunk plus its mask, flattened: the critic sees which steps exist."""
    masked = actions.float() * action_mask.float().unsqueeze(-1)
    return torch.cat((masked.flatten(1), action_mask.float().unsqueeze(-1).expand_as(actions).flatten(1)), dim=-1)


class ValueVectorField(nn.Module):
    """Vector field v(x_t, t | s, a) of a flow from N(0, 1) to the return of chunk ``a`` at ``s``.

    The observation/chunk condition is encoded once per sample; ``returns`` and ``times`` hold
    K query points per sample, shape (B, K). The MLP mirrors the reference ``ValueVectorField``
    (Dense → GELU → LayerNorm per hidden layer).
    """

    def __init__(self, horizon: int = 8, action_dim: int = 7, proprio_dim: int = 8,
                 encoder_factory: Callable[[int], nn.Module] = PixelEncoder,
                 hidden: tuple[int, ...] = (512, 512, 512, 512)):
        super().__init__()
        self.encoder = encoder_factory(proprio_dim)
        width = self.encoder.output_dim + 2 * horizon * action_dim + 2
        layers: list[nn.Module] = []
        for size in hidden:
            layers.extend((nn.Linear(width, size), nn.GELU(approximate="tanh"), nn.LayerNorm(size)))
            width = size
        self.head = nn.Sequential(*layers, nn.Linear(width, 1))

    def condition(self, pixels, proprio, actions, action_mask) -> torch.Tensor:
        return torch.cat((self.encoder(pixels, proprio), action_features(actions, action_mask)), dim=-1)

    def field(self, returns: torch.Tensor, times: torch.Tensor, condition: torch.Tensor) -> torch.Tensor:
        points = returns.shape[1]
        inputs = torch.cat((returns.unsqueeze(-1), times.unsqueeze(-1),
                            condition.unsqueeze(1).expand(-1, points, -1)), dim=-1)
        return self.head(inputs).squeeze(-1)

    def forward(self, returns, times, pixels, proprio, actions, action_mask) -> torch.Tensor:
        return self.field(returns, times, self.condition(pixels, proprio, actions, action_mask))


def _aggregate(first: torch.Tensor, second: torch.Tensor, how: str) -> torch.Tensor:
    return torch.minimum(first, second) if how == "min" else (first + second) / 2


class ValueFlowCritic(nn.Module):
    """Value Flows' distributional critic: twin flows over the chunk return, trained with the
    bootstrapped (BCFM) and distributional (DCFM) conditional flow-matching losses, weighted by
    the confidence sigmoid(-temp / std) + 0.5 where std is the flow's d(return)/d(noise) (jvp).

    Follows chongyi-zheng/value-flows@01833354 ``ValueFlowsAgent.critic_loss``; next chunks come
    from the actor through ``select_next`` (rejection sampling with the online flows).
    """

    MODULES = ("flow1", "flow2", "target_flow1", "target_flow2")

    def __init__(self, *, horizon: int = 8, action_dim: int = 7, proprio_dim: int = 8,
                 discount: float = 0.99, tau: float = 0.005, num_flow_steps: int = 10,
                 bcfm_lambda: float = 1.0, dcfm_lambda: float = 1.0, confidence_weight_temp: float = 0.3,
                 ret_agg: str = "mean", q_agg: str = "mean", clip_flow_returns: bool = True,
                 critic_lr: float = 3e-4, critic_optimizer: str = "adam", critic_weight_decay: float = 0.0,
                 critic_max_grad_norm: float = 10.0,
                 encoder_factory: Callable[[int], nn.Module] = PixelEncoder):
        super().__init__()
        self.flow1 = ValueVectorField(horizon, action_dim, proprio_dim, encoder_factory)
        self.flow2 = ValueVectorField(horizon, action_dim, proprio_dim, encoder_factory)
        self.target_flow1 = copy.deepcopy(self.flow1).requires_grad_(False)
        self.target_flow2 = copy.deepcopy(self.flow2).requires_grad_(False)
        if critic_optimizer not in _OPTIMIZERS or ret_agg not in ("mean", "min") or q_agg not in ("mean", "min"):
            raise ValueError("Invalid flow critic optimizer or aggregation")
        self.geometry = {"horizon": int(horizon), "action_dim": int(action_dim), "proprio_dim": int(proprio_dim)}
        self.discount, self.tau, self.steps = float(discount), float(tau), int(num_flow_steps)
        self.bcfm_lambda, self.dcfm_lambda = float(bcfm_lambda), float(dcfm_lambda)
        self.temperature, self.ret_agg, self.q_agg = float(confidence_weight_temp), ret_agg, q_agg
        self.clip_flow_returns = bool(clip_flow_returns)
        self.max_grad_norm = float(critic_max_grad_norm)
        self.return_bounds: tuple[float, float] | None = None
        self.optimizer_settings = (_OPTIMIZERS[critic_optimizer],
                                   {"lr": critic_lr, "weight_decay": critic_weight_decay})
        optimizer, kwargs = self.optimizer_settings
        self.optimizer = optimizer(list(self.flow1.parameters()) + list(self.flow2.parameters()), **kwargs)

    def set_reward_range(self, low: float, high: float) -> None:
        """Clip flow returns to [low, high] / (1 - γ) like the reference (dataset reward range)."""
        if not self.clip_flow_returns:
            return
        if not self.discount < 1 or not low <= high:
            raise ValueError("Clipping flow returns needs discount < 1 and a valid reward range")
        self.return_bounds = (float(low) / (1 - self.discount), float(high) / (1 - self.discount))

    def _clip(self, returns: torch.Tensor) -> torch.Tensor:
        return returns if self.return_bounds is None else returns.clamp(*self.return_bounds)

    def integrate(self, flow: ValueVectorField, noise: torch.Tensor, condition: torch.Tensor,
                  end: torch.Tensor | None = None, *, tangent: bool = False):
        """Euler integration from t=0 to ``end`` (default 1); optionally d(return)/d(noise) via jvp."""
        returns = noise
        derivative = torch.ones_like(noise) if tangent else None
        step = (torch.ones_like(noise) if end is None else end) / self.steps
        for index in range(self.steps):
            times = index * step
            if tangent:
                field, change = torch.func.jvp(lambda value: flow.field(value, times, condition),
                                               (returns,), (derivative,))
                derivative = derivative + step * change
            else:
                field = flow.field(returns, times, condition)
            returns = self._clip(returns + step * field)
        return returns, derivative

    def one_step_q(self, flows, conditions, noise: torch.Tensor) -> torch.Tensor:
        """Q sample from one Euler step at t=0 (the reference's scoring), aggregated over flows."""
        values = [self._clip(noise + flow.field(noise, torch.zeros_like(noise), condition))
                  for flow, condition in zip(flows, conditions)]
        return _aggregate(*values, self.q_agg)

    @torch.no_grad()
    def select_next(self, pixels, proprio, candidates: torch.Tensor, masks: torch.Tensor) -> torch.Tensor:
        """Index of the best of N encoded candidate chunks per next observation (online flows)."""
        rows, count = candidates.shape[:2]
        flows = (unwrap(self.flow1), unwrap(self.flow2))
        inputs = (pixels.repeat_interleave(count, 0), proprio.repeat_interleave(count, 0),
                  candidates.flatten(0, 1), masks.flatten(0, 1))
        conditions = [flow.condition(*inputs) for flow in flows]
        noise = torch.randn(rows * count, 1, device=pixels.device)
        return self.one_step_q(flows, conditions, noise).view(rows, count).argmax(dim=1)

    def update(self, batch: dict[str, torch.Tensor], next_actions: torch.Tensor,
               next_mask: torch.Tensor) -> dict[str, float]:
        if self.clip_flow_returns and self.return_bounds is None:
            raise RuntimeError("Bind the dataset reward range before training a clipped flow critic")
        here = (batch["pixels"], batch["proprio"], batch["actions"], batch["action_mask"])
        there = (batch["next_pixels"], batch["next_proprio"], next_actions, next_mask)
        reward = batch["reward"].float().unsqueeze(-1)
        discount = (self.discount ** batch["chunk_length"].float()
                    * batch["bootstrap_mask"].float()).unsqueeze(-1)
        targets = (self.target_flow1, self.target_flow2)
        with torch.no_grad():
            target_here = [flow.condition(*here) for flow in targets]
            target_there = [flow.condition(*there) for flow in targets]
            spread = torch.randn_like(reward)
            stds = [self.integrate(flow, spread, condition, tangent=True)[1].abs()
                    for flow, condition in zip(targets, target_here)]
            std = _aggregate(*stds, self.q_agg)
            weights = (torch.sigmoid(-self.temperature / std) + 0.5).squeeze(-1)
            noise, times = torch.randn_like(reward), torch.rand_like(reward)
            next_returns = _aggregate(*[self.integrate(flow, noise, condition)[0]
                                        for flow, condition in zip(targets, target_there)], self.ret_agg)
            returns = reward + discount * next_returns
            bootstrapped = times * returns + (1 - times) * noise
            noisy_next = _aggregate(*[self.integrate(flow, noise, condition, end=times)[0]
                                      for flow, condition in zip(targets, target_there)], self.ret_agg)
            distributional = reward + discount * noisy_next
            target_field = _aggregate(*[flow.field(noisy_next, times, condition)
                                        for flow, condition in zip(targets, target_there)], self.ret_agg)
            goal = torch.cat((returns - noise, target_field), dim=1)
        # One forward per online flow (DDP-safe): column 0 is BCFM, column 1 is DCFM.
        points, point_times = torch.cat((bootstrapped, distributional), dim=1), torch.cat((times, times), dim=1)
        errors = sum((flow(points, point_times, *here) - goal).square() for flow in (self.flow1, self.flow2))
        bcfm, dcfm = errors[:, 0], errors[:, 1]
        loss = (weights * (self.bcfm_lambda * bcfm + self.dcfm_lambda * dcfm)).mean()
        self.optimizer.zero_grad(set_to_none=True)
        loss.backward()
        nn.utils.clip_grad_norm_(list(self.flow1.parameters()) + list(self.flow2.parameters()), self.max_grad_norm)
        self.optimizer.step()
        self.soft_update()
        with torch.no_grad():
            online = (unwrap(self.flow1), unwrap(self.flow2))
            q = self.one_step_q(online, [flow.condition(*here) for flow in online], torch.randn_like(reward))
        return {"critic_loss": float(loss.detach()), "bcfm_loss": float(bcfm.mean().detach()),
                "dcfm_loss": float(dcfm.mean().detach()), "q_mean": float(q.mean()),
                "q_std": float(std.mean()), "confidence_weight_mean": float(weights.mean())}

    @torch.no_grad()
    def soft_update(self) -> None:
        for source, target in ((self.flow1, self.target_flow1), (self.flow2, self.target_flow2)):
            for source_parameter, target_parameter in zip(source.parameters(), target.parameters()):
                target_parameter.lerp_(source_parameter, self.tau)

    def distribute(self, wrap: Callable[[nn.Module], nn.Module], zero_optimizer: Callable) -> list:
        for name in ("flow1", "flow2"):
            setattr(self, name, wrap(getattr(self, name)))
        for online, target in ((self.flow1, self.target_flow1), (self.flow2, self.target_flow2)):
            target.load_state_dict(unwrap(online).state_dict())
        optimizer, kwargs = self.optimizer_settings
        self.optimizer = zero_optimizer(list(self.flow1.parameters()) + list(self.flow2.parameters()),
                                        optimizer, **kwargs)
        return [self.optimizer]

    def inference_payload(self, image_size: int) -> dict[str, Any]:
        bounds = None if self.return_bounds is None else list(self.return_bounds)
        return {"kind": "value_flow",
                "settings": {**self.geometry, "image_size": int(image_size), "q_agg": self.q_agg,
                             "return_bounds": bounds},
                "state": module_state(self, ("flow1", "flow2"))}

    def checkpoint(self) -> dict[str, Any]:
        return {"model": module_state(self, self.MODULES), "optimizer": self.optimizer.state_dict()}

    def restore(self, state: dict[str, Any]) -> None:
        load_module_state(self, self.MODULES, state["model"])
        self.optimizer.load_state_dict(state["optimizer"])


class ValueFlowScorer(nn.Module):
    """Inference view of an exported flow critic: one-step Q sample per candidate, as in ``'rs'``."""

    def __init__(self, settings: dict[str, Any]):
        super().__init__()
        self.settings = dict(settings)
        geometry = (settings["horizon"], settings["action_dim"], settings["proprio_dim"])
        self.flow1, self.flow2 = ValueVectorField(*geometry), ValueVectorField(*geometry)
        bounds = settings["return_bounds"]
        self.bounds = None if bounds is None else tuple(bounds)

    @torch.no_grad()
    def forward(self, batch: dict[str, torch.Tensor]) -> torch.Tensor:
        inputs = (batch["pixels"], batch["proprio"], batch["actions"], batch["action_mask"])
        noise = torch.randn(len(batch["pixels"]), 1, device=batch["pixels"].device)
        values = []
        for flow in (self.flow1, self.flow2):
            value = noise + flow.field(noise, torch.zeros_like(noise), flow.condition(*inputs))
            values.append(value if self.bounds is None else value.clamp(*self.bounds))
        return _aggregate(*values, self.settings["q_agg"]).squeeze(-1)


def advantage_weights(advantage: torch.Tensor, beta: float, maximum: float) -> torch.Tensor:
    return torch.exp(beta * advantage).clamp(max=maximum)
