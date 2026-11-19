"""Inference-time action selection for methods that export a critic (IFQL, Value Flows).

The actor samples N action chunks for one observation, the exported critic scores them
(``critic.pt``, loaded with ``load_critic``) and a selector picks the chunk to execute.
Methods whose actor is used directly (BC, IQL) export neither. Selector names are declared
torch-free in ``core.methods.SELECTOR_KINDS``; the implementations live here.
"""
from __future__ import annotations

from pathlib import Path
from typing import Any, Callable, Protocol

import numpy as np
import torch

from train.methods.critic_inputs import candidate_batch

CRITIC_FILE = "critic.pt"
CRITIC_SCHEMA_VERSION = 1


class Selector(Protocol):
    def __call__(self, candidates: torch.Tensor, scores: torch.Tensor) -> int: ...


def argmax(candidates: torch.Tensor, scores: torch.Tensor) -> int:
    """Highest critic score, as in the official IFQL and Value Flows (``'rs'``) evaluation."""
    if scores.ndim != 1 or len(scores) != len(candidates) or not torch.isfinite(scores).all():
        raise ValueError("Selector needs one finite score per candidate")
    return int(torch.argmax(scores))


SELECTORS: dict[str, Selector] = {"argmax": argmax}


def _scorers() -> dict[str, Callable[[dict[str, Any]], torch.nn.Module]]:
    from train.methods.critics import ExpectileScorer
    return {"expectile": ExpectileScorer}


def load_critic(path: Path, device: torch.device | str) -> torch.nn.Module:
    """Rebuild the exported critic for scoring; the payload holds tensors and plain settings only."""
    payload = torch.load(path, map_location="cpu", weights_only=True)
    if payload.get("schema_version") != CRITIC_SCHEMA_VERSION:
        raise ValueError("Unsupported exported critic schema")
    scorers = _scorers()
    if payload.get("kind") not in scorers:
        raise ValueError(f"Unknown exported critic kind: {payload.get('kind')}")
    scorer = scorers[payload["kind"]](payload["settings"])
    scorer.load_state_dict(payload["state"], strict=True)
    return scorer.to(device).eval()


def check_executed(scorer: torch.nn.Module, executed: Any) -> None:
    """The critic only scores chunks as long as the transitions it was trained on."""
    horizon = scorer.settings["horizon"]
    if type(executed) is not int or not 1 <= executed <= horizon:
        raise ValueError(f"This policy's critic scores 1–{horizon} executed steps per chunk "
                         f"(its training transitions); got open_loop_steps={executed}")


def choose(scorer: torch.nn.Module, selector: dict[str, Any], candidates: np.ndarray, *,
           agent: np.ndarray, wrist: np.ndarray, raw_proprio: np.ndarray, executed: int,
           action_stats: dict[str, Any], proprio_stats: dict[str, Any],
           process_action: Callable[[np.ndarray], np.ndarray]) -> tuple[int, torch.Tensor]:
    """Index of the chunk to execute among ``candidates`` (N, native_horizon, action_dim).

    The critic sees what the environment would execute: the first ``executed`` steps after
    the policy's own action post-processing, encoded exactly as in training replay.
    """
    check_executed(scorer, executed)
    executed_actions = np.stack([np.stack([process_action(step) for step in candidate[:executed]])
                                 for candidate in candidates])
    settings = scorer.settings
    batch = candidate_batch(agent, wrist, raw_proprio, executed_actions, executed,
                            image_size=settings["image_size"], horizon=settings["horizon"],
                            action_stats=action_stats, proprio_stats=proprio_stats)
    device = next(scorer.parameters()).device
    scores = scorer({key: value.to(device) for key, value in batch.items()}).float().cpu()
    return SELECTORS[selector["kind"]](torch.from_numpy(np.asarray(candidates)), scores), scores
