"""Training capabilities shared by configuration and orchestration (no torch import)."""
from __future__ import annotations

from dataclasses import dataclass
from typing import Any


@dataclass(frozen=True)
class TrainingMethod:
    name: str
    requires_rewards: bool
    requires_transitions: bool


METHODS = {
    "iql": TrainingMethod("iql", True, True),
    "bc": TrainingMethod("bc", False, False),
}

COMMON_TRAINING_KEYS = {
    "train_steps", "micro_batch_size", "gradient_accumulation_steps",
    "checkpoint_interval", "resume_checkpoint", "seed", "device", "dtype",
    "policy_peak_lr", "policy_final_lr",
}


def training_method(raw: dict[str, Any]) -> TrainingMethod:
    name = raw.get("training", {}).get("method", "iql")
    if not isinstance(name, str) or name not in METHODS:
        raise ValueError(f"training.method must be one of {', '.join(METHODS)}")
    return METHODS[name]


def actor_lr_warmup(raw: dict[str, Any]) -> int:
    value = raw.get("training", {}).get("actor_lr_warmup_steps")
    return int(raw["iql"]["critic_warmup_steps"] if value is None else value)
