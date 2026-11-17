"""Training capabilities shared by configuration and orchestration (no torch import)."""
from __future__ import annotations

from dataclasses import dataclass
from typing import Any


@dataclass(frozen=True)
class TrainingMethod:
    """What a training method needs and produces; implementations live in ``train.methods``.

    ``config_section``: YAML section holding the method's own settings (critic, value, weighting).
    ``critic``: value model trained with the actor (``"expectile"`` Q/V), or ``None``.
    ``actor_capabilities``: what the model family's actor must offer, e.g. ``"sample"`` for
    methods that draw several action chunks per observation.
    ``exports_critic``: exported policies carry the critic and an inference-time selector.
    ``parameters``: extra platform form fields beyond the shared training/critic/reward ones,
    each ``{"name", "label", "type", "default", "min", "max", "step"}``; stored in ``config_section``.
    """
    name: str
    label: str
    summary: str
    requires_rewards: bool
    requires_transitions: bool
    config_section: str | None = None
    critic: str | None = None
    actor_capabilities: frozenset[str] = frozenset()
    exports_critic: bool = False
    parameters: tuple[dict[str, Any], ...] = ()


# Order is the platform's display order.
METHODS = {
    "iql": TrainingMethod("iql", "IQL", "奖励加权后训练", True, True, config_section="iql", critic="expectile"),
    "bc": TrainingMethod("bc", "BC", "等权行为克隆", False, False),
}
DEFAULT_METHOD = "iql"

COMMON_TRAINING_KEYS = {
    "train_steps", "micro_batch_size", "gradient_accumulation_steps",
    "checkpoint_interval", "resume_checkpoint", "seed", "device", "dtype",
    "policy_peak_lr", "policy_final_lr",
}


def training_method(raw: dict[str, Any]) -> TrainingMethod:
    name = raw.get("training", {}).get("method", DEFAULT_METHOD)
    if not isinstance(name, str) or name not in METHODS:
        raise ValueError(f"training.method must be one of {', '.join(METHODS)}")
    return METHODS[name]


def actor_lr_warmup(raw: dict[str, Any]) -> int:
    return int(raw["training"]["actor_lr_warmup_steps"])


def missing_capabilities(method: TrainingMethod, family_capabilities: frozenset[str]) -> set[str]:
    return set(method.actor_capabilities) - set(family_capabilities)


def method_catalog(family_capabilities: dict[str, frozenset[str]]) -> list[dict[str, Any]]:
    """Platform view of every method, with the model families able to train it."""
    return [{
        "id": method.name, "label": method.label, "summary": method.summary,
        "requires_rewards": method.requires_rewards, "requires_transitions": method.requires_transitions,
        "critic": method.critic, "exports_critic": method.exports_critic,
        "actor_capabilities": sorted(method.actor_capabilities), "parameters": list(method.parameters),
        "families": [family for family, capabilities in family_capabilities.items()
                     if not missing_capabilities(method, capabilities)],
    } for method in METHODS.values()]
