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
    ``critic_keys``: keys of ``config_section`` edited through the platform's shared critic
    fields (optimizers, learning rates, expectile, warmup); empty without a critic.
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
    critic_keys: tuple[str, ...] = ()
    parameters: tuple[dict[str, Any], ...] = ()


# Shared fields of the expectile Q/V critic (IQL and its sampling variants).
EXPECTILE_CRITIC_KEYS = (
    "critic_optimizer", "critic_lr", "critic_weight_decay", "critic_max_grad_norm",
    "value_optimizer", "value_lr", "value_weight_decay", "value_max_grad_norm",
    "expectile", "target_tau",
)

# Inference-time choice among sampled action chunks; implemented in train.methods.selectors.
SELECTOR_KINDS = ("argmax",)

# Order is the platform's display order.
METHODS = {
    "iql": TrainingMethod("iql", "IQL", "奖励加权后训练", True, True, config_section="iql", critic="expectile",
                          critic_keys=("critic_warmup_steps", *EXPECTILE_CRITIC_KEYS, "beta", "max_advantage_weight")),
    "bc": TrainingMethod("bc", "BC", "等权行为克隆", False, False),
    "ifql": TrainingMethod(
        "ifql", "IFQL", "flow 策略采样 + IQL critic 选择", True, True, config_section="ifql",
        critic="expectile", actor_capabilities=frozenset({"sample"}), exports_critic=True,
        critic_keys=EXPECTILE_CRITIC_KEYS,
        parameters=({"name": "num_samples", "label": "推理采样数 N", "type": "number",
                     "default": 32, "min": 1, "max": 256, "step": 1},)),
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


def critic_warmup_steps(raw: dict[str, Any]) -> int:
    """Steps before critic outputs weight the actor; 0 for methods whose actor ignores them."""
    method = training_method(raw)
    if method.critic is None or method.config_section is None:
        return 0
    return int(raw[method.config_section].get("critic_warmup_steps", 0))


def missing_capabilities(method: TrainingMethod, family_capabilities: frozenset[str]) -> set[str]:
    return set(method.actor_capabilities) - set(family_capabilities)


def method_catalog(family_capabilities: dict[str, frozenset[str]]) -> list[dict[str, Any]]:
    """Platform view of every method, with the model families able to train it."""
    return [{
        "id": method.name, "label": method.label, "summary": method.summary,
        "requires_rewards": method.requires_rewards, "requires_transitions": method.requires_transitions,
        "critic": method.critic, "exports_critic": method.exports_critic,
        "critic_keys": list(method.critic_keys),
        "actor_capabilities": sorted(method.actor_capabilities), "parameters": list(method.parameters),
        "families": [family for family, capabilities in family_capabilities.items()
                     if not missing_capabilities(method, capabilities)],
    } for method in METHODS.values()]
