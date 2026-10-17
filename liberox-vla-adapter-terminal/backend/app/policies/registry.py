"""Lightweight access to the shared YAML model registry (no GPU imports)."""
from pathlib import Path
from importlib import import_module

from ..services.inherited_reward_inputs import offline_module

PROJECT = Path(__file__).resolve().parents[4] / "vla-adapter-rynn-iql"


def model_module(name):
    return offline_module(PROJECT, name)


def policy_adapter(family):
    descriptor = model_module("base_models").family_adapter(family)
    return import_module(descriptor.POLICY_ADAPTER)


def provider_factory(family):
    module, name = policy_adapter(family).PROVIDER.split(":")
    return getattr(import_module(module), name)


def configured_models(root: Path = PROJECT):
    models = offline_module(root, "models")
    return {entry.id: models.base_model_config(entry.id)
            for entry in models.BASE_MODELS.values()}
