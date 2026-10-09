"""Lightweight access to the shared YAML model registry (no GPU imports)."""
from pathlib import Path

from ..services.inherited_reward_inputs import offline_module

PROJECT = Path(__file__).resolve().parents[4] / "vla-adapter-rynn-iql"


def model_module(name):
    return offline_module(PROJECT, name)


def configured_models(root: Path = PROJECT):
    models = offline_module(root, "models")
    return {entry.id: models.base_model_config(entry.id)
            for entry in models.BASE_MODELS.values()}
