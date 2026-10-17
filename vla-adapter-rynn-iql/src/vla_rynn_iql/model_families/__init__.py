"""Lightweight family adapters; importing registrations never loads model weights."""
from importlib import import_module


ADAPTERS = {
    "vla_adapter": ".vla_adapter",
    "pi05": ".pi05",
}


def get_adapter(identifier: str):
    if not isinstance(identifier, str) or identifier not in ADAPTERS:
        raise ValueError(f"Unknown model adapter: {identifier!r}")
    return import_module(ADAPTERS[identifier], package=__package__)
