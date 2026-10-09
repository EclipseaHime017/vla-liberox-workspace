"""Load every base component from one immutable HF snapshot without mutating it."""
from .catalog import PolicyEntry
from .registry import model_module


def checkpoint_view(entry: PolicyEntry):
    return model_module("checkpoint_assets").checkpoint_view(entry.base_checkpoint, entry.base_revision)
