"""Model-owned asset pinning; callers never construct family-specific identities."""
from pathlib import Path

from .base_models import model_contract
from .models import model_config


def pin_model(settings: dict) -> dict:
    settings = model_config({"model": settings})
    settings["contract"] = model_contract(settings)
    if settings["family"] == "pi05":
        from .pi05_assets import checkpoint_identity, identity_digest
        identity = checkpoint_identity(Path(settings["base_checkpoint"]),
                                       base_id=settings["base_id"], contract=settings["contract"])
        if f"assets/{settings['stats_key']}/norm_stats.json" not in identity["files"]:
            raise ValueError("Requested normalization is not bound to the selected model identity")
        revision = identity_digest(identity)
        if settings.get("base_revision") not in (None, revision):
            raise ValueError("Selected model weights differ from the configured revision")
    else:
        from .checkpoint_assets import resolve_revision
        revision = resolve_revision(settings["base_checkpoint"], settings.get("base_revision"))
    settings["base_revision"] = revision
    return settings
