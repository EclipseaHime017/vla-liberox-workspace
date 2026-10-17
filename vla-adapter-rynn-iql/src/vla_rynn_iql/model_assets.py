"""Model-owned asset pinning; callers never construct family-specific identities."""
from .base_models import model_contract, family_adapter
from .models import model_config


def pin_model(settings: dict) -> dict:
    settings = model_config({"model": settings})
    settings["contract"] = model_contract(settings)
    revision = family_adapter(settings["family"]).resolve_revision(settings)
    settings["base_revision"] = revision
    return settings
