"""VLA-Adapter configuration and integration contract (no GPU imports)."""
import math


BACKEND = "vla_rynn_iql.vla_adapter"
TRAINING_MODES = ("frozen", "lora", "full")
DEFAULT_KEYS = {"environment", "backbone", "action_head", "proprio_projector", "lora", "use_pro_version", "base_revision"}
REVISION_PATTERN = r"(?:[0-9a-f]{40}|[0-9a-f]{64})"
PIN_SOURCE_REVISION = True
MANIFEST_SCHEMA = 3
MANIFEST_FAMILY_FIELD = False
EXPORT_PREFIX = ""
PARAMETER_PATHS = {
    "model_action_head": ("action_head",), "model_proprio_projector": ("proprio_projector",),
    "model_lora_rank": ("lora", "rank"), "model_lora_alpha": ("lora", "alpha"),
    "model_lora_dropout": ("lora", "dropout"),
}

def validate_contract(io, architecture):
    return (io["image_size"] == 224 and io["native_action_horizon"] == 8 and io["padded_action_dim"] == 7
            and io["action_codec"] == "vla_adapter_pro" and architecture == {})


def configure(settings, *, partial=False):
    if settings["use_pro_version"] is not True:
        raise ValueError("The VLA-Adapter backend requires Pro components")
    for name in ("action_head", "proprio_projector"):
        if settings[name] not in ("train", "frozen"):
            raise ValueError(f"model.{name} must be train or frozen")
    if not partial and all(settings[name] == "frozen" for name in ("backbone", "action_head", "proprio_projector")):
        raise ValueError("At least one model component must be trainable")
    lora = settings["lora"]
    if not isinstance(lora, dict):
        raise TypeError("model.lora must be a mapping")
    if set(lora) != {"rank", "alpha", "dropout"}:
        raise ValueError("Unknown model.lora keys")
    for name in ("rank", "alpha"):
        if type(lora[name]) is not int or lora[name] < 1:
            raise ValueError(f"model.lora.{name} must be a positive integer")
    dropout = lora["dropout"]
    if isinstance(dropout, bool) or not isinstance(dropout, (float, int)) or not math.isfinite(dropout) or not 0 <= dropout < 1:
        raise ValueError("model.lora.dropout must be finite and in [0, 1)")
    return settings


def signature(settings):
    settings.pop("use_pro_version", None)
    if settings["backbone"] != "lora":
        settings.pop("lora", None)
    return settings


def resolve_revision(settings, hash_file=None, *, parent=None):
    from ..checkpoint_assets import resolve_revision as resolve
    from ..model_storage import cached_sha256
    kwargs = {"hash_file": hash_file or cached_sha256}
    return resolve(settings["base_checkpoint"], parent["revision"] if parent else settings.get("base_revision"), **kwargs)
