"""OpenPI configuration and integration contract (no GPU imports)."""
from pathlib import Path


BACKEND = "vla_rynn_iql.pi05"
TRAINING_MODES = ("frozen", "full")
DEFAULT_KEYS = {"environment", "backbone", "base_revision", "num_inference_steps"}
REVISION_PATTERN = r"[0-9a-f]{64}"
PIN_SOURCE_REVISION = False
MANIFEST_SCHEMA = 4
MANIFEST_FAMILY_FIELD = True
EXPORT_PREFIX = "pi05"
PARAMETER_PATHS = {}

def validate_contract(io, architecture):
    return (io["image_size"] == 224 and io["native_action_horizon"] >= io["replay_horizon"] and io["padded_action_dim"] == 32
            and io["action_codec"] == "libero_env_v1"
            and set(architecture) == {"config_name", "discrete_state_input"}
            and architecture["config_name"] == "pi05_libero"
            and type(architecture["discrete_state_input"]) is bool)


def configure(settings, *, partial=False):
    from ..base_models import base_model
    base = base_model(settings["base_id"])
    if settings["stats_key"] != base.stats_key and "contract" not in settings:
        raise ValueError(f"{base.id} requires its own {base.stats_key} normalization assets")
    if type(settings["num_inference_steps"]) is not int or not 1 <= settings["num_inference_steps"] <= 100:
        raise ValueError("model.num_inference_steps must be in 1..100")
    return settings


def signature(settings):
    return settings


def resolve_revision(settings, hash_file=None, *, parent=None):
    from ..pi05_assets import checkpoint_identity, identity_digest
    from ..model_storage import cached_sha256
    kwargs = {"hash_file": hash_file or cached_sha256}
    identity = checkpoint_identity(Path(settings["base_checkpoint"]), base_id=settings["base_id"],
                                   parent=parent, contract=settings.get("contract"), **kwargs)
    if f"assets/{settings['stats_key']}/norm_stats.json" not in identity["files"]:
        raise ValueError("Requested normalization is not bound to the selected model identity")
    revision = identity_digest(identity)
    if settings.get("base_revision") not in (None, revision):
        raise ValueError("Selected model weights differ from the configured revision")
    return revision
