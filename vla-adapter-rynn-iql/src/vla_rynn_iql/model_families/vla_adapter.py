"""VLA-Adapter configuration and integration contract (no GPU imports)."""
import math


BACKEND = "vla_rynn_iql.vla_adapter"
POLICY_ADAPTER = "backend.app.policies.vla_adapter_catalog"
TRAINING_MODES = ("frozen", "lora", "full")
DEFAULT_KEYS = {"environment", "backbone", "action_head", "proprio_projector", "lora", "use_pro_version", "base_revision"}
REVISION_PATTERN = r"(?:[0-9a-f]{40}|[0-9a-f]{64})"
PIN_SOURCE_REVISION = True
MANIFEST_SCHEMA = 3
MANIFEST_FAMILY_FIELD = False
RECORDING_ACTION_CODEC = "vla_adapter_v1"
EXPORT_PREFIX = ""
PARAMETER_PATHS = {
    "model_action_head": ("action_head",), "model_proprio_projector": ("proprio_projector",),
    "model_lora_rank": ("lora", "rank"), "model_lora_alpha": ("lora", "alpha"),
    "model_lora_dropout": ("lora", "dropout"),
}
TRAINING_UI = {
    "fields": [
        {"name": "model_backbone", "label": "Backbone 适配方式", "type": "select", "options": [
            {"value": "frozen", "label": "冻结"}, {"value": "lora", "label": "LoRA"}, {"value": "full", "label": "全量微调"}]},
        *[{"name": name, "label": label, "type": "select", "options": [
            {"value": "train", "label": "训练"}, {"value": "frozen", "label": "冻结"}]}
          for name, label in (("model_action_head", "Action head"), ("model_proprio_projector", "Proprio projector"))],
        *[{"name": name, "label": label, "type": "number", "min": minimum, "step": step,
           "visible_when": {"model_backbone": "lora"}, **({"max": .99} if name == "model_lora_dropout" else {})}
          for name, label, minimum, step in (("model_lora_rank", "LoRA rank", 1, 1),
              ("model_lora_alpha", "LoRA alpha", 1, 1), ("model_lora_dropout", "LoRA dropout", 0, .01))],
    ],
    "trainable_any": ["model_backbone", "model_action_head", "model_proprio_projector"],
    "hints": {
        "frozen": "保持 Backbone 不变，按所选范围训练动作组件。",
        "lora": "使用 all-linear LoRA，并训练 action queries；基础权重保持冻结。",
        "full": "更新完整 Backbone，显存和 checkpoint 开销显著增加。",
    },
    "components": [{"label": "Action head", "parameter": "action_head"},
                   {"label": "Proprio projector", "parameter": "proprio_projector"}],
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
    kwargs = {} if hash_file is None else {"hash_file": hash_file}
    return resolve(settings["base_checkpoint"], parent["revision"] if parent else settings.get("base_revision"), **kwargs)
