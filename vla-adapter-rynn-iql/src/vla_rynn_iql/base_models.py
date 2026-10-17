"""GPU-free YAML registry: model families, base checkpoints and inherited I/O."""
from __future__ import annotations

import copy
import re
from dataclasses import dataclass
from pathlib import Path

import yaml

from .config_sources import UniqueKeyLoader
from .image_transforms import FLIP_KEYS, observation_config, validate_observation
from .model_families import ADAPTERS, get_adapter

REGISTRY_PATH = Path(__file__).resolve().parents[2] / "configs/models/registry.yaml"
DEFAULT_FAMILY = "vla_adapter"
IO_KEYS = {"cameras", "image_size", "proprio", "proprio_dim",
           "native_action_horizon", "padded_action_dim", "action_dim", "replay_horizon",
           "control_hz", "action_codec"}


def validate_contract(contract: dict, family: str, *, adapter: str | None = None,
                      require_observation: bool = True) -> dict:
    if not isinstance(contract, dict) or set(contract) not in ({"io", "architecture"}, {"io", "architecture", "observation"}):
        raise ValueError("Model contract requires io and architecture")
    io, architecture = contract["io"], contract["architecture"]
    if (not isinstance(io, dict) or set(io) not in (IO_KEYS, IO_KEYS | FLIP_KEYS, IO_KEYS | {"image_orientation"})
            or not isinstance(architecture, dict)):
        raise ValueError("Invalid model I/O contract keys")
    # Validate the supported Panda bridge; YAML cannot change the robot controller.
    fixed = {"cameras": ["agentview", "robot0_eye_in_hand"],
             "proprio": "eef_position_axis_angle_gripper",
             "proprio_dim": 8, "action_dim": 7, "replay_horizon": 8, "control_hz": 20}
    if any(io.get(key) != value for key, value in fixed.items()):
        raise ValueError("Model I/O is unsupported by the current Panda adapter")
    if require_observation or "observation" in contract or set(io) != IO_KEYS:
        observation_config(contract)
    for key in ("image_size", "proprio_dim", "native_action_horizon", "padded_action_dim",
                "action_dim", "replay_horizon", "control_hz"):
        if type(io[key]) is not int or io[key] < 1:
            raise ValueError(f"Invalid model I/O integer: {key}")
    definition = globals().get("MODEL_FAMILIES", {}).get(family)
    implementation = get_adapter(adapter or (definition.adapter if definition else family))
    if not implementation.validate_contract(io, architecture):
        raise ValueError(f"Unsupported model contract for {family}")
    return copy.deepcopy(contract)


@dataclass(frozen=True)
class ModelDefinition:
    id: str
    label: str
    adapter: str
    default_base: str
    backbone_modes: tuple[str, ...]
    defaults: dict
    contract: dict
    config_path: Path

    @property
    def backend(self):
        return self.implementation.BACKEND

    @property
    def implementation(self):
        return get_adapter(self.adapter)


@dataclass(frozen=True)
class BaseModel:
    id: str
    family: str
    label: str
    source: str
    stats_key: str
    checkpoint: str
    revision: str | None
    contract: dict
    inference: dict

    @property
    def config_name(self):
        return self.contract["architecture"]["config_name"]

    @property
    def norm_file(self):
        return f"assets/{self.stats_key}/norm_stats.json"


def load_registry(path: Path = REGISTRY_PATH):
    raw = yaml.load(path.read_text(encoding="utf-8"), Loader=UniqueKeyLoader)
    if not isinstance(raw, dict) or set(raw) != {"schema_version", "models"} or raw["schema_version"] != 1:
        raise ValueError("Invalid model registry schema")
    if not isinstance(raw["models"], dict) or not raw["models"]:
        raise ValueError("Registry models must be a non-empty mapping")
    families, bases = {}, {}
    for identifier, filename in raw["models"].items():
        if not isinstance(filename, str) or re.fullmatch(r"[A-Za-z0-9_-]+\.yaml", filename) is None:
            raise ValueError("Model index must reference a family YAML filename")
        family_path = (path.parent / filename).resolve()
        entry = yaml.load(family_path.read_text(encoding="utf-8"), Loader=UniqueKeyLoader)
        required = {"schema_version", "model_name", "label", "adapter", "default_variant",
                    "backbone_modes", "defaults", "io", "architecture", "variants"}
        if (not isinstance(entry, dict) or set(entry) != required or entry["schema_version"] != 1
                or entry["model_name"] != identifier or entry["adapter"] not in ADAPTERS):
            raise ValueError(f"Unsupported model family registration: {identifier}")
        implementation = get_adapter(entry["adapter"])
        modes = entry["backbone_modes"]
        supported = set(implementation.TRAINING_MODES)
        if not isinstance(modes, list) or not modes or len(set(modes)) != len(modes) or set(modes) - supported:
            raise ValueError(f"Invalid training scopes: {identifier}")
        default_keys = implementation.DEFAULT_KEYS
        if not isinstance(entry["defaults"], dict) or set(entry["defaults"]) != default_keys:
            raise ValueError("Invalid model defaults keys")
        if not isinstance(entry["label"], str) or not entry["label"].strip():
            raise ValueError("Model family label must be non-empty")
        if not isinstance(entry["io"], dict) or set(entry["io"]) != IO_KEYS:
            raise ValueError("Family io only defines interfaces; put image flips in variant observation")
        contract = validate_contract({key: entry[key] for key in ("io", "architecture")}, identifier,
                                     adapter=entry["adapter"], require_observation=False)
        families[identifier] = ModelDefinition(identifier, entry["label"], entry["adapter"], entry["default_variant"],
                                                tuple(modes), entry["defaults"], contract, family_path)
        if not isinstance(entry["variants"], dict) or not entry["variants"]:
            raise ValueError(f"Model {identifier} requires a non-empty variants mapping")
        for base_id, base_entry in entry["variants"].items():
            if base_id in bases:
                raise ValueError(f"Duplicate base model ID: {base_id}")
            bases[base_id] = _parse_base(base_id, base_entry, families[identifier], family_path)
    for family in families.values():
        if family.default_base not in bases or bases[family.default_base].family != family.id:
            raise ValueError(f"Missing default base for {family.id}")
    return families, bases


def _parse_base(identifier: str, entry: dict, family: ModelDefinition, path: Path) -> BaseModel:
    required = {"label", "source", "revision", "checkpoint", "stats_key", "inference", "observation"}
    if not isinstance(entry, dict) or not required <= entry.keys() or entry.keys() - required - {"io", "architecture"}:
        raise ValueError(f"Invalid base registration: {identifier}")
    if (not isinstance(identifier, str) or identifier in {".", ".."}
            or re.fullmatch(r"[A-Za-z0-9_.-]+", identifier) is None):
        raise ValueError("Invalid base ID")
    if any(not isinstance(entry[key], str) or not entry[key].strip() for key in required - {"revision", "inference", "observation"}):
        raise ValueError("Base registration fields must be non-empty strings")
    revision = entry["revision"]
    if revision is not None and re.fullmatch(r"[0-9a-f]{40}", str(revision)) is None:
        raise ValueError("Base source revision must be an immutable HF commit or null")
    checkpoint = entry["checkpoint"]
    if checkpoint.startswith(("./", "../", "~/", "/")):
        checkpoint = str((path.parent / Path(checkpoint).expanduser()).resolve())
    contract = copy.deepcopy(family.contract)
    for section in ("io", "architecture"):
        if section in entry:
            if not isinstance(entry[section], dict):
                raise ValueError(f"Base {section} must be a mapping")
            if section == "io" and entry[section].keys() - IO_KEYS:
                raise ValueError("Variant io cannot contain observation settings")
            contract[section].update(entry[section])
    contract["observation"] = validate_observation(entry["observation"])
    contract = validate_contract(contract, family.id, adapter=family.adapter)
    inference = validate_inference(entry["inference"], contract)
    return BaseModel(identifier, family=family.id, **{key: entry[key] for key in required - {"checkpoint", "inference", "observation"}},
                     checkpoint=checkpoint, contract=contract, inference=inference)


def validate_inference(value: dict, contract: dict) -> dict:
    if not isinstance(value, dict) or set(value) != {"open_loop_steps"}:
        raise ValueError("Variant inference requires open_loop_steps")
    validate_open_loop_steps(value["open_loop_steps"], contract["io"]["native_action_horizon"])
    return dict(value)


def validate_open_loop_steps(value: int, maximum: int) -> int:
    if type(value) is not int or not 1 <= value <= maximum:
        raise ValueError(f"open_loop_steps must be an integer in [1, {maximum}] for the selected model")
    return value


MODEL_FAMILIES, BASE_MODELS = load_registry()


def base_model(base_id: str) -> BaseModel:
    if not isinstance(base_id, str) or base_id not in BASE_MODELS:
        raise ValueError(f"Unknown base model: {base_id!r}")
    return BASE_MODELS[base_id]


def model_contract(settings: dict) -> dict:
    family = settings.get("family", DEFAULT_FAMILY)
    if family not in MODEL_FAMILIES:
        raise ValueError(f"Unknown model family: {family}")
    base = base_model(settings.get("base_id", MODEL_FAMILIES[family].default_base))
    if family != base.family:
        raise ValueError("Model family conflicts with base model")
    return validate_contract(settings.get("contract", base.contract), family)


def registered_defaults(family: str, base_id: str | None = None) -> dict:
    definition = MODEL_FAMILIES[family]
    base = base_model(base_id or definition.default_base)
    if base.family != family:
        raise ValueError("Model family conflicts with base model")
    # A local deployment is pinned by content hash, not its source repository commit.
    revision = base.revision if definition.implementation.PIN_SOURCE_REVISION and not Path(base.checkpoint).is_absolute() else None
    return {**copy.deepcopy(definition.defaults), "family": family, "base_id": base.id,
            "base_checkpoint": base.checkpoint, "stats_key": base.stats_key,
            "base_revision": revision}


def model_observation(settings: dict) -> dict:
    return observation_config(model_contract(settings))


def family_adapter(family: str):
    if not isinstance(family, str) or family not in MODEL_FAMILIES:
        raise ValueError(f"Unknown model family: {family!r}")
    return MODEL_FAMILIES[family].implementation


def execution_config(settings: dict) -> dict:
    """Variant execution defaults do not change weights or training configuration."""
    io = model_contract(settings)["io"]
    family = MODEL_FAMILIES[settings.get("family", DEFAULT_FAMILY)]
    base = base_model(settings.get("base_id", family.default_base))
    maximum = io["native_action_horizon"]
    # Replay length defines training supervision, not the model's output shape.
    return {"open_loop_steps": min(base.inference["open_loop_steps"], maximum),
            "max_open_loop_steps": maximum}


def resolve_open_loop_steps(settings: dict, value: int | None) -> int:
    inference = execution_config(settings)
    return validate_open_loop_steps(inference["open_loop_steps"] if value is None else value,
                                    inference["max_open_loop_steps"])
