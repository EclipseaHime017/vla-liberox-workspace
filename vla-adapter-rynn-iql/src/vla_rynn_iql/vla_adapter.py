from __future__ import annotations

import json
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import yaml
from PIL import Image

from .config import LoadedConfig, UniqueKeyLoader
from .io import sha256_file, stable_hash


ACTION_HORIZON = 8
ACTION_DIM = 7
PROPRIO_DIM = 8


def resolve_stats_key(norm_stats: dict[str, Any], requested: str) -> str:
    if requested in norm_stats:
        return requested
    candidate = f"{requested}_no_noops"
    if candidate in norm_stats:
        return candidate
    available = ", ".join(sorted(norm_stats))
    raise KeyError(f"Normalization key {requested!r} is unavailable; choices: {available}")


def normalize_with_stats(values: np.ndarray, stats: dict[str, Any]) -> np.ndarray:
    values = np.asarray(values, dtype=np.float32)
    low = np.asarray(stats["q01"], dtype=np.float32)
    high = np.asarray(stats["q99"], dtype=np.float32)
    mask = np.asarray(stats.get("mask", np.ones_like(low, dtype=bool)), dtype=bool)
    normalized = np.where(mask, 2.0 * (values - low) / (high - low + 1e-8) - 1.0, values)
    return np.clip(normalized, -1.0, 1.0).astype(np.float32)


def denormalize_with_stats(values: np.ndarray, stats: dict[str, Any]) -> np.ndarray:
    values = np.asarray(values, dtype=np.float32)
    low = np.asarray(stats["q01"], dtype=np.float32)
    high = np.asarray(stats["q99"], dtype=np.float32)
    mask = np.asarray(stats.get("mask", np.ones_like(low, dtype=bool)), dtype=bool)
    return np.where(mask, 0.5 * (values + 1.0) * (high - low + 1e-8) + low, values).astype(np.float32)


def env_to_dataset_actions(env_actions: np.ndarray, action_stats: dict[str, Any]) -> np.ndarray:
    """Invert LIBERO execution conversion, then apply VLA training normalization."""
    actions = np.asarray(env_actions, dtype=np.float32).copy()
    if actions.shape[-1] != ACTION_DIM:
        raise ValueError(f"Expected action dimension {ACTION_DIM}, got {actions.shape}")
    # Execution maps model gripper 0=close, 1=open to +1=close, -1=open.
    actions[..., -1] = (1.0 - actions[..., -1]) * 0.5
    return normalize_with_stats(actions, action_stats)


def dataset_to_env_actions(normalized: np.ndarray, action_stats: dict[str, Any]) -> np.ndarray:
    actions = denormalize_with_stats(normalized, action_stats)
    actions[..., -1] = np.where(actions[..., -1] >= 0.5, -1.0, 1.0)
    return actions.astype(np.float32)


def proprio_from_trajectory(trajectory: dict[str, np.ndarray]) -> np.ndarray:
    return np.concatenate((
        np.asarray(trajectory["eef_position"], dtype=np.float32),
        np.asarray(trajectory["eef_axis_angle"], dtype=np.float32),
        np.asarray(trajectory["gripper_qpos"], dtype=np.float32),
    ), axis=-1)


@dataclass
class VLAComponents:
    cfg: Any
    model: Any
    action_head: Any
    proprio_projector: Any
    processor: Any
    stats_key: str
    action_stats: dict[str, Any]
    proprio_stats: dict[str, Any]


def _add_vla_path(config: LoadedConfig) -> None:
    root = Path(config.section("paths")["vla_adapter_root"])
    if not (root / "prismatic").is_dir():
        raise FileNotFoundError(f"Invalid VLA-Adapter root: {root}")
    if str(root) not in sys.path:
        sys.path.insert(0, str(root))


def load_components(
    config: LoadedConfig, overlay: Path | None = None, *, training: bool = True
) -> VLAComponents:
    _add_vla_path(config)
    from experiments.robot.libero.run_libero_eval import GenerateConfig, initialize_model

    vla_cfg = config.section("vla")
    cfg = GenerateConfig(
        pretrained_checkpoint=vla_cfg["base_checkpoint"],
        task_suite_name=vla_cfg["stats_key"],
        use_l1_regression=True, use_minivlm=True, num_images_in_input=2,
        use_proprio=True, use_film=False, use_pro_version=True,
        load_in_8bit=False, load_in_4bit=False, num_open_loop_steps=ACTION_HORIZON,
        seed=int(config.section("iql")["seed"]), phase="Inference",
    )
    model, action_head, proprio_projector, _, processor = initialize_model(cfg)
    stats_key = resolve_stats_key(model.norm_stats, vla_cfg["stats_key"])
    cfg.unnorm_key = stats_key
    if overlay is not None:
        policy = load_overlay(overlay)
        validate_overlay(policy, vla_cfg["base_checkpoint"], stats_key)
        import torch
        action_head.load_state_dict(torch.load(policy.action_head, map_location="cpu", weights_only=True))
        proprio_projector.load_state_dict(torch.load(policy.proprio_projector, map_location="cpu", weights_only=True))
    for parameter in model.parameters():
        parameter.requires_grad_(False)
    model.eval()
    action_head.train(training)
    proprio_projector.train(training)
    return VLAComponents(
        cfg=cfg, model=model, action_head=action_head,
        proprio_projector=proprio_projector, processor=processor,
        stats_key=stats_key, action_stats=model.norm_stats[stats_key]["action"],
        proprio_stats=model.norm_stats[stats_key]["proprio"],
    )


def qwen_prompt(task: str) -> str:
    return (
        "<|im_start|>system\nYou are Qwen, created by Alibaba Cloud. You are a helpful assistant."
        "<|im_end|>\n<|im_start|>user\nWhat action should the robot take to "
        f"{task.lower()}?<|im_end|>\n<|im_start|>assistant\n"
    )


def processor_inputs(components: VLAComponents, prompt: str, agent: np.ndarray, wrist: np.ndarray):
    primary = components.processor(qwen_prompt(prompt), Image.fromarray(agent)).to(
        components.model.device, dtype=components.model.dtype
    )
    secondary = components.processor(qwen_prompt(prompt), Image.fromarray(wrist)).to(
        components.model.device, dtype=components.model.dtype
    )
    primary["pixel_values"] = __import__("torch").cat(
        [primary["pixel_values"], secondary["pixel_values"]], dim=1
    )
    return primary


def extract_action_hidden_states(components: VLAComponents, inputs: Any):
    """Frozen VLA forward up to the Pro action-head conditioning tensor."""
    import torch
    from prismatic.vla.constants import IGNORE_INDEX, NUM_TOKENS, STOP_INDEX

    model = components.model
    input_ids = inputs["input_ids"]
    attention_mask = inputs["attention_mask"]
    pixel_values = inputs["pixel_values"]
    with torch.no_grad(), torch.autocast("cuda", dtype=torch.bfloat16, enabled=input_ids.is_cuda):
        labels = input_ids.clone()
        labels[:] = IGNORE_INDEX
        prompt_tokens = input_ids.shape[-1] - 1
        input_ids, attention_mask = model._prepare_input_for_action_prediction(input_ids, attention_mask)
        labels = model._prepare_labels_for_action_prediction(labels, input_ids)
        input_embeddings = model.get_input_embeddings()(input_ids)
        action_mask = model._process_action_masks(labels)
        language_embeddings = input_embeddings[~action_mask].reshape(
            input_embeddings.shape[0], -1, input_embeddings.shape[2]
        )
        patches = model._process_vision_features(pixel_values, language_embeddings, False)
        queries = model.action_queries.weight.view(1, -1, model.action_queries.weight.shape[-1]).repeat(
            input_embeddings.shape[0], 1, 1
        )
        input_embeddings = model._replace_input_embeddings(input_embeddings, action_mask, queries)
        embeddings, multimodal_mask = model._build_multimodal_attention(
            input_embeddings, patches, attention_mask
        )
        output = model.language_model(
            input_ids=None, attention_mask=multimodal_mask, inputs_embeds=embeddings,
            labels=None, use_cache=False, output_attentions=False,
            output_hidden_states=True, return_dict=True,
        )
        patch_count = model.vision_backbone.get_num_patches() * model.vision_backbone.get_num_images_in_input()
        layers = []
        for hidden in output.hidden_states:
            action_hidden = hidden[:, patch_count + prompt_tokens:patch_count + prompt_tokens + NUM_TOKENS]
            action_hidden = action_hidden.reshape(hidden.shape[0], 1, NUM_TOKENS, -1)
            task_hidden = hidden[:, :patch_count].reshape(hidden.shape[0], 1, patch_count, -1)
            layers.append(torch.cat((task_hidden, action_hidden), dim=2))
        return torch.cat(layers, dim=1).detach()


def predict_normalized(components: VLAComponents, hidden: Any, proprio: Any):
    return components.action_head.predict_action(
        hidden, proprio=proprio, proprio_projector=components.proprio_projector,
        phase="Inference",
    )


@dataclass(frozen=True)
class PolicyOverlay:
    path: Path
    policy_id: str
    label: str
    base_checkpoint: str
    stats_key: str
    action_head: Path
    proprio_projector: Path
    action_horizon: int
    action_dim: int
    proprio_dim: int
    dataset_sha256: str
    reward_sha256: str
    training_step: int
    component_sha256: dict[str, str]
    compatibility_sha256: str


def load_overlay(path: Path) -> PolicyOverlay:
    path = path.expanduser().resolve()
    raw = yaml.load(path.read_text(encoding="utf-8"), Loader=UniqueKeyLoader)
    required = {"schema_version", "policy_id", "label", "base_checkpoint", "stats_key",
                "action_head", "proprio_projector", "action_horizon", "action_dim",
                "proprio_dim", "dataset_sha256", "reward_sha256", "training_step",
                "component_sha256", "compatibility_sha256"}
    if not isinstance(raw, dict) or set(raw) != required or raw["schema_version"] != 1:
        raise ValueError(f"Invalid policy overlay manifest: {path}")
    for key in ("policy_id", "label", "base_checkpoint", "stats_key"):
        if not isinstance(raw[key], str) or not raw[key].strip():
            raise ValueError(f"Invalid policy overlay field {key}: {path}")
    if path.parent.name != raw["policy_id"] or Path(raw["policy_id"]).name != raw["policy_id"]:
        raise ValueError(f"Policy directory must match its safe policy_id: {path}")
    for key in ("action_horizon", "action_dim", "proprio_dim", "training_step"):
        if type(raw[key]) is not int or raw[key] < 1:
            raise ValueError(f"Invalid policy overlay integer {key}: {path}")
    hashes = raw["component_sha256"]
    if not isinstance(hashes, dict) or set(hashes) != {"action_head", "proprio_projector"}:
        raise ValueError(f"Invalid policy overlay component hashes: {path}")
    def component(name: str) -> Path:
        value = Path(raw[name]).expanduser()
        result = (value if value.is_absolute() else path.parent / value).resolve()
        if not result.is_file() or path.parent not in result.parents:
            raise ValueError(f"Overlay component {name} is missing or outside its policy directory")
        return result
    return PolicyOverlay(
        path, str(raw["policy_id"]), str(raw["label"]), str(raw["base_checkpoint"]),
        str(raw["stats_key"]), component("action_head"), component("proprio_projector"),
        int(raw["action_horizon"]), int(raw["action_dim"]), int(raw["proprio_dim"]),
        str(raw["dataset_sha256"]), str(raw["reward_sha256"]), int(raw["training_step"]),
        dict(hashes), str(raw["compatibility_sha256"]),
    )


def validate_overlay(policy: PolicyOverlay, base_checkpoint: str, stats_key: str) -> None:
    expected = (base_checkpoint, stats_key, ACTION_HORIZON, ACTION_DIM, PROPRIO_DIM)
    actual = (policy.base_checkpoint, policy.stats_key, policy.action_horizon, policy.action_dim, policy.proprio_dim)
    if actual != expected:
        raise ValueError(f"Policy overlay is incompatible: expected {expected}, got {actual}")
    compatibility = {
        "base_checkpoint": policy.base_checkpoint,
        "stats_key": policy.stats_key,
        "action_horizon": policy.action_horizon,
        "action_dim": policy.action_dim,
        "proprio_dim": policy.proprio_dim,
    }
    if policy.compatibility_sha256 != stable_hash(compatibility):
        raise ValueError("Policy overlay compatibility hash does not match its manifest")
    for name, path in (
        ("action_head", policy.action_head),
        ("proprio_projector", policy.proprio_projector),
    ):
        expected_hash = policy.component_sha256.get(name)
        if not expected_hash or sha256_file(path) != expected_hash:
            raise ValueError(f"Policy overlay component hash mismatch: {name}")


def overlay_public(path: Path) -> dict[str, Any]:
    policy = load_overlay(path)
    return {
        "policy_id": policy.policy_id, "label": policy.label,
        "base_checkpoint": policy.base_checkpoint, "stats_key": policy.stats_key,
        "training_step": policy.training_step, "manifest": str(policy.path),
    }
