"""OpenPI π₀.₅ model backend. Imports stay inside the isolated model environment."""
from __future__ import annotations

import dataclasses
import json
import os
import subprocess
from pathlib import Path
from typing import Any

import numpy as np
import torch

from .io import atomic_json, sha256_file
from .models import model_config
from .pi05_assets import IDENTITY_FILE, OPENPI_COMMIT, checkpoint_identity, identity_digest


@dataclasses.dataclass
class Components:
    model: torch.nn.Module
    input_transform: Any
    output_transform: Any
    identity: dict
    settings: dict
    action_stats: dict
    proprio_stats: dict
    stats_key: str = "physical-intelligence/libero"


def verify_runtime() -> None:
    os.environ.setdefault("JAX_PLATFORMS", "cpu")
    try:
        import openpi
        import transformers
    except ImportError as exc:
        raise RuntimeError("π₀.₅ requires its separate pi05 environment; see the π₀.₅ setup instructions") from exc
    source = Path(openpi.__file__).resolve().parents[2]
    commit = subprocess.check_output(["git", "-C", str(source), "rev-parse", "HEAD"], text=True).strip()
    if commit != OPENPI_COMMIT or transformers.__version__ != "4.53.2" or torch.__version__.split("+")[0] != "2.7.1":
        raise RuntimeError(f"π₀.₅ requires OpenPI {OPENPI_COMMIT}, torch==2.7.1 and its patched transformers==4.53.2")
    if torch.version.cuda != "12.8":
        raise RuntimeError("π₀.₅ requires the CUDA 12.8 Torch build, including Blackwell support; rerun setup_pi05.py")
    changes = subprocess.check_output(["git", "-C", str(source), "status", "--porcelain", "--", "src",
                                       "examples/convert_jax_model_to_pytorch.py"], text=True)
    if changes.strip():
        raise RuntimeError("OpenPI source differs from the pinned revision; refusing untracked model behavior")
    target = Path(transformers.__file__).resolve().parent
    replacements = source / "src/openpi/models_pytorch/transformers_replace"
    for replacement in replacements.rglob("*.py"):
        installed = target / replacement.relative_to(replacements)
        if not installed.is_file() or sha256_file(installed) != sha256_file(replacement):
            raise RuntimeError("OpenPI's required Transformers patches are missing or changed; rerun setup_pi05.py")


def load_components(config, *, training: bool = True) -> Components:
    verify_runtime()
    from openpi.training import config as configs
    from openpi.training import checkpoints
    from openpi import transforms

    settings = model_config(config.raw)
    root = Path(settings["base_checkpoint"])
    identity = checkpoint_identity(root)
    if settings["base_revision"] is not None and settings["base_revision"] != identity_digest(identity):
        raise ValueError("π₀.₅ base weights/normalization changed after training registration")
    cfg = configs.get_config("pi05_libero")
    cfg = dataclasses.replace(cfg, model=dataclasses.replace(
        cfg.model, dtype=config.section("training")["dtype"], pytorch_compile_mode=None))
    model = cfg.model.load_pytorch(cfg, str(root / "model.safetensors"))
    configure_model(model, settings, training=training)
    model.to(config.section("training")["device"])
    data = cfg.data.create(cfg.assets_dirs, cfg.model)
    stats = checkpoints.load_norm_stats(root / "assets", data.asset_id)
    inputs = transforms.compose([
        *data.data_transforms.inputs,
        transforms.Normalize(stats, use_quantiles=data.use_quantile_norm),
        *data.model_transforms.inputs,
    ])
    outputs = transforms.compose([
        *data.model_transforms.outputs,
        transforms.Unnormalize(stats, use_quantiles=data.use_quantile_norm),
        *data.data_transforms.outputs,
    ])
    # Critic normalization has the same units as the native action inputs; no VLA gripper remapping.
    def norm_dict(name):
        value = stats[name]
        return {"q01": np.asarray(value.q01).tolist(), "q99": np.asarray(value.q99).tolist(),
                "codec": "openpi_quantile"}
    return Components(model, inputs, outputs, identity, settings, norm_dict("actions"), norm_dict("state"))


def configure_model(model, settings, *, training=True):
    model.requires_grad_(training)
    model.train(training)
    if training and settings["backbone"] == "frozen":
        model.paligemma_with_expert.paligemma.requires_grad_(False)
        model.paligemma_with_expert.paligemma.eval()
    if training:
        model.gradient_checkpointing_enable()


def normalize(values: np.ndarray, stats: dict) -> np.ndarray:
    size = values.shape[-1]
    lo, hi = np.asarray(stats["q01"][:size]), np.asarray(stats["q99"][:size])
    return ((values - lo) / (hi - lo + 1e-6) * 2 - 1).astype(np.float32)


def masked_flow_losses(errors: torch.Tensor, mask: torch.Tensor, action_dim: int = 7) -> torch.Tensor:
    """Per-example native flow MSE; padded time and motor coordinates receive no loss."""
    if errors.ndim != 3 or mask.ndim != 2 or errors.shape[0] != mask.shape[0]:
        raise ValueError("Invalid flow loss or action mask dimensions")
    if mask.shape[1] > errors.shape[1] or action_dim > errors.shape[2] or not mask.any(dim=1).all():
        raise ValueError("Invalid native horizon, action dimensions or empty action mask")
    valid = mask.to(errors.device, dtype=errors.dtype)
    return (errors[:, :mask.shape[1], :action_dim] * valid[..., None]).sum((1, 2)) / (valid.sum(1) * action_dim)


def actor_losses(components: Components, batch: dict, device: torch.device):
    from openpi.models.model import Observation
    from torch.utils.data import default_collate
    from torch.utils._pytree import tree_map

    samples = []
    for index, prompt in enumerate(batch["prompt"]):
        actions = np.zeros((10, 7), dtype=np.float32)
        actions[:batch["raw_actions"].shape[1]] = batch["raw_actions"][index].numpy()
        samples.append(components.input_transform({
            "observation/image": batch["agent_image"][index].numpy(),
            "observation/wrist_image": batch["wrist_image"][index].numpy(),
            "observation/state": batch["raw_proprio"][index].numpy(),
            "actions": actions, "prompt": prompt,
        }))
    inputs = tree_map(lambda value: value.to(device), default_collate(samples))
    # Quantile statistics can promote NumPy actions to float64; OpenPI trains with float32 actions.
    inputs["actions"] = inputs["actions"].to(dtype=torch.float32)
    errors = components.model(Observation.from_dict(inputs), inputs["actions"])
    return masked_flow_losses(errors, batch["action_mask"]), None


def trainable_parameters(components):
    return [value for value in components.model.parameters() if value.requires_grad]


def parameter_counts(components):
    return {"policy": {"total": sum(p.numel() for p in components.model.parameters()),
                       "trainable": sum(p.numel() for p in trainable_parameters(components))}}


def save_actor(components, settings, directory):
    torch.save({name: value.detach().cpu() for name, value in components.model.named_parameters()
                if value.requires_grad}, directory / "actor.pt")
    atomic_json(directory / IDENTITY_FILE, components.identity)


def restore_actor(components, settings, directory):
    if json.loads((directory / IDENTITY_FILE).read_text()) != components.identity:
        raise ValueError("π₀.₅ checkpoint uses different base weights/normalization assets")
    state = torch.load(directory / "actor.pt", weights_only=True, map_location="cpu")
    params = dict(components.model.named_parameters())
    expected = {name for name in params if settings["backbone"] == "full"
                or not name.startswith("paligemma_with_expert.paligemma.")}
    if set(state) != expected:
        raise ValueError("π₀.₅ actor keys do not match the selected fine-tuning scope")
    with torch.no_grad():
        for name, value in state.items():
            if value.shape != params[name].shape:
                raise ValueError(f"π₀.₅ actor shape mismatch: {name}")
            params[name].copy_(value)


def export_actor(components, settings, directory):
    # Keep the exact base identity: expert-only exports contain deltas, not a substitute base model.
    save_actor(components, settings, directory)
    return {"family": "pi05", "actor": "actor.pt", "base_identity": IDENTITY_FILE,
            "component_sha256": {"actor": sha256_file(directory / "actor.pt"),
                                 "base_identity": sha256_file(directory / IDENTITY_FILE)}}
