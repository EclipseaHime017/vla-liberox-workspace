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

from core.io import atomic_json, sha256_file
from core.model_config import model_config
from core.registry import model_contract
from core.artifacts import parent_snapshot
from core.model_storage import cached_sha256, configure_openpi_cache, local_model_directory
from core.pi05_assets import IDENTITY_FILE, OPENPI_COMMIT, checkpoint_identity, identity_digest


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
    parent: dict | None = None


def verify_runtime() -> None:
    configure_openpi_cache()
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


def load_components(config, *, training: bool = True, parent: dict | None = None) -> Components:
    verify_runtime()
    from openpi.training import config as configs
    from openpi.training import checkpoints
    from openpi import transforms

    settings = model_config(config.raw)
    contract = model_contract(settings)
    root = local_model_directory(settings["base_checkpoint"])
    identity = checkpoint_identity(root, hash_file=cached_sha256, base_id=settings["base_id"], parent=parent, contract=contract)
    if settings["base_revision"] is not None and settings["base_revision"] != identity_digest(identity):
        raise ValueError("π₀.₅ base weights/normalization changed after training registration")
    cfg = configs.get_config(identity["config_name"])
    cfg = dataclasses.replace(cfg, data=dataclasses.replace(cfg.data, repo_id=settings["stats_key"]), model=dataclasses.replace(
        cfg.model, dtype=config.section("training")["dtype"], pytorch_compile_mode=None,
        action_horizon=contract["io"]["native_action_horizon"], action_dim=contract["io"]["padded_action_dim"],
        discrete_state_input=contract["architecture"]["discrete_state_input"]))
    data = cfg.data.create(cfg.assets_dirs, cfg.model)
    if f"assets/{data.asset_id}/norm_stats.json" not in identity["files"]:
        raise ValueError("Requested normalization is not bound to the selected model identity")
    stats = checkpoints.load_norm_stats(root / "assets", data.asset_id)
    model = cfg.model.load_pytorch(cfg, str(root / "model.safetensors"))
    configure_model(model, settings, training=training)
    model.to(config.section("training")["device"])
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
    return Components(model, inputs, outputs, identity, settings, norm_dict("actions"), norm_dict("state"),
                      settings["stats_key"], parent or parent_snapshot(settings, identity_digest(identity)))


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
    io = model_contract(components.settings)["io"]
    raw_actions = batch.get("actor_raw_actions", batch["raw_actions"])
    for index, prompt in enumerate(batch["prompt"]):
        actions = np.zeros((io["native_action_horizon"], io["action_dim"]), dtype=np.float32)
        actions[:raw_actions.shape[1]] = raw_actions[index].numpy()
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
    return masked_flow_losses(errors, batch.get("actor_action_mask", batch["action_mask"]), io["action_dim"]), None


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


def export_actor(components, settings, directory, *, checkpoint=None):
    # Keep the exact base identity: expert-only exports contain deltas, not a substitute base model.
    save_actor(components, settings, directory)
    return {"schema_version": 4, "family": "pi05", "actor": "actor.pt", "base_identity": IDENTITY_FILE,
            "component_sha256": {"actor": sha256_file(directory / "actor.pt"),
                                 "base_identity": sha256_file(directory / IDENTITY_FILE)}}


@torch.no_grad()
def _sample_shared_prefix(model, device, observation, count: int, num_steps: int, noise=None) -> torch.Tensor:
    """OpenPI ``sample_actions`` with each observation's image/prompt prefix encoded once.

    Mirrors ``PI0Pytorch.sample_actions`` at the pinned OpenPI commit; the prefix KV cache,
    masks and state are repeated ``count`` times per observation (observation-major order)
    before the Euler denoising loop, so N candidates cost one prefix pass plus N suffixes.
    """
    from openpi.models_pytorch.pi0_pytorch import make_att_2d_masks
    images, img_masks, lang_tokens, lang_masks, state = model._preprocess_observation(observation, train=False)
    prefix_embs, prefix_pad_masks, prefix_att_masks = model.embed_prefix(images, img_masks, lang_tokens, lang_masks)
    prefix_att_2d_masks = make_att_2d_masks(prefix_pad_masks, prefix_att_masks)
    prefix_position_ids = torch.cumsum(prefix_pad_masks, dim=1) - 1
    prefix_att_2d_masks_4d = model._prepare_attention_masks_4d(prefix_att_2d_masks)
    model.paligemma_with_expert.paligemma.language_model.config._attn_implementation = "eager"  # noqa: SLF001
    _, past_key_values = model.paligemma_with_expert.forward(
        attention_mask=prefix_att_2d_masks_4d, position_ids=prefix_position_ids,
        past_key_values=None, inputs_embeds=[prefix_embs, None], use_cache=True)
    past_key_values.batch_repeat_interleave(count)
    prefix_pad_masks = prefix_pad_masks.repeat_interleave(count, dim=0)
    state = state.repeat_interleave(count, dim=0)
    total = prefix_pad_masks.shape[0]
    if noise is None:
        noise = model.sample_noise((total, model.config.action_horizon, model.config.action_dim), device)
    dt = torch.tensor(-1.0 / num_steps, dtype=torch.float32, device=device)
    x_t, time = noise, torch.tensor(1.0, dtype=torch.float32, device=device)
    while time >= -dt / 2:
        v_t = model.denoise_step(state, prefix_pad_masks, past_key_values, x_t, time.expand(total))
        x_t = x_t + dt * v_t
        time += dt
    return x_t


def sample_chunks(components, observations: list[dict], count: int, *, noise=None) -> np.ndarray:
    """``count`` environment-space action chunks per observation: (B, count, horizon, action_dim).

    Observations use the inference keys (``observation/image``, ``observation/wrist_image``,
    ``observation/state``, ``prompt``); transforms are the same as ``Policy.infer``.
    """
    from openpi.models.model import Observation
    from torch.utils.data import default_collate
    from torch.utils._pytree import tree_map

    device = next(components.model.parameters()).device
    transformed = [components.input_transform(dict(observation)) for observation in observations]
    inputs = tree_map(lambda value: value.to(device), default_collate(
        [{key: value for key, value in item.items() if key != "actions"} for item in transformed]))
    actions = _sample_shared_prefix(components.model, device, Observation.from_dict(inputs), count,
                                    components.settings["num_inference_steps"], noise=noise)
    actions = actions.float().cpu().numpy().reshape(len(observations), count, *actions.shape[1:])
    contract = model_contract(components.settings)["io"]
    result = np.empty((len(observations), count, contract["native_action_horizon"], contract["action_dim"]),
                      dtype=np.float32)
    for index, item in enumerate(transformed):
        state = np.asarray(item["state"])
        for candidate in range(count):
            result[index, candidate] = components.output_transform(
                {"state": state, "actions": actions[index, candidate]})["actions"]
    return result


def process_action(action):
    return np.clip(action, -1.0, 1.0)


class NextActionSampler:
    """Chunks from the current actor at replay next observations (Value Flows' critic targets)."""

    process_action = staticmethod(process_action)

    def __init__(self, components):
        self.components, self.action_stats = components, components.action_stats

    def __call__(self, batch: dict, rows: list[int], count: int) -> np.ndarray:
        def array(value):
            return value.detach().cpu().numpy() if isinstance(value, torch.Tensor) else np.asarray(value)
        observations = [{"observation/image": array(batch["next_agent_image"][row]),
                         "observation/wrist_image": array(batch["next_wrist_image"][row]),
                         "observation/state": array(batch["next_raw_proprio"][row]).astype(np.float32),
                         "prompt": batch["prompt"][row]} for row in rows]
        return sample_chunks(self.components, observations, count)


def next_action_sampler(components) -> NextActionSampler:
    return NextActionSampler(components)


def choose_chunk(components, critic, selector: dict, observation: dict, executed: int) -> np.ndarray:
    """Sample ``selector['num_samples']`` chunks and return the one the exported critic prefers."""
    from train.methods.selectors import check_executed, choose
    check_executed(critic, executed)  # before sampling: fail fast on an unscoreable length
    candidates = sample_chunks(components, [observation], selector["num_samples"])[0]
    index, _ = choose(critic, selector, candidates, agent=observation["observation/image"],
                      wrist=observation["observation/wrist_image"], raw_proprio=observation["observation/state"],
                      executed=executed, action_stats=components.action_stats,
                      proprio_stats=components.proprio_stats, process_action=process_action)
    return candidates[index]


def load_selection(manifest: Path, device="cuda:0"):
    """(critic, selector) of an exported sampling policy, or None for policies without a critic."""
    import yaml
    from core.config_sources import UniqueKeyLoader
    from train.methods.selectors import load_critic
    payload = yaml.load(Path(manifest).read_text(), Loader=UniqueKeyLoader)
    if "critic" not in payload:
        return None
    return load_critic(Path(manifest).parent / payload["critic"], device), dict(payload["selector"])


def evaluation_policy(config, manifest, parent):
    """Load a validated export with the same inference transforms as main.

    ``predict(observation, prompt, executed)``: policies exported with a critic sample
    several chunks and return the one their critic prefers for the ``executed`` steps.
    """
    from openpi.policies.policy import Policy
    from core.image_transforms import image_orientation, observation_config, orient_image
    from sim.runtime import proprio_from_observation

    settings = model_config(config.raw)
    components = load_components(config, training=False, parent=parent)
    restore_actor(components, settings, manifest.parent)
    policy = Policy(components.model, transforms=[components.input_transform],
                    output_transforms=[components.output_transform], is_pytorch=True,
                    pytorch_device="cuda:0", sample_kwargs={"num_steps": settings["num_inference_steps"]})
    orientation = image_orientation(observation_config(model_contract(settings)))
    selection = load_selection(manifest)

    def predict(observation, prompt, executed=None):
        inputs = {
            "observation/image": orient_image(observation["agentview_image"], orientation),
            "observation/wrist_image": orient_image(observation["robot0_eye_in_hand_image"], orientation),
            "observation/state": proprio_from_observation(observation), "prompt": prompt,
        }
        if selection is None:
            return policy.infer(inputs)["actions"]
        return choose_chunk(components, *selection, inputs, executed)

    return predict, process_action
