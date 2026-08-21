from __future__ import annotations

import json
import importlib.metadata
import logging
import math
import os
import random
import shutil
import subprocess
import time
import uuid
from contextlib import ExitStack
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import numpy as np
import torch
import yaml
from torch.utils.data import default_collate

from .config import LoadedConfig
from .data import load_manifest
from .io import atomic_json, sha256_file, stable_hash
from .iql import PixelIQL, advantage_weights, weighted_masked_l1
from .monitoring import ACTION_NAMES, log_tensorboard_metric
from .replay import ReplayDataset
from .rewards import load_reward_index
from .vla_adapter import (
    ACTION_DIM, ACTION_HORIZON, PROPRIO_DIM, extract_action_hidden_states,
    load_components, predict_normalized, processor_inputs,
)


LOG = logging.getLogger(__name__)


def _action_diagnostics(
    prediction: torch.Tensor,
    target: torch.Tensor,
    mask: torch.Tensor,
) -> dict[str, float]:
    """Return unweighted per-axis errors so gripper learning is observable."""
    valid = mask.float().unsqueeze(-1)
    denominator = valid.sum().clamp_min(1.0)
    error = (prediction.float() - target.float()).abs()
    per_axis = (error * valid).sum(dim=(0, 1)) / denominator
    valid_steps = mask.bool()
    predicted_gripper = prediction.float()[..., -1][valid_steps]
    target_gripper = target.float()[..., -1][valid_steps]
    diagnostics = torch.cat(
        (
            per_axis,
            torch.stack(
                (
                    predicted_gripper.mean(),
                    target_gripper.mean(),
                    (target_gripper < 0.5).float().mean(),
                )
            ),
        )
    ).detach().cpu().tolist()
    result = {
        f"actor_l1_{name}": float(diagnostics[index])
        for index, name in enumerate(ACTION_NAMES)
    }
    result.update(
        actor_gripper_prediction_mean=float(diagnostics[7]),
        actor_gripper_target_mean=float(diagnostics[8]),
        actor_gripper_target_close_fraction=float(diagnostics[9]),
    )
    return result


def _module_parameter_norm(module: torch.nn.Module) -> float:
    """Calculate a module-wide L2 norm without flattening or copying its parameters."""
    norms = [parameter.detach().norm(2).float() for parameter in module.parameters()]
    if not norms:
        return 0.0
    return float(torch.stack(norms).norm(2))


def _device(name: str) -> torch.device:
    device = torch.device(name)
    if device.type == "cuda" and not torch.cuda.is_available():
        raise RuntimeError(f"CUDA device {name} was requested but CUDA is unavailable")
    return device


def _sample_batch(dataset: ReplayDataset, batch_size: int, generator: torch.Generator):
    indices = torch.randint(len(dataset), (batch_size,), generator=generator).tolist()
    return default_collate([dataset[index] for index in indices])


def _code_version() -> dict[str, Any]:
    project = Path(__file__).resolve().parents[2]
    workspace = project.parent
    try:
        commit = subprocess.run(
            ["git", "-C", str(workspace), "rev-parse", "HEAD"],
            check=True, capture_output=True, text=True,
        ).stdout.strip()
    except Exception:
        commit = "unknown"
    versions = {}
    for package in ("vla-rynn-iql", "torch", "transformers", "numpy"):
        try:
            versions[package] = importlib.metadata.version(package)
        except importlib.metadata.PackageNotFoundError:
            versions[package] = "not-installed"
    return {
        "workspace_git_commit": commit,
        "project": str(project),
        "package_versions": versions,
    }


def _actor_lr(step: int, total: int, peak: float, final: float, warmup: int) -> float:
    if warmup and step < warmup:
        return peak * (step + 1) / warmup
    progress = (step - warmup) / max(1, total - warmup - 1)
    return final + 0.5 * (peak - final) * (1.0 + math.cos(math.pi * min(1.0, progress)))


def _state_dict_cpu(module: torch.nn.Module) -> dict[str, torch.Tensor]:
    return {key: value.detach().cpu() for key, value in module.state_dict().items()}


def _save_checkpoint(
    directory: Path,
    step: int,
    components: Any,
    iql: PixelIQL,
    actor_optimizer: torch.optim.Optimizer,
    data_generator: torch.Generator,
    config: LoadedConfig,
    manifest: dict[str, Any],
    reward_index: dict[str, Any],
) -> Path:
    target = directory / f"step_{step:08d}"
    target.mkdir(parents=True, exist_ok=True)
    torch.save(_state_dict_cpu(components.action_head), target / "action_head.pt")
    torch.save(_state_dict_cpu(components.proprio_projector), target / "proprio_projector.pt")
    torch.save({
        "schema_version": 1, "step": step, "iql": iql.checkpoint(),
        "actor_optimizer": actor_optimizer.state_dict(),
        "torch_rng": torch.get_rng_state(),
        "cuda_rng": torch.cuda.get_rng_state_all() if torch.cuda.is_available() else None,
        "numpy_rng": np.random.get_state(), "python_rng": random.getstate(),
        "data_rng": data_generator.get_state(),
    }, target / "trainer.pt")
    checkpoint_metadata = {
        "schema_version": 1, "step": step, "config_sha256": config.digest,
        "dataset_sha256": manifest["dataset_sha256"],
        "reward_sha256": stable_hash(reward_index),
        "base_checkpoint": config.section("vla")["base_checkpoint"],
        "stats_key": components.stats_key,
        "code_version": _code_version(),
    }
    atomic_json(target / "checkpoint.json", checkpoint_metadata)
    (target / "effective_config.yaml").write_text(
        yaml.safe_dump(config.raw, sort_keys=False), encoding="utf-8"
    )
    return target


def _publish_overlay(
    checkpoint: Path,
    registry: Path,
    step: int,
    config: LoadedConfig,
    components: Any,
    manifest: dict[str, Any],
    reward_index: dict[str, Any],
) -> Path:
    policy_id = f"rynn-iql-{datetime.now(timezone.utc).strftime('%Y%m%d-%H%M%S')}-{uuid.uuid4().hex[:8]}"
    target = registry / policy_id
    target.mkdir(parents=True, exist_ok=False)
    shutil.copy2(checkpoint / "action_head.pt", target / "action_head.pt")
    shutil.copy2(checkpoint / "proprio_projector.pt", target / "proprio_projector.pt")
    compatibility = {
        "base_checkpoint": config.section("vla")["base_checkpoint"],
        "stats_key": components.stats_key,
        "action_horizon": ACTION_HORIZON,
        "action_dim": ACTION_DIM,
        "proprio_dim": PROPRIO_DIM,
    }
    payload = {
        "schema_version": 1,
        "policy_id": policy_id,
        "label": f"RynnValue IQL · step {step}",
        "base_checkpoint": config.section("vla")["base_checkpoint"],
        "stats_key": components.stats_key,
        "action_head": "action_head.pt",
        "proprio_projector": "proprio_projector.pt",
        "action_horizon": ACTION_HORIZON,
        "action_dim": ACTION_DIM,
        "proprio_dim": PROPRIO_DIM,
        "dataset_sha256": manifest["dataset_sha256"],
        "reward_sha256": stable_hash(reward_index),
        "training_step": step,
        "component_sha256": {
            "action_head": sha256_file(target / "action_head.pt"),
            "proprio_projector": sha256_file(target / "proprio_projector.pt"),
        },
        "compatibility_sha256": stable_hash(compatibility),
    }
    temporary = target / ".policy.yaml.tmp"
    temporary.write_text(yaml.safe_dump(payload, sort_keys=False), encoding="utf-8")
    os.replace(temporary, target / "policy.yaml")
    latest = registry / "latest"
    temporary_link = registry / f".latest-{uuid.uuid4().hex}"
    temporary_link.symlink_to(target.name, target_is_directory=True)
    os.replace(temporary_link, latest)
    return target / "policy.yaml"


def _restore_checkpoint(
    checkpoint: Path,
    components: Any,
    agent: PixelIQL,
    actor_optimizer: torch.optim.Optimizer,
    data_generator: torch.Generator,
    device: torch.device,
    config: LoadedConfig,
    manifest: dict[str, Any],
    reward_index: dict[str, Any],
) -> int:
    checkpoint = checkpoint.expanduser().resolve()
    metadata = json.loads((checkpoint / "checkpoint.json").read_text(encoding="utf-8"))
    expected = {
        "dataset_sha256": manifest["dataset_sha256"],
        "reward_sha256": stable_hash(reward_index),
        "base_checkpoint": config.section("vla")["base_checkpoint"],
        "stats_key": components.stats_key,
    }
    mismatches = {
        key: (metadata.get(key), value)
        for key, value in expected.items() if metadata.get(key) != value
    }
    if mismatches:
        raise ValueError(f"Resume checkpoint is incompatible: {mismatches}")
    components.action_head.load_state_dict(
        torch.load(checkpoint / "action_head.pt", map_location="cpu", weights_only=True),
        strict=True,
    )
    components.proprio_projector.load_state_dict(
        torch.load(
            checkpoint / "proprio_projector.pt", map_location="cpu", weights_only=True
        ),
        strict=True,
    )
    trainer = torch.load(
        checkpoint / "trainer.pt", map_location=device, weights_only=False
    )
    agent.restore(trainer["iql"])
    actor_optimizer.load_state_dict(trainer["actor_optimizer"])
    torch.set_rng_state(trainer["torch_rng"].cpu())
    if torch.cuda.is_available() and trainer.get("cuda_rng") is not None:
        torch.cuda.set_rng_state_all(trainer["cuda_rng"])
    np.random.set_state(trainer["numpy_rng"])
    random.setstate(trainer["python_rng"])
    data_generator.set_state(trainer["data_rng"].cpu())
    return int(trainer["step"])


def train(config: LoadedConfig) -> Path:
    iql_cfg = config.section("iql")
    seed = int(iql_cfg["seed"])
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    device = _device(iql_cfg["device"])
    if device.type == "cuda":
        torch.cuda.set_device(device)
    manifest = load_manifest(config)
    reward_index = load_reward_index(config)
    components = load_components(config)
    dataset = ReplayDataset(config, components.action_stats, components.proprio_stats, "train")
    data_generator = torch.Generator(device="cpu")
    data_generator.manual_seed(seed + 1)
    agent = PixelIQL(
        horizon=ACTION_HORIZON, action_dim=ACTION_DIM, proprio_dim=PROPRIO_DIM,
        discount=float(config.section("reward")["gamma"]),
        expectile=float(iql_cfg["expectile"]), tau=float(iql_cfg["target_tau"]),
        critic_lr=float(iql_cfg["critic_lr"]), value_lr=float(iql_cfg["value_lr"]),
    ).to(device)
    actor_parameters = list(components.action_head.parameters()) + list(components.proprio_projector.parameters())
    actor_optimizer = torch.optim.AdamW(actor_parameters, lr=float(iql_cfg["policy_peak_lr"]))
    accumulation = int(iql_cfg["gradient_accumulation_steps"])
    total_steps = int(iql_cfg["train_steps"])
    warmup = int(iql_cfg["critic_warmup_steps"])
    start_step = 0
    if iql_cfg["resume_checkpoint"] is not None:
        start_step = _restore_checkpoint(
            Path(iql_cfg["resume_checkpoint"]), components, agent, actor_optimizer,
            data_generator,
            device, config, manifest, reward_index,
        )
        if start_step >= total_steps:
            raise ValueError(
                f"Resume step {start_step} must be smaller than train_steps={total_steps}"
            )
        LOG.info("Resumed training at optimization step %d", start_step)
    run_id = f"{datetime.now(timezone.utc).strftime('%Y-%m-%d_%H%M%S')}__{uuid.uuid4().hex[:8]}"
    run_dir = Path(config.section("paths")["output_dir"]) / run_id
    checkpoint_root = run_dir / "checkpoints"
    checkpoint_root.mkdir(parents=True, exist_ok=False)
    (run_dir / "effective_config.yaml").write_text(
        yaml.safe_dump(config.raw, sort_keys=False), encoding="utf-8"
    )
    atomic_json(run_dir / "provenance.json", {
        "schema_version": 1,
        "code_version": _code_version(),
        "config_sha256": config.digest,
        "dataset_sha256": manifest["dataset_sha256"],
        "reward_sha256": stable_hash(reward_index),
        "base_checkpoint": config.section("vla")["base_checkpoint"],
    })
    actor_optimizer.zero_grad(set_to_none=True)
    start_time = time.monotonic()
    latest_checkpoint: Path | None = None
    metrics_path = run_dir / "metrics.jsonl"
    logging_cfg = config.section("logging")
    with ExitStack() as stack:
        metrics_file = stack.enter_context(metrics_path.open("w", encoding="utf-8"))
        writer = None
        if logging_cfg["tensorboard"]:
            try:
                from torch.utils.tensorboard import SummaryWriter
            except ImportError as exc:
                raise RuntimeError(
                    "TensorBoard logging is enabled but tensorboard is not installed; "
                    "install requirements-train.txt"
                ) from exc
            writer = stack.enter_context(
                SummaryWriter(
                    log_dir=str(run_dir / "tensorboard"),
                    flush_secs=float(logging_cfg["flush_seconds"]),
                )
            )
            writer.add_text(
                "run/effective_config",
                "```yaml\n" + yaml.safe_dump(config.raw, sort_keys=False) + "```",
                start_step,
            )
        current_actor_lr = float(actor_optimizer.param_groups[0]["lr"])
        for step in range(start_step, total_steps):
            batch = _sample_batch(
                dataset, int(iql_cfg["micro_batch_size"]), data_generator
            )
            critic_batch = {
                key: value.to(device, non_blocking=True)
                for key, value in batch.items() if isinstance(value, torch.Tensor)
                and key in {"pixels", "next_pixels", "proprio", "next_proprio", "actions",
                               "action_mask", "reward", "bootstrap_mask", "chunk_length"}
            }
            critic_metrics = agent.update(critic_batch)
            actor_loss_value = None
            weight_mean = None
            # Use ordinary behavior cloning while the critics warm up.  Only
            # the learned advantage weights are delayed; the action head still
            # receives gradients from the first optimization step.
            if step < warmup:
                weights = torch.ones(
                    critic_batch["actions"].shape[0], device=device, dtype=torch.float32
                )
            else:
                with torch.no_grad():
                    advantage = agent.advantage(critic_batch)
                    weights = advantage_weights(
                        advantage, float(iql_cfg["beta"]), float(iql_cfg["max_advantage_weight"])
                    )
            # VLA-Adapter's current continuous path supports one prompt per
            # call. The 16GB profile therefore fixes micro-batch to one and
            # obtains the effective batch through gradient accumulation.
            if len(batch["prompt"]) != 1:
                raise ValueError("VLA actor training currently requires micro_batch_size=1")
            agent_image = batch["agent_image"][0].cpu().numpy()
            wrist_image = batch["wrist_image"][0].cpu().numpy()
            inputs = processor_inputs(components, batch["prompt"][0], agent_image, wrist_image)
            hidden = extract_action_hidden_states(components, inputs)
            proprio = critic_batch["proprio"].to(dtype=torch.bfloat16)
            prediction = predict_normalized(components, hidden, proprio)
            actor_loss = weighted_masked_l1(
                prediction, critic_batch["actions"], critic_batch["action_mask"], weights
            )
            action_metrics = _action_diagnostics(
                prediction,
                critic_batch["actions"],
                critic_batch["action_mask"],
            )
            (actor_loss / accumulation).backward()
            actor_loss_value = float(actor_loss.detach())
            weight_mean = float(weights.mean())
            actor_grad_norm = None
            action_head_parameter_norm = None
            proprio_projector_parameter_norm = None
            if (step + 1) % accumulation == 0 or step == total_steps - 1:
                actor_grad_norm = float(
                    torch.nn.utils.clip_grad_norm_(actor_parameters, 1.0)
                )
                current_actor_lr = _actor_lr(
                    step, total_steps, float(iql_cfg["policy_peak_lr"]),
                    float(iql_cfg["policy_final_lr"]), warmup,
                )
                for group in actor_optimizer.param_groups:
                    group["lr"] = current_actor_lr
                actor_optimizer.step()
                actor_optimizer.zero_grad(set_to_none=True)
                action_head_parameter_norm = _module_parameter_norm(
                    components.action_head
                )
                proprio_projector_parameter_norm = _module_parameter_norm(
                    components.proprio_projector
                )
            metric = {
                "step": step + 1, "q_loss": critic_metrics.q_loss,
                "value_loss": critic_metrics.value_loss, "q_mean": critic_metrics.q_mean,
                "value_mean": critic_metrics.value_mean,
                "advantage_mean": critic_metrics.advantage_mean,
                "actor_loss": actor_loss_value, "advantage_weight_mean": weight_mean,
                "actor_learning_rate": current_actor_lr,
                "actor_grad_norm": actor_grad_norm,
                "action_head_parameter_norm": action_head_parameter_norm,
                "proprio_projector_parameter_norm": proprio_projector_parameter_norm,
                "elapsed_seconds": time.monotonic() - start_time,
                "cuda_peak_memory_bytes": torch.cuda.max_memory_allocated(device) if device.type == "cuda" else 0,
                **action_metrics,
            }
            metric["steps_per_second"] = (metric["step"] - start_step) / max(
                float(metric["elapsed_seconds"]), 1e-9
            )
            metric["cuda_peak_memory_gib"] = float(
                metric["cuda_peak_memory_bytes"]
            ) / (1024.0 ** 3)
            metrics_file.write(json.dumps(metric, sort_keys=True) + "\n")
            metrics_file.flush()
            log_tensorboard_metric(writer, metric)
            if (step + 1) % int(iql_cfg["checkpoint_interval"]) == 0 or step + 1 == total_steps:
                latest_checkpoint = _save_checkpoint(
                    checkpoint_root, step + 1, components, agent, actor_optimizer,
                    data_generator,
                    config, manifest, reward_index,
                )
                LOG.info("Saved checkpoint %s", latest_checkpoint)
    assert latest_checkpoint is not None
    registry = Path(config.section("paths")["policy_registry"])
    registry.mkdir(parents=True, exist_ok=True)
    policy = _publish_overlay(
        latest_checkpoint, registry, total_steps, config, components, manifest, reward_index
    )
    atomic_json(run_dir / "summary.json", {
        "schema_version": 1, "status": "completed", "steps": total_steps,
        "elapsed_seconds": time.monotonic() - start_time,
        "dataset_sha256": manifest["dataset_sha256"],
        "reward_sha256": stable_hash(reward_index), "policy_overlay": str(policy),
        "resumed_from_step": start_step,
        "peak_cuda_memory_bytes": torch.cuda.max_memory_allocated(device) if device.type == "cuda" else 0,
    })
    return policy
