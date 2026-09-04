from __future__ import annotations

import copy
import json
import logging
import os
import random
import signal
import sys
import threading
import time
import uuid
from contextlib import ExitStack, nullcontext
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Any

import numpy as np
import torch
import torch.distributed as dist
import yaml
from torch import nn
from torch.nn import functional as F
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.utils.data import DataLoader

from .config import LoadedConfig
from .data import load_manifest
from .iql import (
    QNetwork,
    ValueNetwork,
    advantage_weights,
    chunk_bellman_target,
    expectile_loss,
    weighted_masked_l1,
)
from .io import atomic_json, stable_hash
from .monitoring import (
    TrainingProgressReporter,
    grouped_scalar_metrics,
    log_tensorboard_metric,
)
from .rewards import load_reward_index
from .server_config import ServerPipelineConfig, validate_global_batch
from .server_replay import (
    CachedReplayDataset,
    DeterministicDistributedBatchSampler,
    replay_cache_path,
    validate_replay_cache,
)
from .training import (
    _action_diagnostics,
    _actor_lr,
    _code_version,
    _initialize_wandb,
    _module_parameter_norm,
    _publish_overlay,
    _state_dict_cpu,
    _validate_single_task_micro_batch,
)
from .vla_adapter import (
    ACTION_DIM,
    ACTION_HORIZON,
    PROPRIO_DIM,
    extract_action_hidden_states,
    load_components,
    processor_inputs,
)


LOG = logging.getLogger(__name__)
_STOP_REQUESTED = threading.Event()


class DistributedTrainingCancelled(RuntimeError):
    pass


class DoubleQModule(nn.Module):
    def __init__(self):
        super().__init__()
        self.q1 = QNetwork(ACTION_HORIZON, ACTION_DIM, PROPRIO_DIM)
        self.q2 = QNetwork(ACTION_HORIZON, ACTION_DIM, PROPRIO_DIM)

    def forward(
        self,
        pixels: torch.Tensor,
        proprio: torch.Tensor,
        actions: torch.Tensor,
        action_mask: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        return (
            self.q1(pixels, proprio, actions, action_mask),
            self.q2(pixels, proprio, actions, action_mask),
        )


class ActorOverlayModule(nn.Module):
    def __init__(self, action_head: nn.Module, proprio_projector: nn.Module):
        super().__init__()
        self.action_head = action_head
        self.proprio_projector = proprio_projector

    def forward(self, hidden: torch.Tensor, proprio: torch.Tensor) -> torch.Tensor:
        return self.action_head.predict_action(
            hidden,
            proprio=proprio,
            proprio_projector=self.proprio_projector,
            phase="Inference",
        )


def request_distributed_stop() -> None:
    _STOP_REQUESTED.set()


def install_distributed_signal_handlers() -> None:
    def stop(_signum: int, _frame: Any) -> None:
        request_distributed_stop()

    signal.signal(signal.SIGINT, stop)
    signal.signal(signal.SIGTERM, stop)


def _optimizer_type(name: str) -> type[torch.optim.Optimizer]:
    values: dict[str, type[torch.optim.Optimizer]] = {
        "adam": torch.optim.Adam,
        "adamw": torch.optim.AdamW,
    }
    try:
        return values[name]
    except KeyError as exc:
        raise ValueError(f"Unsupported optimizer {name!r}") from exc


def _zero_optimizer(
    parameters: Any,
    optimizer_class: type[torch.optim.Optimizer],
    **kwargs: Any,
) -> Any:
    from torch.distributed.optim import ZeroRedundancyOptimizer

    return ZeroRedundancyOptimizer(
        parameters,
        optimizer_class=optimizer_class,
        overlap_with_ddp=False,
        **kwargs,
    )


def _rank_components(config: LoadedConfig, device: torch.device) -> Any:
    # VLA-Adapter's audited upstream helper stores its target in a module-level
    # DEVICE constant. Set it before load_components imports/constructs modules;
    # this is isolated to one torchrun worker process.
    root = Path(config.section("paths")["vla_adapter_root"])
    if str(root) not in sys.path:
        sys.path.insert(0, str(root))
    from experiments.robot import openvla_utils

    openvla_utils.DEVICE = device
    components = load_components(config)
    components.model.to(device)
    components.action_head.to(device)
    components.proprio_projector.to(device)
    model_device = next(components.model.parameters()).device
    if model_device != device:
        raise RuntimeError(f"VLA backbone loaded on {model_device}, expected {device}")
    return components


def _all_reduce(values: dict[str, float], *, operation: str = "mean") -> dict[str, float]:
    keys = sorted(values)
    if not keys:
        return {}
    device = torch.device("cuda", torch.cuda.current_device())
    tensor = torch.tensor([float(values[key]) for key in keys], device=device, dtype=torch.float64)
    if operation == "max":
        dist.all_reduce(tensor, op=dist.ReduceOp.MAX)
    elif operation == "min":
        dist.all_reduce(tensor, op=dist.ReduceOp.MIN)
    elif operation == "mean":
        dist.all_reduce(tensor, op=dist.ReduceOp.SUM)
        tensor /= dist.get_world_size()
    else:
        raise ValueError(f"Unknown reduction operation {operation}")
    return dict(zip(keys, tensor.cpu().tolist()))


def _stop_requested() -> bool:
    flag = torch.tensor(
        [1 if _STOP_REQUESTED.is_set() else 0],
        device=torch.device("cuda", torch.cuda.current_device()),
        dtype=torch.int32,
    )
    dist.all_reduce(flag, op=dist.ReduceOp.MAX)
    return bool(flag.item())


def _ddp_communication_ms(*modules: DDP) -> float:
    total_ns = 0
    for module in modules:
        getter = getattr(module, "_get_ddp_logging_data", None)
        if getter is None:
            continue
        try:
            data = getter()
        except (AttributeError, RuntimeError):
            continue
        value = data.get("avg_backward_comm_time", 0)
        if isinstance(value, (int, float)):
            total_ns += int(value)
    return total_ns / 1_000_000.0


@torch.no_grad()
def _soft_update(source: DoubleQModule, target: DoubleQModule, tau: float) -> None:
    for source_network, target_network in (
        (source.q1, target.q1),
        (source.q2, target.q2),
    ):
        for source_parameter, target_parameter in zip(
            source_network.parameters(), target_network.parameters()
        ):
            target_parameter.lerp_(source_parameter, tau)


def _checkpoint_metadata(
    *,
    step: int,
    config: LoadedConfig,
    server: ServerPipelineConfig,
    manifest: dict[str, Any],
    reward_index: dict[str, Any],
    stats_key: str,
) -> dict[str, Any]:
    return {
        "schema_version": 2,
        "trainer": "server-ddp-zero1",
        "step": step,
        "config_sha256": config.digest,
        "dataset_sha256": manifest["dataset_sha256"],
        "reward_sha256": stable_hash(reward_index),
        "base_checkpoint": config.section("vla")["base_checkpoint"],
        "stats_key": stats_key,
        "distributed": {
            "world_size": server.distributed.world_size,
            "gpu_ids": list(server.distributed.gpu_ids),
            "backend": server.distributed.backend,
            "zero_stage": server.distributed.zero_stage,
        },
        "code_version": _code_version(),
    }


def _save_checkpoint(
    *,
    directory: Path,
    step: int,
    rank: int,
    components: Any,
    q_module: DoubleQModule,
    target_q: DoubleQModule,
    value_module: ValueNetwork,
    q_optimizer: Any,
    value_optimizer: Any,
    actor_optimizer: Any,
    config: LoadedConfig,
    server: ServerPipelineConfig,
    manifest: dict[str, Any],
    reward_index: dict[str, Any],
) -> Path:
    for optimizer in (q_optimizer, value_optimizer, actor_optimizer):
        optimizer.consolidate_state_dict(to=0)
    dist.barrier()
    target = directory / f"step_{step:08d}"
    if rank == 0:
        target.mkdir(parents=True, exist_ok=True)
        torch.save(_state_dict_cpu(components.action_head), target / "action_head.pt")
        torch.save(_state_dict_cpu(components.proprio_projector), target / "proprio_projector.pt")
        torch.save({
            "schema_version": 2,
            "trainer": "server-ddp-zero1",
            "step": step,
            "q": _state_dict_cpu(q_module),
            "target_q": _state_dict_cpu(target_q),
            "value": _state_dict_cpu(value_module),
            "q_optimizer": q_optimizer.state_dict(),
            "value_optimizer": value_optimizer.state_dict(),
            "actor_optimizer": actor_optimizer.state_dict(),
            "torch_rng": torch.get_rng_state(),
            "cuda_rng": torch.cuda.get_rng_state(),
            "numpy_rng": np.random.get_state(),
            "python_rng": random.getstate(),
        }, target / "trainer.pt")
        atomic_json(
            target / "checkpoint.json",
            _checkpoint_metadata(
                step=step,
                config=config,
                server=server,
                manifest=manifest,
                reward_index=reward_index,
                stats_key=components.stats_key,
            ),
        )
        (target / "effective_config.yaml").write_text(
            yaml.safe_dump(config.raw, sort_keys=False), encoding="utf-8"
        )
    dist.barrier()
    return target


def _restore_checkpoint(
    *,
    checkpoint: Path,
    components: Any,
    q_module: DoubleQModule,
    target_q: DoubleQModule,
    value_module: ValueNetwork,
    q_optimizer: Any,
    value_optimizer: Any,
    actor_optimizer: Any,
    config: LoadedConfig,
    manifest: dict[str, Any],
    reward_index: dict[str, Any],
) -> int:
    checkpoint = checkpoint.expanduser().resolve()
    metadata = json.loads((checkpoint / "checkpoint.json").read_text(encoding="utf-8"))
    if metadata.get("schema_version") != 2 or metadata.get("trainer") != "server-ddp-zero1":
        raise ValueError("Server training can resume only a server-ddp-zero1 checkpoint")
    expected = {
        "dataset_sha256": manifest["dataset_sha256"],
        "reward_sha256": stable_hash(reward_index),
        "base_checkpoint": config.section("vla")["base_checkpoint"],
        "stats_key": components.stats_key,
    }
    mismatches = {
        key: (metadata.get(key), value)
        for key, value in expected.items()
        if metadata.get(key) != value
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
    trainer = torch.load(checkpoint / "trainer.pt", map_location="cpu", weights_only=False)
    q_module.load_state_dict(trainer["q"], strict=True)
    target_q.load_state_dict(trainer["target_q"], strict=True)
    value_module.load_state_dict(trainer["value"], strict=True)
    q_optimizer.load_state_dict(trainer["q_optimizer"])
    value_optimizer.load_state_dict(trainer["value_optimizer"])
    actor_optimizer.load_state_dict(trainer["actor_optimizer"])
    return int(trainer["step"])


def _server_tensorboard(writer: Any, metric: dict[str, Any]) -> None:
    log_tensorboard_metric(writer, metric)
    if writer is None:
        return
    step = int(metric["step"])
    for key in (
        "world_size", "global_micro_batch_size", "local_micro_batch_size",
        "actor_global_batch_size", "global_samples_per_second", "data_wait_ms",
        "host_to_device_ms", "actor_input_ms", "backbone_forward_ms", "critic_update_ms",
        "actor_update_ms", "ddp_communication_ms", "slowest_rank_step_ms",
        "cuda_peak_memory_gib_min", "cuda_peak_memory_gib_max",
    ):
        value = metric.get(key)
        if isinstance(value, (int, float)):
            writer.add_scalar(f"server/{key}", float(value), step)


def _server_wandb(run: Any, metric: dict[str, Any]) -> None:
    if run is None:
        return
    step = int(metric["step"])
    payload = grouped_scalar_metrics(metric)
    payload.update({
        f"server/{key}": float(value)
        for key, value in metric.items()
        if isinstance(value, (int, float)) and not isinstance(value, bool)
    })
    payload["train/step"] = step
    run.log(payload, step=step)


def _broadcast_run_directory(config: LoadedConfig, rank: int) -> tuple[str, Path]:
    value: list[Any] = [None, None]
    if rank == 0:
        run_id = (
            f"{datetime.now(timezone.utc).strftime('%Y-%m-%d_%H%M%S')}__"
            f"{uuid.uuid4().hex[:8]}"
        )
        run_dir = Path(config.section("paths")["output_dir"]) / run_id
        run_dir.mkdir(parents=True, exist_ok=False)
        value = [run_id, str(run_dir.resolve())]
    dist.broadcast_object_list(value, src=0)
    return str(value[0]), Path(value[1])


def _configure_rank_cpu_threads(world_size: int, data_workers: int) -> int:
    try:
        available = len(os.sched_getaffinity(0))
    except AttributeError:
        available = os.cpu_count() or world_size
    # Reserve one logical CPU for each DataLoader worker. This prevents eight
    # VLA replicas from each creating a host-sized BLAS/OpenMP pool.
    usable = max(world_size, available - world_size * data_workers)
    threads = max(1, usable // world_size)
    torch.set_num_threads(threads)
    try:
        torch.set_num_interop_threads(1)
    except RuntimeError:
        # PyTorch permits this setting only before inter-op work begins. The
        # worker normally reaches this branch early enough; retain the existing
        # value if an embedding environment initialized the pool first.
        pass
    return threads


def train_distributed(config: LoadedConfig, server: ServerPipelineConfig) -> Path | None:
    _STOP_REQUESTED.clear()
    distributed = server.distributed
    expected_world = distributed.world_size
    world_size = int(os.environ.get("WORLD_SIZE", "1"))
    if world_size != expected_world:
        raise RuntimeError(
            f"torchrun WORLD_SIZE={world_size} does not match configured GPU count {expected_world}"
        )
    rank = int(os.environ.get("RANK", "0"))
    local_rank = int(os.environ.get("LOCAL_RANK", "0"))
    if not torch.cuda.is_available():
        raise RuntimeError("Server DDP training requires CUDA; CPU fallback is disabled")
    if not dist.is_nccl_available():
        raise RuntimeError("This PyTorch build does not provide the required NCCL backend")
    visible_devices = torch.cuda.device_count()
    if visible_devices != expected_world:
        raise RuntimeError(
            "Visible CUDA device count does not match server config: "
            f"torch sees {visible_devices}, expected {expected_world}. "
            "Launch through train_server.py so gpu_ids is mapped to CUDA_VISIBLE_DEVICES."
        )
    if not 0 <= local_rank < visible_devices:
        raise RuntimeError(
            f"LOCAL_RANK={local_rank} is outside the {visible_devices} visible CUDA devices"
        )
    cpu_threads_per_rank = _configure_rank_cpu_threads(
        world_size, distributed.data_workers_per_rank
    )
    torch.cuda.set_device(local_rank)
    device = torch.device("cuda", local_rank)
    dist.init_process_group(
        backend=distributed.backend,
        timeout=timedelta(seconds=distributed.timeout_seconds),
    )
    install_distributed_signal_handlers()
    try:
        iql_cfg = config.section("iql")
        logging_cfg = config.section("logging")
        reward_mode = (
            "cumulative_primitive_steps"
            if config.section("reward")["accumulate_primitive_steps"]
            else "macro_action"
        )
        rynnvalue_enabled = bool(config.section("reward")["rynnvalue"])
        seed = int(iql_cfg["seed"])
        random.seed(seed)
        np.random.seed(seed)
        torch.manual_seed(seed)
        torch.cuda.manual_seed(seed)
        global_batch, local_batch = validate_global_batch(config.raw, distributed)
        manifest = load_manifest(config)
        reward_index = load_reward_index(config)
        _validate_single_task_micro_batch(manifest, global_batch)
        cache_directory = replay_cache_path(config, server.replay_cache, manifest)
        validate_replay_cache(cache_directory, config, manifest)

        if rank == 0:
            print(
                "SERVER TRAIN loading VLA replicas | "
                f"world={world_size} | global_batch={global_batch} | local_batch={local_batch}",
                flush=True,
            )
        components = _rank_components(config, device)
        dataset = CachedReplayDataset(
            config,
            cache_directory,
            components.action_stats,
            components.proprio_stats,
        )
        q_ddp = DDP(
            DoubleQModule().to(device),
            device_ids=[local_rank],
            output_device=local_rank,
            broadcast_buffers=False,
            find_unused_parameters=False,
            gradient_as_bucket_view=True,
        )
        value_ddp = DDP(
            ValueNetwork(PROPRIO_DIM).to(device),
            device_ids=[local_rank],
            output_device=local_rank,
            broadcast_buffers=False,
            find_unused_parameters=False,
            gradient_as_bucket_view=True,
        )
        actor_ddp = DDP(
            ActorOverlayModule(
                components.action_head, components.proprio_projector
            ).to(device),
            device_ids=[local_rank],
            output_device=local_rank,
            broadcast_buffers=False,
            # The upstream Pro head intentionally keeps a legacy FiLM block in
            # its checkpoint even though predict_action does not execute it.
            find_unused_parameters=True,
            gradient_as_bucket_view=True,
        )
        q_module = q_ddp.module
        value_module = value_ddp.module
        actor_module = actor_ddp.module
        target_q = copy.deepcopy(q_module).requires_grad_(False).to(device)

        q_optimizer = _zero_optimizer(
            q_ddp.parameters(),
            _optimizer_type(str(iql_cfg["critic_optimizer"])),
            lr=float(iql_cfg["critic_lr"]),
            weight_decay=float(iql_cfg["critic_weight_decay"]),
        )
        value_optimizer = _zero_optimizer(
            value_ddp.parameters(),
            _optimizer_type(str(iql_cfg["value_optimizer"])),
            lr=float(iql_cfg["value_lr"]),
            weight_decay=float(iql_cfg["value_weight_decay"]),
        )
        actor_optimizer = _zero_optimizer(
            actor_ddp.parameters(),
            torch.optim.AdamW,
            lr=float(iql_cfg["policy_peak_lr"]),
            betas=(0.9, 0.95),
            eps=1e-8,
            weight_decay=1e-10,
        )

        total_steps = int(iql_cfg["train_steps"])
        warmup = int(iql_cfg["critic_warmup_steps"])
        accumulation = int(iql_cfg["gradient_accumulation_steps"])
        start_step = 0
        if iql_cfg["resume_checkpoint"] is not None:
            start_step = _restore_checkpoint(
                checkpoint=Path(iql_cfg["resume_checkpoint"]),
                components=components,
                q_module=q_module,
                target_q=target_q,
                value_module=value_module,
                q_optimizer=q_optimizer,
                value_optimizer=value_optimizer,
                actor_optimizer=actor_optimizer,
                config=config,
                manifest=manifest,
                reward_index=reward_index,
            )
            if start_step >= total_steps:
                raise ValueError(
                    f"Resume step {start_step} must be smaller than train_steps={total_steps}"
                )
        sampler = DeterministicDistributedBatchSampler(
            dataset_size=len(dataset),
            global_batch_size=global_batch,
            rank=rank,
            world_size=world_size,
            seed=seed + 1,
            start_step=start_step,
            end_step=total_steps,
        )
        loader_kwargs: dict[str, Any] = {
            "batch_sampler": sampler,
            "num_workers": distributed.data_workers_per_rank,
            "pin_memory": distributed.pin_memory,
        }
        if distributed.data_workers_per_rank:
            loader_kwargs.update(
                prefetch_factor=distributed.prefetch_factor,
                persistent_workers=distributed.persistent_workers,
            )
        loader = DataLoader(dataset, **loader_kwargs)

        run_id, run_dir = _broadcast_run_directory(config, rank)
        checkpoint_root = run_dir / "checkpoints"
        if rank == 0:
            checkpoint_root.mkdir(parents=True, exist_ok=False)
            (run_dir / "effective_config.yaml").write_text(
                yaml.safe_dump(config.raw, sort_keys=False), encoding="utf-8"
            )
            atomic_json(run_dir / "server_config.json", {
                "schema_version": 1,
                "gpu_ids": list(distributed.gpu_ids),
                "world_size": world_size,
                "backend": distributed.backend,
                "zero_stage": distributed.zero_stage,
                "global_micro_batch_size": global_batch,
                "local_micro_batch_size": local_batch,
                "actor_global_batch_size": global_batch * accumulation,
                "cpu_threads_per_rank": cpu_threads_per_rank,
                "data_workers_per_rank": distributed.data_workers_per_rank,
                "replay_cache": str(cache_directory),
                "reward_mode": reward_mode,
                "rynnvalue_reward_enabled": rynnvalue_enabled,
            })
            atomic_json(run_dir / "provenance.json", {
                "schema_version": 1,
                "trainer": "server-ddp-zero1",
                "code_version": _code_version(),
                "config_sha256": config.digest,
                "dataset_sha256": manifest["dataset_sha256"],
                "reward_sha256": stable_hash(reward_index),
                "base_checkpoint": config.section("vla")["base_checkpoint"],
            })
        dist.barrier()

        actor_optimizer.zero_grad(set_to_none=True)
        progress = TrainingProgressReporter(
            total_steps=total_steps,
            start_step=start_step,
            interval_steps=int(logging_cfg["console_interval_steps"]),
            warmup_steps=warmup,
            samples_per_step=global_batch,
        )
        latest_checkpoint: Path | None = None
        start_time = time.monotonic()
        current_actor_lr = float(actor_optimizer.param_groups[0]["lr"])
        tensor_keys = {
            "pixels", "next_pixels", "proprio", "next_proprio", "actions",
            "action_mask", "reward", "bootstrap_mask", "chunk_length",
        }
        with ExitStack() as stack:
            metrics_file = None
            writer = None
            wandb_run = None
            if rank == 0:
                metrics_file = stack.enter_context(
                    (run_dir / "metrics.jsonl").open("w", encoding="utf-8")
                )
                if logging_cfg["tensorboard"]:
                    from torch.utils.tensorboard import SummaryWriter

                    writer = stack.enter_context(SummaryWriter(
                        log_dir=str(run_dir / "tensorboard"),
                        flush_secs=float(logging_cfg["flush_seconds"]),
                    ))
                wandb_run = _initialize_wandb(config, run_dir, run_id)
                if wandb_run is not None:
                    stack.callback(wandb_run.finish)
                print(
                    "SERVER TRAIN ready | "
                    f"run={run_id} | steps={start_step}->{total_steps} | "
                    f"world={world_size} | global/local batch={global_batch}/{local_batch} | "
                    f"actor_batch={global_batch * accumulation} | "
                    f"reward_mode={reward_mode} | "
                    f"reward_source={'rynnvalue_pbrs' if rynnvalue_enabled else 'sparse_only'}",
                    flush=True,
                )
            iterator = iter(loader)
            for step in range(start_step, total_steps):
                step_started = time.monotonic()
                batch = next(iterator)
                data_wait_ms = (time.monotonic() - step_started) * 1000.0
                transfer_start_event = torch.cuda.Event(enable_timing=True)
                transfer_end_event = torch.cuda.Event(enable_timing=True)
                transfer_start_event.record()
                critic_batch = {
                    key: value.to(device, non_blocking=True)
                    for key, value in batch.items()
                    if key in tensor_keys and isinstance(value, torch.Tensor)
                }
                transfer_end_event.record()

                critic_start_event = torch.cuda.Event(enable_timing=True)
                critic_end_event = torch.cuda.Event(enable_timing=True)
                critic_start_event.record()
                with torch.no_grad():
                    target_q1, target_q2 = target_q(
                        critic_batch["pixels"], critic_batch["proprio"],
                        critic_batch["actions"], critic_batch["action_mask"],
                    )
                    target_q_for_value = torch.minimum(target_q1, target_q2)
                value = value_ddp(critic_batch["pixels"], critic_batch["proprio"])
                value_loss = expectile_loss(
                    target_q_for_value - value, float(iql_cfg["expectile"])
                )
                value_optimizer.zero_grad(set_to_none=True)
                value_loss.backward()
                nn.utils.clip_grad_norm_(
                    value_ddp.parameters(), float(iql_cfg["value_max_grad_norm"])
                )
                value_optimizer.step()

                with torch.no_grad():
                    next_value = value_module(
                        critic_batch["next_pixels"], critic_batch["next_proprio"]
                    )
                    bellman_target = chunk_bellman_target(
                        critic_batch["reward"],
                        next_value,
                        critic_batch["chunk_length"],
                        critic_batch["bootstrap_mask"],
                        float(config.section("reward")["gamma"]),
                        bool(config.section("reward")["accumulate_primitive_steps"]),
                    )
                q1, q2 = q_ddp(
                    critic_batch["pixels"], critic_batch["proprio"],
                    critic_batch["actions"], critic_batch["action_mask"],
                )
                q_loss = 0.5 * (
                    F.mse_loss(q1, bellman_target) + F.mse_loss(q2, bellman_target)
                )
                q_optimizer.zero_grad(set_to_none=True)
                q_loss.backward()
                nn.utils.clip_grad_norm_(
                    q_ddp.parameters(), float(iql_cfg["critic_max_grad_norm"])
                )
                q_optimizer.step()
                _soft_update(q_module, target_q, float(iql_cfg["target_tau"]))
                with torch.no_grad():
                    updated_q1, updated_q2 = q_module(
                        critic_batch["pixels"], critic_batch["proprio"],
                        critic_batch["actions"], critic_batch["action_mask"],
                    )
                    updated_q = torch.minimum(updated_q1, updated_q2)
                    updated_value = value_module(
                        critic_batch["pixels"], critic_batch["proprio"]
                    )
                    advantage = updated_q - updated_value
                    weights = (
                        torch.ones_like(advantage)
                        if step < warmup
                        else advantage_weights(
                            advantage,
                            float(iql_cfg["beta"]),
                            float(iql_cfg["max_advantage_weight"]),
                        )
                    )
                critic_end_event.record()

                actor_input_start_event = torch.cuda.Event(enable_timing=True)
                actor_input_start_event.record()
                backbone_start_event = torch.cuda.Event(enable_timing=True)
                backbone_end_event = torch.cuda.Event(enable_timing=True)
                inputs = processor_inputs(
                    components,
                    list(batch["prompt"]),
                    batch["agent_image"].cpu().numpy(),
                    batch["wrist_image"].cpu().numpy(),
                )
                backbone_start_event.record()
                hidden = extract_action_hidden_states(components, inputs)
                backbone_end_event.record()
                actor_start_event = torch.cuda.Event(enable_timing=True)
                actor_end_event = torch.cuda.Event(enable_timing=True)
                actor_start_event.record()
                actor_step = (step + 1) % accumulation == 0 or step == total_steps - 1
                synchronization = nullcontext() if actor_step else actor_ddp.no_sync()
                with synchronization:
                    prediction = actor_ddp(
                        hidden, critic_batch["proprio"].to(dtype=torch.bfloat16)
                    )
                    actor_loss = weighted_masked_l1(
                        prediction,
                        critic_batch["actions"],
                        critic_batch["action_mask"],
                        weights,
                    )
                    (actor_loss / accumulation).backward()
                action_metrics = _action_diagnostics(
                    prediction, critic_batch["actions"], critic_batch["action_mask"]
                )
                actor_grad_norm = None
                action_head_parameter_norm = None
                proprio_projector_parameter_norm = None
                if actor_step:
                    actor_grad_norm = float(nn.utils.clip_grad_norm_(actor_ddp.parameters(), 1.0))
                    current_actor_lr = _actor_lr(
                        step,
                        total_steps,
                        float(iql_cfg["policy_peak_lr"]),
                        float(iql_cfg["policy_final_lr"]),
                        warmup,
                    )
                    for group in actor_optimizer.param_groups:
                        group["lr"] = current_actor_lr
                    actor_optimizer.step()
                    actor_optimizer.zero_grad(set_to_none=True)
                    action_head_parameter_norm = _module_parameter_norm(actor_module.action_head)
                    proprio_projector_parameter_norm = _module_parameter_norm(
                        actor_module.proprio_projector
                    )
                actor_end_event.record()

                local_means = {
                    "q_loss": float(q_loss.detach()),
                    "value_loss": float(value_loss.detach()),
                    "q_mean": float(updated_q.mean()),
                    "value_mean": float(updated_value.mean()),
                    "advantage_mean": float(advantage.mean()),
                    "actor_loss": float(actor_loss.detach()),
                    "advantage_weight_mean": float(weights.mean()),
                    **action_metrics,
                }
                # Synchronize once at the end of the complete optimization
                # step, never between phases. CUDA events then report device
                # timeline durations without destroying phase overlap.
                actor_end_event.synchronize()
                host_to_device_ms = transfer_start_event.elapsed_time(transfer_end_event)
                critic_update_ms = critic_start_event.elapsed_time(critic_end_event)
                actor_input_ms = actor_input_start_event.elapsed_time(backbone_start_event)
                backbone_forward_ms = backbone_start_event.elapsed_time(backbone_end_event)
                actor_update_ms = actor_start_event.elapsed_time(actor_end_event)
                metric = _all_reduce(local_means, operation="mean")
                slowest = _all_reduce({
                    "data_wait_ms": data_wait_ms,
                    "host_to_device_ms": host_to_device_ms,
                    "actor_input_ms": actor_input_ms,
                    "backbone_forward_ms": backbone_forward_ms,
                    "critic_update_ms": critic_update_ms,
                    "actor_update_ms": actor_update_ms,
                    "ddp_communication_ms": _ddp_communication_ms(
                        q_ddp, value_ddp, actor_ddp
                    ),
                    "slowest_rank_step_ms": (time.monotonic() - step_started) * 1000.0,
                }, operation="max")
                memory_gib = torch.cuda.max_memory_allocated(device) / (1024.0 ** 3)
                memory_min = _all_reduce({"memory": memory_gib}, operation="min")["memory"]
                memory_max = _all_reduce({"memory": memory_gib}, operation="max")["memory"]
                metric.update(slowest)
                metric.update({
                    "step": step + 1,
                    "actor_learning_rate": current_actor_lr,
                    "actor_grad_norm": actor_grad_norm,
                    "action_head_parameter_norm": action_head_parameter_norm,
                    "proprio_projector_parameter_norm": proprio_projector_parameter_norm,
                    "elapsed_seconds": time.monotonic() - start_time,
                    "world_size": world_size,
                    "global_micro_batch_size": global_batch,
                    "local_micro_batch_size": local_batch,
                    "micro_batch_size": global_batch,
                    "actor_global_batch_size": global_batch * accumulation,
                    "actor_effective_batch_size": global_batch * accumulation,
                    "cpu_threads_per_rank": cpu_threads_per_rank,
                    "data_workers_per_rank": distributed.data_workers_per_rank,
                    "cuda_peak_memory_gib": memory_max,
                    "cuda_peak_memory_gib_min": memory_min,
                    "cuda_peak_memory_gib_max": memory_max,
                })
                if rank == 0:
                    should_report = progress.update(metric)
                    metric["global_samples_per_second"] = metric["samples_per_second"]
                    assert metrics_file is not None
                    metrics_file.write(json.dumps(metric, sort_keys=True) + "\n")
                    metrics_file.flush()
                    _server_tensorboard(writer, metric)
                    completed = step + 1 - start_step
                    if (
                        wandb_run is not None
                        and (
                            completed == 1
                            or step + 1 == total_steps
                            or completed % int(logging_cfg["wandb"]["log_interval_steps"]) == 0
                        )
                    ):
                        _server_wandb(wandb_run, metric)
                    if should_report:
                        print(
                            progress.format(metric)
                            + f" | world={world_size} | local_batch={local_batch}"
                            + f" | data={metric['data_wait_ms']:.1f}ms"
                            + f" | backbone={metric['backbone_forward_ms']:.1f}ms",
                            flush=True,
                        )

                checkpoint_due = (
                    (step + 1) % int(iql_cfg["checkpoint_interval"]) == 0
                    or step + 1 == total_steps
                )
                # A server checkpoint does not serialize partially accumulated
                # actor gradients. Keep all ranks running until the next actor
                # optimizer boundary so interruption/resume remains exact.
                stop_pending = _stop_requested()
                stopping = stop_pending and actor_step
                if checkpoint_due or stopping:
                    latest_checkpoint = _save_checkpoint(
                        directory=checkpoint_root,
                        step=step + 1,
                        rank=rank,
                        components=components,
                        q_module=q_module,
                        target_q=target_q,
                        value_module=value_module,
                        q_optimizer=q_optimizer,
                        value_optimizer=value_optimizer,
                        actor_optimizer=actor_optimizer,
                        config=config,
                        server=server,
                        manifest=manifest,
                        reward_index=reward_index,
                    )
                if stopping:
                    if rank == 0:
                        atomic_json(run_dir / "summary.json", {
                            "schema_version": 1,
                            "status": "canceled",
                            "steps": step + 1,
                            "world_size": world_size,
                            "global_micro_batch_size": global_batch,
                            "cpu_threads_per_rank": cpu_threads_per_rank,
                            "reward_mode": reward_mode,
                            "rynnvalue_reward_enabled": rynnvalue_enabled,
                            "cancel_checkpoint": str(latest_checkpoint),
                        })
                    raise DistributedTrainingCancelled(
                        "Distributed training canceled after a safe checkpoint"
                    )
            if rank == 0 and wandb_run is not None:
                wandb_run.summary["status"] = "optimization_completed"
                wandb_run.summary["last_step"] = total_steps

        assert latest_checkpoint is not None
        policy: Path | None = None
        if rank == 0:
            registry = Path(config.section("paths")["policy_registry"])
            registry.mkdir(parents=True, exist_ok=True)
            policy = _publish_overlay(
                latest_checkpoint,
                registry,
                total_steps,
                config,
                components,
                manifest,
                reward_index,
            )
            atomic_json(run_dir / "summary.json", {
                "schema_version": 1,
                "status": "completed",
                "trainer": "server-ddp-zero1",
                "steps": total_steps,
                "world_size": world_size,
                "global_micro_batch_size": global_batch,
                "local_micro_batch_size": local_batch,
                "actor_global_batch_size": global_batch * accumulation,
                "cpu_threads_per_rank": cpu_threads_per_rank,
                "data_workers_per_rank": distributed.data_workers_per_rank,
                "reward_mode": reward_mode,
                "rynnvalue_reward_enabled": rynnvalue_enabled,
                "dataset_sha256": manifest["dataset_sha256"],
                "reward_sha256": stable_hash(reward_index),
                "policy_overlay": str(policy),
                "resumed_from_step": start_step,
            })
        result: list[Any] = [str(policy) if rank == 0 and policy is not None else None]
        dist.broadcast_object_list(result, src=0)
        dist.barrier()
        return Path(result[0]) if rank == 0 else None
    finally:
        if dist.is_initialized():
            dist.destroy_process_group()
