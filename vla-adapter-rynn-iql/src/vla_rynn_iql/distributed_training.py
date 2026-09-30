"""Single-node 1–8 GPU runner, independent of the UI's single-process runner."""
from __future__ import annotations

import json
import os
import random
import signal
import time
from contextlib import ExitStack, nullcontext
from datetime import timedelta
from pathlib import Path

import numpy as np
import torch
import torch.distributed as dist
import yaml
from torch.utils.data import DataLoader

from .config import LoadedConfig
from .data import load_manifest, training_replay_policy
from .distributed_algorithms import ActorModule, CommunicationMeter, DistributedAlgorithm, wrap, zero_optimizer
from .io import atomic_json
from .methods import actor_lr_warmup, training_method
from .model_adaptation import parameter_counts, trainable_parameters
from .models import model_backend, model_signature
from .monitoring import TrainingProgressReporter, log_tensorboard_metric, log_wandb_metric
from .rewards import load_reward_index, reward_manifest_digest
from .server_config import validate_global_batch
from .server_replay import CachedReplayDataset, DistributedBatchSampler, build_replay_cache
from .training import (_actor_lr, _action_diagnostics, _initialize_wandb, _publish_overlay,
                       _restore_checkpoint, _save_checkpoint, _validate_single_task_micro_batch)


def rank_zero_call(function):
    """Communicate startup/IO failures instead of stranding other ranks at a barrier."""
    packet = [None]
    if dist.get_rank() == 0:
        try:
            packet[0] = {"value": function()}
        except Exception as exc:
            packet[0] = {"error": f"{type(exc).__name__}: {exc}"}
    dist.broadcast_object_list(packet, src=0)
    if "error" in packet[0]:
        raise RuntimeError(packet[0]["error"])
    return packet[0]["value"]


def rng_state():
    return {"torch": torch.get_rng_state(), "numpy": np.random.get_state(),
            "python": random.getstate(), "cuda": torch.cuda.get_rng_state().cpu() if torch.cuda.is_available() else None}


def restore_rng(state):
    torch.set_rng_state(state["torch"].cpu())
    random.setstate(state["python"])
    np.random.set_state(state["numpy"])
    if state["cuda"] is not None:
        torch.cuda.set_rng_state(state["cuda"].cpu())


def save_distributed_checkpoint(root, step, components, algorithm, optimizer, generator,
                                config, manifest, reward, global_batch):
    # ZeRO state is collective; only rank 0 serializes the consolidated result.
    states = [None] * dist.get_world_size()
    dist.all_gather_object(states, rng_state())
    for item in [*algorithm.optimizers, optimizer]:
        item.consolidate_state_dict(to=0)

    def save():
        # Do not initialize CUDA contexts for every other rank's GPU while saving.
        state = states[0]
        rng = {"torch_rng": state["torch"], "numpy_rng": state["numpy"], "python_rng": state["python"],
               "cuda_rng": [state["cuda"]] if state["cuda"] is not None else None}
        path = _save_checkpoint(root, step, components, algorithm, optimizer, generator,
                                config, manifest, reward, rng=rng)
        torch.save(states, path / "rank_rng.pt")
        atomic_json(path / "server.json", {"schema_version": 1, "world_size": dist.get_world_size(),
                    "global_micro_batch_size": global_batch,
                    "seed": config.section("training")["seed"], "actor_boundary": True,
                    "gradient_accumulation_steps": config.section("training")["gradient_accumulation_steps"]})
        return str(path)
    return Path(rank_zero_call(save))


def restore_distributed_checkpoint(path, components, algorithm, optimizer, generator,
                                   device, config, manifest, reward, global_batch):
    metadata = json.loads((path / "server.json").read_text())
    training = config.section("training")
    if (metadata["global_micro_batch_size"] != global_batch
            or metadata["gradient_accumulation_steps"] != training["gradient_accumulation_steps"]
            or metadata["seed"] != training["seed"]):
        raise ValueError("Resume must retain global batch, gradient accumulation and seed")
    step = _restore_checkpoint(path, components, algorithm, optimizer, generator,
        device, config, manifest, reward, restore_random=False, checkpoint_device="cpu")
    if not metadata.get("actor_boundary"):
        raise ValueError("Resume requires a complete actor accumulation boundary")
    states = torch.load(path / "rank_rng.pt", map_location="cpu", weights_only=False)
    rank = dist.get_rank()
    if len(states) == dist.get_world_size():
        restore_rng(states[rank])
    else:
        # Optimizers and global sample order survive world-size changes. Random
        # dropout does not have a world-size-independent tensor stream.
        if rank == 0:
            print("RESUME: world size changed; optimizer/sample order retained, dropout RNG is reseeded", flush=True)
        seed = training["seed"] + step * 8 + rank
        random.seed(seed)
        np.random.seed(seed)
        torch.manual_seed(seed)
        if device.type == "cuda":
            torch.cuda.manual_seed(seed)
    return step


def _reduce_metrics(metrics, device, valid_actions=1):
    names = sorted(key for key, value in metrics.items() if isinstance(value, (float, int)))
    # Per-action diagnostics use valid primitive actions, not rank averages;
    # terminal/interrupted chunks can have different lengths on different ranks.
    weights = [valid_actions if key.startswith(("actor_l1_", "actor_gripper_")) else 1 for key in names]
    values = torch.tensor([[metrics[key] * weight, weight] for key, weight in zip(names, weights)],
                          device=device, dtype=torch.float64)
    dist.all_reduce(values)
    means = values[:, 0] / values[:, 1].clamp_min(1)
    return dict(zip(names, means.cpu().tolist()))


def train_distributed(config: LoadedConfig, distributed, cache_root: Path, run_dir: Path):
    if not torch.cuda.is_available() or not dist.is_nccl_available():
        raise RuntimeError("Server training requires CUDA/NCCL; CPU fallback is disabled")
    local_rank = int(os.environ.get("LOCAL_RANK", "-1"))
    if local_rank < 0 or int(os.environ.get("WORLD_SIZE", "0")) != distributed.world_size:
        raise ValueError("Launch through train_server.py / torchrun with the configured GPU count")
    if torch.cuda.device_count() < distributed.world_size:
        raise ValueError(f"Requested {distributed.world_size} GPUs but CUDA sees only {torch.cuda.device_count()}")
    torch.cuda.set_device(local_rank)
    device = torch.device("cuda", local_rank)
    dist.init_process_group("nccl", timeout=timedelta(seconds=distributed.timeout_seconds))
    stop = False
    def request_stop(*_):
        nonlocal stop
        stop = True
    original_handlers = {name: signal.signal(name, request_stop) for name in (signal.SIGINT, signal.SIGTERM)}
    try:
        _train(config, distributed, cache_root, run_dir, device,
               lambda: stop or (run_dir / "cancel.request").exists())
    finally:
        for name, handler in original_handlers.items():
            signal.signal(name, handler)
        dist.destroy_process_group()


def _train(config, distributed, cache_root, run_dir, device, stopping):
    rank, world = dist.get_rank(), dist.get_world_size()
    cfg = config.section("training")
    seed, accumulation, total = cfg["seed"], cfg["gradient_accumulation_steps"], cfg["train_steps"]
    if cfg["checkpoint_interval"] % accumulation:
        raise ValueError("checkpoint_interval must be divisible by gradient_accumulation_steps")
    global_batch, local_batch = validate_global_batch(config.raw, distributed)
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    raw = {**config.raw, "training": {**cfg, "device": str(device)}}
    config = LoadedConfig(config.path, raw)
    method = training_method(raw)
    manifest = load_manifest(config)
    _validate_single_task_micro_batch(manifest, global_batch)
    reward = rank_zero_call(lambda: load_reward_index(config) if method.requires_rewards else None)
    backend = model_backend(raw)
    if rank == 0:
        print(f"MODEL loading | method={method.name} | world={world} | config={model_signature(raw)}", flush=True)
    components = backend.load_components(config, device=device)
    cache_info = rank_zero_call(lambda: tuple(map(str, build_replay_cache(
        config, cache_root, components.action_stats, components.proprio_stats, reward))))
    dataset = CachedReplayDataset(Path(cache_info[0]))
    meter = CommunicationMeter()
    algorithm = DistributedAlgorithm(config, device, meter)
    # Pro retains a legacy FiLM block not used by every prediction path, even
    # when the backbone is frozen. Never assume all registered parameters fire.
    actor = wrap(ActorModule(components, backend, device), device, unused=True, meter=meter)
    parameters = trainable_parameters(components)
    optimizer = zero_optimizer(parameters, torch.optim.AdamW, lr=cfg["policy_peak_lr"],
                               betas=(.9, .95), eps=1e-8, weight_decay=1e-10)
    generator = torch.Generator().manual_seed(seed + 1)
    start = 0
    if cfg["resume_checkpoint"]:
        start = restore_distributed_checkpoint(Path(cfg["resume_checkpoint"]), components, algorithm,
            optimizer, generator, device, config, manifest, reward, global_batch)
    if start >= total:
        raise ValueError("Resume step must be less than train_steps")
    sampler = DistributedBatchSampler(len(dataset), global_batch, rank, world, seed + 1, start, total)
    loader_kwargs = dict(batch_sampler=sampler, num_workers=distributed.data_workers_per_rank,
                         pin_memory=distributed.pin_memory,
                         generator=torch.Generator().manual_seed(seed + rank + 100))
    if distributed.data_workers_per_rank:
        loader_kwargs.update(prefetch_factor=distributed.prefetch_factor,
                             persistent_workers=distributed.persistent_workers,
                             multiprocessing_context="spawn")
    loader = DataLoader(dataset, **loader_kwargs)
    progress = TrainingProgressReporter(total_steps=total, start_step=start,
        interval_steps=config.section("logging")["console_interval_steps"],
        warmup_steps=config.section("iql")["critic_warmup_steps"] if method.name == "iql" else 0,
        samples_per_step=global_batch)
    logs = config.section("logging")
    if rank == 0:
        atomic_json(run_dir / "provenance.json", {"algorithm": method.name, "world_size": world,
            "global_micro_batch_size": global_batch, "local_micro_batch_size": local_batch,
            "actor_effective_batch_size": global_batch * accumulation,
            "model_config": model_signature(raw), "model_parameters": parameter_counts(components),
            "dataset_sha256": manifest["dataset_sha256"],
            "reward_sha256": reward_manifest_digest(reward) if reward else None,
            "replay_policy": training_replay_policy(config.section("data")["include_post_success"]),
            "cache": cache_info[0], "cache_hit": cache_info[1] == "True"})
        print(f"TRAIN ready | {method.name} | steps={start}->{total} | global={global_batch} "
              f"| per-rank={local_batch} | actor batch={global_batch * accumulation}", flush=True)
    optimizer.zero_grad(set_to_none=True)
    started = time.perf_counter()
    checkpoint = None
    cancelled = False
    with ExitStack() as stack:
        writer, wandb, metrics_file = None, None, None
        if rank == 0:
            metrics_file = stack.enter_context((run_dir / "metrics.jsonl").open("w"))
            if logs["tensorboard"]:
                from torch.utils.tensorboard import SummaryWriter
                writer = stack.enter_context(SummaryWriter(str(run_dir / "tensorboard")))
            wandb = _initialize_wandb(config, run_dir, run_dir.name)
            if wandb is not None:
                stack.callback(wandb.finish)
        iterator = iter(loader)
        for step in range(start, total):
            begin = time.perf_counter()
            communication_before = meter.seconds
            batch = next(iterator)
            # Persist the same sampling RNG as single-card training, although
            # workers independently reconstruct batches for prefetch/resume.
            torch.randint(len(dataset), (global_batch,), generator=generator)
            data_time = time.perf_counter() - begin
            tensors = {key: value.to(device, non_blocking=True) for key, value in batch.items()
                       if isinstance(value, torch.Tensor) and key not in {"agent_image", "wrist_image"}}
            torch.cuda.synchronize(device)
            tick = time.perf_counter()
            context, metrics = algorithm.update(tensors, step)
            torch.cuda.synchronize(device)
            critic_time = time.perf_counter() - tick
            boundary = (step + 1) % accumulation == 0 or step + 1 == total
            tick = time.perf_counter()
            with nullcontext() if boundary else actor.no_sync():
                prediction = actor(batch)
                loss = algorithm.actor_loss(prediction, tensors, context)
                (loss / accumulation).backward()
            torch.cuda.synchronize(device)
            actor_time = time.perf_counter() - tick
            metrics.update(_action_diagnostics(prediction, tensors["actions"], tensors["action_mask"]))
            if boundary:
                metrics["actor_grad_norm"] = float(torch.nn.utils.clip_grad_norm_(parameters, 1.0))
                lr = _actor_lr(step, total, cfg["policy_peak_lr"], cfg["policy_final_lr"], actor_lr_warmup(raw))
                for group in optimizer.param_groups:
                    group["lr"] = lr
                optimizer.step()
                optimizer.zero_grad(set_to_none=True)
            torch.cuda.synchronize(device)
            duration = time.perf_counter() - begin
            peaks = torch.tensor([duration, torch.cuda.max_memory_allocated(device) / 1024**3], device=device)
            dist.all_reduce(peaks, op=dist.ReduceOp.MAX)
            metrics.update(actor_loss=float(loss.detach()), actor_learning_rate=optimizer.param_groups[0]["lr"],
                data_wait_seconds=data_time, critic_seconds=critic_time, actor_seconds=actor_time,
                gradient_communication_seconds=meter.seconds - communication_before)
            metrics = _reduce_metrics(metrics, device, float(tensors["action_mask"].sum()))
            metrics.update(step=step + 1, algorithm=method.name, world_size=world,
                micro_batch_size=global_batch, local_micro_batch_size=local_batch,
                actor_effective_batch_size=global_batch * accumulation,
                elapsed_seconds=time.perf_counter() - started, slowest_rank_seconds=float(peaks[0]),
                cuda_peak_memory_gib=float(peaks[1]))
            if rank == 0:
                report = progress.update(metrics)
                metrics_file.write(json.dumps(metrics) + "\n")
                metrics_file.flush()
                atomic_json(run_dir / "progress.json", {"status": "RUNNING", **metrics})
                log_tensorboard_metric(writer, metrics)
                if (step == start or boundary and (step + 1) % logs["wandb"]["log_interval_steps"] == 0
                        or step + 1 == total):
                    log_wandb_metric(wandb, metrics)
                if report:
                    print(progress.format(metrics), flush=True)
            requested = torch.tensor(int(stopping()), device=device)
            dist.all_reduce(requested, op=dist.ReduceOp.MAX)
            cancelled = bool(requested) and boundary
            if (step + 1) % cfg["checkpoint_interval"] == 0 or step + 1 == total or cancelled:
                checkpoint = save_distributed_checkpoint(run_dir / "checkpoints", step + 1, components,
                    algorithm, optimizer, generator, config, manifest, reward, global_batch)
            if cancelled:
                break
    policy = None
    if not cancelled:
        policy = rank_zero_call(lambda: str(_publish_overlay(checkpoint,
            Path(config.section("paths")["policy_registry"]), total, config, components, manifest, reward)))
    if rank == 0:
        result = {"status": "INTERRUPTED" if cancelled else "COMPLETED", "step": step + 1,
                  "checkpoint": str(checkpoint), "policy_overlay": policy,
                  "elapsed_seconds": time.perf_counter() - started}
        atomic_json(run_dir / "summary.json", result)
        atomic_json(run_dir / "progress.json", {**metrics, **result})
    dist.barrier()
