"""Optional observational diagnostics, isolated from optimizer and sampler state."""
from contextlib import contextmanager
import random

import numpy as np
import torch
import torch.distributed as dist
from torch.utils.data import default_collate


def due(logging, step):
    config = logging.get("diagnostics", {})
    return config.get("enabled", False) and step % config.get("interval_steps", 100) == 0


def validation_dataset(config, components):
    # Retain at most one decoded episode, as in replay cache construction.
    from train.replay_cache import _CachedActions
    return _CachedActions(config, components.action_stats, components.proprio_stats, "validation")


def weight_statistics(weights, batch, episodes, *, distributed=False):
    values = weights.detach().float().cpu().reshape(-1).tolist()
    rows = []
    if len(values) != len(batch["run_id"]):
        raise ValueError("Diagnostic weights must align with replay samples")
    for weight, run_id, start, source in zip(values, batch["run_id"], batch["start"], batch["action_source"]):
        episode = episodes[run_id]
        terminal = episode.get("terminal_step")
        post = terminal is not None and int(start) >= int(terminal) + 1
        groups = []
        if source == "human":
            groups.append("human")
            if terminal is not None:
                groups.append("human_post_success" if post else "human_pre_success")
        elif source in {"policy", "policy_requery"}:
            groups.append("policy")
        if post:
            groups.append("post_success")
        if episode.get("kind") != "branch" and terminal is None:
            groups.append("failed_parent")
        rows.append((weight, groups))
    if distributed:
        gathered = [None] * dist.get_world_size()
        dist.all_gather_object(gathered, rows)
        rows = [row for shard in gathered for row in shard]
    return summarize_weights(rows)


def summarize_weights(rows):
    values = np.asarray([row[0] for row in rows], dtype=np.float64)
    if not len(values) or not np.isfinite(values).all():
        raise ValueError("Diagnostic weights must be nonempty and finite")
    result = {"advantage_weight_count": len(values), "advantage_weight_max": float(values.max())}
    for name, q in (("p05", .05), ("p50", .5), ("p95", .95)):
        result[f"advantage_weight_{name}"] = float(np.quantile(values, q))
    for group in ("human", "policy", "post_success", "human_pre_success", "human_post_success", "failed_parent"):
        selected = [weight for weight, groups in rows if group in groups]
        prefix = f"advantage_weight_{group}"
        result.update({prefix + "_count": len(selected), prefix + "_sum": float(sum(selected)),
                       prefix + "_mean": float(np.mean(selected)) if selected else None})
    return result


@contextmanager
def isolated_validation(components, seed=0):
    python_state, numpy_state = random.getstate(), np.random.get_state()
    modules = {}
    for name in ("model", "action_head", "proprio_projector"):
        module = getattr(components, name, None)
        if isinstance(module, torch.nn.Module):
            modules.update({child: child.training for child in module.modules()})
    try:
        devices = [torch.cuda.current_device()] if torch.cuda.is_available() else []
        with torch.random.fork_rng(devices=devices):
            random.seed(seed)
            np.random.seed(seed)
            torch.random.default_generator.manual_seed(seed)
            if devices:
                torch.cuda.manual_seed(seed)
            for module in modules:
                module.training = False
            with torch.no_grad():
                yield
    finally:
        random.setstate(python_state)
        np.random.set_state(numpy_state)
        for module, training in modules.items():
            module.training = training


def _validation_totals(backend, components, dataset, device, batch_size, rank, world):
    total, actions, samples = 0., 0., 0
    # Each rank reads its disjoint validation subset. No DDP forward collectives:
    # call the backend on original components, not the wrapped actor.
    with isolated_validation(components):
        groups = getattr(dataset, "prompt_groups", None) or {"all": list(range(len(dataset)))}
        for group in groups.values():
            indices = list(group)[rank::world]
            for offset in range(0, len(indices), batch_size):
                batch = default_collate([dataset[i] for i in indices[offset:offset + batch_size]])
                losses, _ = backend.actor_losses(components, batch, device)
                valid = batch.get("actor_action_mask", batch["action_mask"]).sum(1).to(losses.device)
                if losses.shape != valid.shape or not torch.isfinite(losses).all() or not (valid > 0).all():
                    raise ValueError("Invalid validation per-sample losses or masks")
                total += float((losses.double() * valid).sum())
                actions += float(valid.sum())
                samples += len(losses)
    return [total, actions, samples]


def validation_metrics(backend, components, dataset, device, batch_size, *, distributed=False):
    rank, world = (dist.get_rank(), dist.get_world_size()) if distributed else (0, 1)
    error, values = None, None
    try:
        values = _validation_totals(backend, components, dataset, device, batch_size, rank, world)
    except Exception as exc:
        if not distributed:
            raise
        error = f"rank {rank}: {type(exc).__name__}: {exc}"
    finally:
        for name in ("_observations", "_trajectory"):
            clear = getattr(getattr(dataset, name, None), "cache_clear", None)
            if clear:
                clear()
    if distributed:
        errors = [None] * world
        dist.all_gather_object(errors, error)
        if any(errors):
            raise RuntimeError("Validation failed: " + "; ".join(item for item in errors if item))
    totals = torch.tensor(values, dtype=torch.float64, device=device)
    if distributed:
        dist.all_reduce(totals)
    total, actions, samples = totals.cpu().tolist()
    return {"validation_actor_loss": total / actions if actions else None,
            "validation_actor_valid_actions": int(actions), "validation_actor_samples": int(samples),
            "validation_actor_status": "complete" if samples else "empty_validation_split"}
