"""DDP/ZeRO execution adapters. Objectives and update order live in the method modules."""
from __future__ import annotations

import time

import torch
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.distributed.optim import ZeroRedundancyOptimizer

from train.methods.base import build_algorithm


def zero_optimizer(parameters, optimizer_class, **kwargs):
    groups = {}
    for parameter in parameters:
        groups.setdefault(parameter.dtype, []).append(parameter)
    if len(groups) == 1:
        return ZeroRedundancyOptimizer(next(iter(groups.values())), optimizer_class=optimizer_class,
                                       overlap_with_ddp=False, **kwargs)
    return ShardedOptimizer(groups, optimizer_class, **kwargs)


class ShardedOptimizer:
    """ZeRO-1 per parameter dtype, preserving native mixed-precision model weights."""
    def __init__(self, groups, optimizer_class, **kwargs):
        self.dtypes = [str(dtype) for dtype in groups]
        self.optimizers = [ZeroRedundancyOptimizer(parameters, optimizer_class=optimizer_class,
                            overlap_with_ddp=False, **kwargs) for parameters in groups.values()]
        if not self.optimizers:
            raise ValueError("Optimizer requires trainable parameters")

    @property
    def param_groups(self):
        return [group for optimizer in self.optimizers for group in optimizer.param_groups]

    def zero_grad(self, *, set_to_none=True):
        for optimizer in self.optimizers:
            optimizer.zero_grad(set_to_none=set_to_none)

    def step(self):
        for optimizer in self.optimizers:
            optimizer.step()

    def consolidate_state_dict(self, to=0):
        for optimizer in self.optimizers:
            optimizer.consolidate_state_dict(to=to)

    def state_dict(self):
        return {"kind": "dtype_sharded_zero", "dtypes": self.dtypes,
                "optimizers": [optimizer.state_dict() for optimizer in self.optimizers]}

    def load_state_dict(self, state):
        if (state.get("kind") != "dtype_sharded_zero" or state.get("dtypes") != self.dtypes
                or len(state.get("optimizers", [])) != len(self.optimizers)):
            raise ValueError("Checkpoint optimizer parameter dtypes differ from the selected model")
        for optimizer, saved in zip(self.optimizers, state["optimizers"]):
            optimizer.load_state_dict(saved)


class CommunicationMeter:
    """Diagnostic sum of gradient all-reduce latency (overlaps compute)."""
    def __init__(self):
        self.seconds = 0.0

    def attach(self, module):
        def hook(state, bucket):
            begin = time.perf_counter()
            future = dist.all_reduce(bucket.buffer(), async_op=True).get_future()
            def finish(result):
                self.seconds += time.perf_counter() - begin
                return result.value()[0].div_(dist.get_world_size())
            return future.then(finish)
        module.register_comm_hook(None, hook)
        return module


def wrap(module, device, *, unused=False, meter=None, move_inputs=True):
    value = DDP(module, device_ids=[device.index] if device.type == "cuda" and move_inputs else None,
                broadcast_buffers=False, find_unused_parameters=unused)
    return meter.attach(value) if meter else value


class DistributedAlgorithm:
    """Run a training method under DDP; the method wraps its own critic through ``distribute``."""
    def __init__(self, config, device, meter=None, *, transition_horizon=None):
        self.inner = build_algorithm(config, device, transition_horizon=transition_horizon)
        distribute = getattr(self.inner, "distribute", None)
        # BC has no critic: nothing to wrap and no optimizer to consolidate.
        self.optimizers = [] if distribute is None else distribute(
            lambda module: wrap(module, device, meter=meter), zero_optimizer)

    def __getattr__(self, name):
        # Optional hooks (critic_batch_keys, export_critic, bind_*) belong to the method.
        if name == "inner":
            raise AttributeError(name)
        return getattr(self.inner, name)

    def update(self, batch, step):
        return self.inner.update(batch, step)

    def actor_loss(self, prediction, batch, context):
        return self.inner.actor_loss(prediction, batch, context)

    def weight_actor_losses(self, losses, context):
        return self.inner.weight_actor_losses(losses, context)

    def checkpoint(self):
        return self.inner.checkpoint()

    def restore(self, state):
        return self.inner.restore(state)


class ActorModule(torch.nn.Module):
    """Register every trainable component, including LoRA/full-backbone parameters."""
    def __init__(self, components, backend, device):
        super().__init__()
        self.backbone = components.model
        for name in ("action_head", "proprio_projector"):
            if hasattr(components, name):
                setattr(self, name, getattr(components, name))
        self.components, self.backend, self.device = components, backend, device

    def forward(self, batch):
        return self.backend.actor_losses(self.components, batch, self.device)
