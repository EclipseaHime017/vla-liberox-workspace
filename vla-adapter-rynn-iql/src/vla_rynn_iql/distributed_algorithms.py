"""DDP/ZeRO execution adapters. All objectives and update order live in algorithms.py."""
from __future__ import annotations

import time

import torch
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.distributed.optim import ZeroRedundancyOptimizer

from .algorithms import build_algorithm


def unwrap(module):
    return module.module if isinstance(module, DDP) else module


def zero_optimizer(parameters, optimizer_class, **kwargs):
    return ZeroRedundancyOptimizer(list(parameters), optimizer_class=optimizer_class,
                                   overlap_with_ddp=False, **kwargs)


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


def wrap(module, device, *, unused=False, meter=None):
    value = DDP(module, device_ids=[device.index] if device.type == "cuda" else None,
                broadcast_buffers=False, find_unused_parameters=unused)
    return meter.attach(value) if meter else value


class DistributedAlgorithm:
    def __init__(self, config, device, meter=None):
        self.inner = build_algorithm(config, device)
        self.optimizers = []
        agent = getattr(self.inner, "agent", None)
        if agent is None:  # BC: no Q/V, no reward evaluation.
            return
        for name in ("q1", "q2", "value"):
            setattr(agent, name, wrap(getattr(agent, name), device, meter=meter))
        for online, target in ((agent.q1, agent.target_q1), (agent.q2, agent.target_q2)):
            target.load_state_dict(unwrap(online).state_dict())
        cfg = config.section("iql")
        optimizers = {"adam": torch.optim.Adam, "adamw": torch.optim.AdamW}
        agent.q_optimizer = zero_optimizer(
            list(agent.q1.parameters()) + list(agent.q2.parameters()),
            optimizers[cfg["critic_optimizer"]], lr=cfg["critic_lr"], weight_decay=cfg["critic_weight_decay"])
        agent.value_optimizer = zero_optimizer(agent.value.parameters(),
            optimizers[cfg["value_optimizer"]], lr=cfg["value_lr"], weight_decay=cfg["value_weight_decay"])
        self.optimizers = [agent.q_optimizer, agent.value_optimizer]

    def update(self, batch, step):
        return self.inner.update(batch, step)

    def actor_loss(self, prediction, batch, context):
        return self.inner.actor_loss(prediction, batch, context)

    def checkpoint(self):
        agent = getattr(self.inner, "agent", None)
        if agent is None:
            return self.inner.checkpoint()
        state = {f"{name}.{key}": value for name in ("q1", "q2", "target_q1", "target_q2", "value")
                 for key, value in unwrap(getattr(agent, name)).state_dict().items()}
        return {"model": state, "q_optimizer": agent.q_optimizer.state_dict(),
                "value_optimizer": agent.value_optimizer.state_dict()}

    def restore(self, state):
        agent = getattr(self.inner, "agent", None)
        if agent is None:
            return self.inner.restore(state)
        for name in ("q1", "q2", "target_q1", "target_q2", "value"):
            prefix = f"{name}."
            unwrap(getattr(agent, name)).load_state_dict({key[len(prefix):]: value
                for key, value in state["model"].items() if key.startswith(prefix)}, strict=True)
        agent.q_optimizer.load_state_dict(state["q_optimizer"])
        agent.value_optimizer.load_state_dict(state["value_optimizer"])


class ActorModule(torch.nn.Module):
    """Register every trainable component, including LoRA/full-backbone parameters."""
    def __init__(self, components, backend, device):
        super().__init__()
        self.backbone = components.model
        self.action_head = components.action_head
        self.proprio_projector = components.proprio_projector
        self.components, self.backend, self.device = components, backend, device

    def forward(self, batch):
        return self.backend.predict_batch(self.components, batch, self.device)
