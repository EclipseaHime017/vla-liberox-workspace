"""CPU/Gloo execution tests exercise real DDP and real ZeRO, not mocked collectives."""
import copy
from datetime import timedelta
from pathlib import Path
from types import SimpleNamespace

import pytest
import torch
import torch.distributed as dist
import torch.multiprocessing as mp

from vla_rynn_iql.algorithms import ImplicitQLearning, BehaviorCloning
from vla_rynn_iql.config import LoadedConfig
from vla_rynn_iql.distributed_algorithms import ActorModule, DistributedAlgorithm, wrap, zero_optimizer
from vla_rynn_iql.model_adaptation import trainable_parameters


class TinyQ(torch.nn.Module):
    def __init__(self, horizon=8, *_):
        super().__init__()
        self.horizon = horizon
        self.linear = torch.nn.Linear(10, 1)

    def forward(self, pixels, proprio, actions, mask):
        x = torch.cat((proprio, pixels.float().mean((1, 2, 3))[:, None],
                       (actions * mask[..., None]).mean((1, 2))[:, None]), dim=1)
        return self.linear(x).squeeze(-1)


class TinyValue(torch.nn.Module):
    def __init__(self, *_):
        super().__init__()
        self.linear = torch.nn.Linear(9, 1)

    def forward(self, pixels, proprio):
        return self.linear(torch.cat((proprio, pixels.float().mean((1, 2, 3))[:, None]), 1)).squeeze(-1)


def tiny_predict(components, batch, device):
    x = components.model(components.proprio_projector(batch["proprio"]))
    return components.action_head(x).reshape(-1, 8, 7)


def tiny_actor_losses(components, batch, device):
    prediction = tiny_predict(components, batch, device)
    mask = batch.get("actor_action_mask", batch["action_mask"])[..., None]
    loss = ((prediction - batch.get("actor_actions", batch["actions"])).abs() * mask).sum((1, 2)) / (mask.sum((1, 2)) * 7)
    return loss, prediction


def _worker(rank, world, rendezvous, raw, result, horizon=8):
    torch.set_num_threads(1)
    dist.init_process_group("gloo", init_method=f"file://{rendezvous}", rank=rank,
                            world_size=world, timeout=timedelta(seconds=45))
    try:
        from vla_rynn_iql.distributed_training import _reduce_metrics
        reduced = _reduce_metrics({"actor_l1_x": float(rank + 1), "q_mean": float(rank + 1)},
                                  torch.device("cpu"), valid_actions=rank + 1)
        assert reduced["q_mean"] == 1.5
        assert reduced["actor_l1_x"] == pytest.approx(5 / 3)
        from vla_rynn_iql import iql
        iql.QNetwork, iql.ValueNetwork = TinyQ, TinyValue
        config = LoadedConfig(Path("test.yaml"), raw)
        torch.manual_seed(8)
        reference = ImplicitQLearning(config, torch.device("cpu"), transition_horizon=horizon)
        torch.manual_seed(8)
        parallel = DistributedAlgorithm(config, torch.device("cpu"), transition_horizon=horizon)
        assert parallel.inner.agent.q1.module.horizon == reference.agent.q1.horizon == horizon
        torch.manual_seed(21)
        batch = {"pixels": torch.rand(4, 6, 4, 4), "next_pixels": torch.rand(4, 6, 4, 4),
            "proprio": torch.randn(4, 8), "next_proprio": torch.randn(4, 8),
            "actions": torch.randn(4, horizon, 7), "action_mask": torch.ones(4, horizon),
            "actor_actions": torch.randn(4, 8, 7), "actor_action_mask": torch.ones(4, 8),
            "reward": -torch.rand(4), "chunk_length": torch.tensor([horizon, 3, horizon, 4]),
            "bootstrap_mask": torch.tensor([1., 1., 0., 1.])}
        local = {key: value[rank * 2:(rank + 1) * 2] for key, value in batch.items()}
        for step in range(3):
            weights, _ = reference.update(batch, 2000 + step)
            actual, _ = parallel.update(local, 2000 + step)
            torch.testing.assert_close(actual, weights[rank * 2:(rank + 1) * 2], atol=2e-6, rtol=2e-5)
            for name in ("q1", "q2", "target_q1", "target_q2", "value"):
                for left, right in zip(getattr(reference.agent, name).parameters(),
                                       getattr(parallel.inner.agent, name).parameters()):
                    torch.testing.assert_close(left, right, atol=2e-6, rtol=2e-5)
        for optimizer in parallel.optimizers:
            optimizer.consolidate_state_dict(to=0)
        packet = [copy.deepcopy(parallel.checkpoint()) if rank == 0 else None]
        dist.broadcast_object_list(packet, src=0)
        replacement = DistributedAlgorithm(config, torch.device("cpu"), transition_horizon=horizon)
        replacement.restore(packet[0])
        parallel.update(local, 2003)
        replacement.update(local, 2003)
        for left, right in zip(parallel.inner.agent.parameters(), replacement.inner.agent.parameters()):
            torch.testing.assert_close(left, right, atol=1e-7, rtol=1e-6)

        # BC actor includes a trainable backbone and an intentionally unused head
        # parameter (like the upstream Pro FiLM path); accumulation is identical.
        torch.manual_seed(4)
        components = SimpleNamespace(model=torch.nn.Linear(3, 3),
            proprio_projector=torch.nn.Linear(8, 3), action_head=torch.nn.Linear(3, 56))
        components.action_head.register_parameter("unused", torch.nn.Parameter(torch.ones(1)))
        single = copy.deepcopy(components)
        backend = SimpleNamespace(actor_losses=tiny_actor_losses)
        actor = wrap(ActorModule(components, backend, torch.device("cpu")), torch.device("cpu"), unused=True)
        zero = zero_optimizer(trainable_parameters(components), torch.optim.AdamW,
                              lr=.001, betas=(.9, .95), weight_decay=1e-10)
        optimizer = torch.optim.AdamW(trainable_parameters(single), lr=.001, betas=(.9, .95), weight_decay=1e-10)
        bc = BehaviorCloning()
        for update in range(2):
            optimizer.zero_grad(set_to_none=True)
            zero.zero_grad(set_to_none=True)
            from contextlib import nullcontext
            for micro in range(2):
                (bc.actor_loss(tiny_predict(single, batch, None), batch, None) / 2).backward()
                with actor.no_sync() if micro == 0 else nullcontext():
                    losses, _ = actor(local)
                    (bc.weight_actor_losses(losses, None) / 2).backward()
            torch.nn.utils.clip_grad_norm_(trainable_parameters(single), 1)
            torch.nn.utils.clip_grad_norm_(trainable_parameters(components), 1)
            optimizer.step()
            zero.step()
            for left, right in zip(trainable_parameters(single), trainable_parameters(components)):
                torch.testing.assert_close(left, right, atol=2e-6, rtol=2e-5)
        if rank == 0:
            Path(result).write_text("ok")
    finally:
        dist.destroy_process_group()


@pytest.mark.parametrize("horizon", [5, 8])
def test_two_rank_iql_and_bc_numerical_alignment(configured, tmp_path, horizon):
    mp.spawn(_worker, args=(2, str(tmp_path / "rendezvous"), configured.raw, str(tmp_path / "result"), horizon),
             nprocs=2, join=True)
    assert (tmp_path / "result").read_text() == "ok"


class TinyFlow(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.paligemma_with_expert = torch.nn.Module()
        self.paligemma_with_expert.paligemma = torch.nn.Linear(8, 3)
        self.paligemma_with_expert.paligemma.requires_grad_(False)
        self.paligemma_with_expert.gemma_expert = torch.nn.Linear(3, 320)
        self.action_in_proj = torch.nn.Linear(8, 3)

    def forward(self, proprio):
        prefix = self.paligemma_with_expert.paligemma(proprio)
        hidden = prefix + self.action_in_proj(proprio)
        expert = self.paligemma_with_expert.gemma_expert
        return expert(hidden.to(expert.weight.dtype)).reshape(-1, 10, 32)


def flow_losses(components, batch, device):
    from vla_rynn_iql.pi05 import masked_flow_losses
    prediction = components.model(batch["proprio"])
    target = torch.zeros_like(prediction)
    target[:, :8, :7] = batch["actions"]
    return masked_flow_losses((prediction - target).square(), batch["action_mask"]), None


def _native_worker(rank, world, rendezvous, raw):
    from contextlib import nullcontext
    from vla_rynn_iql import iql, pi05
    from vla_rynn_iql.algorithms import build_algorithm
    torch.set_num_threads(1)
    dist.init_process_group("gloo", init_method=f"file://{rendezvous}", rank=rank,
                            world_size=world, timeout=timedelta(seconds=45))
    try:
        iql.QNetwork, iql.ValueNetwork = TinyQ, TinyValue
        config = LoadedConfig(Path("test.yaml"), raw)
        torch.manual_seed(8)
        single_algorithm = build_algorithm(config, torch.device("cpu"))
        torch.manual_seed(8)
        parallel = DistributedAlgorithm(config, torch.device("cpu"))
        torch.manual_seed(9)
        components = SimpleNamespace(model=TinyFlow())
        single = copy.deepcopy(components)
        frozen = {key: value.clone() for key, value in components.model.paligemma_with_expert.paligemma.state_dict().items()}
        actor = wrap(ActorModule(components, SimpleNamespace(actor_losses=flow_losses), torch.device("cpu")),
                     torch.device("cpu"), unused=True)
        optimizer = zero_optimizer(pi05.trainable_parameters(components), torch.optim.AdamW, lr=.001)
        reference_optimizer = torch.optim.AdamW(pi05.trainable_parameters(single), lr=.001)
        lengths = torch.tensor([8, 3, 5, 1])
        batch = {"pixels": torch.rand(4, 6, 4, 4), "next_pixels": torch.rand(4, 6, 4, 4),
                 "proprio": torch.randn(4, 8), "next_proprio": torch.randn(4, 8),
                 "actions": torch.randn(4, 8, 7), "action_mask": torch.arange(8)[None, :] < lengths[:, None],
                 "reward": -torch.rand(4), "chunk_length": lengths,
                 "bootstrap_mask": torch.tensor([1., 1., 0., 1.])}
        local = {key: value[rank * 2:(rank + 1) * 2] for key, value in batch.items()}
        for update in range(2):
            optimizer.zero_grad(set_to_none=True)
            reference_optimizer.zero_grad(set_to_none=True)
            for micro in range(2):
                context, _ = single_algorithm.update(batch, 2000 + update * 2 + micro)
                local_context, _ = parallel.update(local, 2000 + update * 2 + micro)
                losses, _ = flow_losses(single, batch, None)
                (single_algorithm.weight_actor_losses(losses, context) / 2).backward()
                with actor.no_sync() if micro == 0 else nullcontext():
                    losses, prediction = actor(local)
                    assert prediction is None
                    (parallel.weight_actor_losses(losses, local_context) / 2).backward()
            torch.nn.utils.clip_grad_norm_(pi05.trainable_parameters(single), 1)
            torch.nn.utils.clip_grad_norm_(pi05.trainable_parameters(components), 1)
            optimizer.step()
            reference_optimizer.step()
            for left, right in zip(single.model.parameters(), components.model.parameters()):
                torch.testing.assert_close(left, right, atol=2e-6, rtol=2e-5)
        for key, value in components.model.paligemma_with_expert.paligemma.state_dict().items():
            assert torch.equal(value, frozen[key])
    finally:
        dist.destroy_process_group()


@pytest.mark.parametrize("method", ["bc", "iql"])
def test_model_only_native_flow_ddp_and_accumulation(configured, tmp_path, method):
    raw = copy.deepcopy(configured.raw)
    raw["training"]["method"] = method
    mp.spawn(_native_worker, args=(2, str(tmp_path / "rendezvous"), raw), nprocs=2, join=True)


def _checkpoint_worker(rank, world, rendezvous, raw, root, resume):
    from vla_rynn_iql.distributed_training import save_distributed_checkpoint, restore_distributed_checkpoint
    torch.set_num_threads(1)
    dist.init_process_group("gloo", init_method=f"file://{rendezvous}", rank=rank,
                            world_size=world, timeout=timedelta(seconds=45))
    try:
        torch.manual_seed(9)
        config = LoadedConfig(Path(root) / "config.yaml", raw)
        native = raw["model"]["family"] == "pi05"
        if native:
            from vla_rynn_iql.pi05 import trainable_parameters as parameters
            components = SimpleNamespace(model=TinyFlow(), stats_key=raw["model"]["stats_key"],
                                         identity={"test_model": "pi05"})
            components.model.paligemma_with_expert.gemma_expert.to(torch.bfloat16)
        else:
            parameters = trainable_parameters
            components = SimpleNamespace(model=torch.nn.Linear(3, 3),
                proprio_projector=torch.nn.Linear(8, 3), action_head=torch.nn.Linear(3, 56), stats_key="test")
        actor = wrap(ActorModule(components, SimpleNamespace(actor_losses=flow_losses if native else tiny_actor_losses),
                               torch.device("cpu")), torch.device("cpu"))
        optimizer = zero_optimizer(parameters(components), torch.optim.AdamW, lr=.001)
        algorithm = DistributedAlgorithm(config, torch.device("cpu"))
        generator = torch.Generator().manual_seed(raw["training"]["seed"] + 1)
        manifest = {"dataset_sha256": "dataset", "action_horizon": 8, "episodes": []}
        step = 0
        if resume:
            step = restore_distributed_checkpoint(Path(resume), components, algorithm, optimizer,
                generator, torch.device("cpu"), config, manifest, None, 4)
            for name in (() if native else ("action_head", "proprio_projector")):
                saved = torch.load(Path(resume) / f"{name}.pt", weights_only=True)
                for key, value in getattr(components, name).state_dict().items():
                    assert torch.equal(value, saved[key])
            if native:
                saved = torch.load(Path(resume) / "actor.pt", weights_only=True)
                for name, value in components.model.named_parameters():
                    if value.requires_grad:
                        assert torch.equal(value, saved[name])
            rng = torch.load(Path(resume) / "trainer.pt", weights_only=False)["data_rng"]
            assert torch.equal(generator.get_state(), rng)
        torch.manual_seed(100)
        batch = {"proprio": torch.randn(4, 8), "actions": torch.randn(4, 8, 7), "action_mask": torch.ones(4, 8)}
        local = {key: value[rank * (4 // world):(rank + 1) * (4 // world)] for key, value in batch.items()}
        optimizer.zero_grad()
        losses, _ = actor(local)
        algorithm.weight_actor_losses(losses, None).backward()
        optimizer.step()
        torch.randint(100, (8,), generator=generator)
        save_distributed_checkpoint(Path(root) / "checkpoints", step + 2, components, algorithm,
                                     optimizer, generator, config, manifest, None, 4)
    finally:
        dist.destroy_process_group()


@pytest.mark.parametrize("family", ["vla_adapter", "pi05"])
def test_checkpoint_one_two_one_and_same_world(configured, tmp_path, family):
    from vla_rynn_iql.models import model_config
    raw = copy.deepcopy(configured.raw)
    raw["training"].update(method="bc", gradient_accumulation_steps=2)
    raw["model"] = model_config({"model": {"family": family,
                               "backbone": "full" if family == "vla_adapter" else "frozen"}})
    previous = None
    for index, world in enumerate((1, 2, 2, 1)):
        root = tmp_path / f"run{index}"
        root.mkdir()
        mp.spawn(_checkpoint_worker, args=(world, str(root / "rendezvous"), raw, str(root), previous),
                 nprocs=world, join=True)
        previous = str(root / "checkpoints" / f"step_{(index + 1) * 2:08d}")
        assert (Path(previous) / "server.json").is_file()
