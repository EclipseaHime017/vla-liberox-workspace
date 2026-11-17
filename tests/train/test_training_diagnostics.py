"""CPU-only diagnostic tests; no model downloads or training machines."""
import copy
from datetime import timedelta
import json
import os
from pathlib import Path
import random
import sys
import tempfile
from types import SimpleNamespace
import unittest

import numpy as np
import torch
import torch.distributed as dist
import torch.multiprocessing as mp

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from train.diagnostics import due, isolated_validation, validation_metrics, weight_statistics


class Dataset:
    prompt_groups = {"one": [0, 1, 2]}

    def __len__(self):
        return 3

    def __getitem__(self, index):
        return {"index": torch.tensor(index), "action_mask": torch.arange(3) < index + 1}


class Backend:
    @staticmethod
    def actor_losses(components, batch, device):
        assert not components.model.training
        assert not torch.is_grad_enabled()
        random.random()
        np.random.rand()
        torch.rand(3)
        return (batch["index"].float() + 1).to(device), None


def components():
    result = SimpleNamespace(model=torch.nn.Sequential(torch.nn.Linear(1, 1), torch.nn.Dropout()),
                             action_head=torch.nn.Linear(1, 1))
    result.model[0].eval()
    return result


def distributed_probe(rank, rendezvous, output):
    dist.init_process_group("gloo", init_method="file://" + rendezvous, rank=rank, world_size=2,
                            timeout=timedelta(seconds=30))
    try:
        weights = torch.tensor([0., 1., 2.] if rank == 0 else [10.])
        run_id = "human" if rank == 0 else "parent"
        batch = {"run_id": [run_id] * len(weights), "start": [0] * len(weights),
                 "action_source": ["human" if rank == 0 else "policy"] * len(weights)}
        episodes = {"human": {"kind": "branch", "terminal_step": 2},
                    "parent": {"kind": "original", "terminal_step": None}}
        metrics = weight_statistics(weights, batch, episodes, distributed=True)
        metrics.update(validation_metrics(Backend, components(), Dataset(), torch.device("cpu"), 2, distributed=True))
        class BrokenBackend(Backend):
            @staticmethod
            def actor_losses(components, batch, device):
                if rank == 1:
                    raise ValueError("one rank validation probe")
                return Backend.actor_losses(components, batch, device)
        try:
            validation_metrics(BrokenBackend, components(), Dataset(), torch.device("cpu"), 2, distributed=True)
        except RuntimeError as exc:
            metrics["validation_failure_propagated"] = "rank 1: ValueError: one rank validation probe" in str(exc)
        if rank == 0:
            Path(output).write_text(json.dumps(metrics))
    finally:
        dist.destroy_process_group()


class DiagnosticsTests(unittest.TestCase):
    def test_config_defaults_and_invalid_values(self):
        from core.config import load_train_config
        from core.config_sources import validate_overrides
        assert validate_overrides({"logging": {"diagnostics": {"enabled": True}}})["logging"]["diagnostics"]["enabled"]
        legacy = load_train_config()
        self.assertNotIn("diagnostics", legacy.section("logging"))
        enabled = load_train_config(overrides={"logging": {"diagnostics": {"enabled": True}}})
        self.assertEqual(enabled.section("logging")["diagnostics"], {"enabled": True, "interval_steps": 100})
        disabled = load_train_config(overrides={"logging": {"diagnostics": {}}})
        self.assertFalse(disabled.section("logging")["diagnostics"]["enabled"])
        for value in (0, -1, True, 2.5):
            with self.assertRaises((ValueError, TypeError)):
                load_train_config(overrides={"logging": {"diagnostics": {"interval_steps": value}}})
        with self.assertRaises(TypeError):
            load_train_config(overrides={"logging": {"diagnostics": {"enabled": "true"}}})
        with self.assertRaises(ValueError):
            load_train_config(overrides={"logging": {"diagnostics": {"unexpected": 1}}})

    def test_disabled_and_fixed_interval(self):
        self.assertFalse(due({}, 100))
        self.assertFalse(due({"diagnostics": {"enabled": False}}, 100))
        self.assertFalse(due({"diagnostics": {"enabled": True, "interval_steps": 20}}, 19))
        self.assertTrue(due({"diagnostics": {"enabled": True, "interval_steps": 20}}, 20))

    def test_actual_server_yaml_override_loads_into_training_config(self):
        import yaml
        from core.config import PROJECT_ROOT, load_train_config
        from cluster.config import load_server_config
        raw = yaml.safe_load((PROJECT_ROOT / "configs/cluster/pipeline.yaml").read_text())
        raw["training_config"] = str(PROJECT_ROOT / "configs/methods/iql.yaml")
        raw["overrides"]["logging"] = {"diagnostics": {"enabled": True, "interval_steps": 20}}
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "server.yaml"
            path.write_text(yaml.safe_dump(raw))
            server = load_server_config(path)
            training = load_train_config(server.training_config, overrides=server.overrides, overrides_path=server.path)
        self.assertEqual(training.section("logging")["diagnostics"], {"enabled": True, "interval_steps": 20})

    def test_categories_and_empty_groups(self):
        episodes = {"b": {"kind": "branch", "terminal_step": 9},
                    "p": {"kind": "original", "terminal_step": None}}
        result = weight_statistics(torch.tensor([1., 3., .2]),
            {"run_id": ["b", "b", "p"], "start": [0, 10, 2], "action_source": ["human", "human", "policy"]}, episodes)
        self.assertEqual(result["advantage_weight_human_pre_success_mean"], 1.)
        self.assertEqual(result["advantage_weight_post_success_mean"], 3.)
        self.assertEqual(result["advantage_weight_human_count"], 2)
        self.assertAlmostEqual(result["advantage_weight_failed_parent_mean"], .2)
        empty = weight_statistics(torch.ones(1), {"run_id": ["p"], "start": [0], "action_source": ["policy"]}, episodes)
        self.assertIsNone(empty["advantage_weight_human_pre_success_mean"])

    def test_validation_mask_weighted_and_rng_modes_unchanged(self):
        model = components()
        before = {m: m.training for m in (*model.model.modules(), *model.action_head.modules())}
        state = copy.deepcopy(model.model.state_dict())
        random.seed(42)
        np.random.seed(42)
        torch.manual_seed(42)
        expected = (random.random(), np.random.rand(), torch.rand(3))
        random.seed(42)
        np.random.seed(42)
        torch.manual_seed(42)
        result = validation_metrics(Backend, model, Dataset(), torch.device("cpu"), 2)
        actual = (random.random(), np.random.rand(), torch.rand(3))
        self.assertEqual(actual[:2], expected[:2])
        torch.testing.assert_close(actual[2], expected[2], rtol=0, atol=0)
        self.assertEqual(before, {m: m.training for m in before})
        for name, tensor in state.items():
            torch.testing.assert_close(tensor, model.model.state_dict()[name], rtol=0, atol=0)
        self.assertAlmostEqual(result["validation_actor_loss"], 14 / 6)
        self.assertEqual(result["validation_actor_samples"], 3)

    def test_exception_restores_rng_and_modes(self):
        model = components()
        state = torch.get_rng_state().clone()
        with self.assertRaisesRegex(ValueError, "probe"):
            with isolated_validation(model):
                torch.rand(50)
                raise ValueError("probe")
        self.assertTrue(model.model.training)
        self.assertFalse(model.model[0].training)
        torch.testing.assert_close(state, torch.get_rng_state())

    def test_empty_validation_not_zero_loss(self):
        class Empty(Dataset):
            prompt_groups = {}

            def __len__(self):
                return 0
        result = validation_metrics(Backend, components(), Empty(), torch.device("cpu"), 2)
        self.assertIsNone(result["validation_actor_loss"])
        self.assertEqual(result["validation_actor_status"], "empty_validation_split")

    def test_real_two_rank_global_quantiles_and_validation(self):
        with tempfile.TemporaryDirectory() as directory:
            result = str(Path(directory) / "result.json")
            mp.spawn(distributed_probe, args=(str(Path(directory) / "rendezvous"), result), nprocs=2, join=True)
            metrics = json.loads(Path(result).read_text())
        self.assertAlmostEqual(metrics["advantage_weight_p50"], 1.5)
        self.assertAlmostEqual(metrics["advantage_weight_p95"], 8.8)
        self.assertEqual(metrics["advantage_weight_count"], 4)
        self.assertEqual(metrics["advantage_weight_failed_parent_mean"], 10.)
        self.assertAlmostEqual(metrics["validation_actor_loss"], 14 / 6)
        self.assertEqual(metrics["validation_actor_samples"], 3)
        self.assertTrue(metrics["validation_failure_propagated"])


if __name__ == "__main__":
    unittest.main()
