from __future__ import annotations

import copy
import importlib.util
import json
import sys
from pathlib import Path
from types import SimpleNamespace

import pytest
import torch
import yaml

from vla_rynn_iql.algorithms import BehaviorCloning, ImplicitQLearning
from vla_rynn_iql.config import DEFAULT_TRAIN_CONFIG, load_train_config
from vla_rynn_iql.data import prepare_dataset
from vla_rynn_iql.iql import weighted_masked_l1
from vla_rynn_iql.methods import actor_lr_warmup, training_method
from vla_rynn_iql.replay import ActionDataset, ReplayDataset
from vla_rynn_iql.rewards import annotate_manifest
from vla_rynn_iql import training
from vla_rynn_iql.vla_adapter import load_overlay, validate_overlay
from test_replay import FakeAnnotator, _stats
from test_terminal_pipeline import _terminal_config


def test_bc_config_without_reward_or_iql_and_strict_method(configured, tmp_path):
    raw = copy.deepcopy(configured.raw)
    raw.pop("reward")
    raw.pop("iql")
    raw["training"] = {"method": "bc", "train_steps": 20, "micro_batch_size": 2,
                       "actor_lr_warmup_steps": 5, "resume_checkpoint": "checkpoint"}
    path = tmp_path / "bc.yaml"
    path.write_text(yaml.safe_dump(raw))
    config = load_train_config(path)
    assert not training_method(config.raw).requires_rewards
    assert config.section("iql")["train_steps"] == 20
    assert config.section("iql")["resume_checkpoint"] == str(tmp_path / "checkpoint")
    assert config.section("training")["resume_checkpoint"] == str(tmp_path / "checkpoint")
    assert actor_lr_warmup(config.raw) == 5
    raw["training"]["method"] = "unknown"
    path.write_text(yaml.safe_dump(raw))
    with pytest.raises(ValueError, match="training.method"):
        load_train_config(path)
    raw["training"] = {"method": "bc", "typo": 1}
    path.write_text(yaml.safe_dump(raw))
    with pytest.raises(ValueError, match="Unknown training"):
        load_train_config(path)


def test_bc_samples_match_iql_without_reward_access(configured, monkeypatch):
    prepare_dataset(configured)
    bc = ActionDataset(configured, _stats(7), _stats(8))
    assert {episode["success"] for episode, _, _ in bc.items} == {False, True}
    annotate_manifest(configured, FakeAnnotator())
    replay = ReplayDataset(configured, _stats(7), _stats(8))
    assert len(bc) == len(replay)
    for i in range(len(bc)):
        sample, transition = bc[i], replay[i]
        assert "reward" not in sample and "next_pixels" not in sample and "pixels" not in sample
        for key, value in sample.items():
            if isinstance(value, torch.Tensor):
                assert torch.equal(value, transition[key]), key
            else:
                assert value == transition[key]
    monkeypatch.setattr("vla_rynn_iql.replay.load_reward_index", lambda *_: pytest.fail("reward read"))
    assert len(ActionDataset(configured, _stats(7), _stats(8))) == len(bc)


def test_bc_loss_is_equal_weight_masked_l1():
    prediction = torch.randn(2, 8, 7, requires_grad=True)
    batch = {"actions": torch.randn_like(prediction),
             "action_mask": torch.tensor([[True] * 8, [True] * 3 + [False] * 5])}
    bc = BehaviorCloning()
    context, metrics = bc.update(batch, 10000)
    assert metrics == {} and bc.checkpoint() == {}
    loss = bc.actor_loss(prediction, batch, context)
    assert torch.equal(loss, weighted_masked_l1(prediction, batch["actions"], batch["action_mask"], torch.ones(2)))
    loss.backward()
    assert not prediction.grad[1, 3:].any()


def test_terminal_bc_skips_all_rewards_and_reuses_prepare(configured, tmp_path, monkeypatch):
    path = _terminal_config(tmp_path, configured.path)
    raw = yaml.safe_load(path.read_text())
    raw["overrides"]["training"] = {"method": "bc"}
    raw["overrides"]["reward"] = {"source": "final", "alpha": .5, "shaping_weight": .1}
    path.write_text(yaml.safe_dump(raw))
    script = DEFAULT_TRAIN_CONFIG.parents[1] / "scripts" / "train_terminal.py"
    spec = importlib.util.spec_from_file_location("bc_terminal_test", script)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    monkeypatch.setattr(module, "load_stage_annotations", lambda *_: pytest.fail("stage evaluation"))
    monkeypatch.setattr(module, "annotation_cache_valid", lambda *_: pytest.fail("reward cache"))
    monkeypatch.setattr(module, "reward_cache_valid", lambda *_: pytest.fail("reward cache"))
    monkeypatch.setattr(module, "_verify_conda_environments", lambda names: names == {"vla-liberox"} or pytest.fail(str(names)))
    monkeypatch.setattr(module.StageRunner, "install_signal_handlers", lambda _: None)
    stages = []

    def stage(self, stage_id, label, environment, script, config_path, extra=None):
        stages.append(stage_id)
        if stage_id == "prepare":
            prepare_dataset(load_train_config(config_path))
        elif stage_id == "train":
            Path(extra[1]).write_text(json.dumps({"policy_overlay": "/test/policy.yaml"}))
        else:
            pytest.fail(f"Unexpected BC stage {stage_id}")
        self.state["stages"][stage_id]["status"] = "COMPLETED"

    monkeypatch.setattr(module.StageRunner, "stage", stage)
    monkeypatch.setattr(sys, "argv", [str(script), "--config", str(path), "--yes", "--force-annotate"])
    assert module.main() == 0
    assert stages == ["prepare", "train"]
    stages.clear()
    assert module.main() == 0
    assert stages == ["train"]
    for state_path in (tmp_path / "pipelines" / "runs").glob("*/pipeline.json"):
        state = json.loads(state_path.read_text())
        assert all(state["stages"][name]["status"] == "SKIPPED" for name in ("annotate", "rewards", "bind"))


def test_terminal_explicit_legacy_override_wins_over_base_common_values(configured, tmp_path):
    from vla_rynn_iql.terminal_pipeline import load_terminal_config, merged_training_config
    raw = copy.deepcopy(configured.raw)
    raw["training"] = {"method": "bc", "train_steps": 10000, "resume_checkpoint": "old"}
    configured.path.write_text(yaml.safe_dump(raw))
    path = _terminal_config(tmp_path, configured.path)
    terminal = yaml.safe_load(path.read_text())
    terminal["overrides"]["iql"] = {"train_steps": 77, "resume_checkpoint": "new"}
    path.write_text(yaml.safe_dump(terminal))
    merged = merged_training_config(load_terminal_config(path))
    assert merged["iql"]["train_steps"] == merged["training"]["train_steps"] == 77
    assert merged["iql"]["resume_checkpoint"] == merged["training"]["resume_checkpoint"] == str(tmp_path / "new")


def test_iql_adapter_preserves_update_and_actor_objective(configured):
    from vla_rynn_iql.iql import advantage_weights
    torch.manual_seed(12)
    method = ImplicitQLearning(configured, torch.device("cpu"))
    original = copy.deepcopy(method.agent)
    batch = {
        "pixels": torch.zeros(2, 6, 16, 16), "next_pixels": torch.ones(2, 6, 16, 16),
        "proprio": torch.zeros(2, 8), "next_proprio": torch.ones(2, 8),
        "actions": torch.zeros(2, 8, 7), "action_mask": torch.ones(2, 8),
        "reward": -torch.ones(2), "chunk_length": torch.tensor([8, 3]),
        "bootstrap_mask": torch.tensor([1., 0.]),
    }
    original.update(batch)
    weights, _ = method.update(batch, 2000)
    for key, value in original.state_dict().items():
        assert torch.equal(value, method.agent.state_dict()[key]), key
    cfg = configured.section("iql")
    assert torch.equal(weights, advantage_weights(original.advantage(batch), cfg["beta"], cfg["max_advantage_weight"]))


def test_bc_cpu_training_resume_and_overlay(configured, monkeypatch):
    config = copy.deepcopy(configured)
    config.raw["training"] = {"method": "bc", "actor_lr_warmup_steps": 0}
    config.section("iql").update(train_steps=4, gradient_accumulation_steps=2, checkpoint_interval=2,
                                 micro_batch_size=2, policy_peak_lr=.01, policy_final_lr=.01)
    config.section("logging")["tensorboard"] = False
    prepare_dataset(config)
    instances = []

    def load_components(_):
        components = SimpleNamespace(
            action_head=torch.nn.Linear(3, 56), proprio_projector=torch.nn.Linear(8, 3),
            model=torch.nn.Linear(3, 3).requires_grad_(False),
            stats_key="test", action_stats=_stats(7), proprio_stats=_stats(8),
        )
        instances.append((components, copy.deepcopy(components.model.state_dict()),
                          copy.deepcopy(components.action_head.state_dict()),
                          copy.deepcopy(components.proprio_projector.state_dict())))
        return components

    monkeypatch.setattr(training, "load_components", load_components)
    monkeypatch.setattr(training, "_device", lambda _: torch.device("cpu"))
    monkeypatch.setattr(torch.cuda, "is_available", lambda: False)
    monkeypatch.setattr(training, "load_reward_index", lambda _: pytest.fail("BC loaded rewards"))
    monkeypatch.setattr("vla_rynn_iql.algorithms.PixelIQL", lambda **_: pytest.fail("BC created Q/V"))
    monkeypatch.setattr(training, "processor_inputs", lambda c, prompts, *args: len(prompts))
    monkeypatch.setattr(training, "extract_action_hidden_states", lambda c, n: c.model(torch.ones(n, 3)).detach())
    calls = 0
    interrupt = False

    def predict(c, hidden, proprio):
        nonlocal calls
        calls += 1
        if interrupt and calls == 1:  # Save only after the second micro-batch (safe boundary).
            training.request_training_stop()
        return c.action_head(hidden + c.proprio_projector(proprio.float())).reshape(-1, 8, 7)

    monkeypatch.setattr(training, "predict_normalized", predict)
    full = load_overlay(training.train(config))
    validate_overlay(full, config.section("vla")["base_checkpoint"], "test")
    assert full.reward_sha256 is None
    full_weights = torch.load(full.action_head, weights_only=True)
    calls, interrupt = 0, True
    with pytest.raises(training.TrainingCancelled):
        training.train(config)
    summaries = list(Path(config.section("paths")["output_dir"]).glob("*/summary.json"))
    canceled = next(json.loads(p.read_text()) for p in summaries if json.loads(p.read_text())["status"] == "canceled")
    assert canceled["steps"] == 2
    interrupt = False
    config.section("iql")["resume_checkpoint"] = canceled["cancel_checkpoint"]
    resumed = load_overlay(training.train(config))
    for key, value in torch.load(resumed.action_head, weights_only=True).items():
        assert torch.equal(value, full_weights[key])
    for components, backbone, head, proprio in instances:
        assert all(torch.equal(value, components.model.state_dict()[key]) for key, value in backbone.items())
        assert any(not torch.equal(value, components.action_head.state_dict()[key]) for key, value in head.items())
        assert any(not torch.equal(value, components.proprio_projector.state_dict()[key]) for key, value in proprio.items())
    for summary in summaries:
        assert not (summary.parent / "reward_manifest.json").exists()
        metrics = [json.loads(line) for line in (summary.parent / "metrics.jsonl").read_text().splitlines()]
        assert all(row["algorithm"] == "bc" and "q_mean" not in row for row in metrics)
    config.raw["training"]["method"] = "iql"
    with pytest.raises(ValueError, match="different training method"):
        training._restore_checkpoint(Path(canceled["cancel_checkpoint"]), instances[0][0], None,
                                     None, None, torch.device("cpu"), config, {}, None)
