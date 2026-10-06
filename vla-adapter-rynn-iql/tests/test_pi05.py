"""CPU contracts; no checkpoint download or OpenPI installation required."""
from __future__ import annotations

import copy
import sys
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest
import torch
import yaml

from vla_rynn_iql import pi05, vla_adapter
from vla_rynn_iql.algorithms import BehaviorCloning, ImplicitQLearning
from vla_rynn_iql.config import PROJECT_ROOT, load_train_config
from vla_rynn_iql.data import prepare_dataset
from vla_rynn_iql.iql import weighted_masked_l1
from vla_rynn_iql.models import apply_model_parameters, model_config, model_parameters
from vla_rynn_iql.replay import ActionDataset


@pytest.mark.parametrize("method", ["bc", "iql"])
def test_model_family_configuration_is_independent(method, tmp_path):
    path = PROJECT_ROOT / f"configs/training/{method}.yaml"
    raw = load_train_config(path, family="pi05").raw
    assert raw["model"]["family"] == "pi05"
    assert raw["data"]["action_horizon"] == 8
    assert set(model_parameters(raw)) == {"model_family", "model_backbone"}
    assert "lora" not in raw["model"] and "proprio_projector" not in raw["model"]
    saved = tmp_path / "vla.yaml"
    saved.write_text(yaml.safe_dump(load_train_config(path).raw))
    assert load_train_config(saved, family="pi05").raw["model"] == raw["model"]
    via_override = load_train_config(path, overrides={"model": {"family": "pi05"}}).raw
    assert via_override["model"] == raw["model"]
    changed = apply_model_parameters({}, {"model_family": "pi05"})
    assert changed["stats_key"] == "physical-intelligence/libero"
    for fields in ({"backbone": "lora"}, {"action_head": "train"}, {"family": "unknown"}):
        with pytest.raises(ValueError):
            model_config({"model": {**raw["model"], **fields}})


def test_flow_mask_and_detached_iql_weights():
    errors = torch.ones(2, 10, 32, requires_grad=True)
    mask = torch.tensor([[True] * 8, [True] * 3 + [False] * 5])
    losses = pi05.masked_flow_losses(errors, mask)
    assert torch.equal(losses, torch.ones(2))
    weights = torch.tensor([1., 4.], requires_grad=True)
    algorithm = object.__new__(ImplicitQLearning)
    algorithm.weight_actor_losses(losses, weights).backward()
    assert weights.grad is None
    assert errors.grad[0, :8, :7].count_nonzero() == 8 * 7
    assert not errors.grad[:, :, 7:].any()
    assert not errors.grad[1, 3:].any()
    assert not errors.grad[0, 8:].any()
    assert errors.grad[1, :3, :7].sum() == 2
    assert BehaviorCloning().weight_actor_losses(losses, None) == 1
    with pytest.raises(ValueError, match="empty action"):
        pi05.masked_flow_losses(errors, torch.zeros(2, 8, dtype=torch.bool))


def test_vla_objective_is_numerically_unchanged(monkeypatch):
    prediction = torch.randn(3, 8, 7, requires_grad=True)
    batch = {"actions": torch.randn_like(prediction),
             "action_mask": torch.tensor([[True] * 8, [True] * 3 + [False] * 5, [True] * 5 + [False] * 3])}
    weights = torch.tensor([.5, 2., 1.])
    monkeypatch.setattr(vla_adapter, "predict_batch", lambda *args: prediction)
    per_sample, actual = vla_adapter.actor_losses(None, batch, torch.device("cpu"))
    assert actual is prediction
    assert torch.equal((per_sample * weights).mean(), weighted_masked_l1(prediction, batch["actions"], batch["action_mask"], weights))


def test_native_flow_actions_use_official_float32_dtype(monkeypatch):
    observation = SimpleNamespace(from_dict=lambda inputs: inputs)
    monkeypatch.setitem(sys.modules, "openpi.models.model", SimpleNamespace(Observation=observation))
    def transform(sample):
        return {"actions": np.pad(sample["actions"].astype(np.float64), ((0, 0), (0, 25)))}
    def model(inputs, actions):
        assert actions.dtype == torch.float32
        assert actions.shape == (1, 10, 32)
        return actions.square()
    components = SimpleNamespace(input_transform=transform, model=model)
    batch = {"prompt": ["place the bowl"], "raw_actions": torch.ones(1, 8, 7),
             "agent_image": torch.zeros(1, 8, 8, 3, dtype=torch.uint8),
             "wrist_image": torch.zeros(1, 8, 8, 3, dtype=torch.uint8),
             "raw_proprio": torch.zeros(1, 8), "action_mask": torch.ones(1, 8, dtype=torch.bool)}
    losses, prediction = pi05.actor_losses(components, batch, torch.device("cpu"))
    assert torch.equal(losses, torch.ones(1))
    assert prediction is None


class TinyPi(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.paligemma_with_expert = torch.nn.Module()
        self.paligemma_with_expert.paligemma = torch.nn.Linear(2, 2)
        self.paligemma_with_expert.gemma_expert = torch.nn.Linear(2, 2)
        self.action_in_proj = torch.nn.Linear(2, 2)
        self.time_mlp_in = torch.nn.Linear(2, 2)
        self.action_out_proj = torch.nn.Linear(2, 2)

    def gradient_checkpointing_enable(self):
        self.checkpointing = True

    def forward(self, x):
        return self.action_out_proj(self.paligemma_with_expert.gemma_expert(
            self.paligemma_with_expert.paligemma(self.action_in_proj(x)) + self.time_mlp_in(x)))


@pytest.mark.parametrize("scope", ["frozen", "full"])
def test_scope_checkpoint_and_inference_restore(scope, tmp_path):
    torch.manual_seed(7)
    original = TinyPi()
    model = copy.deepcopy(original)
    settings = model_config({"model": {"family": "pi05", "backbone": scope}})
    pi05.configure_model(model, settings)
    components = SimpleNamespace(model=model, identity={"weights": "fixed", "norm": "fixed"})
    optimizer = torch.optim.AdamW(pi05.trainable_parameters(components), lr=.01)
    model(torch.ones(2, 2)).square().mean().backward()
    for name, parameter in model.named_parameters():
        assert (parameter.grad is None) == (scope == "frozen" and name.startswith("paligemma_with_expert.paligemma."))
    optimizer.step()
    pi05.save_actor(components, settings, tmp_path)
    restored = SimpleNamespace(model=copy.deepcopy(original), identity=components.identity)
    pi05.configure_model(restored.model, settings, training=False)
    pi05.restore_actor(restored, settings, tmp_path)
    for name, value in model.state_dict().items():
        assert torch.equal(value, restored.model.state_dict()[name])
    restored.identity = {"weights": "changed"}
    with pytest.raises(ValueError, match="different base"):
        pi05.restore_actor(restored, settings, tmp_path)


def test_pi_replay_uses_raw_actions_and_native_quantiles(configured):
    prepare_dataset(configured)
    def stats(size):
        return {"q01": [-1.] * size, "q99": [1.] * size, "codec": "openpi_quantile"}
    dataset = ActionDataset(configured, stats(7), stats(8))
    for i in range(len(dataset)):
        sample = dataset[i]
        valid = sample["action_mask"]
        assert torch.allclose(sample["actions"][valid], sample["raw_actions"][valid], atol=1e-5)
        assert sample["raw_proprio"].shape == (8,)


def test_prepare_accepts_mixed_family_clipped_actions(configured):
    root = Path(configured.section("paths")["dataset_sources"][0])
    path = next(root.rglob("root/episodes/episode_000/trajectory.npz"))
    with np.load(path) as source:
        arrays = dict(source)
    count = len(arrays["env_action"])
    codecs = np.full(count, "vla_adapter_v1")
    codecs[8:] = "libero_env_v1"
    arrays["raw_action"][8:] = arrays["env_action"][8:]
    arrays["raw_action"][8:, 0] = 2
    arrays["env_action"][8:, 0] = 1
    arrays["action_codec"] = codecs
    np.savez(path, **arrays)
    assert prepare_dataset(configured)
