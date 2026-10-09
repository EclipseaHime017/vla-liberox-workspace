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
from vla_rynn_iql.base_models import base_model
from vla_rynn_iql.replay import ActionDataset


@pytest.mark.parametrize("method", ["bc", "iql"])
def test_model_family_configuration_is_independent(method, tmp_path):
    path = PROJECT_ROOT / f"configs/training/{method}.yaml"
    raw = load_train_config(path, family="pi05").raw
    assert raw["model"]["family"] == "pi05"
    assert raw["data"]["action_horizon"] == 8
    assert set(model_parameters(raw)) == {"model_family", "model_backbone", "model_base_id"}
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


@pytest.mark.parametrize("method", ["bc", "iql"])
def test_liberox_base_uses_same_backend_with_independent_identity(method, tmp_path):
    path = PROJECT_ROOT / f"configs/training/{method}.yaml"
    original = load_train_config(path, family="pi05").raw
    override = {"model": {"family": "pi05", "base_id": "pi05-liberox-base"}}
    selected = load_train_config(path, overrides=override).raw
    base = base_model("pi05-liberox-base")
    assert selected["model"]["base_checkpoint"] == base.checkpoint
    assert selected["model"]["stats_key"] == "meituan/LIBERO-X"
    assert selected["model"]["family"] == "pi05"
    for section in ("data", "training", "iql", "bc", "reward"):
        assert selected[section] == original[section]
    assert apply_model_parameters(original, {"model_base_id": base.id}) == selected["model"]
    assert apply_model_parameters(selected, {"model_base_id": "pi05-libero-base"}) == original["model"]
    saved = tmp_path / "sealed.yaml"
    saved.write_text(yaml.safe_dump(selected))
    assert load_train_config(saved).raw == selected
    for parameters in ({"model_family": "vla_adapter", "model_base_id": base.id},
                       {"model_family": "pi05", "model_base_id": "unknown"}):
        with pytest.raises(ValueError):
            apply_model_parameters({}, parameters)
    with pytest.raises(ValueError, match="normalization"):
        model_config({"model": {**selected["model"], "stats_key": "physical-intelligence/libero"}})


def test_pi05_load_uses_selected_base_normalization(monkeypatch, tmp_path):
    from dataclasses import dataclass
    from vla_rynn_iql.models import base_model_config

    requested = {}
    @dataclass(frozen=True)
    class Model:
        dtype: str = "float32"
        pytorch_compile_mode: str | None = "default"
        action_horizon: int = 10
        action_dim: int = 32
        discrete_state_input: bool = False

        def load_pytorch(self, cfg, path):
            requested["model_repo"] = cfg.data.repo_id
            return torch.nn.Linear(1, 1)

    @dataclass(frozen=True)
    class Data:
        repo_id: str = "physical-intelligence/libero"

        def create(self, assets_dirs, model):
            empty = SimpleNamespace(inputs=[], outputs=[])
            return SimpleNamespace(asset_id=self.repo_id, use_quantile_norm=True,
                                   data_transforms=empty, model_transforms=empty)

    @dataclass(frozen=True)
    class Config:
        model: Model = Model()
        data: Data = Data()
        assets_dirs: str = "assets"

    def stats(root, key):
        requested["norm_key"] = key
        value = SimpleNamespace(q01=np.zeros(8), q99=np.ones(8))
        return {"actions": value, "state": value}
    transforms = SimpleNamespace(compose=lambda values: values,
                                 Normalize=lambda *a, **k: None, Unnormalize=lambda *a, **k: None)
    configs = SimpleNamespace(get_config=lambda name: Config())
    monkeypatch.setitem(sys.modules, "openpi", SimpleNamespace(transforms=transforms))
    monkeypatch.setitem(sys.modules, "openpi.training", SimpleNamespace(config=configs,
                         checkpoints=SimpleNamespace(load_norm_stats=stats)))
    monkeypatch.setattr(pi05, "verify_runtime", lambda: None)
    monkeypatch.setattr(pi05, "configure_model", lambda *a, **k: None)
    monkeypatch.setattr(pi05, "checkpoint_identity", lambda root, **kw: {
        "config_name": "pi05_libero", "files": {"assets/meituan/LIBERO-X/norm_stats.json": "test"}})
    settings = base_model_config("pi05-liberox-base")
    raw = {"model": settings, "training": {"device": "cpu", "dtype": "bfloat16"}}
    result = pi05.load_components(SimpleNamespace(raw=raw, section=raw.__getitem__), training=False)
    from vla_rynn_iql.base_models import model_contract
    settings["contract"] = model_contract(settings)
    settings["stats_key"] = "unbound/stats"
    with pytest.raises(ValueError, match="not bound"):
        pi05.load_components(SimpleNamespace(raw=raw, section=raw.__getitem__), training=False)
    assert requested == {"model_repo": "meituan/LIBERO-X", "norm_key": "meituan/LIBERO-X"}
    assert result.stats_key == "meituan/LIBERO-X"


def test_liberox_conversion_pins_source_and_copies_its_own_stats(tmp_path, monkeypatch):
    import importlib.util
    from vla_rynn_iql import pi05_assets

    spec = importlib.util.spec_from_file_location("prepare_pi05", PROJECT_ROOT / "scripts/prepare_pi05.py")
    script = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(script)
    base = base_model("pi05-liberox-base")
    source, output = tmp_path / "source", tmp_path / "converted"
    norm = source / base.norm_file
    norm.parent.mkdir(parents=True)
    norm.write_text('{"stats": "liberox"}')
    calls = []
    def download(repo, revision, **kwargs):
        kwargs["revision"] = revision
        calls.append((repo, kwargs))
        return source
    def convert(command, check):
        assert command[command.index("--config-name") + 1] == "pi05_libero"
        destination = Path(command[command.index("--output-path") + 1])
        (destination / "model.safetensors").write_bytes(b"test-conversion")
    monkeypatch.setattr(pi05, "verify_runtime", lambda: None)
    monkeypatch.setattr(script.subprocess, "run", convert)
    monkeypatch.setitem(sys.modules, "openpi", SimpleNamespace(__file__=str(tmp_path / "openpi/src/openpi/__init__.py")))
    monkeypatch.setitem(sys.modules, "openpi.shared.download", SimpleNamespace(maybe_download=lambda _: pytest.fail("wrong source")))
    from vla_rynn_iql import model_storage
    monkeypatch.setattr(model_storage, "download_repository", download)
    monkeypatch.setattr(sys, "argv", ["prepare_pi05.py", "--base-model", base.id, "--output", str(output)])
    script.main()
    assert calls == [(base.source, {"revision": base.revision, "allow_patterns": ["params/**", base.norm_file]})]
    identity = pi05_assets.checkpoint_identity(output, base_id=base.id)
    assert identity["source_revision"] == base.revision
    assert identity["conversion_precision"] == "bfloat16"
    assert (output / base.norm_file).read_text() == norm.read_text()
    assert not (output / pi05_assets.base_model("pi05-libero-base").norm_file).exists()
    with pytest.raises(SystemExit, match="Refusing to overwrite"):
        script.main()
    assert len(calls) == 1


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
    components = SimpleNamespace(input_transform=transform, model=model, settings={"family": "pi05"})
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
