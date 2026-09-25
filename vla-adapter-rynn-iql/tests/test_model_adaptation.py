from __future__ import annotations

import copy
from types import SimpleNamespace

import pytest
import torch
import yaml
from transformers import PretrainedConfig, PreTrainedModel

from vla_rynn_iql.config import load_train_config
from vla_rynn_iql.models import model_config, model_signature
from vla_rynn_iql.model_adaptation import (
    configure_components, export_backbone, parameter_counts, restore_backbone,
    save_backbone, trainable_parameters,
)


class TinyVLA(PreTrainedModel):
    config_class = PretrainedConfig

    def __init__(self):
        super().__init__(PretrainedConfig())
        self.action_queries = torch.nn.Embedding(2, 4)
        self.projector = torch.nn.Linear(4, 4)

    def forward(self, inputs):
        return self.projector(inputs + self.action_queries.weight.mean(0))


def components():
    return SimpleNamespace(model=TinyVLA(), action_head=torch.nn.Linear(4, 7),
                           proprio_projector=torch.nn.Linear(8, 4))


def predict(c):
    return c.action_head(c.model(torch.ones(2, 4)) + c.proprio_projector(torch.ones(2, 8)))


@pytest.mark.parametrize("mode,head,proprio", [
    ("frozen", "train", "train"), ("frozen", "frozen", "train"), ("frozen", "train", "frozen"),
    ("full", "train", "train"), ("full", "frozen", "frozen"),
    ("lora", "train", "train"), ("lora", "frozen", "frozen"),
])
def test_component_freezing_and_gradient_flow(mode, head, proprio):
    torch.manual_seed(7)
    c = components()
    settings = model_config({"model": {"backbone": mode, "action_head": head, "proprio_projector": proprio}})
    configure_components(c, settings, training=True)
    before = {name: {key: p.detach().clone() for key, p in module.named_parameters()}
              for name, module in (("model", c.model), ("action_head", c.action_head), ("proprio_projector", c.proprio_projector))}
    params = trainable_parameters(c)
    optimizer = torch.optim.AdamW(params, lr=.01)
    loss = predict(c).square().mean()
    loss.backward()
    if mode == "lora":
        assert c.model.action_queries.weight.grad.abs().sum() > 0
        assert any(p.grad is not None and p.grad.abs().sum() > 0 for n, p in c.model.named_parameters() if "lora_B" in n)
    optimizer.step()
    for name, module, enabled in (("model", c.model, mode != "frozen"),
                                 ("action_head", c.action_head, head == "train"),
                                 ("proprio_projector", c.proprio_projector, proprio == "train")):
        changed = []
        for key, parameter in module.named_parameters():
            same = torch.equal(parameter, before[name][key])
            if not parameter.requires_grad:
                assert same, key
            changed.append(not same)
        assert any(changed) is enabled
        assert module.training is enabled
    assert sum(item["trainable"] for item in parameter_counts(c).values()) == sum(p.numel() for p in params)


@pytest.mark.parametrize("mode", ["full", "lora"])
def test_adapted_backbone_checkpoint_resume_and_merged_export(tmp_path, mode):
    torch.manual_seed(9)
    initial = components()
    c, resumed = copy.deepcopy(initial), copy.deepcopy(initial)
    settings = model_config({"model": {"backbone": mode}})
    configure_components(c, settings, training=True)
    optimizer = torch.optim.AdamW(trainable_parameters(c), lr=.01)
    predict(c).square().mean().backward()
    optimizer.step()
    optimizer.zero_grad(set_to_none=True)
    save_backbone(c, settings, tmp_path)
    checkpoint = torch.load(tmp_path / "backbone.pt", weights_only=True)
    configure_components(resumed, settings, training=True)
    restore_backbone(resumed, settings, tmp_path)
    resumed.action_head.load_state_dict(c.action_head.state_dict())
    resumed.proprio_projector.load_state_dict(c.proprio_projector.state_dict())
    new_optimizer = torch.optim.AdamW(trainable_parameters(resumed), lr=.01)
    new_optimizer.load_state_dict(copy.deepcopy(optimizer.state_dict()))
    for obj, opt in ((c, optimizer), (resumed, new_optimizer)):
        predict(obj).square().mean().backward()
        opt.step()
    for name in ("model", "action_head", "proprio_projector"):
        for key, tensor in getattr(c, name).state_dict().items():
            assert torch.equal(tensor, getattr(resumed, name).state_dict()[key]), key
    expected = predict(c).detach()
    export = tmp_path / "export"
    export.mkdir()
    assert export_backbone(c, settings, export)
    fresh = components()
    fresh.model.load_state_dict(torch.load(export / "backbone.pt", weights_only=True), strict=True)
    fresh.action_head.load_state_dict(c.action_head.state_dict())
    fresh.proprio_projector.load_state_dict(c.proprio_projector.state_dict())
    torch.testing.assert_close(predict(fresh), expected, atol=1e-6, rtol=1e-5)
    assert not any("lora_" in key for key in fresh.model.state_dict())
    assert all(torch.equal(value, torch.load(tmp_path / "backbone.pt", weights_only=True)[key]) for key, value in checkpoint.items())
    if mode == "lora":
        assert any("action_queries" in name for name in checkpoint)
        checkpoint.pop(next(iter(checkpoint)))
        torch.save(checkpoint, tmp_path / "backbone.pt")
        with pytest.raises(ValueError, match="checkpoint keys"):
            restore_backbone(resumed, settings, tmp_path)


def test_model_config_compatibility_and_strict_validation(configured):
    raw = copy.deepcopy(configured.raw)
    raw.pop("model")
    assert model_config(raw)["backbone"] == "frozen"
    raw["vla"] = {"freeze_backbone": False}
    assert model_config(raw)["backbone"] == "full"
    raw["model"] = {"backbone": "lora", "lora": {"rank": 4}}
    configured.path.write_text(yaml.safe_dump(raw))
    loaded = load_train_config(configured.path)
    assert loaded.section("model")["lora"]["rank"] == 4
    assert "vla" not in loaded.raw
    for invalid in ({"family": "unknown"}, {"backbone": "unknown"}, {"typo": 1},
                    {"action_head": False}, {"lora": {"rank": True}}, {"lora": {"dropout": float("nan")}},
                    {"lora": {"dropout": 1}}, {"lora": {"typo": 3}},
                    {"action_head": "frozen", "proprio_projector": "frozen"}):
        with pytest.raises((ValueError, TypeError)):
            model_config({"model": invalid})
    assert model_signature({}) == model_signature({"model": {"lora": {"rank": 1}}})


def test_resume_rejects_model_configuration_changes(configured, tmp_path):
    import json
    from vla_rynn_iql.training import _restore_checkpoint
    (tmp_path / "checkpoint.json").write_text(json.dumps({"algorithm": "iql"}))
    configured.raw["model"]["backbone"] = "lora"
    with pytest.raises(ValueError, match="model training configuration"):
        _restore_checkpoint(tmp_path, None, None, None, None, torch.device("cpu"), configured, {}, None)


def test_terminal_preserves_nested_lora_defaults_and_explicit_legacy_override(configured, tmp_path):
    from test_terminal_pipeline import _terminal_config
    from vla_rynn_iql.terminal_pipeline import load_terminal_config, merged_training_config
    raw = copy.deepcopy(configured.raw)
    raw["model"]["lora"].update(rank=8, alpha=16)
    configured.path.write_text(yaml.safe_dump(raw))
    path = _terminal_config(tmp_path, configured.path)
    terminal = yaml.safe_load(path.read_text())
    terminal["overrides"]["model"] = {"backbone": "lora", "lora": {"dropout": .1}}
    path.write_text(yaml.safe_dump(terminal))
    merged = merged_training_config(load_terminal_config(path))
    assert merged["model"]["lora"] == {"rank": 8, "alpha": 16, "dropout": .1}
    assert "vla" not in merged
    terminal["overrides"]["model"] = {}
    terminal["overrides"]["vla"] = {"freeze_backbone": False}
    path.write_text(yaml.safe_dump(terminal))
    assert merged_training_config(load_terminal_config(path))["model"]["backbone"] == "full"


@pytest.mark.parametrize("mode", ["frozen", "full", "lora"])
def test_overlay_cli_round_trip_including_backbone(configured, tmp_path, monkeypatch, mode):
    import sys
    import types
    from vla_rynn_iql.training import _publish_overlay, _save_checkpoint
    from vla_rynn_iql.algorithms import BehaviorCloning
    from vla_rynn_iql import vla_adapter
    from test_replay import _stats

    config = copy.deepcopy(configured)
    config.raw["training"]["method"] = "bc"
    config.raw["model"]["backbone"] = mode
    initial = components()
    initial.model.norm_stats = {config.section("model")["stats_key"]: {"action": _stats(7), "proprio": _stats(8)}}
    c = copy.deepcopy(initial)
    c.stats_key = config.section("model")["stats_key"]
    configure_components(c, config.section("model"), training=True)
    optimizer = torch.optim.AdamW(trainable_parameters(c), lr=.01)
    predict(c).square().mean().backward()
    optimizer.step()
    checkpoint = _save_checkpoint(tmp_path / "checkpoints", 1, c, BehaviorCloning(), optimizer,
        torch.Generator(), config, {"dataset_sha256": "d" * 64}, None)
    expected = predict(c).detach()
    overlay_path = _publish_overlay(checkpoint, tmp_path / "registry", 1, config, c,
                                   {"dataset_sha256": "d" * 64}, None)
    overlay = vla_adapter.load_overlay(overlay_path)
    vla_adapter.validate_overlay(overlay, config.section("model")["base_checkpoint"], c.stats_key)
    assert (overlay.backbone is not None) == (mode != "frozen")
    module = types.ModuleType("experiments.robot.libero.run_libero_eval")
    module.GenerateConfig = SimpleNamespace
    def initialize(_):
        fresh = copy.deepcopy(initial)
        return fresh.model, fresh.action_head, fresh.proprio_projector, None, None
    module.initialize_model = initialize
    monkeypatch.setitem(sys.modules, module.__name__, module)
    monkeypatch.setattr(vla_adapter, "_add_vla_path", lambda _: None)
    loaded = vla_adapter.load_components(config, overlay_path, training=False)
    torch.testing.assert_close(predict(loaded), expected, atol=1e-6, rtol=1e-5)
    assert not any(parameter.requires_grad for parameter in loaded.model.parameters())
    if mode != "frozen":
        raw = yaml.safe_load(overlay_path.read_text())
        raw["backbone"] = ""
        raw["component_sha256"].pop("backbone")
        overlay_path.write_text(yaml.safe_dump(raw))
        with pytest.raises(ValueError, match="backbone"):
            vla_adapter.load_overlay(overlay_path)
