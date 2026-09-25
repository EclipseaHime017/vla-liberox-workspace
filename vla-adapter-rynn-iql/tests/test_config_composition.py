from __future__ import annotations

import copy
from pathlib import Path

import pytest
import yaml

from vla_rynn_iql import config_sources
from vla_rynn_iql.config import DEFAULT_TRAIN_CONFIG, PROJECT_ROOT, load_train_config
from vla_rynn_iql.methods import COMMON_TRAINING_KEYS


def write(path: Path, value: dict) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(yaml.safe_dump(value, sort_keys=False))
    return path


def test_entries_are_independent_and_bc_never_reads_iql_or_reward(tmp_path, monkeypatch):
    original = Path.read_text

    def independent(path, *args, **kwargs):
        if path in (DEFAULT_TRAIN_CONFIG, PROJECT_ROOT / "configs/reward.yaml"):
            pytest.fail("BC consulted IQL or reward defaults")
        return original(path, *args, **kwargs)

    monkeypatch.setattr(Path, "read_text", independent)
    wrapper = write(tmp_path / "run.yaml", {"extends": str(DEFAULT_TRAIN_CONFIG),
                                           "training": {"method": "bc", "train_steps": 456}})
    for path in (DEFAULT_TRAIN_CONFIG, PROJECT_ROOT / "configs/training/bc.yaml", wrapper):
        raw = load_train_config(path, method="bc").raw
        assert raw["training"]["method"] == "bc"
        assert raw["iql"] == raw["reward"] == {}
        assert "vla" not in raw
    assert load_train_config(wrapper).raw["training"]["train_steps"] == 456
    raw = load_train_config(DEFAULT_TRAIN_CONFIG, overrides={"training": {"method": "bc"}}).raw
    assert raw["training"]["method"] == "bc"
    assert raw["paths"]["work_dir"] == str(PROJECT_ROOT / "outputs/bc-work")


def test_runtime_then_training_entry_then_explicit_override_precedence(tmp_path):
    run = write(tmp_path / "run.yaml", {
        "extends": str(PROJECT_ROOT / "configs/runtime.yaml"),
        "presets": {"model": str(PROJECT_ROOT / "configs/models/vla_adapter.yaml")},
        "training": {"method": "bc", "actor_lr_warmup_steps": 17,
                     "train_steps": 512, "policy_peak_lr": .004, "micro_batch_size": 2},
        "bc": {},
    })
    raw = load_train_config(run).raw
    assert raw["training"]["train_steps"] == 512
    assert raw["training"]["policy_peak_lr"] == .004
    assert raw["training"]["actor_lr_warmup_steps"] == 17
    child = write(tmp_path / "child.yaml", {
        "extends": "run.yaml", "iql": {"train_steps": 800},
        "training": {"policy_peak_lr": .006},
    })
    derived = load_train_config(child).raw
    assert derived["training"]["train_steps"] == 800
    assert derived["training"]["policy_peak_lr"] == .006
    assert derived["training"]["actor_lr_warmup_steps"] == 17
    explicit = load_train_config(child, overrides={"iql": {"train_steps": 111}, "training": {"train_steps": 222}}).raw
    assert explicit["training"]["train_steps"] == 222


def test_reward_is_independent_and_training_overrides_share_one_gamma(tmp_path):
    reward = yaml.safe_load((PROJECT_ROOT / "configs/reward.yaml").read_text())
    reward["reward"].update(gamma=.92, stage_exponent=4.)
    reward_path = write(tmp_path / "evaluation/reward.yaml", reward)
    run = write(tmp_path / "experiment/iql.yaml", {
        "extends": str(DEFAULT_TRAIN_CONFIG),
        "presets": {"reward": "../evaluation/reward.yaml"},
    })
    raw = load_train_config(run).raw
    assert raw["reward"]["gamma"] == .92
    assert raw["reward"]["stage_exponent"] == 4.
    assert "gamma" not in raw["iql"] and "gamma" not in raw["training"]
    overridden = load_train_config(run, overrides={"reward": {"gamma": .9}}).raw
    assert overridden["reward"]["gamma"] == .9
    assert overridden["reward"]["stage_exponent"] == 4.
    assert load_train_config().raw["reward"]["gamma"] == .99
    assert yaml.safe_load(reward_path.read_text())["reward"]["gamma"] == .92


def test_relative_paths_follow_declaring_file_and_overrides(tmp_path):
    base = write(tmp_path / "base/run.yaml", {
        "extends": str(DEFAULT_TRAIN_CONFIG),
        "paths": {"work_dir": "work"}, "training": {"resume_checkpoint": "checkpoint"},
        "model": {"base_checkpoint": "./checkpoint"},
    })
    child = write(tmp_path / "experiment/run.yaml", {
        "extends": "../base/run.yaml", "paths": {"output_dir": "output"},
    })
    raw = load_train_config(child).raw
    assert raw["paths"]["work_dir"] == str(base.parent / "work")
    assert raw["paths"]["output_dir"] == str(child.parent / "output")
    assert raw["training"]["resume_checkpoint"] == str(base.parent / "checkpoint")
    assert raw["model"]["base_checkpoint"] == str(base.parent / "checkpoint")
    raw = load_train_config(child, overrides={"paths": {"output_dir": "output"}},
                            overrides_path=tmp_path / "terminal.yaml").raw
    assert raw["paths"]["output_dir"] == str(tmp_path / "output")


def test_old_flat_config_keeps_numeric_settings_and_new_snapshot_is_self_contained(tmp_path, monkeypatch):
    canonical = load_train_config().raw
    legacy = copy.deepcopy(canonical)
    legacy["schema_version"] = 1
    legacy["vla"] = {key: legacy["model"].pop(key) for key in ("base_checkpoint", "stats_key", "use_pro_version")}
    legacy["vla"]["freeze_backbone"] = True
    for key in COMMON_TRAINING_KEYS:
        legacy["iql"][key] = legacy["training"].pop(key)
    raw = load_train_config(write(tmp_path / "legacy.yaml", legacy)).raw
    assert raw == canonical
    assert not COMMON_TRAINING_KEYS & raw["iql"].keys()
    snapshot = write(tmp_path / "effective.yaml", raw)
    monkeypatch.setattr(config_sources, "read_source", lambda *_: pytest.fail("snapshot read presets"))
    assert load_train_config(snapshot).raw == canonical


def test_rejects_cycles_unknown_keys_and_cross_layer_parameters(tmp_path):
    first = write(tmp_path / "a.yaml", {"extends": "b.yaml"})
    write(tmp_path / "b.yaml", {"extends": "a.yaml"})
    with pytest.raises(ValueError, match="Cyclic"):
        load_train_config(first)
    bad_model = write(tmp_path / "model.yaml", {"model": {"family": "vla_adapter"}, "iql": {"beta": 1}})
    run = write(tmp_path / "run.yaml", {"extends": str(DEFAULT_TRAIN_CONFIG), "presets": {"model": str(bad_model)}})
    with pytest.raises(ValueError, match="unrelated sections"):
        load_train_config(run)
    bad_reward = write(tmp_path / "reward.yaml", {"reward": {}, "iql": {"beta": 1}})
    write(run, {"extends": str(DEFAULT_TRAIN_CONFIG), "presets": {"reward": str(bad_reward)}})
    with pytest.raises(ValueError, match="unrelated sections"):
        load_train_config(run)
    run.write_text("extends: ./missing.yaml\nextends: ./another.yaml\n")
    with pytest.raises(yaml.YAMLError, match="duplicate key"):
        load_train_config(run)
    write(run, {"extends": str(DEFAULT_TRAIN_CONFIG), "training": {"critic_lr": 1}})
    with pytest.raises(ValueError, match="Unknown training"):
        load_train_config(run)


def test_post_success_option_defaults_true_and_requires_boolean(tmp_path):
    raw = load_train_config().raw
    raw["data"].pop("include_post_success")
    path = write(tmp_path / "legacy.yaml", raw)
    assert load_train_config(path).raw["data"]["include_post_success"] is True
    for invalid in (None, "false", 0, 1):
        with pytest.raises(TypeError, match="include_post_success"):
            load_train_config(path, overrides={"data": {"include_post_success": invalid}})
