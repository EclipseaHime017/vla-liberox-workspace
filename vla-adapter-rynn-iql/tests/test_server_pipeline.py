import copy
import json
import signal
from dataclasses import replace
from pathlib import Path

import pytest
import yaml

from vla_rynn_iql.config import PROJECT_ROOT
from vla_rynn_iql.server_config import configure_server, load_server_config, distributed_config, validate_global_batch
from vla_rynn_iql.server_pipeline import ServerRun, training_settings, execution_plan, visible_devices
from test_server_inputs import global_runs


def configuration(tmp_path, dataset, configured):
    raw = yaml.safe_load((PROJECT_ROOT / "configs/server_pipeline.yaml").read_text())
    raw.update(training_config=str(configured.path), runs_root=configured.raw["paths"]["dataset_sources"][0],
               task_id=dataset.task_id, output_root="./runs", cache_root="./cache")
    path = tmp_path / "server.yaml"
    path.write_text(yaml.safe_dump(raw))
    return load_server_config(path)


@pytest.mark.parametrize("patch", [{"gpu_ids": [0, 0]}, {"gpu_ids": []}, {"zero_stage": 2},
    {"backend": "gloo"}, {"data_workers_per_rank": 0}, {"pin_memory": "yes"}])
def test_reject_invalid_distributed(patch):
    with pytest.raises((TypeError, ValueError)):
        distributed_config(patch)


def test_gpu_selection_respects_scheduler_allocation():
    assert visible_devices((1, 3), {}) == "1,3"
    assert visible_devices((0, 1), {"CUDA_VISIBLE_DEVICES": "3,7"}) == "3,7"
    assert visible_devices((1,), {"CUDA_VISIBLE_DEVICES": "GPU-first,GPU-second"}) == "GPU-second"
    for inherited in ("", "-1", "3"):
        with pytest.raises(ValueError, match="allocation"):
            visible_devices((0, 1), {"CUDA_VISIBLE_DEVICES": inherited})


def test_legacy_server_config_migrates_environment_to_model_ownership(configured, tmp_path):
    dataset, _, _ = global_runs(configured)
    server = configuration(tmp_path, dataset, configured)
    raw = yaml.safe_load(server.path.read_text())
    raw.update(schema_version=3, environment="vla-liberox")
    raw["overrides"]["model"] = {"family": "pi05", "base_id": "pi05-libero-base"}
    server.path.write_text(yaml.safe_dump(raw))
    loaded = load_server_config(server.path)
    assert training_settings(loaded, dataset)["model"]["environment"] == "pi05"


def test_rank_component_loading_ignores_cuda_checkpoint_tags(monkeypatch):
    from types import SimpleNamespace
    import torch
    from vla_rynn_iql.vla_adapter import _rank_component_loader
    calls = []
    def load(path, **kwargs):
        calls.append(kwargs)
        return {"module.weight": torch.ones(1), "bias": torch.zeros(1)}
    original = lambda _: None
    utils = SimpleNamespace(load_component_state_dict=original)
    monkeypatch.setattr(torch, "load", load)
    with pytest.raises(RuntimeError, match="construction failed"):
        with _rank_component_loader(utils, torch.device("cuda:1")):
            assert set(utils.load_component_state_dict("tagged.pt")) == {"weight", "bias"}
            assert utils.DEVICE == torch.device("cuda:1")
            raise RuntimeError("construction failed")
    assert utils.load_component_state_dict is original
    assert calls == [{"map_location": "cpu", "weights_only": True}]


def test_batch_sigint_during_start_is_deferred_and_reaped(tmp_path, monkeypatch):
    import importlib.util
    spec = importlib.util.spec_from_file_location("server_batch", PROJECT_ROOT / "scripts/train_server.py")
    script = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(script)

    class FakeRun:
        directory = tmp_path
        cancel_at = None
        state = {"status": "RUNNING"}
        process = None
        closed = False

        def start(self):
            signal.raise_signal(signal.SIGINT)
            self.process = self

        def poll(self):
            return 130 if self.cancel_at else None

        def cancel(self):
            self.cancel_at = 1
            self.state["status"] = "INTERRUPTED"

        def progress(self):
            return {}

        def shutdown(self):
            self.closed = True

    monkeypatch.setattr(script.time, "sleep", lambda _: None)
    previous = signal.getsignal(signal.SIGINT)
    run = FakeRun()
    assert script.run_batch(run) == 130
    assert run.closed and signal.getsignal(signal.SIGINT) == previous


def test_config_preserves_global_final_and_overrides_training_only(configured, tmp_path):
    dataset, source, _ = global_runs(configured)
    server = configuration(tmp_path, dataset, configured)
    raw = training_settings(server, dataset)
    assert raw["training"]["train_steps"] == 10000
    assert raw["training"]["micro_batch_size"] == 64
    assert raw["data"]["include_post_success"] == configured.raw["data"]["include_post_success"]
    assert raw["reward"]["source"] == "final"
    assert raw["reward"]["final_normalization"] == "initial_chunk_v1"
    assert validate_global_batch(raw, server.distributed) == (64, 8)
    raw["training"]["micro_batch_size"] = 7
    with pytest.raises(ValueError, match="divisible"):
        validate_global_batch(raw, server.distributed)
    changed = copy.deepcopy(server.overrides)
    changed["reward"]["gamma"] = .92
    changed["training"]["method"] = "bc"
    assert training_settings(server, dataset, overrides=changed)["training"]["method"] == "bc"
    changed["training"]["method"] = "iql"
    assert training_settings(server, dataset, overrides=changed)["reward"]["gamma"] == .92


@pytest.mark.parametrize("method", ["bc", "iql"])
@pytest.mark.parametrize("model,family,batch,mode", [
    ("base", "vla_adapter", 64, "lora"),
    ("pi05-libero-base", "pi05", 32, "full"),
    ("pi05-liberox-base", "pi05", 32, "full"),
])
def test_server_model_profiles_select_batch_and_adaptation(configured, tmp_path, method, model, family, batch, mode):
    dataset, _, _ = global_runs(configured)
    server = configuration(tmp_path, dataset, configured)
    selected = configure_server(server.path, model=model, method=method)
    raw = training_settings(selected, dataset)
    assert raw["model"]["family"] == family
    assert raw["model"]["base_id"] == model
    assert raw["model"]["backbone"] == mode
    if family == "vla_adapter":
        assert raw["model"]["lora"] == {"rank": 64, "alpha": 64, "dropout": 0.0}
        assert raw["model"]["action_head"] == raw["model"]["proprio_projector"] == "train"
    assert raw["training"]["method"] == method
    assert raw["training"]["gradient_accumulation_steps"] == 1
    assert validate_global_batch(raw, selected.distributed) == (batch, batch // 8)
    assert execution_plan(selected, dataset, raw, selected.distributed, "final")["actor_effective_batch_size"] == batch
    reference = training_settings(replace(selected, model_profiles={}), dataset)
    assert {key: value for key, value in raw["training"].items()
            if key not in {"micro_batch_size", "gradient_accumulation_steps"}} == {
        key: value for key, value in reference["training"].items()
        if key not in {"micro_batch_size", "gradient_accumulation_steps"}}
    for section in ("iql", "bc", "reward", "data", "logging", "paths"):
        assert raw[section] == reference[section]


def test_explicit_server_settings_override_model_profile(configured, tmp_path):
    dataset, _, _ = global_runs(configured)
    server = configuration(tmp_path, dataset, configured)
    source = yaml.safe_load(server.path.read_text())
    source["overrides"]["training"].update(micro_batch_size=16, gradient_accumulation_steps=2)
    source["overrides"]["model"].update(backbone="lora", lora={"rank": 8})
    server.path.write_text(yaml.safe_dump(source))
    selected = configure_server(server.path, model="vla-adapter-pro-liberox-base")
    raw = training_settings(selected, dataset)
    assert raw["model"]["base_id"] == "vla-adapter-pro-liberox-base"
    assert raw["model"]["lora"]["rank"] == 8
    assert raw["training"]["gradient_accumulation_steps"] == 2
    assert validate_global_batch(raw, selected.distributed) == (16, 2)
    changed = copy.deepcopy(selected.overrides)
    changed["model"]["backbone"] = "frozen"
    assert training_settings(selected, dataset, overrides=changed)["model"]["backbone"] == "frozen"
    pi05 = training_settings(configure_server(server.path, model="pi05-libero-base"), dataset)
    assert pi05["model"]["backbone"] == "full"
    assert "lora" not in pi05["model"]


@pytest.mark.parametrize("model", ["vla-adapter-pro-liberox-base", "pi05-liberox-base"])
def test_cli_model_replaces_sealed_checkpoint_stats_and_contract(configured, tmp_path, model):
    from vla_rynn_iql.base_models import model_contract
    from vla_rynn_iql.models import base_model_config
    dataset, _, _ = global_runs(configured)
    original = copy.deepcopy(configured.raw)
    original["model"]["contract"] = model_contract(original["model"])
    original["model"]["base_revision"] = "f" * 64
    configured.path.write_text(yaml.safe_dump(original))
    server = configuration(tmp_path, dataset, configured)
    raw = training_settings(configure_server(server.path, model=model), dataset)
    expected = base_model_config(model)
    for key in ("family", "base_id", "base_checkpoint", "base_revision", "stats_key"):
        assert raw["model"][key] == expected[key]
    assert raw["model"]["contract"] == model_contract(expected)
    assert raw["model"]["contract"]["observation"]["image_flip_horizontal"] is False


def test_server_config_without_profiles_retains_legacy_settings(configured, tmp_path):
    dataset, _, _ = global_runs(configured)
    server = configuration(tmp_path, dataset, configured)
    source = yaml.safe_load(server.path.read_text())
    source.pop("model_profiles")
    source["overrides"]["training"].update(micro_batch_size=8, gradient_accumulation_steps=4)
    source["overrides"]["model"]["backbone"] = "frozen"
    server.path.write_text(yaml.safe_dump(source))
    loaded = load_server_config(server.path)
    raw = training_settings(loaded, dataset)
    assert loaded.model_profiles == {}
    assert raw["training"]["gradient_accumulation_steps"] == 4
    assert validate_global_batch(raw, loaded.distributed) == (8, 1)
    assert raw["model"]["backbone"] == "frozen"


@pytest.mark.parametrize("profiles", [
    None, {"unknown": {}}, {"pi05": {"model": {"backbone": "lora"}}},
    {"vla_adapter": {"training": {"micro_batch_size": True}}},
    {"vla_adapter": {"training": {"gradient_accumulation_steps": 0}}},
    {"vla_adapter": {"training": {"policy_peak_lr": .01}}},
    {"vla_adapter": {"model": {"lora": {"rank": 0}}}},
    {"vla_adapter": {"model": {"base_id": "base"}}},
    {"vla_adapter": {"reward": {"gamma": .9}}},
])
def test_reject_invalid_server_model_profiles(configured, tmp_path, profiles):
    dataset, _, _ = global_runs(configured)
    server = configuration(tmp_path, dataset, configured)
    source = yaml.safe_load(server.path.read_text())
    source["model_profiles"] = profiles
    server.path.write_text(yaml.safe_dump(source))
    with pytest.raises((TypeError, ValueError)):
        load_server_config(server.path)


@pytest.mark.parametrize("section,field,value", [("reward", "alpha", .5),
    ("data", "success_consecutive_steps", 10), ("training", "device", "cuda:1")])
def test_server_rejects_settings_owned_by_dataset_or_launcher(configured, tmp_path, section, field, value):
    dataset, _, _ = global_runs(configured)
    server = configuration(tmp_path, dataset, configured)
    raw = yaml.safe_load(server.path.read_text())
    raw["overrides"].setdefault(section, {})[field] = value
    server.path.write_text(yaml.safe_dump(raw))
    with pytest.raises(ValueError):
        load_server_config(server.path)


def test_dry_run_writes_no_training_results(configured, tmp_path, monkeypatch):
    import importlib.util
    dataset, _, _ = global_runs(configured)
    server = configuration(tmp_path, dataset, configured)
    spec = importlib.util.spec_from_file_location("server_cli", PROJECT_ROOT / "scripts/train_server.py")
    script = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(script)
    assert script.main(["--config", str(server.path), "--dry-run"]) == 0
    assert not server.output_root.exists()
    monkeypatch.setattr(script.sys.stdin, "isatty", lambda: False)
    calls = []
    monkeypatch.setattr(script, "run_batch", lambda run: calls.append(run) or 0)
    assert script.main(["--config", str(server.path), "--model", "pi05-libero-base", "--method", "bc"]) == 0
    assert calls[0].raw["model"]["family"] == "pi05"
    assert calls[0].raw["model"]["backbone"] == "full"
    assert calls[0].raw["training"]["micro_batch_size"] == 32
    assert calls[0].raw["training"]["gradient_accumulation_steps"] == 1
    assert calls[0].raw["training"]["method"] == "bc"
    assert calls[0].raw["reward"] == {}


def test_server_preflight_has_no_ui_simulation_or_model_inference_dependencies(configured, tmp_path):
    import subprocess
    import sys

    dataset, _, _ = global_runs(configured)
    server = configuration(tmp_path, dataset, configured)
    # A fresh interpreter ensures previously imported test modules cannot mask
    # accidental dependencies on the personal-device stack.
    program = """
import importlib.abc
import runpy
import sys
class HeadlessImports(importlib.abc.MetaPathFinder):
    def find_spec(self, fullname, path=None, target=None):
        if fullname.split('.')[0] in {
            'backend', 'fastapi', 'robosuite', 'mujoco', 'libero',
            'transformers', 'robometer',
        }:
            raise RuntimeError('Unexpected server preflight dependency: ' + fullname)
sys.meta_path.insert(0, HeadlessImports())
sys.argv = sys.argv[1:]
runpy.run_path(sys.argv[0], run_name='__main__')
"""
    result = subprocess.run(
        [sys.executable, "-c", program, str(PROJECT_ROOT / "scripts/train_server.py"),
         "--config", str(server.path), "--dry-run"],
        text=True, capture_output=True, timeout=30,
    )
    assert result.returncode == 0, result.stdout + result.stderr
    assert not server.output_root.exists()




def test_launch_failure_has_durable_status(configured, tmp_path, monkeypatch):
    from vla_rynn_iql import server_resources
    monkeypatch.setattr(server_resources, "LEASE_ROOT", tmp_path / "lease")
    dataset, _, _ = global_runs(configured)
    server = configuration(tmp_path, dataset, configured)
    run = ServerRun(server, dataset, training_settings(server, dataset), server.distributed, "final")
    def fail(*_, **__):
        raise OSError("conda unavailable")
    monkeypatch.setattr("vla_rynn_iql.server_pipeline.subprocess.Popen", fail)
    with pytest.raises(OSError, match="conda unavailable"):
        run.start()
    status = json.loads((run.directory / "pipeline.json").read_text())
    assert status["status"] == "FAILED"
    assert "conda unavailable" in status["error"]
    assert run.log is None




def test_cancellation_checkpoint_request_and_escalation(configured, tmp_path, monkeypatch):
    from types import SimpleNamespace
    dataset, _, _ = global_runs(configured)
    server = configuration(tmp_path, dataset, configured)
    run = ServerRun(server, dataset, training_settings(server, dataset), server.distributed, "final")
    run.directory = tmp_path / "process"
    run.directory.mkdir()
    run.process = SimpleNamespace(pid=123, poll=lambda: None)
    now, killed = [0.], []
    monkeypatch.setattr("vla_rynn_iql.server_pipeline.time.monotonic", lambda: now[0])
    monkeypatch.setattr("vla_rynn_iql.server_pipeline.os.killpg", lambda pid, sig: killed.append(sig))
    run.cancel()
    assert (run.directory / "cancel.request").is_file()
    now[0] = 121
    run.poll()
    now[0] = 132
    run.poll()
    assert killed == [signal.SIGTERM, signal.SIGKILL]
