import copy
import json
import signal
from dataclasses import replace
from pathlib import Path

import pytest
import yaml

from vla_rynn_iql.config import PROJECT_ROOT
from vla_rynn_iql.server_config import load_server_config, distributed_config, validate_global_batch
from vla_rynn_iql.server_pipeline import ServerRun, training_settings, execution_plan, visible_devices
from vla_rynn_iql.server_tui import ServerApp, set_value
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

        def close(self):
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
    assert raw["training"]["micro_batch_size"] == 8
    assert raw["data"]["include_post_success"] == configured.raw["data"]["include_post_success"]
    assert raw["reward"]["source"] == "final"
    assert raw["reward"]["final_normalization"] == "initial_chunk_v1"
    assert validate_global_batch(raw, server.distributed) == (8, 1)
    raw["training"]["micro_batch_size"] = 7
    with pytest.raises(ValueError, match="divisible"):
        validate_global_batch(raw, server.distributed)
    changed = copy.deepcopy(server.overrides)
    changed["reward"]["gamma"] = .92
    changed["training"]["method"] = "bc"
    assert training_settings(server, dataset, overrides=changed)["training"]["method"] == "bc"
    changed["training"]["method"] = "iql"
    assert training_settings(server, dataset, overrides=changed)["reward"]["gamma"] == .92


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
    with pytest.raises(SystemExit):
        script.main(["--config", str(server.path)])


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


def test_tui_never_silently_changes_method_or_reward(configured, tmp_path):
    dataset, _, _ = global_runs(configured)
    server = configuration(tmp_path, dataset, configured)
    app = ServerApp(replace(server, task_id=None, reward_source="stage"))
    app.activate(None)
    assert "Method: iql" in app._lines()
    assert app.source == "stage"
    app.cursor = 0
    with pytest.raises(ValueError, match="No eligible"):
        app.activate(None)
    app.cursor = 2
    app.activate(None)
    lines = app._lines()
    assert app.raw["training"]["method"] == "bc"
    assert "Method: bc" in lines


def test_launch_failure_has_durable_status(configured, tmp_path, monkeypatch):
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


def test_bad_legacy_json_does_not_block_bc_selection(configured, tmp_path):
    dataset, _, _ = global_runs(configured)
    for run in dataset.runs:
        (run.episode_dir / "trajectory_reward.json").write_text("bad json")
    server = configuration(tmp_path, dataset, configured)
    app = ServerApp(replace(server, task_id=None, reward_source="stage"))
    app.activate(None)
    app._lines()
    assert app.settings_error
    assert app.page == "settings"
    with pytest.raises(ValueError):
        app.activate(None)
    app.cursor = 2
    app.activate(None)
    assert "Method: bc" in app._lines()
    assert app.settings_error is None


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
