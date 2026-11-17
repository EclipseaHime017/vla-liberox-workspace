from collections import Counter
import json
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest
import yaml

from vla_rynn_iql import evaluation_runtime as runtime, server_evaluation as server, server_resources
from vla_rynn_iql.base_models import registered_defaults, model_contract
from vla_rynn_iql.evaluation_metrics import aggregate_trials, build_schedule
from vla_rynn_iql.io import sha256_file, stable_hash
from vla_rynn_iql.model_artifacts import parent_snapshot


@pytest.fixture
def config_path(tmp_path):
    raw = yaml.safe_load((Path(__file__).resolve().parents[1] / "configs/server_evaluation.yaml").read_text())
    for name in ("liberox_root", "vla_adapter_root", "output_root"):
        raw[name] = str(tmp_path / name)
    for task in raw["tasks"]:
        for kind in ("bddl", "init"):
            file = Path(raw["liberox_root"]) / "libero/libero_x" / kind / task["level"] / f"{task['task_name']}.{kind}"
            file.parent.mkdir(parents=True)
            file.write_text(kind)
    path = tmp_path / "evaluate.yaml"
    path.write_text(yaml.safe_dump(raw))
    return path


def child(tmp_path, base_id):
    family = "pi05" if base_id.startswith("pi05") else "vla_adapter"
    settings = registered_defaults(family, base_id)
    settings["contract"] = model_contract(settings)
    parent = parent_snapshot(settings, "a" * 64)
    io = parent["contract"]["io"]
    payload = {"schema_version": 5, "family": family, "parent": parent, "model_config": settings,
               "base_checkpoint": parent["checkpoint"], "stats_key": parent["stats_key"], "label": "test policy",
               "action_horizon": io["replay_horizon"], "action_dim": 7, "proprio_dim": 8}
    path = tmp_path / f"{base_id}.yaml"
    path.write_text(yaml.safe_dump(payload))
    return path, payload


def test_configuration_freezes_task_assets_and_balanced_schedule(config_path):
    config = server.load_evaluation(config_path)
    task = config["tasks"][0]
    assert task["bddl_sha256"] == sha256_file(Path(task["bddl_path"]))
    assert task["schedule_sha256"] == stable_hash(task["schedule"])
    assert len(task["schedule"]) == 30
    assert set(Counter(row["init_state_index"] for row in task["schedule"]).values()) == {10}
    assert config == server.load_evaluation(config_path)


@pytest.mark.parametrize("change", [{"unknown": True}, {"trials": True}, {"open_loop_steps": True},
                                    {"render_backend": "glfw"}, {"success_streak": 0}])
def test_bad_configuration_rejected(config_path, change):
    raw = yaml.safe_load(config_path.read_text())
    raw.update(change)
    config_path.write_text(yaml.safe_dump(raw))
    with pytest.raises((ValueError, TypeError)):
        server.load_evaluation(config_path)


def test_duplicate_yaml_keys_rejected(config_path):
    config_path.write_text(config_path.read_text() + "trials: 10\n")
    with pytest.raises(yaml.constructor.ConstructorError, match="duplicate"):
        server.load_evaluation(config_path)


@pytest.mark.parametrize("base_id,expected", [("base", 8), ("pi05-libero-base", 5), ("pi05-liberox-base", 10)])
def test_checkpoint_uses_variant_execution_and_frozen_parent(tmp_path, base_id, expected):
    path, payload = child(tmp_path, base_id)
    result = server.policy_request(path, {"open_loop_steps": None})
    assert result["open_loop_steps"] == expected
    assert result["policy_sha256"] == sha256_file(path)
    with pytest.raises(ValueError, match="open_loop_steps"):
        server.policy_request(path, {"open_loop_steps": 999})
    payload["model_config"]["contract"]["observation"]["image_flip_vertical"] = False
    path.write_text(yaml.safe_dump(payload))
    with pytest.raises(ValueError, match="inherited"):
        server.policy_request(path, {"open_loop_steps": None})


def test_schedule_matches_platform_and_balances_pairs():
    kwargs = dict(trials=103, init_state_indices=list(range(10)), base_seed=20, seed_count=None, schedule_seed=7)
    schedule = build_schedule(**kwargs)
    assert schedule == build_schedule(**kwargs)
    for values in (Counter(r["seed"] for r in schedule), Counter(r["init_state_index"] for r in schedule),
                   Counter((r["seed"], r["init_state_index"]) for r in schedule)):
        assert max(values.values()) - min(values.values()) <= 1


class Env:
    def __init__(self):
        self.steps = 0

    def regenerate_obs_from_state(self, _state):
        return {"step": self.steps}

    def step(self, action):
        self.steps += 1
        return {"step": self.steps}, 0, self.steps <= 5, {}


def run_episode(monkeypatch, predict, process=lambda x: x):
    monkeypatch.setattr(runtime, "capture", lambda env, obs: obs)
    return runtime.episode(Env(), np.zeros(1), predict, process, "task",
                           {"trial_index": 0, "init_state_index": 0, "seed": 1}, steps=8,
                           open_loop_steps=3, native_horizon=10, success_streak=5, stopping=lambda: False)


def test_closed_loop_queries_new_state_and_runs_full_horizon(monkeypatch):
    seen = []
    def predict(obs, prompt):
        seen.append(obs["step"])
        return np.zeros((10, 7))
    result = run_episode(monkeypatch, predict)
    assert seen == [0, 3, 6]
    assert result["success"] and result["first_success_step"] == 5
    assert result["steps"] == 8 and not result["final_done"]
    assert result["policy_queries"] == 3


@pytest.mark.parametrize("chunk", [np.zeros((8, 7)), np.full((10, 7), np.nan), np.zeros((10, 6))])
def test_invalid_chunk_fails_without_stepping(monkeypatch, chunk):
    result = run_episode(monkeypatch, lambda *_: chunk)
    assert result["steps"] == 0 and not result["success"] and result["error"]


def test_error_after_success_is_not_counted_as_success(monkeypatch):
    def predict(obs, _):
        if obs["step"] >= 6:
            raise RuntimeError("policy lost")
        return np.zeros((10, 7))
    result = run_episode(monkeypatch, predict)
    assert result["first_success_step"] == 5
    assert not result["success"] and result["error"]
    totals = aggregate_trials([result], scheduled_trials=3)
    assert totals["errors"] == 1 and totals["successes"] == 0
    assert totals["completion_coverage"] == pytest.approx(1/3)


def test_worker_records_render_failure_before_loading_policy(config_path, monkeypatch):
    request = server.load_evaluation(config_path)
    request.update(check_render=False, lease_token="test", launcher_identity=server_resources.process_identity(runtime.os.getpid()))
    path = config_path.parent / "request.json"
    path.write_text(json.dumps(request))
    monkeypatch.setattr(server_resources, "register_worker", lambda _: None)
    monkeypatch.setattr(runtime, "initialize_simulation", lambda *_: (_ for _ in ()).throw(RuntimeError("missing EGL")))
    monkeypatch.setattr(runtime, "validate_policy", lambda *_: pytest.fail("Must not load model"))
    before = {sig: runtime.signal.getsignal(sig) for sig in (runtime.signal.SIGINT, runtime.signal.SIGTERM)}
    try:
        assert runtime.run_worker(path) == 1
    finally:
        for sig, handler in before.items():
            runtime.signal.signal(sig, handler)
    result = json.loads((path.parent / "result.json").read_text())
    assert result["status"] == "FAILED" and "missing EGL" in result["error"]


def test_evaluation_respects_training_lease(config_path, monkeypatch, tmp_path):
    monkeypatch.setattr(server_resources, "LEASE_ROOT", tmp_path / "lease")
    with server_resources.TrainingLease():
        with pytest.raises(server_resources.TrainingBusy):
            server.run_evaluations(server.load_evaluation(config_path), check_render=True)
    assert not Path(server.load_evaluation(config_path)["output_root"]).exists()


@pytest.mark.parametrize("needs_kill", [False, True])
def test_worker_cleanup_waits_and_escalates_if_needed(monkeypatch, needs_kill):
    signals = []
    waits = []

    class Process:
        pid = 123

        def poll(self):
            return None

        def wait(self, timeout=None):
            waits.append(timeout)
            if timeout is not None and needs_kill:
                raise server.subprocess.TimeoutExpired("worker", timeout)
            return 0

    monkeypatch.setattr(server.os, "killpg", lambda pid, sig: signals.append((pid, sig)))
    server._stop_worker(Process())
    assert signals == [(123, server.signal.SIGTERM)] + ([(123, server.signal.SIGKILL)] if needs_kill else [])
    assert waits == [15, None]


def test_matrix_continues_after_failure_and_selects_model_environment(config_path, monkeypatch, tmp_path):
    monkeypatch.setattr(server_resources, "LEASE_ROOT", tmp_path / "lease")
    config = server.load_evaluation(config_path)
    config["policies"] = [str(child(tmp_path, name)[0]) for name in ("base", "pi05-libero-base")]
    seen = []

    class Process:
        def __init__(self, command, **kwargs):
            request = Path(command[-1])
            frozen = json.loads(request.read_text())
            seen.append((command, frozen))
            self.code = 1 if len(seen) == 1 else 0
            (request.parent / "result.json").write_text(json.dumps({
                "status": "FAILED" if self.code else "COMPLETED", "tasks": [],
                "error": "bad checkpoint" if self.code else None,
            }))
        def poll(self):
            return self.code
        def wait(self, **kwargs):
            return self.code

    monkeypatch.setattr(server.subprocess, "Popen", Process)
    assert server.run_evaluations(config) == 1
    assert [cmd[4] for cmd, _ in seen] == ["vla-liberox", "pi05"]
    assert seen[0][1]["tasks"] == seen[1][1]["tasks"]
    summary = json.loads(next(Path(config["output_root"]).glob("*/summary.json")).read_text())
    assert summary["status"] == "FAILED"
    assert [r["status"] for r in summary["evaluations"]] == ["FAILED", "COMPLETED"]


def test_worker_records_reset_error_and_recovers(config_path, monkeypatch, tmp_path):
    import torch
    config = server.load_evaluation(config_path)
    config["tasks"][0]["schedule"] = build_schedule(trials=2, init_state_indices=[0], base_seed=0, seed_count=1, schedule_seed=0)
    policy_path, payload = child(tmp_path, "pi05-libero-base")
    request = {**config, **server.policy_request(policy_path, config), "check_render": False, "lease_token": "test",
               "launcher_identity": server_resources.process_identity(runtime.os.getpid())}
    path = tmp_path / "request.json"
    path.write_text(json.dumps(request))
    monkeypatch.setattr(server_resources, "register_worker", lambda _: None)
    monkeypatch.setattr(runtime, "initialize_simulation", lambda *_: (None, {}))
    monkeypatch.setattr(runtime, "render_preflight", lambda *_: {"renderer": "fixture"})
    monkeypatch.setattr(runtime, "validate_policy", lambda *_: None)
    monkeypatch.setattr(runtime, "load_states", lambda _: [np.zeros(1)])
    adapter = SimpleNamespace(BACKEND="fixture", resolve_revision=lambda *a, **kw: "a" * 64)
    monkeypatch.setattr(runtime, "family_adapter", lambda _: adapter)
    backend = SimpleNamespace(evaluation_policy=lambda *a: (lambda *_: np.zeros((10, 7)), lambda a: a))
    monkeypatch.setattr(runtime.importlib, "import_module", lambda _: backend)
    monkeypatch.setattr(torch.cuda, "is_available", lambda: True)
    monkeypatch.setattr(torch.cuda, "get_device_name", lambda _: "fixture")
    monkeypatch.setattr(runtime, "capture", lambda env, obs: obs)
    calls = []
    def create(*_):
        calls.append(True)
        if len(calls) == 1:
            raise RuntimeError("reset failed")
        env = Env()
        env.language_instruction = "task"
        env.close = lambda: None
        return env
    monkeypatch.setattr(runtime, "create_env", create)
    before = {sig: runtime.signal.getsignal(sig) for sig in (runtime.signal.SIGINT, runtime.signal.SIGTERM)}
    try:
        assert runtime.run_worker(path) == 1
    finally:
        for sig, handler in before.items():
            runtime.signal.signal(sig, handler)
    result = json.loads((tmp_path / "result.json").read_text())
    aggregate = result["tasks"][0]["aggregate"]
    assert aggregate["attempted"] == 2 and aggregate["errors"] == 1
    assert aggregate["successes"] == 1 and aggregate["success_rate"] == .5


def test_changed_task_is_rejected_before_rendering(config_path, monkeypatch, tmp_path):
    request = server.load_evaluation(config_path)
    request.update(check_render=True, lease_token="test", launcher_identity=server_resources.process_identity(runtime.os.getpid()))
    Path(request["tasks"][0]["bddl_path"]).write_text("changed scene")
    path = tmp_path / "request.json"
    path.write_text(json.dumps(request))
    monkeypatch.setattr(server_resources, "register_worker", lambda _: None)
    monkeypatch.setattr(runtime, "initialize_simulation", lambda *_: pytest.fail("must reject changed task before rendering"))
    before = {sig: runtime.signal.getsignal(sig) for sig in (runtime.signal.SIGINT, runtime.signal.SIGTERM)}
    try:
        assert runtime.run_worker(path) == 1
    finally:
        for sig, handler in before.items():
            runtime.signal.signal(sig, handler)
    result = json.loads((tmp_path / "result.json").read_text())
    assert result["status"] == "FAILED" and "Task asset changed" in result["error"]
