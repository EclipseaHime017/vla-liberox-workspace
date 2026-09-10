from __future__ import annotations

import dataclasses
import json
import io
from pathlib import Path
import subprocess
import sys
import time
from types import SimpleNamespace

import numpy as np
import pytest

import test_factr as cli
from backend.app.devices.factr import FactrSnapshot, load_factr_config


class FakeService:
    def __init__(self, config):
        self.config = config
        self.steps = []
        self.state = "UNCALIBRATED"
        self.closed = False
        self.sequence = 0
        self.gravity = False
        self.fingerprint = {"test": True}

    def status(self):
        return {"state": self.state, "calibrated": True, "gravity_enabled": self.gravity}

    def enable_gravity(self):
        self.gravity = True
        self.steps.append("enable")

    def disable_gravity(self):
        self.gravity = False
        self.steps.append("disable")

    def calibration_snapshot(self):
        return {"reference": True, "open": 0, "closed": 1}

    def arm(self, *args, **kwargs):
        self.state = "ARMED"
        return self.snapshot(cli.OWNER)

    def snapshot(self, _owner):
        return self.latest_snapshot()

    def latest_snapshot(self):
        now = time.monotonic()
        self.sequence += 1
        return FactrSnapshot(sequence=self.sequence, captured_monotonic=now,
                             sample_monotonic=now-0.001, connected=True, stale=False,
                             gripper_fraction=0.0)

    def disarm(self, _owner):
        self.state = "READY"

    def diagnostics(self):
        return {"fake": True}

    def close(self):
        if self.gravity:
            self.disable_gravity()
        self.closed = True


def config(tmp_path, **changes):
    original = load_factr_config()
    return dataclasses.replace(original,
                                runtime={**original.runtime, "calibration_file": str(tmp_path/"calibration.json")},
                                countdown_seconds=0, **changes)


def test_passive_menu_waits_for_new_sample_not_old_timestamp(tmp_path):
    service = FakeService(config(tmp_path))
    fresh = service.latest_snapshot()
    samples = iter([dataclasses.replace(fresh, stale=True), fresh])
    service.latest_snapshot = lambda: next(samples)
    cli._check_menu_snapshot(service, sleep=lambda _: None)


@pytest.mark.parametrize("condition", ["gravity", "armed", "disconnected", "error"])
def test_menu_never_waits_through_active_or_disconnected_fault(tmp_path, condition):
    service = FakeService(config(tmp_path))
    service.gravity = condition == "gravity"
    if condition == "armed":
        service.state = "ARMED"
    sample = dataclasses.replace(service.latest_snapshot(), stale=True,
        connected=condition != "disconnected", error="bus failed" if condition == "error" else None)
    service.latest_snapshot = lambda: sample
    def forbidden_sleep(_):
        pytest.fail("Must not wait through active/stopped-bus fault")
    with pytest.raises(cli.ControllerUnavailable):
        cli._check_menu_snapshot(service, sleep=forbidden_sleep)


def test_passive_menu_wait_is_bounded(tmp_path):
    service = FakeService(config(tmp_path))
    sample = dataclasses.replace(service.latest_snapshot(), stale=True)
    service.latest_snapshot = lambda: sample
    times = iter([0., 0., 2.])
    with pytest.raises(cli.ControllerUnavailable, match="age=.*sequence="):
        cli._check_menu_snapshot(service, timeout=1., clock=lambda: next(times), sleep=lambda _: None)


def test_background_fault_is_reported_without_pressing_enter(tmp_path, monkeypatch, caplog):
    service = FakeService(config(tmp_path))
    stream = io.StringIO("g\n")
    monkeypatch.setattr(cli.sys, "stdin", stream)
    select_calls = []
    def select(readers, writers, errors, timeout):
        select_calls.append(timeout)
        if len(select_calls) == 1:
            return readers, [], []
        sample = dataclasses.replace(service.latest_snapshot(), error="read_ms=64 exceeds deadline", connected=False)
        service.latest_snapshot = lambda: sample
        return [], [], []  # Operator has entered nothing after g.
    monkeypatch.setattr(cli.select, "select", select)
    assert cli.run_test(service.config, service_factory=lambda _: service) == 1
    assert service.closed and not service.gravity
    assert service.steps == ["enable", "disable"]
    assert "read_ms=64 exceeds deadline" in caplog.text
    assert select_calls == [.1, .1]
    assert not list(tmp_path.iterdir())


def test_terminal_eof_has_distinct_exit_reason(tmp_path, caplog):
    service = FakeService(config(tmp_path))
    def eof(_):
        raise EOFError
    assert cli.run_test(service.config, service_factory=lambda _: service, input_fn=eof) == 1
    assert "Terminal input closed (EOF)" in caplog.text
    assert service.closed


def test_terminal_reader_handles_normal_command_and_eof(tmp_path):
    service = FakeService(config(tmp_path))
    def ready(readers, *args):
        return readers, [], []
    stream = io.StringIO("i\n")
    assert cli._read_terminal_input("FACTR> ", service, stream=stream, select_fn=ready) == "i"
    with pytest.raises(EOFError):
        cli._read_terminal_input("FACTR> ", service, stream=stream, select_fn=ready)



def test_calibration_is_one_shared_official_call(tmp_path):
    from backend.app.devices.factr_calibration import make_profile
    settings = config(tmp_path)
    service = FakeService(settings)
    calls, prompts = [], []
    profile = make_profile(settings, [0]*7, 0., -.8, service.fingerprint)
    service.calibrate = lambda: calls.append("official") or profile
    result = cli.calibrate_interactively(service, settings, input_fn=lambda p: prompts.append(p) or "")
    assert result is profile and calls == ["official"] and len(prompts) == 1
    assert "RELEASE" in prompts[0]

def test_calibration_cannot_run_while_gravity_is_on(tmp_path):
    service = FakeService(config(tmp_path))
    service.gravity = True
    with pytest.raises(RuntimeError, match="disable gravity"):
        cli.calibrate_interactively(service, service.config)



def test_quit_creates_no_test_records(tmp_path):
    service = FakeService(config(tmp_path))
    assert cli.run_test(service.config, service_factory=lambda _: service, input_fn=lambda _: "q") == 0
    assert service.closed and service.steps == []
    assert not list(tmp_path.iterdir())


def test_g_enables_directly_without_enable_prompt(tmp_path):
    service = FakeService(config(tmp_path))
    commands, prompts = iter(["g", "q"]), []
    def answer(prompt):
        prompts.append(prompt)
        return next(commands)
    assert cli.run_test(service.config, service_factory=lambda _: service, input_fn=answer) == 0
    assert service.steps == ["enable", "disable"]
    assert len(prompts) == 2
    assert not list(tmp_path.iterdir())


def test_disable_is_immediate_and_returns_to_menu_without_confirmation(tmp_path):
    service = FakeService(config(tmp_path))
    commands = iter(["g", "d", "q"])
    def answer(_):
        command = next(commands)
        if command == "q":
            assert not service.gravity and not service.closed
        return command
    assert cli.run_test(service.config, service_factory=lambda _: service, input_fn=answer) == 0
    assert service.closed and service.steps == ["enable", "disable"]


@pytest.mark.parametrize("during_simulation", [False, True])
def test_ctrl_c_disables_and_exits_without_another_prompt(tmp_path, monkeypatch, during_simulation):
    service = FakeService(config(tmp_path, mode="simulation" if during_simulation else "device"))
    service.enable_gravity()
    prompts = []
    def answer(prompt):
        prompts.append(prompt)
        if during_simulation:
            assert len(prompts) == 1
            return "s"
        raise KeyboardInterrupt
    def interrupt(*args, **kwargs):
        raise KeyboardInterrupt
    monkeypatch.setattr(cli, "run_simulation_test", interrupt)
    assert cli.run_test(service.config, service_factory=lambda _: service, input_fn=answer) == 130
    assert len(prompts) == 1 and service.closed and not service.gravity
    assert service.steps == ["enable", "disable"]


def test_repeated_device_tests_do_not_record_even_with_legacy_flags(tmp_path):
    service = FakeService(config(tmp_path, test_duration_seconds=.001))
    commands = iter(["s", "s", "q"])
    assert cli.run_test(service.config, service_factory=lambda _: service,
                        input_fn=lambda _: next(commands)) == 0
    assert service.closed and "enable" not in service.steps
    assert not list(tmp_path.iterdir())


def test_counter_does_not_accumulate_trajectory_data():
    counter = cli._StepCounter()
    counter.record_initial(None, None)
    for _ in range(1000):
        counter.record_transition(None, None, None, None, 0, False)
    assert counter.action_count == 1000
    assert counter.dones == (False,) and not hasattr(counter, "actions")


def test_uncalibrated_start_is_rejected_without_creating_results(tmp_path, caplog):
    service = FakeService(config(tmp_path))
    service.status = lambda: {"calibrated": False, "gravity_enabled": False}
    commands = iter(["s", "q"])
    assert cli.run_test(service.config, service_factory=lambda _: service,
                        input_fn=lambda _: next(commands)) == 0
    assert "calibration (c)" in caplog.text and not list(tmp_path.iterdir())


def test_device_failure_closes_without_writing_logs(tmp_path):
    service = FakeService(config(tmp_path))
    service.arm = lambda *args, **kwargs: None
    service.snapshot = lambda _: dataclasses.replace(service.latest_snapshot(), error="read failed")
    commands = iter(["s", "q"])
    assert cli.run_test(service.config, service_factory=lambda _: service,
                        input_fn=lambda _: next(commands)) == 1
    assert service.closed and not list(tmp_path.iterdir())


def test_non_tty_refuses_without_creating_data(tmp_path):
    result = subprocess.run([sys.executable, str(Path(cli.__file__))], stdin=subprocess.DEVNULL,
                            text=True, capture_output=True, cwd=tmp_path)
    assert result.returncode == 2
    assert "interactive terminal" in result.stderr
    assert list(tmp_path.iterdir()) == []


def test_device_entry_does_not_import_simulator_or_policy_stack():
    project = Path(cli.__file__).resolve().parents[1]
    program = f"""
import sys
sys.path[:0] = [{str(project)!r}, {str(project/'scripts')!r}]
class Blocker:
    def find_spec(self, fullname, path=None, target=None):
        if fullname.split('.')[0] in {{'torch', 'mujoco', 'libero', 'prismatic', 'eval_pickplace_direct', 'simulation_core'}}:
            raise RuntimeError('Forbidden eager import: ' + fullname)
sys.meta_path.insert(0, Blocker())
import test_factr
"""
    result = subprocess.run([sys.executable, "-c", program], text=True, capture_output=True)
    assert result.returncode == 0, result.stderr


@pytest.mark.parametrize("stop", ["error", "interrupt", "viewer"])
def test_simulation_stop_never_saves_or_postprocesses(tmp_path, monkeypatch, stop):
    settings = config(tmp_path, mode="simulation", max_steps=5)
    root = tmp_path/"liberox"
    (root/"libero").mkdir(parents=True)

    def observation(index):
        return {"robot0_eef_pos": np.array([index, 0, 1], dtype=float),
                "robot0_eef_quat": np.array([0, 0, 0, 1], dtype=float),
                "robot0_gripper_qpos": np.array([0.04, -0.04])}

    class Env:
        index = 0
        closed = False
        def get_sim_state(self):
            return np.array([self.index], dtype=float)
        def step(self, action):
            if self.index == 1 and stop != "viewer":
                if stop == "interrupt":
                    raise KeyboardInterrupt()
                raise RuntimeError("fake MuJoCo failure")
            self.index += 1
            return observation(self.index), 0.0, False, {}
    env = Env()
    env.viewer = SimpleNamespace(is_running=lambda: not (stop == "viewer" and env.index >= 1))

    class Follower:
        def __init__(self, _env):
            self.sim = SimpleNamespace(data=SimpleNamespace(qpos=np.asarray(settings.reference_joint_positions)))
            self.joint_indices = np.arange(7)
        def current_pose(self): return np.zeros(3), np.eye(3)
        def joints(self): return np.asarray(settings.reference_joint_positions)
        def check_aligned(self, _q): pass
        def step(self, q, grip): return env.step(np.r_[q, grip])
        def pose(self, _joints): return self.current_pose()
        def mapper(self, _joints): return self
        def action(self, *args):
            return np.array([0, 0, 0, 0, 0, 0, -1], dtype=float), {
                "saturated": False, "target_position": [0, 0, 0],
                "target_rotation_vector": [0, 0, 0]}

    eval_config = SimpleNamespace(control_hz=20, liberox_root=root, level="LEVEL1", task_name="test",
                                  env_resolution=32, seed=0, video_camera="both", video_width=32,
                                  video_height=32, main_view_video_width=32, main_view_video_height=32,
                                  headless=True, video_fps=20)
    def postprocess(*args):
        raise AssertionError("Live FACTR test must never generate recordings")
    fake_direct = SimpleNamespace(
        load_config=lambda _: eval_config, apply_runtime_environment=lambda _: None,
        load_runtime=lambda: SimpleNamespace(parse_bddl_file=lambda _: {"language": "test"}),
        resolve_task=lambda *args: (tmp_path/"test.bddl", tmp_path/"init"),
        load_initial_states=lambda *args: [np.zeros(1)], prewarm_simulation_control=lambda *args: None,
        make_env=lambda *args: env, restore_state=lambda *args: observation(0),
        render_live_window=lambda _: None, close_native_mujoco_viewer=lambda _: None,
        close_env=lambda _: setattr(env, "closed", True), NATIVE_VIEWER_ATTRIBUTE="viewer",
        RealTimeControlLimiter=lambda *args: SimpleNamespace(wait_before_step=time.monotonic),
        postprocess_recorded_trajectory=postprocess)
    monkeypatch.setitem(sys.modules, "eval_pickplace_direct", fake_direct)
    monkeypatch.setitem(sys.modules, "backend.app.devices.factr_joint_control", SimpleNamespace(
        JointFollower=Follower, align_for_takeover=lambda *args: True))
    service = FakeService(settings)
    service.finish_alignment = lambda _: None
    service.enable_gravity()
    def close_env(_):
        assert service.gravity == (stop == "viewer")  # Error/interrupt withdraw before scene teardown.
        env.closed = True
    fake_direct.close_env = close_env
    if stop == "interrupt":
        with pytest.raises(KeyboardInterrupt):
            cli.run_simulation_test(service, settings)
    else:
        result = cli.run_simulation_test(service, settings)
        assert result["action_count"] == 1 and result["state_count"] == 2
    assert env.closed
    assert service.closed == (stop != "viewer")
    assert not list(tmp_path.rglob("*.npz"))
    assert not list(tmp_path.rglob("*.csv"))
    assert not list(tmp_path.rglob("*.mp4"))
    assert not list(tmp_path.rglob("summary.json"))
