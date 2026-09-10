"""Recorded FACTR transitions; fake device only, no serial or motor writes."""
import threading
from types import SimpleNamespace

import numpy as np
import pytest

from backend.app.devices.factr import FactrSnapshot
from backend.app.devices.factr_joint_control import EndEffectorActionEncoder
from backend.app.domain.run import SimulationSession
from backend.app.workers.factr_control_worker import run_factr_loop
from trajectory_utils import TrajectoryRecorder


def encoder():
    return EndEffectorActionEncoder(SimpleNamespace(use_delta=True, control_dim=6,
        input_min=-1., input_max=1., output_min=np.array([-.05]*3+[-.5]*3),
        output_max=np.array([.05]*3+[.5]*3)))


def test_inverse_osc_world_rotation_and_clipping():
    from scipy.spatial.transform import Rotation
    encode = encoder()
    position = np.array([.2, .3, .5])
    rotation = Rotation.from_euler("xyz", [.6, -.4, 1.2]).as_matrix()
    delta = np.array([.1, -.2, .05])
    after = (position + [.025, -.01, .1], Rotation.from_rotvec(delta).as_matrix() @ rotation)
    raw, action = encode.encode((position, rotation), after, -1.)
    np.testing.assert_allclose(raw, [.5, -.2, 2., .2, -.4, .1, -1.], atol=1e-6)
    np.testing.assert_allclose(action, [.5, -.2, 1., .2, -.4, .1, -1.], atol=1e-6)
    assert encode.diagnostics()["clipped_fraction"] == 1.
    assert encode.diagnostics()["clipped_axes"] == [0, 0, 1, 0, 0, 0]
    raw, action = encode.encode((position, rotation), (position, rotation), 1.)
    np.testing.assert_allclose(action, [0.]*6+[1.], atol=1e-7)
    assert encode.diagnostics()["clipped_fraction"] == .5


@pytest.mark.parametrize("failure", [False, True])
def test_loop_preserves_prefix_post_done_steps_and_partial_recording(tmp_path, failure):
    class Env:
        step_index = 0
        def get_sim_state(self): return np.array([self.step_index], dtype=float)
        def obs(self):
            return {"robot0_eef_pos": np.array([.01*self.step_index, 0, 0]),
                    "robot0_eef_quat": np.array([0., 0., 0., 1.]),
                    "robot0_gripper_qpos": np.array([.01, -.01])}
    env = Env()
    parent = TrajectoryRecorder(capture_images=False)
    parent.record_initial(env, env.obs())
    env.step_index = 1
    parent.record_transition(env, env.obs(), np.zeros(7), np.zeros(7), 0., False, "policy")
    recorder = TrajectoryRecorder.from_prefix(parent.arrays(), 1, 20.)
    class Follower:
        encoder = encoder()
        def pose(self): return env.obs()["robot0_eef_pos"], np.eye(3)
        def step(self, joints, gripper):
            assert joints == tuple(range(7)) and gripper == 1.
            if failure and env.step_index == 2: raise RuntimeError("device fault")
            env.step_index += 1
            return env.obs(), 1., True, {}
    follower = Follower()
    follower.env = env
    record = SimulationSession(id="s", kind="branch", output_dir=tmp_path, max_steps=4,
                               open_loop_steps=8, resume_step=1)
    snapshot = FactrSnapshot(connected=True, stale=False, sample_monotonic=1.,
        captured_monotonic=1.002, joint_positions=tuple(range(7)), gripper_command=1.)
    call = dict(record=record, controller=SimpleNamespace(snapshot=lambda _: snapshot),
        follower=follower, recorder=recorder, initial_observation=env.obs(),
        rate_limiter=SimpleNamespace(wait_before_step=lambda: None), on_transition=lambda *args: None)
    if failure:
        with pytest.raises(RuntimeError, match="device fault"): run_factr_loop(**call)
        assert recorder.action_count == 2  # Valid prefix plus completed action survive.
    else:
        result = run_factr_loop(**call)
        assert result.executed_steps == 3 and result.success
        assert recorder.action_count == 4 and recorder.state_count == 5
    data = recorder.arrays()
    np.testing.assert_array_equal(data["env_action"][0], parent.arrays()["env_action"][0])
    np.testing.assert_allclose(data["env_action"][1:, 0], .2)
    np.testing.assert_array_equal(data["action_source"][1:], "human")
    assert data["done"][1:].all()


def test_snapshot_stale_and_stop_prevent_physics(tmp_path):
    record = SimulationSession(id="s", kind="branch", output_dir=tmp_path, max_steps=1, open_loop_steps=8)
    recorder = TrajectoryRecorder(capture_images=False)
    call = dict(record=record, controller=SimpleNamespace(snapshot=lambda _: FactrSnapshot()),
        follower=None, recorder=recorder, initial_observation={},
        rate_limiter=SimpleNamespace(wait_before_step=lambda: None), on_transition=lambda *args: None)
    with pytest.raises(RuntimeError, match="stale"): run_factr_loop(**call)
    record.stop_event.set()
    assert run_factr_loop(**call).stopped_reason == "user_stop"


@pytest.mark.parametrize("failure", [None, "prepare", "step", "postprocess", "cancel"])
def test_shared_session_publishes_factr_like_other_manual_runs(tmp_path, monkeypatch, failure):
    import eval_pickplace_direct as direct
    import backend.app.devices.factr_joint_control as joint_module
    from backend.app.workers.simulation_worker import SimulationManager
    from backend.app.recording.episode_recorder import EpisodeRecorderFactory
    from trajectory_utils import save_trajectory_bundle, load_trajectory
    calls = []
    class Env:
        env = SimpleNamespace(control_freq=20)
        index = 0
        def get_sim_state(self): return np.array([self.index], dtype=float)
        def obs(self):
            return {"robot0_eef_pos": np.array([self.index*.01, 0, 0]),
                    "robot0_eef_quat": np.array([0., 0., 0., 1.]),
                    "robot0_gripper_qpos": np.array([.01, -.01])}
    env = Env()
    parent = TrajectoryRecorder(capture_images=False)
    parent.record_initial(env, env.obs())
    env.index = 1
    parent.record_transition(env, env.obs(), np.ones(7), np.ones(7), 0., False, "human")
    save_trajectory_bundle(parent, tmp_path/"source", {"seed": 7}, create_plot=False)
    record = SimulationSession(id="s", kind="branch", output_dir=tmp_path/"run", max_steps=4,
        open_loop_steps=8, seed=7, resume_step=1, control_mode="manual", manual_source="factr",
        source_trajectory=str(tmp_path/"source.npz"), manual_translation_gain=.25, manual_rotation_gain=.25)
    record.episode_dir.mkdir(parents=True)
    class Stop:
        def is_set(self): return False
        def wait(self, seconds): calls.append("countdown"); return False
    record.stop_event = Stop()
    class Controller:
        gravity = True
        def calibration_snapshot(self): return {"id": "calibrated"}
        def start_alignment(self, *args): calls.append("align")
        def finish_alignment(self, *args): calls.append("release")
        def status(self): return {"connected": True, "gravity_enabled": failure != "cancel",
                                  "alignment": {"state": "COMPLETE"}}
        def arm(self, *args, gripper):
            assert calls.count("countdown") == 3 and gripper == 1.
            calls.append("arm"); return self.snapshot("s")
        def snapshot(self, _): return FactrSnapshot(connected=True, stale=False,
            captured_monotonic=1., sample_monotonic=1., joint_positions=(0.,)*7, gripper_command=1.)
        def disarm(self, *args): calls.append("disarm")
        def emergency_stop(self, reason): self.gravity = False; calls.append("off")
    controller = Controller()
    class Follower:
        def __init__(self, e): self.env = e; self.encoder = encoder()
        def joints(self): return np.zeros(7)
        def check_aligned(self, _): pass
        def pose(self): return env.obs()["robot0_eef_pos"], np.eye(3)
        def step(self, *args):
            if failure == "step" and env.index == 2: raise RuntimeError("step fault")
            env.index += 1
            return env.obs(), 1., True, {}
    monkeypatch.setattr(joint_module, "JointFollower", Follower)
    monkeypatch.setattr(direct, "RealTimeControlLimiter", lambda *a: SimpleNamespace(wait_before_step=lambda: None))
    def postprocess(**kw):
        calls.append("postprocess")
        if failure == "postprocess": raise RuntimeError("render fault")
        rec = kw["recorder"]
        rec.agentview_images = [np.zeros((2, 2, 3), dtype=np.uint8) for _ in rec.sim_states]
        rec.wrist_images = [image.copy() for image in rec.agentview_images]
        rec.capture_images = True
        kw["combined_path"].write_bytes(b"paired video")
        kw["main_view_path"].write_bytes(b"main video")
    monkeypatch.setattr(direct, "postprocess_recorded_trajectory", postprocess)
    def preview(r, state):
        r.preview_error = "prepare fault" if failure == "prepare" else None
        r.preview_event.set()
    worker = object.__new__(SimulationManager)
    worker.lock = threading.RLock()
    worker.active_session_id = "s"
    worker.factr_controller = controller
    worker.controller = None
    worker.eval_config = direct.load_config(direct.DEFAULT_CONFIG_PATH)
    worker.runtime = None
    worker.evaluator = SimpleNamespace(success=bool)
    worker.recorder_factory = SimpleNamespace(branch=EpisodeRecorderFactory.branch,
        save=EpisodeRecorderFactory.save, compare=lambda *a, **kw: {})
    worker.catalog = SimpleNamespace(entry=lambda _: SimpleNamespace(prompt="bowl"), paths=lambda _: (None, None))
    worker.simulator = SimpleNamespace(prewarm=lambda *a: None, create=lambda *a, **kw: env,
        restore=lambda *a: env.obs(), observations=lambda e, o: o, close=lambda e: calls.append("close"))
    worker.preview = SimpleNamespace(submit=preview)
    worker._persist_manifest = lambda r: None
    worker._metadata = lambda r, *a: {"manual_source": r.manual_source, "seed": 7,
                                    "controller_diagnostics": r.controller_diagnostics}
    worker._run_session(record)
    assert worker.active_session_id is None and "disarm" in calls and "close" in calls
    assert record.trajectory and (record.output_dir/"summary.json").exists()
    data, meta = load_trajectory(record.episode_dir/"trajectory.npz")
    assert meta["manual_source"] == "factr"
    if failure in ("prepare", "cancel"):
        assert len(data["env_action"]) == 1 and "postprocess" not in calls
    else:
        expected = 2 if failure == "step" else 4
        assert len(data["env_action"]) == expected
        assert data["action_source"].tolist() == ["human"]*expected
        if failure != "postprocess":
            with np.load(record.episode_dir/"trajectory_observations.npz") as images:
                assert len(images["agentview_image"]) == expected+1
                assert len(images["wrist_image"]) == expected+1
            assert (record.episode_dir/"agentview.mp4").exists()
    assert record.status == ("ERROR" if failure in ("prepare", "step", "postprocess") else "COMPLETED")
    assert controller.gravity == (failure in (None, "cancel"))
