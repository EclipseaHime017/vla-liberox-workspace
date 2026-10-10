"""Opt-in real LIBERO / OSC smoke test with deterministic synthetic HID input.

RUN_MUJOCO_SMOKE=1 MUJOCO_GL=egl PYTHONPATH=.:scripts pytest -s tests/test_spacemouse_simulation.py
No physical SpaceMouse or VLA checkpoint is required; no motor device is opened.
"""
from __future__ import annotations

import dataclasses
import os
import sys
from pathlib import Path

import numpy as np
import pytest

from backend.app.devices.spacemouse import SpaceMouseInput, load_spacemouse_config
from backend.app.devices.spacemouse_motion import SpaceMouseActionMapper
from simulation_core import run_control_loop
from trajectory_utils import TrajectoryRecorder, load_trajectory, save_trajectory_bundle


@pytest.mark.skipif(os.environ.get("RUN_MUJOCO_SMOKE") != "1", reason="opt-in MuJoCo smoke test")
@pytest.mark.parametrize("frame", ["world", "tool"])
def test_exclusive_commands_step_real_osc_and_save_complete_partial_trajectory(tmp_path, monkeypatch, frame):
    import eval_pickplace_direct as direct

    root = Path(__file__).resolve().parents[2]
    monkeypatch.syspath_prepend(str(root / "third_party" / "LIBERO-X"))
    runtime = direct.load_runtime()
    bddl, init_path = direct.resolve_task(root / "third_party" / "LIBERO-X", "LEVEL1",
        "EXTENSION_KITCHEN_SCENE11_place_the_black_bowl_on_the_flat_stove")
    state = direct.load_initial_states(runtime, init_path)[0]
    env = direct.make_env(runtime, bddl, 64, 25, 20, 0, "vla_views", 128, 64, True)
    clock = [10.]
    mouse = SpaceMouseInput(dataclasses.replace(load_spacemouse_config(),
        translation_gain=.1, rotation_gain=.1, smoothing_alpha=.5), clock=lambda: clock[0])
    mouse._connected = True
    mouse.reset_for_arm(gripper=1., control_frame=frame)
    mapper = SpaceMouseActionMapper(env)
    executed = []
    snapshots = []
    recorder = TrajectoryRecorder(control_hz=20., capture_images=False)
    try:
        initial = direct.restore_state(env, state)
        recorder.record_initial(env, initial)

        def query(step):
            clock[0] += .05
            if step < 20:
                axes = ((0, .8, 0, .1, 0, 0) if step < 8 else
                        (.1, 0, 0, 0, 0, .8) if step < 16 else (0,) * 6)
                mouse._accept_event(step + 1, axes, (0, 0))
            elif step == 20:
                clock[0] += .3  # Stale: hold gripper but no motion.
            else:
                mouse._connected = False
            snapshot = mouse.latest_snapshot()
            snapshots.append(snapshot)
            if not snapshot.connected:
                return None
            action = mapper.convert(snapshot.action, snapshot.control_frame)
            executed.append(action.copy())
            return action

        result = run_control_loop(env=env, recorder=recorder, initial_observation=initial,
            target_action_count=24, rate_limiter=direct.RealTimeControlLimiter(20., False),
            action_source="human", manual_query=query, stop_on_success=False)
        assert result.stopped_reason == "controller_stop"
        assert recorder.action_count == 21 and recorder.state_count == 22
        actions = np.stack(recorder.env_actions)
        assert np.array_equal(actions, executed)
        local = np.asarray([s.action for s in snapshots[:-1]], dtype=np.float32)
        assert np.array_equal(actions, local) if frame == "world" else not np.allclose(actions, local)
        assert np.all(actions[:8, 3:6] == 0) and np.any(actions[:8, :3])
        assert np.all(actions[8:16, :3] == 0) and np.any(actions[8:16, 3:6])
        assert np.all(actions[16:, :6] == 0) and np.all(actions[:, 6] == 1)
        assert np.all(np.isfinite(recorder.sim_states))
        assert np.linalg.norm(recorder.eef_positions[8] - recorder.eef_positions[0]) > 1e-4
        assert np.linalg.norm(recorder.eef_quaternions[16] - recorder.eef_quaternions[8]) > 1e-4
        # Reconstruct cameras after control, just as the GUI records observations.
        direct.postprocess_recorded_trajectory(runtime, env, recorder,
            combined_path=None, main_view_path=None, save_observations=True,
            fps=20, video_camera="vla_views", video_width=128, video_height=64,
            main_view_width=64, main_view_height=64)
        paths = save_trajectory_bundle(recorder, tmp_path / "trajectory",
            {"manual_source": "spacemouse", "motion_mode": "exclusive", "manual_control_frame": frame}, create_plot=False)
        trajectory, metadata = load_trajectory(Path(paths["trajectory"]))
        assert metadata["control_hz"] == 20
        assert metadata["manual_control_frame"] == frame
        assert metadata["action_count"] == 21 and metadata["state_count"] == 22
        assert np.array_equal(trajectory["env_action"], actions)
        assert set(trajectory["action_source"]) == {"human"}
        with np.load(paths["trajectory_observations"]) as images:
            assert images["agentview_image"].shape == images["wrist_image"].shape == (22, 64, 64, 3)
        assert not any(name.startswith("prismatic") for name in sys.modules)
    finally:
        env.close()
