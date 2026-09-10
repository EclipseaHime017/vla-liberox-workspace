"""No serial port or motors: preparation and simulator-only joint following."""
import os
import sys
from dataclasses import replace

import numpy as np
import pytest

from backend.app.devices.factr_alignment import LeaderAlignment
from backend.app.devices.factr_joint_control import align_for_takeover


def test_alignment_ramp_blocked_error_and_settling():
    move = LeaderAlignment(np.zeros(7), np.ones(7), now=0.)
    error = move.update(np.zeros(7), np.zeros(7), .002)
    np.testing.assert_allclose(error, .0003)
    for now in np.arange(.004, 3., .002):
        assert np.max(np.abs(move.update(np.zeros(7), np.zeros(7), now))) <= .12
    assert not move.done
    move.update(np.ones(7), np.zeros(7), 3.)
    move.update(np.ones(7), np.zeros(7), 3.6)
    assert move.done and move.status()["state"] == "COMPLETE"


def test_alignment_timeout_and_invalid_feedback():
    move = LeaderAlignment(np.zeros(7), np.ones(7), now=0.)
    with pytest.raises(RuntimeError, match="timed out"):
        move.update(np.zeros(7), np.zeros(7), 61.)
    with pytest.raises(ValueError):
        LeaderAlignment([0]*6, [0]*7, now=0.)
    with pytest.raises(ValueError):
        move.update(np.full(7, np.nan), np.zeros(7), .002)


def test_alignment_completion_holds_until_explicit_release():
    class Controller:
        calls = []
        def start_alignment(self, owner, target): self.calls.append("align")
        def finish_alignment(self, owner): self.calls.append("release")
        def status(self):
            return {"connected": True, "gravity_enabled": True,
                    "alignment": {"state": "COMPLETE"}}
    controller = Controller()
    assert not align_for_takeover(controller, "s", [0]*7, lambda: True)
    assert not controller.calls
    assert align_for_takeover(controller, "s", [0]*7, lambda: False)
    assert controller.calls == ["align"]  # PD hold remains through countdown.
    controller.finish_alignment("s")
    assert controller.calls == ["align", "release"]


@pytest.mark.skipif(os.environ.get("RUN_FACTR_MUJOCO_SMOKE") != "1", reason="explicit simulator opt-in")
def test_real_panda_joint_following_recording_and_state_reconstruction(tmp_path):
    import eval_pickplace_direct as direct
    from backend.app.devices.factr_joint_control import JointFollower
    from trajectory_utils import TrajectoryRecorder, save_trajectory_bundle, load_trajectory
    config = replace(direct.load_config(direct.DEFAULT_CONFIG_PATH), headless=True, max_steps=40)
    direct.apply_runtime_environment(config)
    sys.path.insert(0, str(config.liberox_root))
    runtime = direct.load_runtime()
    bddl, init = direct.resolve_task(config.liberox_root, config.level, config.task_name)
    initial = direct.load_initial_states(runtime, init)[0]
    direct.prewarm_simulation_control(runtime, bddl, initial, config)
    env = direct.make_env(runtime, bddl, 128, 50, 20, config.seed, "vla_views", 256, 128, True)
    try:
        observation = direct.restore_state(env, initial)
        recorder = TrajectoryRecorder(capture_images=False)
        recorder.record_initial(env, observation)
        before = np.array(env.get_sim_state()).copy()
        follower = JointFollower(env)
        np.testing.assert_allclose(env.get_sim_state(), before, atol=1e-12, rtol=0)
        start = follower.joints()
        follower.check_aligned(start)
        with pytest.raises(RuntimeError, match="aligned"):
            follower.check_aligned(start + .3)
        target = start.copy()
        target[0] += .06
        for _ in range(40):
            pose = follower.pose()
            observation, reward, done, _ = follower.step(target, -1.)
            after = follower.pose()
            np.testing.assert_allclose(after[0], observation["robot0_eef_pos"], atol=1e-7)
            raw, action = follower.encoder.encode(pose, after, -1.)
            recorder.record_transition(env, observation, raw, action, reward, done, "human")
        assert follower.robot.action_dim == 8
        assert follower.joints()[0] > start[0] + .03
        np.testing.assert_allclose(follower.robot.controller.goal_qpos, target)
        assert np.isfinite(observation["robot0_eef_pos"]).all()
        # Renderer restores physical states, never attempts to replay OSC labels
        # through the joint controller. Dual camera observations have N+1 frames.
        direct.postprocess_recorded_trajectory(runtime, env, recorder, True,
            tmp_path/"vla_views.mp4", tmp_path/"agentview.mp4", 20, "vla_views", 256, 128, 128, 128)
        paths = save_trajectory_bundle(recorder, tmp_path/"trajectory", {"manual_source": "factr"},
                                       save_observations=True, create_plot=False)
        arrays, _ = load_trajectory(tmp_path/"trajectory.npz")
        assert arrays["env_action"].shape == (40, 7)
        assert len(arrays["sim_state"]) == 41
        np.testing.assert_allclose(arrays["raw_action"][:, :3] * .05,
                                   np.diff(arrays["eef_position"], axis=0), atol=1e-7)
        assert (tmp_path/"agentview.mp4").stat().st_size > 0
        assert (tmp_path/"vla_views.mp4").stat().st_size > 0
        images = np.load(paths["trajectory_observations"])
        assert images["agentview_image"].shape[0] == images["wrist_image"].shape[0] == 41
    finally:
        direct.close_env(env)
