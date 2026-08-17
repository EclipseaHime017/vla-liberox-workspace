from __future__ import annotations

import time

import numpy as np

from simulation_core import run_control_loop
from trajectory_utils import TrajectoryRecorder


def observation(step: int) -> dict:
    return {
        "robot0_eef_pos": np.asarray([step, 0, 1], dtype=np.float64),
        "robot0_eef_quat": np.asarray([0, 0, 0, 1], dtype=np.float64),
        "robot0_gripper_qpos": np.asarray([-0.1, 0.1], dtype=np.float64),
    }


class FakeEnv:
    def __init__(self):
        self.step_index = 0

    def get_sim_state(self):
        return np.asarray([self.step_index], dtype=np.float64)

    def step(self, action):
        self.step_index += 1
        return observation(self.step_index), 0.0, self.step_index == 3, {}


class FakeLimiter:
    def wait_before_step(self):
        return time.monotonic()


def test_policy_loop_chunks_and_stops_on_success():
    env = FakeEnv()
    recorder = TrajectoryRecorder(control_hz=20, capture_images=False)
    initial = observation(0)
    recorder.record_initial(env, initial)
    queries = []

    def query(obs, step):
        queries.append(step)
        return obs, [np.ones(7, dtype=np.float32) * value for value in range(8)]

    result = run_control_loop(
        env=env,
        recorder=recorder,
        initial_observation=initial,
        target_action_count=20,
        rate_limiter=FakeLimiter(),
        action_source="policy",
        open_loop_steps=2,
        policy_query=query,
        policy_action_transform=lambda value: value,
    )
    assert result.success
    assert result.executed_steps == 3
    assert result.policy_queries == 2
    assert queries == [0, 2]
    assert recorder.state_count == 4


def test_stop_is_checked_after_policy_query():
    env = FakeEnv()
    recorder = TrajectoryRecorder(control_hz=20, capture_images=False)
    initial = observation(0)
    recorder.record_initial(env, initial)
    stopped = False

    def query(obs, _step):
        nonlocal stopped
        stopped = True
        return obs, [np.zeros(7, dtype=np.float32)]

    result = run_control_loop(
        env=env,
        recorder=recorder,
        initial_observation=initial,
        target_action_count=1,
        rate_limiter=FakeLimiter(),
        action_source="policy",
        policy_query=query,
        policy_action_transform=lambda value: value,
        stop_requested=lambda: stopped,
    )
    assert result.executed_steps == 0
    assert result.stopped_reason == "user_stop"


def test_manual_input_is_sampled_after_the_control_boundary_wait():
    order = []

    class OrderedEnv(FakeEnv):
        def step(self, action):
            order.append("step")
            return super().step(action)

    class OrderedLimiter:
        def wait_before_step(self):
            order.append("wait")
            return time.monotonic()

    env = OrderedEnv()
    recorder = TrajectoryRecorder(control_hz=20, capture_images=False)
    initial = observation(0)
    recorder.record_initial(env, initial)

    def manual_query(_step):
        order.append("query")
        return np.zeros(7, dtype=np.float32)

    run_control_loop(
        env=env,
        recorder=recorder,
        initial_observation=initial,
        target_action_count=1,
        rate_limiter=OrderedLimiter(),
        action_source="manual",
        manual_query=manual_query,
        stop_on_success=False,
    )
    assert order == ["wait", "query", "step"]
