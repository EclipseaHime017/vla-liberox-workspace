from __future__ import annotations

import time

import numpy as np
import pytest

from simulation_core import confirmed_success_step, run_control_loop, success_confirm_steps
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


def test_policy_loop_latches_success_and_runs_to_target_steps():
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
        target_action_count=5,
        rate_limiter=FakeLimiter(),
        action_source="policy",
        open_loop_steps=2,
        policy_query=query,
        policy_action_transform=lambda value: value,
    )
    assert result.success
    assert result.executed_steps == 5
    assert result.policy_queries == 3
    assert result.stopped_reason == "horizon"
    assert queries == [0, 2, 4]
    assert recorder.state_count == 6


def test_continuation_keeps_success_reached_earlier_in_the_prefix():
    env = FakeEnv()
    recorder = TrajectoryRecorder(capture_images=False)
    recorder.record_initial(env, observation(0))
    for _ in range(5):
        obs, reward, done, _ = env.step([0.] * 7)
        recorder.record_transition(env, obs, np.zeros(7), np.zeros(7), reward, done, "policy")
    assert recorder.dones == [False, False, True, False, False]
    result = run_control_loop(env=env, recorder=recorder, initial_observation=observation(5),
        target_action_count=6, rate_limiter=FakeLimiter(), action_source="human",
        manual_query=lambda _: np.zeros(7))
    assert result.success and not recorder.dones[-1]


@pytest.mark.parametrize("horizon,execution_steps", [(8, 8), (10, 5), (10, 10), (50, 25)])
def test_policy_loop_executes_requested_prefix_then_replans(horizon, execution_steps):
    env = FakeEnv()
    recorder = TrajectoryRecorder(control_hz=20, capture_images=False)
    recorder.record_initial(env, observation(0))
    queries, actions = [], []

    def query(obs, step):
        queries.append(step)
        return obs, np.arange(horizon * 7, dtype=np.float32).reshape(horizon, 7)

    def transform(action):
        actions.append(action[0])
        return action

    result = run_control_loop(
        env=env, recorder=recorder, initial_observation=observation(0),
        target_action_count=execution_steps * 2 + 1, rate_limiter=FakeLimiter(),
        action_source="policy", open_loop_steps=execution_steps,
        policy_query=query, policy_action_transform=transform,
    )
    assert queries == [0, execution_steps, execution_steps * 2]
    assert actions == [float(i * 7) for i in range(execution_steps)] * 2 + [0.0]
    assert result.executed_steps == execution_steps * 2 + 1


def test_short_prediction_fails_before_executing_an_action():
    env = FakeEnv()
    recorder = TrajectoryRecorder(control_hz=20, capture_images=False)
    recorder.record_initial(env, observation(0))
    with pytest.raises(ValueError, match="actions"):
        run_control_loop(
            env=env, recorder=recorder, initial_observation=observation(0),
            target_action_count=20, rate_limiter=FakeLimiter(), action_source="policy",
            open_loop_steps=10, policy_query=lambda obs, step: (obs, np.zeros((8, 7))),
            policy_action_transform=lambda action: action,
        )
    assert env.step_index == 0


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


def test_manual_stop_during_wait_does_not_execute_an_extra_action():
    stopped = False

    class StopDuringWait:
        def wait_before_step(self):
            nonlocal stopped
            stopped = True

    env = FakeEnv()
    recorder = TrajectoryRecorder(control_hz=20, capture_images=False)
    recorder.record_initial(env, observation(0))

    def forbidden_query(_step):
        raise AssertionError("Stopped manual control must not sample or step")

    result = run_control_loop(
        env=env, recorder=recorder, initial_observation=observation(0),
        target_action_count=1, rate_limiter=StopDuringWait(), action_source="human",
        manual_query=forbidden_query, stop_requested=lambda: stopped,
    )
    assert result.stopped_reason == "user_stop"
    assert result.executed_steps == recorder.action_count == env.step_index == 0


def test_confirmed_success_requires_consecutive_done_values():
    assert success_confirm_steps() == 5  # vla-adapter-rynn-iql/configs/runtime.yaml
    assert confirmed_success_step([False, True, False, True, True, True, True]) is None
    assert confirmed_success_step([True] * 4 + [False] + [True] * 5) == 9
    assert confirmed_success_step(np.ones(5, dtype=bool)) == 4
    assert confirmed_success_step([]) is None
    with pytest.raises(ValueError):
        confirmed_success_step([True], required=0)


def test_success_rule_follows_the_system_setting(monkeypatch):
    import vla_rynn_iql.config as config
    monkeypatch.setattr(config, "success_consecutive_steps", lambda: 3)
    assert success_confirm_steps() == 3
    assert confirmed_success_step([False, True, True, True]) == 3
    assert confirmed_success_step([False, True, True, True], required=5) is None
