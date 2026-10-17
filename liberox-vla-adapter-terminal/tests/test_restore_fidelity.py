"""Opt-in real MuJoCo checks for environment construction and branch restoration."""
import os
import time
from pathlib import Path

import numpy as np
import pytest

from backend.app.simulators.control_state import restore_control_state

pytestmark = pytest.mark.skipif(os.environ.get("RUN_MUJOCO_SMOKE") != "1", reason="opt-in MuJoCo smoke test")

TASK = ("LEVEL1", "EXTENSION_KITCHEN_SCENE11_place_the_black_bowl_on_the_flat_stove")


def _actions(count):
    rng = np.random.default_rng(3)
    actions = np.zeros((count, 7), dtype=np.float32)
    actions[:, :3] = rng.uniform(-.4, .4, (count, 3))
    actions[:, 3:6] = rng.uniform(-.2, .2, (count, 3))
    actions[:, 6] = np.where(np.arange(count) < count // 3, -1., 1.)  # open, then close
    return actions


def _setup(monkeypatch):
    import eval_pickplace_direct as direct
    root = Path(__file__).resolve().parents[2]
    monkeypatch.syspath_prepend(str(root / "third_party/LIBERO-X"))
    runtime = direct.load_runtime()
    bddl, init = direct.resolve_task(root / "third_party/LIBERO-X", *TASK)
    return direct, runtime, bddl, direct.load_initial_states(runtime, init)[0]


def _rollout(direct, env, state, actions):
    direct.restore_state(env, state)
    states = [env.get_sim_state().copy()]
    for action in actions:
        env.step(action.tolist())
        states.append(env.get_sim_state().copy())
    return np.asarray(states)


def test_soft_reset_environment_matches_hard_reset_rollout(monkeypatch):
    direct, runtime, bddl, state = _setup(monkeypatch)
    actions = _actions(40)
    results, seconds = {}, {}
    for hard in (True, False):
        started = time.perf_counter()
        env = direct.make_env(runtime, bddl, 64, 60, 20, 0, "vla_views", 128, 64, True, hard_reset=hard)
        seconds[hard] = time.perf_counter() - started
        try:
            results[hard] = _rollout(direct, env, state, actions)
        finally:
            direct.close_env(env)
    print(f"make_env seconds: hard_reset={seconds[True]:.2f} soft_reset={seconds[False]:.2f}")
    np.testing.assert_allclose(results[False], results[True], atol=1e-9, rtol=0)


def test_policy_branch_resume_reproduces_the_source_rollout(monkeypatch):
    direct, runtime, bddl, state = _setup(monkeypatch)
    actions, resume = _actions(40), 25
    env = direct.make_env(runtime, bddl, 64, 60, 20, 0, "vla_views", 128, 64, True)
    try:
        source = _rollout(direct, env, state, actions)
    finally:
        direct.close_env(env)
    deviations = {}
    for reanchor in (False, True):
        env = direct.make_env(runtime, bddl, 64, 60, 20, 0, "vla_views", 128, 64, True)
        try:
            direct.restore_state(env, source[resume])
            restore_control_state(env, actions[:resume], 20., reanchor_nullspace=reanchor)
            branch = [env.get_sim_state().copy()]
            for action in actions[resume:]:
                env.step(action.tolist())
                branch.append(env.get_sim_state().copy())
        finally:
            direct.close_env(env)
        deviations[reanchor] = float(np.max(np.abs(np.asarray(branch) - source[resume:])))
    print(f"max state deviation after resume: keep_nullspace={deviations[False]:.2e} "
          f"reanchor={deviations[True]:.2e}")
    assert deviations[False] < 1e-6


def _fixtures(env):
    model = getattr(env, "env", env).sim.model
    return np.array(model.body_pos).copy(), np.array(model.body_quat).copy()


def test_reused_environment_after_reset_matches_a_fresh_one(monkeypatch):
    direct, runtime, bddl, state = _setup(monkeypatch)
    first, second = _actions(40), _actions(40)[::-1].copy()
    reused = direct.make_env(runtime, bddl, 64, 41, 20, 0, "vla_views", 128, 64, True)
    try:
        _rollout(direct, reused, state, first)
        started = time.perf_counter()
        reused.seed(1)
        reused.reset()
        reset_seconds = time.perf_counter() - started
        again = _rollout(direct, reused, state, second)
        reused_fixtures = _fixtures(reused)
    finally:
        direct.close_env(reused)
    fresh = direct.make_env(runtime, bddl, 64, 41, 20, 1, "vla_views", 128, 64, True)
    try:
        expected = _rollout(direct, fresh, state, second)
        fresh_fixtures = _fixtures(fresh)
    finally:
        direct.close_env(fresh)
    print(f"soft reset of a used environment: {reset_seconds:.3f}s")
    np.testing.assert_allclose(again, expected, atol=1e-9, rtol=0)
    # Seed-placed fixtures live in the model, not in the restored state.
    for left, right in zip(reused_fixtures, fresh_fixtures):
        np.testing.assert_allclose(left, right, atol=1e-12, rtol=0)


def test_seed_changes_fixture_placement(monkeypatch):
    direct, runtime, bddl, _ = _setup(monkeypatch)
    placements = []
    for seed in (0, 1):
        env = direct.make_env(runtime, bddl, 64, 41, 20, seed, "vla_views", 128, 64, True)
        try:
            placements.append(_fixtures(env)[0])
        finally:
            direct.close_env(env)
    offset = float(np.max(np.abs(placements[0] - placements[1])))
    print(f"max fixture offset between seeds: {offset:.4f} m")
    # Measured 1.56 cm for this scene; 1 mm guards against seeds silently no longer mattering.
    assert offset > 1e-3
