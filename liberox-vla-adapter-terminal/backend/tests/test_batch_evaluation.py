from __future__ import annotations

from collections import Counter
from pathlib import Path
from types import SimpleNamespace
import sys

TERMINAL_ROOT = Path(__file__).resolve().parents[2]
SCRIPTS_ROOT = TERMINAL_ROOT / "scripts"
if str(SCRIPTS_ROOT) not in sys.path:
    sys.path.insert(0, str(SCRIPTS_ROOT))

import numpy as np
import pytest
import yaml

from backend.app.evaluation.batch import (
    ConsecutiveSuccess,
    _apply_close_error,
    _episode,
    aggregate_trials,
    build_evaluation_preview,
    build_schedule,
    load_effective_config,
    run_evaluation,
)


class _FakeEnvironment:
    def __init__(self, dones):
        self.dones = list(dones)
        self.steps = 0

    def step(self, _action):
        done = self.dones[self.steps]
        self.steps += 1
        return {"observation": self.steps}, 0.0, done, {}


class _FakeSimulator:
    @staticmethod
    def restore(_env, _state):
        return {"observation": 0}

    @staticmethod
    def observations(_env, observation):
        return observation


class _FakeProvider:
    @staticmethod
    def predict(_observation, _prompt, _disabled):
        return [np.zeros(7, dtype=np.float32) for _ in range(8)]

    @staticmethod
    def process_action(action):
        return action


def test_schedule_is_reproducible_and_balanced_across_all_dimensions():
    first = build_schedule(
        trials=103,
        init_state_indices=list(range(10)),
        base_seed=20,
        seed_count=None,
        schedule_seed=7,
    )
    second = build_schedule(
        trials=103,
        init_state_indices=list(range(10)),
        base_seed=20,
        seed_count=None,
        schedule_seed=7,
    )
    assert first == second
    assert [item["trial_index"] for item in first] == list(range(103))
    for values in (
        Counter(item["init_state_index"] for item in first).values(),
        Counter(item["seed"] for item in first).values(),
        Counter((item["init_state_index"], item["seed"]) for item in first).values(),
    ):
        values = list(values)
        assert max(values) - min(values) <= 1


def test_schedule_supports_fixed_seed_and_rejects_bad_inputs():
    schedule = build_schedule(
        trials=7,
        init_state_indices=[1, 4, 9],
        base_seed=42,
        seed_count=1,
        schedule_seed=0,
    )
    assert {item["seed"] for item in schedule} == {42}
    assert max(Counter(item["init_state_index"] for item in schedule).values()) == 3
    with pytest.raises(ValueError, match="duplicates"):
        build_schedule(
            trials=1, init_state_indices=[0, 0], base_seed=0,
            seed_count=1, schedule_seed=0,
        )


def test_preview_exposes_distribution_hash_and_duration():
    preview = build_evaluation_preview(
        trials=12, init_state_indices=[0, 1, 2], base_seed=3,
        seed_count=2, schedule_seed=9, control_hz=20, max_steps=100,
    )
    assert len(preview["schedule"]) == 12
    assert len(preview["schedule_sha256"]) == 64
    assert preview["estimated_simulation_seconds"] == 60
    assert preview["init_state_counts"] == preview["distribution"]["init_state_counts"]


def test_success_requires_streak_then_latches_without_ending_horizon():
    tracker = ConsecutiveSuccess(5)
    for step, done in enumerate([True, True, False, True, True, True, True], 1):
        assert tracker.observe(done, step) is False
    assert tracker.observe(True, 8) is True
    assert tracker.first_confirmed_step == 8
    assert tracker.maximum == 5
    assert tracker.observe(False, 9) is True
    assert tracker.current == 0


def test_episode_runs_full_horizon_after_success_latches():
    env = _FakeEnvironment([True] * 5 + [False] * 3)
    result = _episode(
        simulator=_FakeSimulator(), provider=_FakeProvider(), env=env,
        initial_state=np.zeros(1), prompt="task", trial_index=0,
        init_state_index=2, seed=9, max_steps=8, open_loop_steps=8,
        control_hz=20, realtime=False, success_streak=5,
        disabled_policy_cameras=(), stop_requested=lambda: False,
    )
    assert env.steps == 8
    assert result["steps"] == 8
    assert result["success"] is True
    assert result["first_success_step"] == 5
    assert result["final_done"] is False


def test_environment_close_failure_becomes_trial_error():
    trial = {"success": True, "error": None}
    assert _apply_close_error(trial, RuntimeError("EGL cleanup")) is True
    assert trial["success"] is False
    assert "EGL cleanup" in trial["error"]
    assert _apply_close_error(trial, RuntimeError("second")) is False
    assert "second" in trial["error"]


def test_aggregation_counts_errors_in_denominator_and_groups_results():
    trials = [
        {
            "init_state_index": 0, "seed": 1, "success": True, "error": None,
            "first_success_step": 8, "policy_queries": 2,
            "inference_latency_ms": 10.0, "measured_control_hz": 20.0,
            "elapsed_seconds": 1.0,
        },
        {
            "init_state_index": 0, "seed": 2, "success": False, "error": None,
            "first_success_step": None, "policy_queries": 2,
            "inference_latency_ms": 12.0, "measured_control_hz": 19.0,
            "elapsed_seconds": 1.2,
        },
        {
            "init_state_index": 1, "seed": 1, "success": True, "error": "boom",
            "first_success_step": 4, "policy_queries": 0,
            "inference_latency_ms": None, "measured_control_hz": None,
            "elapsed_seconds": 0.1,
        },
    ]
    result = aggregate_trials(trials, scheduled_trials=6)
    assert result["success_rate"] == pytest.approx(1 / 3)
    assert result["errors"] == 1
    assert result["failures"] == 1
    assert result["completion_coverage"] == 0.5
    assert result["by_seed"]["1"]["success_rate"] == 0.5
    assert result["wilson_95"][0] < result["success_rate"] < result["wilson_95"][1]
    assert result["first_success_step_mean"] == 8


def test_effective_yaml_is_strict_and_schedule_is_frozen(tmp_path: Path):
    bddl = tmp_path / "task.bddl"
    init = tmp_path / "task.init"
    bddl.write_text("task", encoding="utf-8")
    init.write_bytes(b"states")
    preview = build_evaluation_preview(
        trials=2, init_state_indices=[0, 1], base_seed=0,
        seed_count=1, schedule_seed=2,
    )
    payload = {
        "schema_version": 1,
        "evaluation_id": "eval-1",
        "result_path": str(tmp_path / "evaluation.json"),
        "task_snapshot": {
            "task_id": "LEVEL1::task", "level": "LEVEL1", "task_name": "task",
            "prompt": "do task", "bddl_path": str(bddl), "init_path": str(init),
        },
        "policy_snapshot": {
            "policy_id": "base", "label": "Base", "base_checkpoint": "repo/model",
            "stats_key": "libero_object", "manifest": None, "action_head": None,
            "proprio_projector": None, "training_step": None,
            "compatibility_sha256": None,
        },
        "config": {
            "trials": 2, "max_steps": 300, "open_loop_steps": 8,
            "realtime": True, "init_state_indices": [0, 1], "base_seed": 0,
            "seed_count": 1, "schedule_seed": 2, "control_hz": 20,
            "success_streak": 5, "consecutive_error_limit": 3,
            "disabled_policy_cameras": [],
        },
        "schedule": preview["schedule"],
        "schedule_sha256": preview["schedule_sha256"],
    }
    path = tmp_path / "effective.yaml"
    path.write_text(yaml.safe_dump(payload), encoding="utf-8")
    assert load_effective_config(path)["evaluation_id"] == "eval-1"
    payload["config"]["trials"] = 1
    path.write_text(yaml.safe_dump(payload), encoding="utf-8")
    with pytest.raises(ValueError, match="Frozen schedule"):
        load_effective_config(path)


def test_setup_failure_persists_terminal_manifest(tmp_path: Path, monkeypatch):
    schedule = build_schedule(
        trials=1, init_state_indices=[0], base_seed=0,
        seed_count=1, schedule_seed=0,
    )
    result_path = tmp_path / "evaluation.json"
    effective = {
        "evaluation_id": "eval-setup-failure",
        "result_path": str(result_path),
        "task_snapshot": {
            "task_id": "LEVEL1::task", "level": "LEVEL1", "task_name": "task",
            "prompt": "do task", "bddl_path": str(tmp_path / "task.bddl"),
            "init_path": str(tmp_path / "task.init"),
        },
        "policy_snapshot": {
            "policy_id": "base", "label": "Base", "base_checkpoint": "model",
            "stats_key": "stats", "manifest": None, "action_head": None,
            "proprio_projector": None, "training_step": None,
            "compatibility_sha256": None,
        },
        "config": {
            "trials": 1, "max_steps": 1, "open_loop_steps": 1,
            "realtime": False, "init_state_indices": [0], "base_seed": 0,
            "seed_count": 1, "schedule_seed": 0, "control_hz": 20,
            "success_streak": 5, "consecutive_error_limit": 3,
            "disabled_policy_cameras": [],
        },
        "schedule": schedule,
        "schedule_sha256": "x" * 64,
    }
    monkeypatch.setitem(
        sys.modules,
        "eval_pickplace_direct",
        SimpleNamespace(
            load_config=lambda _path: (_ for _ in ()).throw(RuntimeError("setup boom"))
        ),
    )
    result = run_evaluation(effective)
    persisted = yaml.safe_load(result_path.read_text(encoding="utf-8"))
    assert result["status"] == "FAILED"
    assert persisted["status"] == "FAILED"
    assert "setup boom" in persisted["error"]
