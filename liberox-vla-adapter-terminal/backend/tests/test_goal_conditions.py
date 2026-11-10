from types import SimpleNamespace
import copy
import random
import importlib.util
import os
from pathlib import Path

import numpy as np
import pytest

from backend.app.evaluation.goal_conditions import GoalConditionTracker, aggregate_goal_conditions
from backend.app.evaluation.batch import _episode, aggregate_trials
from test_batch_evaluation import _FakeEnvironment, _FakeProvider, _FakeSimulator


class Predicates:
    def __init__(self, goal):
        self.parsed_problem = {"goal_state": goal}
        self.values = {}
        self.calls = []

    def _eval_predicate(self, condition):
        self.calls.append(tuple(condition))
        value = self.values[condition[-1]]
        if isinstance(value, Exception):
            raise value
        return value


def test_initial_ever_first_and_final_and_readonly_rng():
    env = Predicates([["open", "drawer"], ["in", "bowl", "drawer_region"]])
    initial = copy.deepcopy(env.parsed_problem)
    env.values = {"drawer": True, "drawer_region": False}
    tracker = GoalConditionTracker(SimpleNamespace(env=env))
    python_rng = random.getstate()
    numpy_rng = np.random.get_state()
    tracker.observe(0)
    env.values.update(drawer=False, drawer_region=True)
    tracker.observe(1)
    env.values["drawer_region"] = False
    tracker.observe(2)
    first, second = tracker.snapshot()["conditions"]
    assert (first["initial_satisfied"], first["first_satisfied_step"], first["ever_satisfied"], first["final_satisfied"]) == (True, 0, True, False)
    assert (second["initial_satisfied"], second["first_satisfied_step"], second["ever_satisfied"], second["final_satisfied"]) == (False, 1, True, False)
    assert initial == env.parsed_problem
    assert random.getstate() == python_rng
    assert np.array_equal(np.random.get_state()[1], numpy_rng[1])


def test_or_not_remain_required_expressions_not_separate_required_leaves():
    goal = ["and", ["or", ["in", "bowl", "left"], ["in", "bowl", "right"]], ["not", ["open", "drawer"]]]
    env = Predicates(goal)
    env.values = {"left": False, "right": True, "drawer": False}
    tracker = GoalConditionTracker(env)
    tracker.observe(0)
    result = tracker.snapshot()
    assert result["goal_expression"] == goal
    assert len(result["conditions"]) == 2
    assert all(item["final_satisfied"] for item in result["conditions"])
    assert result["conditions"][0]["expression"][0] == "or"
    assert result["conditions"][1]["expression"][0] == "not"


def test_unavailable_unsupported_and_reader_errors_are_not_false_labels():
    assert GoalConditionTracker(object()).snapshot()["status"] == "unavailable"
    assert GoalConditionTracker(Predicates(["not", ["open", "a"], ["open", "b"]])).snapshot()["status"] == "unsupported"
    env = Predicates([["open", "drawer"]])
    env.values = {"drawer": True}
    tracker = GoalConditionTracker(env)
    tracker.observe(0)
    env.values["drawer"] = RuntimeError("broken predicate")
    tracker.observe(1)
    result = tracker.snapshot()
    assert result["status"] == "error" and "broken predicate" in result["error"]
    assert result["last_observed_step"] == 0
    assert result["conditions"][0]["final_satisfied"] is True
    summary = aggregate_goal_conditions([{"success": False, "error": None, "steps": 1, "goal_conditions": result}])
    assert summary["unavailable_failures"] == 1 and summary["by_goal"] == {}


def test_real_episode_entry_records_goals_without_changing_streak_or_horizon():
    env = _FakeEnvironment([True] * 5 + [False] * 3)
    env.parsed_problem = {"goal_state": [["in", "bowl", "drawer"], ["close", "drawer"]]}
    env._eval_predicate = lambda condition: (env.steps >= 2 if condition[0] == "in" else 3 <= env.steps <= 6)
    report = _episode(simulator=_FakeSimulator(), provider=_FakeProvider(), env=env,
        initial_state=np.zeros(1), prompt="task", trial_index=0, init_state_index=2,
        seed=9, max_steps=8, open_loop_steps=8, control_hz=20, realtime=False,
        success_streak=5, disabled_policy_cameras=(), stop_requested=lambda: False)
    assert report["success"] is True and report["first_success_step"] == 5
    assert report["steps"] == 8 and report["final_done"] is False
    first, second = report["goal_conditions"]["conditions"]
    assert first["first_satisfied_step"] == 2 and first["final_satisfied"]
    assert second["first_satisfied_step"] == 3 and not second["final_satisfied"]
    assert report["goal_conditions"]["last_observed_step"] == 8
    # Failure distributions are descriptive and only include completed failures.
    failure = {**report, "success": False}
    aggregate = aggregate_trials([failure, report, {**failure, "error": "failed"}], scheduled_trials=3)
    goals = aggregate["goal_conditions"]
    assert goals["completed_failures"] == 1 and goals["unavailable_failures"] == 0
    group, = goals["by_goal"].values()
    assert group["conditions"][second["id"]]["satisfied_then_lost"] == 1
    assert group["conditions"][first["id"]]["not_satisfied_at_end"] == 0


def test_episode_predicate_error_does_not_fail_policy_or_change_success():
    env = _FakeEnvironment([False] * 8)
    env.parsed_problem = {"goal_state": [["open", "drawer"]]}
    def broken(_):
        raise RuntimeError("diagnostic read failed")
    env._eval_predicate = broken
    report = _episode(simulator=_FakeSimulator(), provider=_FakeProvider(), env=env,
        initial_state=np.zeros(1), prompt="task", trial_index=0, init_state_index=0,
        seed=7, max_steps=8, open_loop_steps=8, control_hz=20, realtime=False,
        success_streak=5, disabled_policy_cameras=(), stop_requested=lambda: False)
    assert report["steps"] == 8 and report["error"] is None and not report["success"]
    assert report["goal_conditions"]["status"] == "error"


@pytest.mark.skipif(not os.environ.get("LIBERO_X_ROOT"), reason="Optional local benchmark checkout")
def test_middle_task_uses_actual_bddl_goal_structure():
    root = Path(os.environ["LIBERO_X_ROOT"])
    parser_path = root / "libero/libero/envs/bddl_utils.py"
    spec = importlib.util.spec_from_file_location("local_bddl_parser", parser_path)
    parser = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(parser)
    path = root / "libero/libero_x/bddl/LEVEL2/EXTENSION_KITCHEN_SCENE5_place_the_yellow_bowl_in_the_bottom_drawer_of_the_white_cabinet_and_close_the_drawer.bddl"
    parsed = parser.robosuite_parse_problem(str(path))
    env = Predicates(parsed["goal_state"])
    env.values = {"white_cabinet_1_bottom_region": True}
    tracker = GoalConditionTracker(SimpleNamespace(env=env))
    tracker.observe(0)
    assert tracker.snapshot()["status"] == "ok"
    conditions = tracker.snapshot()["conditions"]
    assert {item["expression"][0].lower() for item in conditions} == {"in", "close"}
    assert len(conditions) == 2
