"""Observe benchmark goal predicates without changing the episode success rule."""
from __future__ import annotations

import copy
import hashlib
import json


def _logical(expression):
    return (isinstance(expression, (list, tuple)) and bool(expression)
            and isinstance(expression[0], str) and expression[0].lower() in {"and", "or", "not"})


def _validate(expression):
    if _logical(expression):
        if len(expression) < 2:
            raise ValueError("Empty logical goal")
        if expression[0].lower() == "not" and len(expression) != 2:
            raise ValueError("Negation needs exactly one child")
        for child in expression[1:]:
            _validate(child)
    elif (not isinstance(expression, (list, tuple)) or len(expression) not in (2, 3)
          or not all(isinstance(token, str) and token for token in expression)):
        raise ValueError("Unsupported goal expression")


def _required(expression):
    if _logical(expression) and expression[0].lower() == "and":
        return [item for child in expression[1:] for item in _required(child)]
    return [expression]


def _identity(expression):
    return hashlib.sha256(json.dumps(expression, separators=(",", ":")).encode()).hexdigest()


class GoalConditionTracker:
    def __init__(self, env):
        self.reader = None
        self.result = {"schema_version": 1, "status": "unavailable", "error": None,
                       "last_observed_step": None, "conditions": []}
        seen = set()
        while env is not None and id(env) not in seen:
            seen.add(id(env))
            if isinstance(getattr(env, "parsed_problem", None), dict) and callable(getattr(env, "_eval_predicate", None)):
                self.reader = env._eval_predicate
                break
            env = getattr(env, "env", None)
        if self.reader is None:
            self.result["error"] = "Environment has no readable LIBERO goal predicates"
            return
        try:
            goal = copy.deepcopy(env.parsed_problem["goal_state"])
            if not goal:
                raise ValueError("Empty benchmark goal")
            # Flat BDDL lists are conjunctions; an OR stays one required condition.
            expression = goal if _logical(goal) or isinstance(goal[0], str) else ["and", *goal]
            _validate(expression)
            conditions = list({_identity(item): item for item in _required(expression)}.values())
            self.result.update(status="ok", goal_expression=expression, goal_sha256=_identity(expression))
            self.result["conditions"] = [{"id": _identity(item), "expression": item,
                "ever_satisfied": False, "first_satisfied_step": None,
                "initial_satisfied": None, "final_satisfied": None} for item in conditions]
        except Exception as error:
            self.result.update(status="unsupported", error=f"{type(error).__name__}: {error}")

    def _evaluate(self, expression):
        if _logical(expression):
            values = [self._evaluate(child) for child in expression[1:]]
            if expression[0].lower() == "not":
                return not values[0]
            return all(values) if expression[0].lower() == "and" else any(values)
        return bool(self.reader(expression))

    def observe(self, step):
        if self.result["status"] != "ok":
            return
        try:
            # Read all values before publishing a step, avoiding partial observations.
            values = [self._evaluate(item["expression"]) for item in self.result["conditions"]]
            for item, value in zip(self.result["conditions"], values):
                if step == 0:
                    item["initial_satisfied"] = value
                if value and not item["ever_satisfied"]:
                    item["first_satisfied_step"] = step
                item["ever_satisfied"] |= value
                item["final_satisfied"] = value
            self.result["last_observed_step"] = step
        except Exception as error:
            self.result.update(status="error", error=f"{type(error).__name__}: {error}")

    def snapshot(self):
        return copy.deepcopy(self.result)


def aggregate_goal_conditions(trials):
    failures = [item for item in trials if not item.get("success") and not item.get("error")]
    grouped = {}
    unavailable = 0
    for trial in failures:
        report = trial.get("goal_conditions") or {}
        if report.get("status") != "ok" or report.get("last_observed_step") != trial.get("steps"):
            unavailable += 1
            continue
        group = grouped.setdefault(report["goal_sha256"], {"failures": 0, "conditions": {}})
        group["failures"] += 1
        for condition in report["conditions"]:
            row = group["conditions"].setdefault(condition["id"], {"expression": condition["expression"],
                "never_satisfied": 0, "not_satisfied_at_end": 0, "satisfied_then_lost": 0})
            row["never_satisfied"] += not condition["ever_satisfied"]
            row["not_satisfied_at_end"] += not condition["final_satisfied"]
            row["satisfied_then_lost"] += condition["ever_satisfied"] and not condition["final_satisfied"]
    return {"completed_failures": len(failures), "unavailable_failures": unavailable,
            "by_goal": grouped,
            "semantics": "Raw goal predicates at observation 0 and after each action; not streak-confirmed or causal stages."}
