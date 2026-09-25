"""Metrics-only batch evaluation for LIBERO-X policies.

This module deliberately keeps schedule construction and aggregation free of
MuJoCo / Torch imports so the FastAPI process can use them for previews.  Heavy
runtime dependencies are imported only inside :func:`run_evaluation`.
"""

from __future__ import annotations

from collections import Counter, deque
from dataclasses import replace
from datetime import datetime, timezone
import hashlib
import json
import math
import os
from pathlib import Path
import random
import re
import statistics
import time
from typing import Any, Callable, Sequence

import numpy as np
import yaml


SCHEMA_VERSION = 1
CONFIG_KEYS = frozenset(
    {
        "trials",
        "max_steps",
        "open_loop_steps",
        "realtime",
        "init_state_indices",
        "base_seed",
        "seed_count",
        "schedule_seed",
        "control_hz",
        "success_streak",
        "consecutive_error_limit",
        "disabled_policy_cameras",
    }
)
TASK_KEYS = frozenset(
    {"task_id", "level", "task_name", "prompt", "bddl_path", "init_path"}
)
POLICY_KEYS = frozenset(
    {
        "policy_id",
        "label",
        "base_checkpoint",
        "stats_key",
        "manifest",
        "action_head",
        "proprio_projector",
        "training_step",
        "compatibility_sha256",
    }
)
ROOT_KEYS = frozenset(
    {
        "schema_version",
        "evaluation_id",
        "result_path",
        "task_snapshot",
        "policy_snapshot",
        "config",
        "schedule",
        "schedule_sha256",
    }
)


class _UniqueKeyLoader(yaml.SafeLoader):
    pass


def _unique_mapping(
    loader: _UniqueKeyLoader, node: yaml.MappingNode, deep: bool = False
) -> dict[Any, Any]:
    result: dict[Any, Any] = {}
    for key_node, value_node in node.value:
        key = loader.construct_object(key_node, deep=deep)
        if key in result:
            raise ValueError(f"Duplicate evaluation configuration key: {key!r}")
        result[key] = loader.construct_object(value_node, deep=deep)
    return result


_UniqueKeyLoader.add_constructor(
    yaml.resolver.BaseResolver.DEFAULT_MAPPING_TAG, _unique_mapping
)


def utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def _stable_hash(value: Any) -> str:
    return hashlib.sha256(
        json.dumps(value, sort_keys=True, separators=(",", ":")).encode("utf-8")
    ).hexdigest()


def _strict_int(value: Any, name: str, minimum: int | None = None) -> int:
    if type(value) is not int:
        raise TypeError(f"{name} must be an integer")
    if minimum is not None and value < minimum:
        raise ValueError(f"{name} must be >= {minimum}")
    return value


def _strict_string(value: Any, name: str) -> str:
    if not isinstance(value, str) or not value.strip():
        raise TypeError(f"{name} must be a non-empty string")
    return value


def build_schedule(
    *,
    trials: int,
    init_state_indices: Sequence[int],
    base_seed: int,
    seed_count: int | None,
    schedule_seed: int,
) -> list[dict[str, int]]:
    """Build a reproducible schedule balanced over states, seeds and pairs.

    Complete Cartesian-product rounds give every pair the same count.  The
    remainder is a balanced bipartite degree sequence, so state, seed and pair
    counts each differ by at most one.
    """

    trials = _strict_int(trials, "trials", 1)
    if trials > 1000:
        raise ValueError("trials must be <= 1000")
    base_seed = _strict_int(base_seed, "base_seed", 0)
    schedule_seed = _strict_int(schedule_seed, "schedule_seed", 0)
    if base_seed > 2_147_483_647 or schedule_seed > 2_147_483_647:
        raise ValueError("base_seed and schedule_seed must be <= 2147483647")
    if not isinstance(init_state_indices, Sequence) or isinstance(
        init_state_indices, (str, bytes)
    ):
        raise TypeError("init_state_indices must be a sequence of integers")
    states = list(init_state_indices)
    if not states:
        raise ValueError("init_state_indices must not be empty")
    if any(type(value) is not int or value < 0 for value in states):
        raise ValueError("init_state_indices must contain non-negative integers")
    if len(states) != len(set(states)):
        raise ValueError("init_state_indices must not contain duplicates")
    count = math.ceil(trials / len(states)) if seed_count is None else seed_count
    count = _strict_int(count, "seed_count", 1)
    if count > 1000:
        raise ValueError("seed_count must be <= 1000")
    if base_seed + count - 1 > 2_147_483_647:
        raise ValueError("base_seed + seed_count exceeds 2147483647")
    seeds = [base_seed + offset for offset in range(count)]

    rng = random.Random(schedule_seed)
    state_order = states.copy()
    seed_order = seeds.copy()
    rng.shuffle(state_order)
    rng.shuffle(seed_order)
    pair_count = len(states) * len(seeds)
    full_rounds, remainder = divmod(trials, pair_count)
    pairs: list[tuple[int, int]] = [
        pair
        for _ in range(full_rounds)
        for pair in ((state, seed) for state in state_order for seed in seed_order)
    ]

    # Balanced target degrees for the partial Cartesian round.  Havel-Hakimi
    # realizes these balanced bipartite degrees without repeating a pair.
    row_degrees = [
        remainder // len(states) + (index < remainder % len(states))
        for index in range(len(states))
    ]
    column_degrees = [
        remainder // len(seeds) + (index < remainder % len(seeds))
        for index in range(len(seeds))
    ]
    rows = sorted(range(len(states)), key=lambda index: (-row_degrees[index], index))
    for row in rows:
        degree = row_degrees[row]
        columns = sorted(
            range(len(seeds)), key=lambda index: (-column_degrees[index], index)
        )[:degree]
        if any(column_degrees[column] <= 0 for column in columns):
            raise RuntimeError("Unable to construct a balanced evaluation schedule")
        for column in columns:
            pairs.append((state_order[row], seed_order[column]))
            column_degrees[column] -= 1
    if any(column_degrees):
        raise RuntimeError("Balanced schedule construction left unassigned seeds")

    rng.shuffle(pairs)
    return [
        {"trial_index": index, "init_state_index": state, "seed": seed}
        for index, (state, seed) in enumerate(pairs)
    ]


def schedule_distribution(schedule: Sequence[dict[str, int]]) -> dict[str, Any]:
    states = Counter(int(item["init_state_index"]) for item in schedule)
    seeds = Counter(int(item["seed"]) for item in schedule)
    combinations = Counter(
        (int(item["init_state_index"]), int(item["seed"])) for item in schedule
    )
    return {
        "init_state_counts": {str(key): states[key] for key in sorted(states)},
        "seed_counts": {str(key): seeds[key] for key in sorted(seeds)},
        "combination_counts": {
            f"{state}:{seed}": combinations[(state, seed)]
            for state, seed in sorted(combinations)
        },
    }


def build_evaluation_preview(
    *,
    trials: int,
    init_state_indices: Sequence[int],
    base_seed: int,
    seed_count: int | None,
    schedule_seed: int,
    control_hz: int = 20,
    max_steps: int = 300,
) -> dict[str, Any]:
    control_hz = _strict_int(control_hz, "control_hz", 1)
    max_steps = _strict_int(max_steps, "max_steps", 1)
    schedule = build_schedule(
        trials=trials,
        init_state_indices=init_state_indices,
        base_seed=base_seed,
        seed_count=seed_count,
        schedule_seed=schedule_seed,
    )
    distribution = schedule_distribution(schedule)
    effective_seed_count = (
        math.ceil(trials / len(init_state_indices))
        if seed_count is None else seed_count
    )
    seed_pool = [base_seed + offset for offset in range(effective_seed_count)]
    for state in init_state_indices:
        distribution["init_state_counts"].setdefault(str(state), 0)
    for seed in seed_pool:
        distribution["seed_counts"].setdefault(str(seed), 0)
    for state in init_state_indices:
        for seed in seed_pool:
            distribution["combination_counts"].setdefault(f"{state}:{seed}", 0)
    for key in tuple(distribution):
        distribution[key] = dict(
            sorted(distribution[key].items(), key=lambda item: tuple(
                int(value) for value in item[0].split(":")
            ))
        )
    return {
        "schedule": schedule,
        "schedule_sha256": _stable_hash(schedule),
        "distribution": distribution,
        **distribution,
        "schedule_preview": schedule[: min(20, len(schedule))],
        "estimated_simulation_seconds": len(schedule) * max_steps / control_hz,
    }


class ConsecutiveSuccess:
    """Latch success after the configured number of consecutive done values."""

    def __init__(self, required: int = 5):
        self.required = _strict_int(required, "success_streak", 1)
        self.current = 0
        self.maximum = 0
        self.first_confirmed_step: int | None = None

    @property
    def success(self) -> bool:
        return self.first_confirmed_step is not None

    def observe(self, done: bool, step: int) -> bool:
        if done:
            self.current += 1
            self.maximum = max(self.maximum, self.current)
            if self.current >= self.required and self.first_confirmed_step is None:
                self.first_confirmed_step = int(step)
        else:
            self.current = 0
        return self.success


def wilson_interval(successes: int, total: int, z: float = 1.959963984540054) -> list[float]:
    if total <= 0:
        return [0.0, 0.0]
    proportion = successes / total
    denominator = 1.0 + z * z / total
    center = (proportion + z * z / (2.0 * total)) / denominator
    margin = (
        z
        * math.sqrt(
            proportion * (1.0 - proportion) / total + z * z / (4.0 * total * total)
        )
        / denominator
    )
    return [max(0.0, center - margin), min(1.0, center + margin)]


def _group_summary(items: Sequence[dict[str, Any]]) -> dict[str, Any]:
    attempted = len(items)
    successes = sum(
        bool(item.get("success")) and not bool(item.get("error")) for item in items
    )
    errors = sum(bool(item.get("error")) for item in items)
    return {
        "trials": attempted,
        "attempted": attempted,
        "successes": successes,
        "failures": attempted - successes - errors,
        "errors": errors,
        "success_rate": successes / attempted if attempted else 0.0,
        "wilson_95": wilson_interval(successes, attempted),
    }


def _numeric_summary(values: Sequence[float]) -> dict[str, float | None]:
    clean = sorted(float(value) for value in values if math.isfinite(float(value)))
    if not clean:
        return {"count": 0, "mean": None, "p50": None, "p95": None, "max": None}

    def percentile(fraction: float) -> float:
        if len(clean) == 1:
            return clean[0]
        position = fraction * (len(clean) - 1)
        lower = math.floor(position)
        upper = math.ceil(position)
        weight = position - lower
        return clean[lower] * (1.0 - weight) + clean[upper] * weight

    return {
        "count": len(clean),
        "mean": statistics.fmean(clean),
        "p50": percentile(0.5),
        "p95": percentile(0.95),
        "max": clean[-1],
    }


def _apply_close_error(trial: dict[str, Any], error: Exception) -> bool:
    """Attach environment cleanup failure; return whether it is a new error."""
    message = f"Environment close failed: {type(error).__name__}: {error}"
    if trial.get("error"):
        trial["error"] = f"{trial['error']}; {message}"
        return False
    trial["success"] = False
    trial["error"] = message
    return True


def aggregate_trials(
    trials: Sequence[dict[str, Any]], *, scheduled_trials: int
) -> dict[str, Any]:
    scheduled_trials = _strict_int(scheduled_trials, "scheduled_trials", 1)
    items = list(trials)
    overall = _group_summary(items)

    def groups(key: Callable[[dict[str, Any]], str]) -> dict[str, Any]:
        grouped: dict[str, list[dict[str, Any]]] = {}
        for item in items:
            grouped.setdefault(key(item), []).append(item)
        return {name: _group_summary(grouped[name]) for name in sorted(grouped)}

    overall.update(
        {
            "total_trials": scheduled_trials,
            "attempted_trials": len(items),
            "completed_trials": len(items) - overall["errors"],
            "scheduled_trials": scheduled_trials,
            "completion_coverage": len(items) / scheduled_trials,
            "completion_rate": len(items) / scheduled_trials,
            "by_init_state": groups(lambda item: str(item["init_state_index"])),
            "by_seed": groups(lambda item: str(item["seed"])),
            "by_combination": groups(
                lambda item: f"{item['init_state_index']}:{item['seed']}"
            ),
            "first_success_step": _numeric_summary(
                [
                    float(item["first_success_step"])
                    for item in items
                    if (
                        item.get("success")
                        and not item.get("error")
                        and item.get("first_success_step") is not None
                    )
                ]
            ),
            "policy_queries": _numeric_summary(
                [float(item["policy_queries"]) for item in items if not item.get("error")]
            ),
            "inference_latency_ms": _numeric_summary(
                [
                    float(item["inference_latency_ms"])
                    for item in items
                    if item.get("inference_latency_ms") is not None
                ]
            ),
            "control_hz": _numeric_summary(
                [
                    float(item["measured_control_hz"])
                    for item in items
                    if item.get("measured_control_hz") is not None
                ]
            ),
            "wall_seconds": _numeric_summary(
                [float(item["elapsed_seconds"]) for item in items]
            ),
        }
    )
    overall["wilson_lower"], overall["wilson_upper"] = overall["wilson_95"]
    overall["first_success_step_mean"] = overall["first_success_step"]["mean"]
    overall["policy_queries_mean"] = overall["policy_queries"]["mean"]
    overall["inference_latency_ms_mean"] = overall["inference_latency_ms"]["mean"]
    overall["measured_control_hz_mean"] = overall["control_hz"]["mean"]
    overall["elapsed_seconds_mean"] = overall["wall_seconds"]["mean"]
    return overall


def _validate_exact_mapping(value: Any, keys: frozenset[str], name: str) -> dict[str, Any]:
    if not isinstance(value, dict):
        raise TypeError(f"{name} must be a mapping")
    missing = sorted(keys - set(value))
    unknown = sorted(set(value) - keys)
    if missing or unknown:
        raise ValueError(f"{name} keys mismatch; missing={missing}, unknown={unknown}")
    return value


def load_effective_config(path: Path) -> dict[str, Any]:
    """Load the backend-generated YAML and reject all undeclared input."""
    path = path.expanduser().resolve()
    raw = yaml.load(path.read_text(encoding="utf-8"), Loader=_UniqueKeyLoader)
    root = _validate_exact_mapping(raw, ROOT_KEYS, "evaluation config")
    if root["schema_version"] != SCHEMA_VERSION:
        raise ValueError("Unsupported evaluation config schema_version")
    evaluation_id = _strict_string(root["evaluation_id"], "evaluation_id")
    if Path(evaluation_id).name != evaluation_id or ".." in evaluation_id:
        raise ValueError("evaluation_id is unsafe")
    task = _validate_exact_mapping(root["task_snapshot"], TASK_KEYS, "task_snapshot")
    policy_keys = POLICY_KEYS | ({"backbone"} if "backbone" in root["policy_snapshot"] else set())
    policy = _validate_exact_mapping(root["policy_snapshot"], policy_keys, "policy_snapshot")
    config = _validate_exact_mapping(root["config"], CONFIG_KEYS, "config")
    _strict_string(task["task_id"], "task_snapshot.task_id")
    _strict_string(task["level"], "task_snapshot.level")
    _strict_string(task["task_name"], "task_snapshot.task_name")
    _strict_string(task["prompt"], "task_snapshot.prompt")
    policy_id = _strict_string(policy["policy_id"], "policy_snapshot.policy_id")
    _strict_string(policy["label"], "policy_snapshot.label")
    _strict_string(policy["base_checkpoint"], "policy_snapshot.base_checkpoint")
    _strict_string(policy["stats_key"], "policy_snapshot.stats_key")
    result_path = Path(_strict_string(root["result_path"], "result_path")).expanduser()
    if not result_path.is_absolute() or result_path.name != "evaluation.json":
        raise ValueError("result_path must be an absolute evaluation.json path")
    root["result_path"] = str(result_path.resolve())
    for key in ("bddl_path", "init_path"):
        source = Path(_strict_string(task[key], f"task_snapshot.{key}")).expanduser()
        if not source.is_absolute() or not source.is_file():
            raise FileNotFoundError(f"task_snapshot.{key} is not an existing absolute file")
        task[key] = str(source.resolve())
    if not isinstance(config["realtime"], bool):
        raise TypeError("config.realtime must be true or false")
    for key, minimum in (
        ("trials", 1),
        ("max_steps", 1),
        ("open_loop_steps", 1),
        ("base_seed", 0),
        ("schedule_seed", 0),
        ("control_hz", 1),
        ("success_streak", 1),
        ("consecutive_error_limit", 1),
    ):
        _strict_int(config[key], f"config.{key}", minimum)
    if config["trials"] > 1000:
        raise ValueError("config.trials must be <= 1000")
    if config["max_steps"] > 10000:
        raise ValueError("config.max_steps must be <= 10000")
    if config["open_loop_steps"] > 8:
        raise ValueError("config.open_loop_steps must be <= 8")
    if config["control_hz"] != 20:
        raise ValueError("config.control_hz must remain 20 Hz")
    if config["success_streak"] != 5:
        raise ValueError("config.success_streak must remain 5")
    if config["consecutive_error_limit"] != 3:
        raise ValueError("config.consecutive_error_limit must remain 3")
    seed_count = config["seed_count"]
    if seed_count is not None:
        _strict_int(seed_count, "config.seed_count", 1)
    cameras = config["disabled_policy_cameras"]
    if not isinstance(cameras, list) or any(not isinstance(item, str) for item in cameras):
        raise TypeError("config.disabled_policy_cameras must be a list of strings")
    if len(cameras) != len(set(cameras)):
        raise ValueError("config.disabled_policy_cameras must not contain duplicates")
    allowed_cameras = {"agentview", "robot0_eye_in_hand"}
    if set(cameras) - allowed_cameras or len(cameras) >= len(allowed_cameras):
        raise ValueError("config.disabled_policy_cameras must leave one VLA camera enabled")
    if policy_id == "base":
        if policy.get("backbone") is not None:
            raise ValueError("Base policy cannot contain a backbone overlay")
        for key in (
            "manifest", "action_head", "proprio_projector", "training_step",
            "compatibility_sha256",
        ):
            if policy[key] is not None:
                raise ValueError(f"Base policy snapshot must set {key} to null")
    else:
        for key in ("manifest", "action_head", "proprio_projector") + (("backbone",) if "backbone" in policy else ()):
            component = Path(
                _strict_string(policy[key], f"policy_snapshot.{key}")
            ).expanduser()
            if not component.is_absolute() or not component.is_file() or component.is_symlink():
                raise FileNotFoundError(
                    f"policy_snapshot.{key} is not a safe existing absolute file"
                )
            policy[key] = str(component.resolve())
        _strict_int(policy["training_step"], "policy_snapshot.training_step", 1)
        compatibility = _strict_string(
            policy["compatibility_sha256"], "policy_snapshot.compatibility_sha256"
        )
        if re.fullmatch(r"[0-9a-f]{64}", compatibility) is None:
            raise ValueError("policy_snapshot.compatibility_sha256 must be SHA256")
    indices = config["init_state_indices"]
    expected = build_schedule(
        trials=config["trials"],
        init_state_indices=indices,
        base_seed=config["base_seed"],
        seed_count=seed_count,
        schedule_seed=config["schedule_seed"],
    )
    if root["schedule"] != expected:
        raise ValueError("Frozen schedule does not match the effective config")
    if root["schedule_sha256"] != _stable_hash(expected):
        raise ValueError("schedule_sha256 mismatch")
    return root


def _atomic_write_json(path: Path, value: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.tmp")
    with temporary.open("w", encoding="utf-8") as stream:
        json.dump(value, stream, ensure_ascii=False, indent=2)
        stream.write("\n")
        stream.flush()
        os.fsync(stream.fileno())
    os.replace(temporary, path)


def _result_base(effective: dict[str, Any]) -> dict[str, Any]:
    path = Path(effective["result_path"])
    previous: dict[str, Any] = {}
    if path.is_file():
        try:
            loaded = json.loads(path.read_text(encoding="utf-8"))
            if isinstance(loaded, dict) and loaded.get("id") == effective["evaluation_id"]:
                previous = loaded
        except (OSError, json.JSONDecodeError):
            pass
    return {
        "schema_version": SCHEMA_VERSION,
        "id": effective["evaluation_id"],
        "status": "STARTING",
        "created_at": previous.get("created_at", utc_now()),
        "started_at": None,
        "completed_at": None,
        "task_snapshot": effective["task_snapshot"],
        "policy_snapshot": effective["policy_snapshot"],
        "config": effective["config"],
        "schedule": effective["schedule"],
        "schedule_sha256": effective["schedule_sha256"],
        "success_rule": {
            "done_consecutive_steps": effective["config"]["success_streak"],
            "success_latched": True,
            "run_full_horizon": True,
            "errors_in_denominator": True,
        },
        "trials": [],
        "aggregate": aggregate_trials([], scheduled_trials=effective["config"]["trials"]),
        "timing": {
            "model_load_seconds": None,
            "wall_time_seconds": 0.0,
            "simulated_time_seconds": 0.0,
        },
        "error": None,
    }


def _episode(
    *,
    simulator: Any,
    provider: Any,
    env: Any,
    initial_state: Any,
    prompt: str,
    trial_index: int,
    init_state_index: int,
    seed: int,
    max_steps: int,
    open_loop_steps: int,
    control_hz: int,
    realtime: bool,
    success_streak: int,
    disabled_policy_cameras: tuple[str, ...],
    stop_requested: Callable[[], bool],
    progress: dict[str, Any] | None = None,
) -> dict[str, Any]:
    import eval_pickplace_direct as direct

    started = time.monotonic()
    observation = simulator.restore(env, initial_state)
    observation = simulator.observations(env, observation)
    limiter = direct.RealTimeControlLimiter(float(control_hz), realtime)
    queue: deque[np.ndarray] = deque()
    tracker = ConsecutiveSuccess(success_streak)
    query_count = 0
    query_times: list[float] = []
    step_times: list[float] = []
    final_done = False
    deadline_misses = 0
    progress = progress if progress is not None else {}
    progress.update(
        steps=0,
        first_success_step=None,
        max_done_streak=0,
        final_done=False,
        policy_queries=0,
        query_times=[],
        step_times=[],
        deadline_misses=0,
    )
    for step_index in range(max_steps):
        if stop_requested():
            raise InterruptedError("Evaluation stopped by user")
        if not queue:
            observation = simulator.observations(env, observation)
            query_started = time.monotonic()
            chunk = provider.predict(observation, prompt, disabled_policy_cameras)
            query_times.append(time.monotonic() - query_started)
            if stop_requested():
                raise InterruptedError("Evaluation stopped after policy inference")
            validated = []
            for index, value in enumerate(chunk):
                action = np.asarray(value, dtype=np.float32)
                if action.shape != (7,) or not np.isfinite(action).all():
                    raise ValueError(f"Policy action {index} is invalid")
                validated.append(action)
            if not validated:
                raise RuntimeError("Policy returned an empty action chunk")
            queue.extend(validated[:open_loop_steps])
            query_count += 1
            progress["policy_queries"] = query_count
            progress["query_times"] = list(query_times)
        step_started = limiter.wait_before_step()
        env_action = np.asarray(provider.process_action(queue.popleft()), dtype=np.float32)
        if env_action.shape != (7,) or not np.isfinite(env_action).all():
            raise ValueError("Environment action is invalid")
        observation, _reward, done, _info = env.step(env_action.tolist())
        final_done = bool(done)
        tracker.observe(final_done, step_index + 1)
        step_times.append(step_started)
        if len(step_times) >= 2 and step_times[-1] - step_times[-2] > 1.1 / control_hz:
            deadline_misses += 1
        progress.update(
            steps=step_index + 1,
            first_success_step=tracker.first_confirmed_step,
            max_done_streak=tracker.maximum,
            final_done=final_done,
            step_times=list(step_times),
            deadline_misses=deadline_misses,
        )
    timing = direct.summarize_control_timing(step_times, control_hz)
    query_ms = [value * 1000.0 for value in query_times]
    return {
        "trial_index": trial_index,
        "init_state_index": init_state_index,
        "seed": seed,
        "success": tracker.success,
        "error": None,
        "steps": max_steps,
        "first_success_step": tracker.first_confirmed_step,
        "max_done_streak": tracker.maximum,
        "final_done": final_done,
        "policy_queries": query_count,
        "inference_latency_ms": statistics.fmean(query_ms) if query_ms else None,
        "inference_latency_p95_ms": _numeric_summary(query_ms)["p95"],
        "measured_control_hz": timing["measured_control_hz"],
        "deadline_misses": deadline_misses,
        "elapsed_seconds": time.monotonic() - started,
    }


def run_evaluation(
    effective: dict[str, Any], *, stop_requested: Callable[[], bool] | None = None
) -> dict[str, Any]:
    """Run all episodes and persist only the configured ``evaluation.json``."""
    stop_requested = stop_requested or (lambda: False)
    result_path = Path(effective["result_path"])
    record = _result_base(effective)
    _atomic_write_json(result_path, record)
    config = effective["config"]
    task = effective["task_snapshot"]
    policy = effective["policy_snapshot"]
    started = time.monotonic()
    consecutive_errors = 0
    provider: Any = None
    try:
        if stop_requested():
            raise InterruptedError("Evaluation stopped before setup")
        # Heavy imports and every setup operation are covered by this terminal
        # state guard. A missing dependency or invalid snapshot can never leave
        # the durable manifest stuck in STARTING.
        import eval_pickplace_direct as direct
        from backend.app.policies.catalog import PolicyCatalog
        from backend.app.policies.vla_adapter import VLAAdapterPolicyProvider
        from backend.app.simulators.libero_x import LiberoXSimulator

        workspace_root = Path(__file__).resolve().parents[4]
        base = direct.load_config(workspace_root / "configs" / "config.yaml")
        eval_config = replace(
            base,
            checkpoint=policy["base_checkpoint"],
            stats_key=policy["stats_key"],
            level=task["level"],
            task_name=task["task_name"],
            trials=1,
            max_steps=config["max_steps"],
            seed=config["base_seed"],
            control_hz=config["control_hz"],
            realtime_control=config["realtime"],
            disabled_policy_cameras=tuple(config["disabled_policy_cameras"]),
            open_loop_steps=config["open_loop_steps"],
            headless=True,
            no_video=True,
            save_trajectory=False,
            save_observation_images=False,
            trajectory_plot=False,
        )
        direct.apply_runtime_environment(eval_config)
        direct.add_repo_paths(eval_config.vla_root, eval_config.liberox_root)
        runtime = direct.load_runtime()
        simulator = LiberoXSimulator(runtime)
        init_states = direct.load_initial_states(runtime, Path(task["init_path"]))
        if max(config["init_state_indices"]) >= len(init_states):
            raise ValueError("Configured init_state_index exceeds the task state count")
        manifest = policy.get("manifest")
        registry = (
            Path(manifest).resolve().parent.parent
            if manifest
            else result_path.parent / ".empty-policy-registry"
        )
        catalog = PolicyCatalog(
            registry, policy["base_checkpoint"], policy["stats_key"]
        )
        entry = catalog.entry(policy["policy_id"])
        if manifest:
            expected_components = (
                Path(manifest).resolve(),
                Path(policy["action_head"]).resolve(),
                Path(policy["proprio_projector"]).resolve(),
                policy["training_step"],
                policy["compatibility_sha256"],
                Path(policy["backbone"]).resolve() if policy.get("backbone") else None,
            )
            actual_components = (
                entry.manifest,
                entry.action_head,
                entry.proprio_projector,
                entry.training_step,
                entry.compatibility_sha256,
                entry.backbone,
            )
            if actual_components != expected_components:
                raise ValueError("Policy snapshot does not match the validated registry entry")
        provider = VLAAdapterPolicyProvider(runtime, eval_config, catalog)
        record["status"] = "RUNNING"
        record["started_at"] = utc_now()
        load_started = time.monotonic()
        if stop_requested():
            raise InterruptedError("Evaluation stopped before policy loading")
        print(
            json.dumps(
                {"level": "INFO", "message": "加载评测策略模型", "evaluation_id": record["id"]},
                ensure_ascii=False,
            ),
            flush=True,
        )
        provider.load(config["open_loop_steps"], policy["policy_id"])
        record["timing"]["model_load_seconds"] = time.monotonic() - load_started
        print(
            json.dumps(
                {
                    "level": "DONE",
                    "message": (
                        "评测策略模型加载完成 · "
                        f"{record['timing']['model_load_seconds']:.3f}s"
                    ),
                    "model_load_seconds": record["timing"]["model_load_seconds"],
                },
                ensure_ascii=False,
            ),
            flush=True,
        )
        if stop_requested():
            raise InterruptedError("Evaluation stopped after policy loading")
        first_schedule = effective["schedule"][0]
        simulator.prewarm(
            Path(task["bddl_path"]),
            init_states[first_schedule["init_state_index"]],
            replace(eval_config, seed=int(first_schedule["seed"])),
        )
        print(
            json.dumps(
                {"level": "DONE", "message": "MuJoCo 控制器预热完成"},
                ensure_ascii=False,
            ),
            flush=True,
        )
        _atomic_write_json(result_path, record)
        for item in effective["schedule"]:
            if stop_requested():
                record["status"] = "CANCELED"
                break
            env = None
            episode_started = time.monotonic()
            episode_progress: dict[str, Any] = {}
            close_error: Exception | None = None
            print(
                json.dumps(
                    {
                        "level": "INFO",
                        "message": (
                            f"开始测试回合 {item['trial_index'] + 1}/{config['trials']} · "
                            f"init {item['init_state_index']} · seed {item['seed']}"
                        ),
                        **item,
                    },
                    ensure_ascii=False,
                ),
                flush=True,
            )
            try:
                seed = int(item["seed"])
                set_seed = getattr(runtime, "set_seed_everywhere", None)
                if callable(set_seed):
                    set_seed(seed)
                episode_config = replace(eval_config, seed=seed)
                env = simulator.create(
                    Path(task["bddl_path"]),
                    episode_config,
                    max_steps=config["max_steps"],
                    seed=seed,
                )
                trial = _episode(
                    simulator=simulator,
                    provider=provider,
                    env=env,
                    initial_state=init_states[item["init_state_index"]],
                    prompt=task["prompt"],
                    trial_index=item["trial_index"],
                    init_state_index=item["init_state_index"],
                    seed=seed,
                    max_steps=config["max_steps"],
                    open_loop_steps=config["open_loop_steps"],
                    control_hz=config["control_hz"],
                    realtime=config["realtime"],
                    success_streak=config["success_streak"],
                    disabled_policy_cameras=tuple(config["disabled_policy_cameras"]),
                    stop_requested=stop_requested,
                    progress=episode_progress,
                )
                consecutive_errors = 0
            except InterruptedError:
                record["status"] = "CANCELED"
                break
            except Exception as exc:
                consecutive_errors += 1
                query_ms = [
                    float(value) * 1000.0
                    for value in episode_progress.get("query_times", [])
                ]
                timing = direct.summarize_control_timing(
                    episode_progress.get("step_times", []), config["control_hz"]
                )
                trial = {
                    "trial_index": item["trial_index"],
                    "init_state_index": item["init_state_index"],
                    "seed": item["seed"],
                    "success": False,
                    "error": f"{type(exc).__name__}: {exc}",
                    "steps": int(episode_progress.get("steps", 0)),
                    "first_success_step": episode_progress.get("first_success_step"),
                    "max_done_streak": int(
                        episode_progress.get("max_done_streak", 0)
                    ),
                    "final_done": bool(episode_progress.get("final_done", False)),
                    "policy_queries": int(episode_progress.get("policy_queries", 0)),
                    "inference_latency_ms": (
                        statistics.fmean(query_ms) if query_ms else None
                    ),
                    "inference_latency_p95_ms": _numeric_summary(query_ms)["p95"],
                    "measured_control_hz": timing["measured_control_hz"],
                    "deadline_misses": int(
                        episode_progress.get("deadline_misses", 0)
                    ),
                    "elapsed_seconds": time.monotonic() - episode_started,
                }
            finally:
                if env is not None:
                    try:
                        simulator.close(env)
                    except Exception as exc:
                        close_error = exc
            if close_error is not None:
                if _apply_close_error(trial, close_error):
                    consecutive_errors += 1
            record["trials"].append(trial)
            record["aggregate"] = aggregate_trials(
                record["trials"], scheduled_trials=config["trials"]
            )
            record["timing"]["wall_time_seconds"] = time.monotonic() - started
            record["timing"]["simulated_time_seconds"] = sum(
                int(value["steps"]) for value in record["trials"]
            ) / config["control_hz"]
            _atomic_write_json(result_path, record)
            print(
                json.dumps(
                    {
                        "event": "evaluation_progress",
                        "level": "INFO",
                        "message": (
                            f"完成测试回合 {len(record['trials'])}/{config['trials']} · "
                            f"{'错误' if trial['error'] else '成功' if trial['success'] else '失败'} · "
                            f"成功率 {record['aggregate']['success_rate'] * 100:.1f}% · "
                            f"{trial['elapsed_seconds']:.2f}s"
                        ),
                        "trial_index": trial["trial_index"],
                        "init_state_index": trial["init_state_index"],
                        "seed": trial["seed"],
                        "attempted_trials": len(record["trials"]),
                        "total_trials": config["trials"],
                        "successes": record["aggregate"]["successes"],
                        "success_rate": record["aggregate"]["success_rate"],
                        "progress_percent": 100.0 * len(record["trials"]) / config["trials"],
                        "elapsed_seconds": time.monotonic() - started,
                        "measured_control_hz": trial["measured_control_hz"],
                        "trial_success": trial["success"],
                        "trial_error": trial["error"],
                        "trial_elapsed_seconds": trial["elapsed_seconds"],
                    },
                    ensure_ascii=False,
                ),
                flush=True,
            )
            if consecutive_errors >= config["consecutive_error_limit"]:
                record["status"] = "FAILED"
                record["error"] = (
                    f"Stopped after {consecutive_errors} consecutive episode errors"
                )
                break
        else:
            record["status"] = "COMPLETED"
    except BaseException as exc:
        if isinstance(exc, (KeyboardInterrupt, InterruptedError)) or stop_requested():
            record["status"] = "CANCELED"
        else:
            record["status"] = "FAILED"
            record["error"] = f"{type(exc).__name__}: {exc}"
    finally:
        if provider is not None:
            try:
                provider.unload()
            except Exception as exc:
                if record["status"] == "COMPLETED":
                    record["status"] = "FAILED"
                    record["error"] = f"Policy unload failed: {type(exc).__name__}: {exc}"
        record["completed_at"] = utc_now()
        record["timing"]["wall_time_seconds"] = time.monotonic() - started
        record["aggregate"] = aggregate_trials(
            record["trials"], scheduled_trials=config["trials"]
        )
        _atomic_write_json(result_path, record)
        print(
            json.dumps(
                {
                    "level": "DONE" if record["status"] == "COMPLETED" else "ERROR"
                    if record["status"] == "FAILED" else "WARN",
                    "message": (
                        f"测试结束 · {record['status']} · "
                        f"{record['aggregate']['successes']}/{record['aggregate']['attempted']} 成功 · "
                        f"成功率 {record['aggregate']['success_rate'] * 100:.1f}%"
                    ),
                    "status": record["status"],
                    "attempted_trials": record["aggregate"]["attempted"],
                    "successes": record["aggregate"]["successes"],
                    "success_rate": record["aggregate"]["success_rate"],
                    "wall_time_seconds": record["timing"]["wall_time_seconds"],
                },
                ensure_ascii=False,
            ),
            flush=True,
        )
    return record


def run_from_path(path: Path, *, stop_requested: Callable[[], bool] | None = None) -> dict[str, Any]:
    return run_evaluation(load_effective_config(path), stop_requested=stop_requested)
