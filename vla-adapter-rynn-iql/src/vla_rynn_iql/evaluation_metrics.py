"""Batch schedule and metrics ported from main c62dcc9 (no platform imports)."""
from collections import Counter
import math
import random
import statistics
from typing import Any, Callable, Sequence

def _strict_int(value: Any, name: str, minimum: int | None = None) -> int:
    if type(value) is not int:
        raise TypeError(f"{name} must be an integer")
    if minimum is not None and value < minimum:
        raise ValueError(f"{name} must be >= {minimum}")
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


def init_state_summary(items: Sequence[dict[str, Any]]) -> dict[str, Any]:
    """Per-init-state view of the success rate, without an interval.

    Trials sharing an init state differ only in seed-placed fixtures and policy
    sampling, so their outcomes can be strongly correlated, and the overall
    Wilson interval (which assumes independent trials) can be too narrow. With
    the few init states a batch test uses, no simple clustered interval has
    reliable coverage, so only the unweighted mean of per-state rates is given
    next to ``by_init_state``.
    """
    grouped: dict[int, list[bool]] = {}
    for item in items:
        grouped.setdefault(int(item["init_state_index"]), []).append(
            bool(item.get("success")) and not bool(item.get("error")))
    rates = [sum(values) / len(values) for values in grouped.values()]
    return {"states": len(grouped),
            "state_mean_success_rate": statistics.fmean(rates) if rates else None}


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
            "init_state_summary": init_state_summary(items),
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
