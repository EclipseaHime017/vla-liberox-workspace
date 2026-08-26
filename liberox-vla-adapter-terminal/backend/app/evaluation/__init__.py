"""Task evaluation adapters and metrics-only batch evaluation primitives."""

from .batch import (
    ConsecutiveSuccess,
    aggregate_trials,
    build_evaluation_preview,
    build_schedule,
    load_effective_config,
    run_evaluation,
)

__all__ = [
    "ConsecutiveSuccess",
    "aggregate_trials",
    "build_evaluation_preview",
    "build_schedule",
    "load_effective_config",
    "run_evaluation",
]
