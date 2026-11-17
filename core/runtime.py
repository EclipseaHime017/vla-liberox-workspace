from __future__ import annotations

from collections.abc import Callable
from typing import TypeVar


T = TypeVar("T")


def run_cuda_stage(stage: str, operation: Callable[[], T]) -> T:
    """Add a stable stage name to CUDA OOM failures without falling back to CPU."""
    try:
        return operation()
    except RuntimeError as exc:
        message = str(exc)
        if "out of memory" in message.lower() and "cuda" in message.lower():
            raise RuntimeError(
                f"{stage} failed: CUDA out of memory. This pipeline does not fall back "
                "to CPU; reduce only the stage-specific YAML load or use a larger GPU. "
                f"Original error: {message}"
            ) from exc
        raise
