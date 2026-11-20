"""Dense progress reward interpolated between annotated keyframes."""
from __future__ import annotations

from typing import Any, Sequence

import numpy as np

from rewards.stage.keyframes import checked_exponent, confirmed_success_step, stage_anchors


def stage_annotation_context(annotation: dict[str, Any], *, done: Sequence[bool],
                             success_consecutive_steps: int,
                             exponent: float = 2.0) -> dict[str, Any]:
    """Attach dataset-local evaluation settings without changing human labels."""
    if len(done) != annotation["action_count"]:
        raise ValueError("Stage evaluation action count does not match the keyframe labels")
    return {
        **annotation,
        "success_consecutive_steps": success_consecutive_steps,
        "success_step": confirmed_success_step(done, success_consecutive_steps),
        "exponent": checked_exponent(exponent),
    }


def stage_scores(annotation: dict[str, Any], exponent: float | None = None) -> np.ndarray:
    power = checked_exponent(annotation.get("exponent", 2.0) if exponent is None else exponent)
    anchors = stage_anchors(annotation)
    scores = np.full(int(annotation["action_count"]) + 1, anchors[-1]["score"], dtype=np.float64)
    scores[0] = -1.0
    for left, right in zip(anchors, anchors[1:]):
        start, end = left["step"], right["step"]
        x = np.arange(end - start + 1, dtype=np.float64) / (end - start)
        scores[start:end + 1] = left["score"] + (right["score"] - left["score"]) * x ** power
    return scores.astype(np.float32)


def stage_chunk_reward(scores: np.ndarray, start: int, length: int, gamma: float) -> float:
    """Discounted per-step Stage scores over [start, start + length)."""
    if length < 1 or start < 0 or start + length >= len(scores):
        raise ValueError("Invalid stage reward chunk bounds")
    return float(np.dot(np.power(gamma, np.arange(length)), scores[start + 1:start + length + 1]))
