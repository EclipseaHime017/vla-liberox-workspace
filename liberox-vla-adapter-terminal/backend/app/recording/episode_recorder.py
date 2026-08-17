"""Recorder factory and publishing boundary for one episode."""

from __future__ import annotations

from pathlib import Path
from typing import Any

from trajectory_utils import (
    TrajectoryRecorder,
    plot_action_comparison,
    save_trajectory_bundle,
)


class EpisodeRecorderFactory:
    @staticmethod
    def original(control_hz: float) -> TrajectoryRecorder:
        return TrajectoryRecorder(control_hz=control_hz, capture_images=False)

    @staticmethod
    def branch(source: dict[str, Any], resume_step: int, control_hz: float) -> TrajectoryRecorder:
        return TrajectoryRecorder.from_prefix(source, resume_step, control_hz)

    @staticmethod
    def save(recorder: TrajectoryRecorder, prefix: Path, metadata: dict[str, Any], **kwargs: Any):
        return save_trajectory_bundle(recorder, prefix, metadata, **kwargs)

    @staticmethod
    def compare(*args: Any, **kwargs: Any):
        return plot_action_comparison(*args, **kwargs)
