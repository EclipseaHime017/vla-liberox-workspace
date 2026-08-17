"""LIBERO-X / robosuite adapter. It never persists data."""

from __future__ import annotations

from pathlib import Path
from types import SimpleNamespace
from typing import Any

import eval_pickplace_direct as direct


class LiberoXSimulator:
    def __init__(self, runtime: SimpleNamespace):
        self.runtime = runtime

    def prewarm(self, bddl: Path, state: Any, config: direct.EvalConfig) -> None:
        direct.prewarm_simulation_control(self.runtime, bddl, state, config)

    def create(
        self,
        bddl: Path,
        config: direct.EvalConfig,
        *,
        max_steps: int,
        seed: int,
    ) -> Any:
        return direct.make_env(
            self.runtime,
            bddl,
            config.env_resolution,
            max_steps + 1,
            config.control_hz,
            seed,
            config.video_camera,
            config.video_width,
            config.video_height,
            True,
        )

    @staticmethod
    def restore(env: Any, state: Any) -> dict[str, Any]:
        return direct.restore_state(env, state)

    @staticmethod
    def observations(env: Any, observation: dict[str, Any]) -> dict[str, Any]:
        result = direct.capture_camera_observations(env, observation)
        direct.validate_observation(result)
        return result

    @staticmethod
    def render(env: Any, camera: str, width: int, height: int):
        return direct.render_high_resolution_camera_frame(env, camera, width, height)

    @staticmethod
    def close(env: Any) -> None:
        direct.close_env(env)
