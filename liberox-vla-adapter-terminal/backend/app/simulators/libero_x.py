"""LIBERO-X / robosuite adapter. It never persists data."""

from __future__ import annotations

from pathlib import Path
from types import SimpleNamespace
from typing import Any

import cv2
import numpy as np

import eval_pickplace_direct as direct


OPERATOR_PREVIEW_CAMERAS = (
    ("agentview", "Agent view"),
    ("robot0_eye_in_hand", "Wrist view"),
    ("oblique_minus_45", "-45 deg view"),
    ("oblique_plus_45", "+45 deg view"),
)


def _look_at_quaternion(position: np.ndarray, target: np.ndarray) -> np.ndarray:
    """Return a MuJoCo camera quaternion whose local -Z axis looks at target."""
    import mujoco

    forward = np.asarray(target, dtype=np.float64) - np.asarray(position, dtype=np.float64)
    norm = float(np.linalg.norm(forward))
    if norm <= 1e-12:
        raise ValueError("Camera position and target must differ")
    forward /= norm
    world_up = np.array([0.0, 0.0, 1.0], dtype=np.float64)
    right = np.cross(forward, world_up)
    right_norm = float(np.linalg.norm(right))
    if right_norm <= 1e-12:
        raise ValueError("Camera forward vector must not be parallel to world up")
    right /= right_norm
    up = np.cross(right, forward)
    rotation = np.column_stack((right, up, -forward))
    quaternion = np.empty(4, dtype=np.float64)
    mujoco.mju_mat2Quat(quaternion, rotation.reshape(-1))
    return quaternion


def _oblique_camera_pose(
    reference_position: np.ndarray,
    reference_quaternion: np.ndarray,
    angle_degrees: float,
) -> tuple[np.ndarray, np.ndarray]:
    """Place a fixed camera at a signed horizontal angle around its look target."""
    import mujoco

    position = np.asarray(reference_position, dtype=np.float64)
    quaternion = np.asarray(reference_quaternion, dtype=np.float64)
    rotation = np.empty(9, dtype=np.float64)
    mujoco.mju_quat2Mat(rotation, quaternion)
    forward = -rotation.reshape(3, 3)[:, 2]
    target = position + 1.5 * forward
    # LIBERO's named side camera points slightly off the table centre. The
    # operator pair must be symmetric, so use the scene's Y=0 centre line.
    target[1] = 0.0
    horizontal_radius = float(np.linalg.norm(position[:2] - target[:2]))
    camera_height = float(position[2] - target[2])
    radians = np.deg2rad(angle_degrees)
    oblique_position = target + np.array(
        [
            horizontal_radius * np.cos(radians),
            horizontal_radius * np.sin(radians),
            camera_height,
        ],
        dtype=np.float64,
    )
    return oblique_position, _look_at_quaternion(oblique_position, target)


def compose_operator_preview(
    frames: list[np.ndarray],
    labels: tuple[str, ...] | None = None,
) -> np.ndarray:
    """Compose four equally-sized RGB views into one labelled 2x2 image."""
    if len(frames) != 4:
        raise ValueError(f"Operator preview requires four frames, got {len(frames)}")
    normalized = [np.asarray(frame, dtype=np.uint8).copy() for frame in frames]
    shape = normalized[0].shape
    if len(shape) != 3 or shape[2] != 3 or any(frame.shape != shape for frame in normalized):
        raise ValueError("Operator preview frames must have equal HxWx3 shapes")
    names = labels or tuple(label for _, label in OPERATOR_PREVIEW_CAMERAS)
    if len(names) != 4:
        raise ValueError("Operator preview requires four labels")
    for frame, label in zip(normalized, names, strict=True):
        scale = max(0.45, min(frame.shape[0], frame.shape[1]) / 640.0)
        thickness = max(1, round(scale * 2))
        (text_width, text_height), baseline = cv2.getTextSize(
            label, cv2.FONT_HERSHEY_SIMPLEX, scale, thickness
        )
        cv2.rectangle(
            frame,
            (10, 10),
            (22 + text_width, 20 + text_height + baseline),
            (0, 0, 0),
            thickness=-1,
        )
        cv2.putText(
            frame,
            label,
            (16, 16 + text_height),
            cv2.FONT_HERSHEY_SIMPLEX,
            scale,
            (255, 255, 255),
            thickness,
            cv2.LINE_AA,
        )
    return np.concatenate(
        (np.concatenate(normalized[:2], axis=1), np.concatenate(normalized[2:], axis=1)),
        axis=0,
    )


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
    def _render_oblique_side(
        env: Any,
        width: int,
        height: int,
        angle_degrees: float,
    ) -> np.ndarray:
        """Render an operator camera at a signed angle around the workspace."""
        wrapped_env = getattr(env, "env", env)
        sim = getattr(wrapped_env, "sim", None)
        if sim is None:
            raise AttributeError("LIBERO environment does not expose a MuJoCo sim renderer")
        camera_id = sim.model.camera_name2id("sideview")
        original_position = np.asarray(sim.model.cam_pos[camera_id], dtype=np.float64).copy()
        original_quaternion = np.asarray(sim.model.cam_quat[camera_id], dtype=np.float64).copy()

        oblique_position, oblique_quaternion = _oblique_camera_pose(
            original_position,
            original_quaternion,
            angle_degrees,
        )

        try:
            sim.model.cam_pos[camera_id] = oblique_position
            sim.model.cam_quat[camera_id] = oblique_quaternion
            sim.forward()
            return direct.render_high_resolution_camera_frame(
                env, "sideview", width, height
            )
        finally:
            sim.model.cam_pos[camera_id] = original_position
            sim.model.cam_quat[camera_id] = original_quaternion
            sim.forward()

    def render_operator_preview(self, env: Any, width: int, height: int) -> np.ndarray:
        """Render transient operator-only views; no frame is added to a recorder."""
        frames = [
            self.render(env, "agentview", width, height),
            self.render(env, "robot0_eye_in_hand", width, height),
            self._render_oblique_side(env, width, height, -45.0),
            self._render_oblique_side(env, width, height, 45.0),
        ]
        return compose_operator_preview(frames)

    @staticmethod
    def close(env: Any) -> None:
        direct.close_env(env)
