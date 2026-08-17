#!/usr/bin/env python3
"""Trajectory recording, persistence, plotting, and exact-state replay helpers."""

from __future__ import annotations

import csv
import json
import math
import os
import tempfile
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import numpy as np


TRAJECTORY_SCHEMA_VERSION = 2
SUPPORTED_TRAJECTORY_SCHEMA_VERSIONS = frozenset({1, 2})
ACTION_COMPONENTS = (
    ("dx", 0, "Translation X", "Normalized command [-]"),
    ("dy", 1, "Translation Y", "Normalized command [-]"),
    ("dz", 2, "Translation Z", "Normalized command [-]"),
    ("drx", 3, "Rotation X", "Normalized command [-]"),
    ("dry", 4, "Rotation Y", "Normalized command [-]"),
    ("drz", 5, "Rotation Z", "Normalized command [-]"),
    ("gripper", 6, "Gripper", "Normalized command [-]"),
)


def quaternion_to_axis_angle(quaternion: np.ndarray) -> np.ndarray:
    """Convert an (x, y, z, w) quaternion to a three-dimensional axis-angle vector."""
    quat = np.asarray(quaternion, dtype=np.float64).copy()
    if quat.shape != (4,):
        raise ValueError(f"Expected quaternion shape (4,), got {quat.shape}")
    quat[3] = np.clip(quat[3], -1.0, 1.0)
    denominator = math.sqrt(max(0.0, 1.0 - quat[3] * quat[3]))
    if math.isclose(denominator, 0.0):
        return np.zeros(3, dtype=np.float64)
    return quat[:3] * (2.0 * math.acos(quat[3]) / denominator)


def _copy_vector(value: Any, shape: tuple[int, ...], name: str) -> np.ndarray:
    array = np.asarray(value).copy()
    if array.shape != shape:
        raise ValueError(f"Expected {name} shape {shape}, got {array.shape}")
    if not np.isfinite(array).all():
        raise ValueError(f"{name} contains NaN or Inf")
    return array


def _sim_state(env: Any) -> np.ndarray:
    if hasattr(env, "get_sim_state"):
        state = env.get_sim_state()
    else:
        state = env.sim.get_state().flatten()
    state = np.asarray(state).copy()
    if state.ndim != 1 or not np.isfinite(state).all():
        raise ValueError(f"Invalid flattened MuJoCo state with shape {state.shape}")
    return state


@dataclass
class TrajectoryRecorder:
    """Records N actions together with the N+1 states that bound those transitions."""

    control_hz: float = 20.0
    capture_images: bool = True
    sim_states: list[np.ndarray] = field(default_factory=list)
    eef_positions: list[np.ndarray] = field(default_factory=list)
    eef_quaternions: list[np.ndarray] = field(default_factory=list)
    eef_axis_angles: list[np.ndarray] = field(default_factory=list)
    gripper_qpos: list[np.ndarray] = field(default_factory=list)
    agentview_images: list[np.ndarray] = field(default_factory=list)
    wrist_images: list[np.ndarray] = field(default_factory=list)
    raw_actions: list[np.ndarray] = field(default_factory=list)
    env_actions: list[np.ndarray] = field(default_factory=list)
    rewards: list[float] = field(default_factory=list)
    dones: list[bool] = field(default_factory=list)
    action_sources: list[str] = field(default_factory=list)
    inference_query_steps: list[int] = field(default_factory=list)
    inference_action_chunks: list[np.ndarray] = field(default_factory=list)

    def _append_state(self, env: Any, observation: dict[str, Any]) -> None:
        position = _copy_vector(observation["robot0_eef_pos"], (3,), "robot0_eef_pos")
        quaternion = _copy_vector(observation["robot0_eef_quat"], (4,), "robot0_eef_quat")
        gripper = _copy_vector(observation["robot0_gripper_qpos"], (2,), "robot0_gripper_qpos")
        self.sim_states.append(_sim_state(env))
        self.eef_positions.append(position.astype(np.float64, copy=False))
        self.eef_quaternions.append(quaternion.astype(np.float64, copy=False))
        self.eef_axis_angles.append(quaternion_to_axis_angle(quaternion))
        self.gripper_qpos.append(gripper.astype(np.float64, copy=False))
        if self.capture_images:
            self.agentview_images.append(np.asarray(observation["agentview_image"], dtype=np.uint8).copy())
            self.wrist_images.append(
                np.asarray(observation["robot0_eye_in_hand_image"], dtype=np.uint8).copy()
            )

    def record_initial(self, env: Any, observation: dict[str, Any]) -> None:
        if self.sim_states:
            raise RuntimeError("Initial trajectory state has already been recorded")
        self._append_state(env, observation)

    def record_transition(
        self,
        env: Any,
        observation: dict[str, Any],
        raw_action: np.ndarray,
        env_action: np.ndarray,
        reward: float,
        done: bool,
        action_source: str,
    ) -> None:
        if not self.sim_states:
            raise RuntimeError("record_initial() must be called before recording a transition")
        self.raw_actions.append(_copy_vector(raw_action, (7,), "raw_action").astype(np.float32))
        self.env_actions.append(_copy_vector(env_action, (7,), "env_action").astype(np.float32))
        self.rewards.append(float(reward))
        self.dones.append(bool(done))
        self.action_sources.append(str(action_source))
        self._append_state(env, observation)

    def record_inference(self, query_step: int, action_chunk: Any) -> None:
        """Record every action proposed by one VLA query, including actions not executed later."""
        if isinstance(query_step, bool) or not isinstance(query_step, int) or query_step < 0:
            raise ValueError("query_step must be a non-negative integer")
        chunk = np.asarray(action_chunk, dtype=np.float32)
        if chunk.ndim != 2 or chunk.shape[0] < 1 or chunk.shape[1] != 7:
            raise ValueError(f"Inference action chunk must have shape (N, 7), got {chunk.shape}")
        if not np.isfinite(chunk).all():
            raise ValueError("Inference action chunk contains NaN or Inf")
        self.inference_query_steps.append(query_step)
        self.inference_action_chunks.append(chunk.copy())

    @classmethod
    def from_prefix(
        cls,
        trajectory: dict[str, np.ndarray],
        resume_step: int,
        control_hz: float,
    ) -> "TrajectoryRecorder":
        """Copy source states [0, resume_step] and actions [0, resume_step) into a new branch."""
        recorder = cls(control_hz=control_hz, capture_images=False)
        state_slice = slice(0, resume_step + 1)
        action_slice = slice(0, resume_step)
        recorder.sim_states = [row.copy() for row in trajectory["sim_state"][state_slice]]
        recorder.eef_positions = [row.copy() for row in trajectory["eef_position"][state_slice]]
        recorder.eef_quaternions = [row.copy() for row in trajectory["eef_quaternion"][state_slice]]
        recorder.eef_axis_angles = [row.copy() for row in trajectory["eef_axis_angle"][state_slice]]
        recorder.gripper_qpos = [row.copy() for row in trajectory["gripper_qpos"][state_slice]]
        recorder.raw_actions = [row.copy() for row in trajectory["raw_action"][action_slice]]
        recorder.env_actions = [row.copy() for row in trajectory["env_action"][action_slice]]
        recorder.rewards = [float(value) for value in trajectory["reward"][action_slice]]
        recorder.dones = [bool(value) for value in trajectory["done"][action_slice]]
        recorder.action_sources = [str(value) for value in trajectory["action_source"][action_slice]]
        inference_keys = {
            "inference_query_step",
            "inference_chunk_offset",
            "inference_action",
        }
        if inference_keys <= set(trajectory):
            query_steps = trajectory["inference_query_step"]
            offsets = trajectory["inference_chunk_offset"]
            inference_actions = trajectory["inference_action"]
            for index, query_step in enumerate(query_steps):
                if int(query_step) >= resume_step:
                    continue
                start = int(offsets[index])
                end = int(offsets[index + 1])
                recorder.inference_query_steps.append(int(query_step))
                recorder.inference_action_chunks.append(inference_actions[start:end].copy())
        recorder._validate_lengths()
        return recorder

    @property
    def state_count(self) -> int:
        return len(self.sim_states)

    @property
    def action_count(self) -> int:
        return len(self.env_actions)

    def _validate_lengths(self) -> None:
        state_count = self.state_count
        action_count = self.action_count
        if state_count == 0:
            raise ValueError("Cannot persist an empty trajectory")
        state_series = (
            self.eef_positions,
            self.eef_quaternions,
            self.eef_axis_angles,
            self.gripper_qpos,
        )
        if any(len(series) != state_count for series in state_series):
            raise ValueError("Trajectory state arrays have inconsistent lengths")
        action_series = (self.raw_actions, self.rewards, self.dones, self.action_sources)
        if any(len(series) != action_count for series in action_series):
            raise ValueError("Trajectory action arrays have inconsistent lengths")
        if state_count != action_count + 1:
            raise ValueError(
                f"Trajectory must contain N+1 states for N actions, got {state_count} states and {action_count} actions"
            )
        if self.capture_images and (
            len(self.agentview_images) != state_count or len(self.wrist_images) != state_count
        ):
            raise ValueError("Trajectory image arrays have inconsistent lengths")
        if len(self.inference_query_steps) != len(self.inference_action_chunks):
            raise ValueError("Inference query steps and action chunks have inconsistent lengths")
        previous_query_step = -1
        for query_step, chunk in zip(self.inference_query_steps, self.inference_action_chunks):
            if not previous_query_step <= query_step <= action_count:
                raise ValueError(f"Invalid or unsorted inference query step: {query_step}")
            if chunk.ndim != 2 or chunk.shape[0] < 1 or chunk.shape[1] != 7:
                raise ValueError(f"Invalid inference action chunk shape: {chunk.shape}")
            previous_query_step = query_step

    def arrays(self) -> dict[str, np.ndarray]:
        self._validate_lengths()
        action_count = self.action_count
        inference_offsets = [0]
        for chunk in self.inference_action_chunks:
            inference_offsets.append(inference_offsets[-1] + len(chunk))
        inference_actions = (
            np.concatenate(self.inference_action_chunks, axis=0)
            if self.inference_action_chunks
            else np.empty((0, 7), dtype=np.float32)
        )
        return {
            "step": np.arange(self.state_count, dtype=np.int64),
            "time_seconds": np.arange(self.state_count, dtype=np.float64) / self.control_hz,
            "sim_state": np.stack(self.sim_states, axis=0),
            "eef_position": np.stack(self.eef_positions, axis=0),
            "eef_quaternion": np.stack(self.eef_quaternions, axis=0),
            "eef_axis_angle": np.stack(self.eef_axis_angles, axis=0),
            "gripper_qpos": np.stack(self.gripper_qpos, axis=0),
            "raw_action": np.stack(self.raw_actions, axis=0)
            if action_count
            else np.empty((0, 7), dtype=np.float32),
            "env_action": np.stack(self.env_actions, axis=0)
            if action_count
            else np.empty((0, 7), dtype=np.float32),
            "reward": np.asarray(self.rewards, dtype=np.float32),
            "done": np.asarray(self.dones, dtype=np.bool_),
            "action_source": np.asarray(self.action_sources, dtype=np.str_),
            "inference_query_step": np.asarray(self.inference_query_steps, dtype=np.int64),
            "inference_chunk_offset": np.asarray(inference_offsets, dtype=np.int64),
            "inference_action": inference_actions,
        }

    def observation_arrays(self) -> dict[str, np.ndarray] | None:
        self._validate_lengths()
        if not self.capture_images:
            return None
        return {
            "agentview_image": np.stack(self.agentview_images, axis=0),
            "wrist_image": np.stack(self.wrist_images, axis=0),
        }


def _validate_loaded_trajectory(trajectory: dict[str, np.ndarray]) -> None:
    required = {
        "step",
        "time_seconds",
        "sim_state",
        "eef_position",
        "eef_quaternion",
        "eef_axis_angle",
        "gripper_qpos",
        "raw_action",
        "env_action",
        "reward",
        "done",
        "action_source",
    }
    missing = sorted(required - set(trajectory))
    if missing:
        raise ValueError(f"Trajectory is missing arrays: {missing}")
    state_count = len(trajectory["sim_state"])
    action_count = len(trajectory["env_action"])
    if state_count != action_count + 1:
        raise ValueError(
            f"Invalid trajectory lengths: {state_count} states for {action_count} actions"
        )
    expected_state_shapes = {
        "eef_position": (state_count, 3),
        "eef_quaternion": (state_count, 4),
        "eef_axis_angle": (state_count, 3),
        "gripper_qpos": (state_count, 2),
    }
    for key, shape in expected_state_shapes.items():
        if trajectory[key].shape != shape:
            raise ValueError(f"Invalid {key} shape {trajectory[key].shape}; expected {shape}")
    for key in ("raw_action", "env_action"):
        if trajectory[key].shape != (action_count, 7):
            raise ValueError(
                f"Invalid {key} shape {trajectory[key].shape}; expected {(action_count, 7)}"
            )
    inference_keys = {
        "inference_query_step",
        "inference_chunk_offset",
        "inference_action",
    }
    present_inference_keys = inference_keys & set(trajectory)
    if present_inference_keys and present_inference_keys != inference_keys:
        raise ValueError("Trajectory contains an incomplete VLA inference trace")
    if present_inference_keys:
        query_steps = trajectory["inference_query_step"]
        offsets = trajectory["inference_chunk_offset"]
        inference_actions = trajectory["inference_action"]
        if offsets.shape != (len(query_steps) + 1,):
            raise ValueError("Invalid inference_chunk_offset length")
        if len(offsets) == 0 or int(offsets[0]) != 0 or int(offsets[-1]) != len(inference_actions):
            raise ValueError("Inference chunk offsets do not cover inference_action")
        if np.any(np.diff(offsets) <= 0):
            raise ValueError("Every inference query must contain at least one action")
        if inference_actions.shape != (int(offsets[-1]), 7):
            raise ValueError(f"Invalid inference_action shape: {inference_actions.shape}")
        if np.any(np.diff(query_steps) < 0):
            raise ValueError("Inference query steps must be sorted")


def load_trajectory(trajectory_path: Path) -> tuple[dict[str, np.ndarray], dict[str, Any]]:
    """Load and validate a trajectory core archive without enabling pickle."""
    trajectory_path = trajectory_path.expanduser().resolve()
    if not trajectory_path.is_file():
        raise FileNotFoundError(f"Source trajectory not found: {trajectory_path}")
    with np.load(trajectory_path, allow_pickle=False) as archive:
        if "metadata_json" not in archive.files:
            raise ValueError(f"Trajectory has no metadata_json: {trajectory_path}")
        metadata = json.loads(str(archive["metadata_json"].item()))
        trajectory = {key: archive[key].copy() for key in archive.files if key != "metadata_json"}
    if metadata.get("schema_version") not in SUPPORTED_TRAJECTORY_SCHEMA_VERSIONS:
        raise ValueError(
            f"Unsupported trajectory schema {metadata.get('schema_version')}; "
            f"supported={sorted(SUPPORTED_TRAJECTORY_SCHEMA_VERSIONS)}"
        )
    _validate_loaded_trajectory(trajectory)
    return trajectory, metadata


def _write_csv(csv_path: Path, trajectory: dict[str, np.ndarray]) -> None:
    fieldnames = [
        "step",
        "time_seconds",
        "eef_x",
        "eef_y",
        "eef_z",
        "axis_angle_x",
        "axis_angle_y",
        "axis_angle_z",
        "quat_x",
        "quat_y",
        "quat_z",
        "quat_w",
        "gripper_left",
        "gripper_right",
        "vla_action_dx",
        "vla_action_dy",
        "vla_action_dz",
        "vla_action_drx",
        "vla_action_dry",
        "vla_action_drz",
        "vla_action_gripper",
        "action_dx",
        "action_dy",
        "action_dz",
        "action_drx",
        "action_dry",
        "action_drz",
        "action_gripper",
        "action_source",
        "reward",
        "done",
    ]
    with csv_path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        action_count = len(trajectory["env_action"])
        for index in range(len(trajectory["sim_state"])):
            position = trajectory["eef_position"][index]
            axis_angle = trajectory["eef_axis_angle"][index]
            quaternion = trajectory["eef_quaternion"][index]
            gripper = trajectory["gripper_qpos"][index]
            row: dict[str, Any] = {
                "step": int(trajectory["step"][index]),
                "time_seconds": float(trajectory["time_seconds"][index]),
                "eef_x": float(position[0]),
                "eef_y": float(position[1]),
                "eef_z": float(position[2]),
                "axis_angle_x": float(axis_angle[0]),
                "axis_angle_y": float(axis_angle[1]),
                "axis_angle_z": float(axis_angle[2]),
                "quat_x": float(quaternion[0]),
                "quat_y": float(quaternion[1]),
                "quat_z": float(quaternion[2]),
                "quat_w": float(quaternion[3]),
                "gripper_left": float(gripper[0]),
                "gripper_right": float(gripper[1]),
            }
            if index < action_count:
                raw_action = trajectory["raw_action"][index]
                action = trajectory["env_action"][index]
                row.update(
                    {
                        "vla_action_dx": float(raw_action[0]),
                        "vla_action_dy": float(raw_action[1]),
                        "vla_action_dz": float(raw_action[2]),
                        "vla_action_drx": float(raw_action[3]),
                        "vla_action_dry": float(raw_action[4]),
                        "vla_action_drz": float(raw_action[5]),
                        "vla_action_gripper": float(raw_action[6]),
                        "action_dx": float(action[0]),
                        "action_dy": float(action[1]),
                        "action_dz": float(action[2]),
                        "action_drx": float(action[3]),
                        "action_dry": float(action[4]),
                        "action_drz": float(action[5]),
                        "action_gripper": float(action[6]),
                        "action_source": str(trajectory["action_source"][index]),
                        "reward": float(trajectory["reward"][index]),
                        "done": bool(trajectory["done"][index]),
                    }
                )
            writer.writerow(row)


def _write_inference_csv(csv_path: Path, trajectory: dict[str, np.ndarray], control_hz: float) -> None:
    fieldnames = [
        "query_index",
        "query_step",
        "query_time_seconds",
        "chunk_index",
        "target_step",
        "target_time_seconds",
        "vla_action_dx",
        "vla_action_dy",
        "vla_action_dz",
        "vla_action_drx",
        "vla_action_dry",
        "vla_action_drz",
        "vla_action_gripper",
    ]
    with csv_path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        offsets = trajectory["inference_chunk_offset"]
        for query_index, query_step_value in enumerate(trajectory["inference_query_step"]):
            query_step = int(query_step_value)
            start = int(offsets[query_index])
            end = int(offsets[query_index + 1])
            for chunk_index, action in enumerate(trajectory["inference_action"][start:end]):
                target_step = query_step + chunk_index
                writer.writerow(
                    {
                        "query_index": query_index,
                        "query_step": query_step,
                        "query_time_seconds": query_step / control_hz,
                        "chunk_index": chunk_index,
                        "target_step": target_step,
                        "target_time_seconds": target_step / control_hz,
                        "vla_action_dx": float(action[0]),
                        "vla_action_dy": float(action[1]),
                        "vla_action_dz": float(action[2]),
                        "vla_action_drx": float(action[3]),
                        "vla_action_dry": float(action[4]),
                        "vla_action_drz": float(action[5]),
                        "vla_action_gripper": float(action[6]),
                    }
                )


def plot_trajectory(
    trajectory: dict[str, np.ndarray],
    plot_path: Path,
    metadata: dict[str, Any],
    intervention_step: int | None = None,
) -> None:
    """Generate one PNG containing position, orientation, and both gripper joint traces."""
    cache_dir = Path(tempfile.gettempdir()) / "liberox-matplotlib-cache"
    cache_dir.mkdir(parents=True, exist_ok=True)
    os.environ.setdefault("MPLCONFIGDIR", str(cache_dir))
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    time_seconds = trajectory["time_seconds"]
    figure, axes = plt.subplots(3, 1, figsize=(12, 10), sharex=True, constrained_layout=True)
    for index, label in enumerate(("x", "y", "z")):
        axes[0].plot(time_seconds, trajectory["eef_position"][:, index], label=label)
    axes[0].set_ylabel("EEF position [m]")
    axes[0].legend(loc="upper right", ncol=3)
    axes[0].grid(alpha=0.3)

    for index, label in enumerate(("rx", "ry", "rz")):
        axes[1].plot(time_seconds, trajectory["eef_axis_angle"][:, index], label=label)
    axes[1].set_ylabel("Axis-angle [rad]")
    axes[1].legend(loc="upper right", ncol=3)
    axes[1].grid(alpha=0.3)

    axes[2].plot(time_seconds, trajectory["gripper_qpos"][:, 0], label="left finger")
    axes[2].plot(time_seconds, trajectory["gripper_qpos"][:, 1], label="right finger")
    axes[2].set_ylabel("Gripper qpos")
    axes[2].set_xlabel("Time [s]")
    axes[2].legend(loc="upper right", ncol=2)
    axes[2].grid(alpha=0.3)

    if intervention_step is not None:
        intervention_time = float(time_seconds[intervention_step])
        for axis in axes:
            axis.axvline(intervention_time, color="black", linestyle="--", alpha=0.7, label="intervention")

    title = str(metadata.get("task", "LIBERO-X trajectory"))
    control_mode = metadata.get("control_mode")
    if control_mode:
        title = f"{title} | branch={control_mode}"
    figure.suptitle(title)
    figure.savefig(plot_path, dpi=160)
    plt.close(figure)


def _add_step_and_time_axes(axis: Any, control_hz: float) -> None:
    axis.set_xlabel("Control step [frame]")
    time_axis = axis.secondary_xaxis(
        "top",
        functions=(lambda step: step / control_hz, lambda seconds: seconds * control_hz),
    )
    time_axis.set_xlabel("Simulation time [s]")


def plot_action_trajectories(
    trajectory: dict[str, np.ndarray],
    plot_prefix: Path,
    metadata: dict[str, Any],
    intervention_step: int | None = None,
) -> dict[str, str]:
    """Create seven continuous action plots with all inferred chunks."""
    cache_dir = Path(tempfile.gettempdir()) / "liberox-matplotlib-cache"
    cache_dir.mkdir(parents=True, exist_ok=True)
    os.environ.setdefault("MPLCONFIGDIR", str(cache_dir))
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    control_hz = float(metadata["control_hz"])
    action_steps = trajectory["step"][:-1].astype(np.float64)
    raw_actions = trajectory["raw_action"]
    inference_keys = {
        "inference_query_step",
        "inference_chunk_offset",
        "inference_action",
    }
    output_paths: dict[str, str] = {}
    for component_name, action_index, title, unit_label in ACTION_COMPONENTS:
        figure, axis = plt.subplots(figsize=(13, 5.5), constrained_layout=True)
        if inference_keys <= set(trajectory):
            offsets = trajectory["inference_chunk_offset"]
            for query_index, query_step_value in enumerate(trajectory["inference_query_step"]):
                query_step = int(query_step_value)
                start = int(offsets[query_index])
                end = int(offsets[query_index + 1])
                chunk = trajectory["inference_action"][start:end]
                chunk_steps = query_step + np.arange(len(chunk), dtype=np.float64)
                axis.plot(
                    chunk_steps,
                    chunk[:, action_index],
                    color="tab:orange",
                    alpha=0.16,
                    linewidth=0.9,
                )

        axis.plot(
            action_steps,
            raw_actions[:, action_index],
            color="tab:blue",
            linewidth=1.5,
            label=f"executed {component_name}",
        )
        axis.set_ylabel(f"{title} action\n{unit_label}")
        axis.grid(alpha=0.3)
        if intervention_step is not None:
            axis.axvline(
                intervention_step,
                color="black",
                linestyle="--",
                alpha=0.7,
                label="intervention",
            )
        axis.legend(loc="upper right")
        _add_step_and_time_axes(axis, control_hz)
        figure.suptitle(
            f"{metadata.get('task', 'LIBERO-X')} | VLA {component_name}\n"
            "blue: executed raw action; faint orange: complete proposed chunks"
        )
        plot_path = plot_prefix.with_name(f"{plot_prefix.name}_{component_name}.png")
        figure.savefig(plot_path, dpi=160)
        plt.close(figure)
        output_paths[f"trajectory_action_{component_name}_plot"] = str(plot_path)
    return output_paths


def plot_action_comparison(
    source_trajectory: dict[str, np.ndarray],
    branch_trajectory: dict[str, np.ndarray],
    resume_step: int,
    control_hz: float,
    plot_prefix: Path,
    branch_label: str = "re-inference",
) -> dict[str, str]:
    """Plot the complete original curve and overlay the branch only from resume_step."""
    source_actions = source_trajectory["raw_action"]
    branch_actions = branch_trajectory["raw_action"]
    end_step = min(len(source_actions), len(branch_actions))
    if not 0 <= resume_step < end_step:
        raise ValueError(
            f"Cannot compare actions at resume_step={resume_step}; overlapping action range ends at {end_step}"
        )

    cache_dir = Path(tempfile.gettempdir()) / "liberox-matplotlib-cache"
    cache_dir.mkdir(parents=True, exist_ok=True)
    os.environ.setdefault("MPLCONFIGDIR", str(cache_dir))
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    original_steps = np.arange(len(source_actions), dtype=np.float64)
    branch_steps = np.arange(resume_step, end_step, dtype=np.float64)
    output_paths: dict[str, str] = {}
    for component_name, action_index, title, unit_label in ACTION_COMPONENTS:
        figure, axis = plt.subplots(figsize=(13, 5.5), constrained_layout=True)
        axis.plot(
            original_steps,
            source_actions[:, action_index],
            linestyle="--",
            linewidth=1.2,
            label="original inference",
        )
        axis.plot(
            branch_steps,
            branch_actions[resume_step:end_step, action_index],
            linewidth=1.2,
            label=branch_label,
        )
        axis.axvline(resume_step, color="black", linestyle=":", alpha=0.8, label="resume")
        axis.set_ylabel(f"{title} action\n{unit_label}")
        axis.grid(alpha=0.3)
        axis.legend(loc="upper right")
        _add_step_and_time_axes(axis, control_hz)
        figure.suptitle(
            f"Original vs branch VLA {component_name} | resume step={resume_step} "
            f"({resume_step / control_hz:.3f} s)\n"
            f"original: frames 0-{len(source_actions) - 1}; branch: frames {resume_step}-{end_step - 1}"
        )
        plot_path = plot_prefix.with_name(f"{plot_prefix.name}_{component_name}.png")
        figure.savefig(plot_path, dpi=160)
        plt.close(figure)
        output_paths[f"trajectory_action_comparison_{component_name}"] = str(plot_path)
    return output_paths


def save_trajectory_bundle(
    recorder: TrajectoryRecorder,
    prefix: Path,
    metadata: dict[str, Any],
    save_observations: bool = True,
    create_plot: bool = True,
    intervention_step: int | None = None,
    save_metadata_json: bool = True,
) -> dict[str, str]:
    """Persist core NPZ, readable CSV, and optional metadata/observations/plots.

    The NPZ always embeds ``metadata_json``.  Callers that already write a
    human-facing summary may disable the separate metadata JSON to avoid
    storing the same fields twice.
    """
    prefix = prefix.expanduser().resolve()
    prefix.parent.mkdir(parents=True, exist_ok=True)
    trajectory = recorder.arrays()
    metadata = dict(metadata)
    metadata.update(
        {
            "schema_version": TRAJECTORY_SCHEMA_VERSION,
            "control_hz": recorder.control_hz,
            "state_count": recorder.state_count,
            "action_count": recorder.action_count,
            "inference_query_count": len(recorder.inference_query_steps),
            "inference_proposed_action_count": sum(
                len(chunk) for chunk in recorder.inference_action_chunks
            ),
            "state_action_alignment": "state[i] --action[i]--> state[i+1]",
            "manual_action_convention": "LIBERO OSC_POSE environment action; final value -1=open, +1=close",
        }
    )

    core_path = prefix.with_suffix(".npz")
    csv_path = prefix.with_suffix(".csv")
    inference_csv_path = prefix.with_name(f"{prefix.name}_inference.csv")
    metadata_path = prefix.with_suffix(".json")
    plot_path = prefix.with_name(f"{prefix.name}_plot.png")
    action_plot_prefix = prefix.with_name(f"{prefix.name}_action")
    observation_path = prefix.with_name(f"{prefix.name}_observations.npz")

    np.savez_compressed(
        core_path,
        **trajectory,
        metadata_json=np.asarray(json.dumps(metadata, ensure_ascii=False), dtype=np.str_),
    )
    _write_csv(csv_path, trajectory)

    paths = {
        "trajectory": str(core_path),
        "trajectory_csv": str(csv_path),
    }
    if save_metadata_json:
        metadata_path.write_text(
            json.dumps(metadata, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
        )
        paths["trajectory_metadata"] = str(metadata_path)
    if len(trajectory["inference_query_step"]):
        _write_inference_csv(inference_csv_path, trajectory, recorder.control_hz)
        paths["trajectory_inference_csv"] = str(inference_csv_path)
    observations = recorder.observation_arrays()
    if save_observations and observations is not None:
        np.savez_compressed(observation_path, **observations)
        paths["trajectory_observations"] = str(observation_path)
    if create_plot:
        plot_trajectory(trajectory, plot_path, metadata, intervention_step=intervention_step)
        paths["trajectory_plot"] = str(plot_path)
        paths.update(
            plot_action_trajectories(
                trajectory,
                action_plot_prefix,
                metadata,
                intervention_step=intervention_step,
            )
        )
    return paths
