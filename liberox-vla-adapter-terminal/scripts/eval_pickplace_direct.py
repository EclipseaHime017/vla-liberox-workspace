#!/usr/bin/env python3
"""Direct, single-process VLA-Adapter evaluation on one LIBERO-X pick/place task."""

from __future__ import annotations

import argparse
import json
import logging
import os
import sys
import time
from dataclasses import dataclass, fields
from pathlib import Path
from types import SimpleNamespace
from typing import Any

import numpy as np
import yaml

from simulation_core import run_control_loop
from trajectory_utils import (
    TrajectoryRecorder,
    quaternion_to_axis_angle,
    save_trajectory_bundle,
)


WORKSPACE_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_CONFIG_PATH = WORKSPACE_ROOT / "configs" / "config.yaml"
VALID_LEVELS = frozenset({"LEVEL1", "LEVEL2", "LEVEL3", "LEVEL4"})
VLA_VIDEO_VIEW = "vla_views"
VLA_OBSERVATION_CAMERAS = ("agentview", "robot0_eye_in_hand")
MAIN_VIEW_CAMERA = "agentview"
NATIVE_VIEWER_ATTRIBUTE = "_liberox_native_mujoco_viewer"
NATIVE_VIEWER_CLOSED_ATTRIBUTE = "_liberox_native_mujoco_viewer_closed"
CAMERA_DIMENSIONS_ATTRIBUTE = "_liberox_camera_dimensions"
_CONTROL_STACK_WARMED = False
VALID_VIDEO_CAMERAS = frozenset(
    {VLA_VIDEO_VIEW, "frontview", "birdview", "sideview", "galleryview"}
)
LOGGER = logging.getLogger("liberox_vla_adapter")


class UniqueKeyLoader(yaml.SafeLoader):
    """Safe YAML loader that rejects duplicate mapping keys."""


def _construct_unique_mapping(loader: UniqueKeyLoader, node: yaml.MappingNode, deep: bool = False) -> dict:
    mapping = {}
    for key_node, value_node in node.value:
        key = loader.construct_object(key_node, deep=deep)
        if key in mapping:
            raise yaml.constructor.ConstructorError(
                "while constructing a mapping",
                node.start_mark,
                f"found duplicate key {key!r}",
                key_node.start_mark,
            )
        mapping[key] = loader.construct_object(value_node, deep=deep)
    return mapping


UniqueKeyLoader.add_constructor(
    yaml.resolver.BaseResolver.DEFAULT_MAPPING_TAG,
    _construct_unique_mapping,
)


@dataclass(frozen=True)
class EvalConfig:
    vla_root: Path
    liberox_root: Path
    checkpoint: str
    use_pro_version: bool | None
    stats_key: str
    level: str
    task_name: str
    trials: int
    max_steps: int
    seed: int
    control_hz: int
    realtime_control: bool
    env_resolution: int
    disabled_policy_cameras: tuple[str, ...]
    video_camera: str
    video_width: int
    video_height: int
    main_view_video_width: int
    main_view_video_height: int
    video_fps: int
    open_loop_steps: int
    output: Path
    headless: bool
    env_only: bool
    no_video: bool
    save_trajectory: bool
    save_observation_images: bool
    trajectory_plot: bool
    cuda_visible_devices: str | None
    mujoco_gl: str | None


def validate_command_line() -> None:
    """Reject runtime parameters: evaluation settings live in the fixed YAML file."""
    parser = argparse.ArgumentParser(
        description=__doc__,
        epilog=f"Configuration is always loaded from: {DEFAULT_CONFIG_PATH}",
    )
    parser.parse_args()


def _require_string(config: dict[str, Any], key: str) -> str:
    value = config[key]
    if not isinstance(value, str) or not value.strip():
        raise TypeError(f"Config key '{key}' must be a non-empty string")
    return value


def _require_int(config: dict[str, Any], key: str) -> int:
    value = config[key]
    if isinstance(value, bool) or not isinstance(value, int):
        raise TypeError(f"Config key '{key}' must be an integer")
    return value


def _require_bool(config: dict[str, Any], key: str) -> bool:
    value = config[key]
    if not isinstance(value, bool):
        raise TypeError(f"Config key '{key}' must be true or false")
    return value


def _require_camera_list(config: dict[str, Any], key: str) -> tuple[str, ...]:
    value = config[key]
    if not isinstance(value, list) or any(not isinstance(item, str) for item in value):
        raise TypeError(f"Config key '{key}' must be a list of camera names")
    cameras = tuple(value)
    if len(cameras) != len(set(cameras)):
        raise ValueError(f"Config key '{key}' must not contain duplicate camera names")
    unknown = sorted(set(cameras) - set(VLA_OBSERVATION_CAMERAS))
    if unknown:
        raise ValueError(
            f"Config key '{key}' contains unknown policy cameras: {unknown}; "
            f"allowed: {list(VLA_OBSERVATION_CAMERAS)}"
        )
    if len(cameras) >= len(VLA_OBSERVATION_CAMERAS):
        raise ValueError("At least one VLA policy camera must remain enabled")
    return cameras


def _resolve_path(value: str, config_dir: Path) -> Path:
    path = Path(value).expanduser()
    if not path.is_absolute():
        path = config_dir / path
    return path.resolve()


def _resolve_checkpoint(value: str, config_dir: Path) -> str:
    """Resolve explicit local paths while preserving Hugging Face repository IDs."""
    path = Path(value).expanduser()
    if path.is_absolute():
        resolved_path = path.resolve()
    elif value.startswith(("./", "../", "~")):
        resolved_path = (config_dir / path).resolve()
    else:
        return value
    if not resolved_path.is_dir():
        raise FileNotFoundError(f"Local checkpoint directory not found: {resolved_path}")
    return str(resolved_path)


def load_config(config_path: Path) -> EvalConfig:
    if not config_path.is_file():
        raise FileNotFoundError(f"Configuration file not found: {config_path}")

    try:
        raw_config = yaml.load(config_path.read_text(encoding="utf-8"), Loader=UniqueKeyLoader)
    except yaml.YAMLError as exc:
        raise ValueError(f"Invalid YAML in {config_path}: {exc}") from exc

    if not isinstance(raw_config, dict):
        raise TypeError(f"Configuration root must be a mapping: {config_path}")
    if any(not isinstance(key, str) for key in raw_config):
        raise TypeError("All configuration keys must be strings")

    expected_keys = {field.name for field in fields(EvalConfig)}
    actual_keys = set(raw_config)
    missing_keys = sorted(expected_keys - actual_keys)
    unknown_keys = sorted(actual_keys - expected_keys)
    if missing_keys:
        raise ValueError(f"Missing configuration keys: {missing_keys}")
    if unknown_keys:
        raise ValueError(f"Unknown configuration keys: {unknown_keys}")

    config_dir = config_path.parent
    use_pro_version = raw_config["use_pro_version"]
    if use_pro_version is not None and not isinstance(use_pro_version, bool):
        raise TypeError("Config key 'use_pro_version' must be true, false, or null")

    cuda_visible_devices = raw_config["cuda_visible_devices"]
    if cuda_visible_devices is not None and not isinstance(cuda_visible_devices, str):
        raise TypeError("Config key 'cuda_visible_devices' must be a quoted string or null")

    mujoco_gl = raw_config["mujoco_gl"]
    if mujoco_gl is not None and (not isinstance(mujoco_gl, str) or not mujoco_gl.strip()):
        raise TypeError("Config key 'mujoco_gl' must be a non-empty string or null")

    config = EvalConfig(
        vla_root=_resolve_path(_require_string(raw_config, "vla_root"), config_dir),
        liberox_root=_resolve_path(_require_string(raw_config, "liberox_root"), config_dir),
        checkpoint=_resolve_checkpoint(_require_string(raw_config, "checkpoint"), config_dir),
        use_pro_version=use_pro_version,
        stats_key=_require_string(raw_config, "stats_key"),
        level=_require_string(raw_config, "level"),
        task_name=_require_string(raw_config, "task_name"),
        trials=_require_int(raw_config, "trials"),
        max_steps=_require_int(raw_config, "max_steps"),
        seed=_require_int(raw_config, "seed"),
        control_hz=_require_int(raw_config, "control_hz"),
        realtime_control=_require_bool(raw_config, "realtime_control"),
        env_resolution=_require_int(raw_config, "env_resolution"),
        disabled_policy_cameras=_require_camera_list(
            raw_config, "disabled_policy_cameras"
        ),
        video_camera=_require_string(raw_config, "video_camera"),
        video_width=_require_int(raw_config, "video_width"),
        video_height=_require_int(raw_config, "video_height"),
        main_view_video_width=_require_int(raw_config, "main_view_video_width"),
        main_view_video_height=_require_int(raw_config, "main_view_video_height"),
        video_fps=_require_int(raw_config, "video_fps"),
        open_loop_steps=_require_int(raw_config, "open_loop_steps"),
        output=_resolve_path(_require_string(raw_config, "output"), config_dir),
        headless=_require_bool(raw_config, "headless"),
        env_only=_require_bool(raw_config, "env_only"),
        no_video=_require_bool(raw_config, "no_video"),
        save_trajectory=_require_bool(raw_config, "save_trajectory"),
        save_observation_images=_require_bool(raw_config, "save_observation_images"),
        trajectory_plot=_require_bool(raw_config, "trajectory_plot"),
        cuda_visible_devices=cuda_visible_devices,
        mujoco_gl=mujoco_gl,
    )

    if config.level not in VALID_LEVELS:
        raise ValueError(f"Config key 'level' must be one of {sorted(VALID_LEVELS)}")
    if config.trials < 1:
        raise ValueError("Config key 'trials' must be >= 1")
    if config.max_steps < 1:
        raise ValueError("Config key 'max_steps' must be >= 1")
    if not 1 <= config.control_hz <= 240:
        raise ValueError("Config key 'control_hz' must be in [1, 240]")
    if config.env_resolution < 1:
        raise ValueError("Config key 'env_resolution' must be >= 1")
    if config.video_camera not in VALID_VIDEO_CAMERAS:
        raise ValueError(
            f"Config key 'video_camera' must be one of {sorted(VALID_VIDEO_CAMERAS)}"
        )
    if not 64 <= config.video_width <= 2048 or not 64 <= config.video_height <= 2048:
        raise ValueError("Config keys 'video_width' and 'video_height' must be in [64, 2048]")
    if config.video_camera == VLA_VIDEO_VIEW:
        expected_video_size = (2 * config.env_resolution, config.env_resolution)
        if (config.video_width, config.video_height) != expected_video_size:
            raise ValueError(
                "video_camera: vla_views must preserve the raw policy observations; "
                f"set video_width={expected_video_size[0]} and "
                f"video_height={expected_video_size[1]} for "
                f"env_resolution={config.env_resolution}"
            )
    if not 64 <= config.main_view_video_width <= 2048 or not 64 <= config.main_view_video_height <= 2048:
        raise ValueError(
            "Config keys 'main_view_video_width' and 'main_view_video_height' must be in [64, 2048]"
        )
    if config.main_view_video_width != config.main_view_video_height:
        raise ValueError(
            "The agentview policy camera is square; set main_view_video_width equal to "
            "main_view_video_height to preserve the inference view"
        )
    if not 1 <= config.video_fps <= 120:
        raise ValueError("Config key 'video_fps' must be in [1, 120]")
    if config.video_fps != config.control_hz:
        raise ValueError(
            "One video frame is recorded per control step; set video_fps equal to "
            "control_hz so playback duration matches simulated time"
        )
    if not 1 <= config.open_loop_steps <= 8:
        raise ValueError("Config key 'open_loop_steps' must be in [1, 8]")
    if config.seed < 0:
        raise ValueError("Config key 'seed' must be >= 0")
    if config.trajectory_plot and not config.save_trajectory:
        raise ValueError("Config key 'trajectory_plot' requires save_trajectory: true")
    if config.save_observation_images and not config.save_trajectory:
        raise ValueError("Config key 'save_observation_images' requires save_trajectory: true")
    if (
        config.task_name.endswith(".bddl")
        or "/" in config.task_name
        or "\\" in config.task_name
        or config.task_name in {".", ".."}
    ):
        raise ValueError("Config key 'task_name' must be a BDDL basename without directories or '.bddl'")
    return config


def apply_runtime_environment(config: EvalConfig) -> None:
    """Apply GPU and rendering settings before importing PyTorch or MuJoCo."""
    if config.cuda_visible_devices is not None:
        os.environ["CUDA_VISIBLE_DEVICES"] = config.cuda_visible_devices
    if config.mujoco_gl is not None:
        os.environ["MUJOCO_GL"] = config.mujoco_gl
    if (
        not config.headless
        and sys.platform.startswith("linux")
        and not os.environ.get("DISPLAY")
        and not os.environ.get("WAYLAND_DISPLAY")
    ):
        raise RuntimeError(
            "headless: false requires an active graphical session; DISPLAY and "
            "WAYLAND_DISPLAY are both unset"
        )
    matplotlib_cache = Path("/tmp/liberox-matplotlib-cache")
    matplotlib_cache.mkdir(parents=True, exist_ok=True)
    os.environ.setdefault("MPLCONFIGDIR", str(matplotlib_cache))


def add_repo_paths(vla_root: Path, liberox_root: Path) -> tuple[Path, Path]:
    vla_root = vla_root.expanduser().resolve()
    liberox_root = liberox_root.expanduser().resolve()
    for root, marker in ((vla_root, "prismatic"), (liberox_root, "libero")):
        if not (root / marker).exists():
            raise FileNotFoundError(f"Repository root looks invalid: {root} (missing {marker}/)")
    sys.path.insert(0, str(liberox_root))
    sys.path.insert(0, str(vla_root))
    return vla_root, liberox_root


def load_runtime() -> SimpleNamespace:
    import imageio.v2 as imageio
    import torch
    from libero.libero.envs import OffScreenRenderEnv
    from libero.libero.utils.parse_bddl import parse_bddl_file

    return SimpleNamespace(
        imageio=imageio,
        torch=torch,
        OffScreenRenderEnv=OffScreenRenderEnv,
        parse_bddl_file=parse_bddl_file,
    )


def load_policy_runtime(runtime: SimpleNamespace) -> SimpleNamespace:
    """Load VLA/TensorFlow modules only when a policy rollout is requested."""
    from experiments.robot.libero.run_libero_eval import (
        GenerateConfig,
        initialize_model,
        prepare_observation,
        process_action,
    )
    from experiments.robot.robot_utils import get_action, get_image_resize_size, set_seed_everywhere

    runtime.GenerateConfig = GenerateConfig
    runtime.initialize_model = initialize_model
    runtime.prepare_observation = prepare_observation
    runtime.process_action = process_action
    runtime.get_action = get_action
    runtime.get_image_resize_size = get_image_resize_size
    runtime.set_seed_everywhere = set_seed_everywhere
    return runtime


def resolve_task(liberox_root: Path, level: str, task_name: str) -> tuple[Path, Path]:
    bddl = liberox_root / "libero" / "libero_x" / "bddl" / level / f"{task_name}.bddl"
    init = liberox_root / "libero" / "libero_x" / "init" / level / f"{task_name}.init"
    if not bddl.is_file():
        raise FileNotFoundError(f"BDDL task not found: {bddl}")
    if not init.is_file():
        raise FileNotFoundError(f"Initial-state file not found: {init}")
    return bddl, init


def load_initial_states(runtime: SimpleNamespace, init_path: Path):
    states = runtime.torch.load(init_path, map_location="cpu")
    if len(states) == 0:
        raise ValueError(f"No initial states in {init_path}")
    return states


def make_env(
    runtime: SimpleNamespace,
    bddl_path: Path,
    resolution: int,
    horizon: int,
    control_hz: int,
    seed: int,
    video_camera: str,
    video_width: int,
    video_height: int,
    headless: bool,
):
    camera_dimensions = {
        "agentview": [resolution, resolution],
        "robot0_eye_in_hand": [resolution, resolution],
    }
    if video_camera != VLA_VIDEO_VIEW:
        camera_dimensions[video_camera] = [video_width, video_height]
    env_kwargs = dict(
        bddl_file_name=str(bddl_path),
        use_camera_obs=False,
        camera_names=list(camera_dimensions),
        camera_widths=[dimensions[0] for dimensions in camera_dimensions.values()],
        camera_heights=[dimensions[1] for dimensions in camera_dimensions.values()],
        horizon=horizon,
        control_freq=control_hz,
    )
    # The policy always needs offscreen cameras. For interactive runs we attach
    # MuJoCo's native passive viewer to the same model/data after reset instead
    # of asking robosuite to create its OpenCVRenderer.
    env = runtime.OffScreenRenderEnv(**env_kwargs)
    setattr(env, CAMERA_DIMENSIONS_ATTRIBUTE, camera_dimensions)
    env.seed(seed)
    env.reset()
    if not headless:
        try:
            launch_native_mujoco_viewer(env)
        except Exception:
            env.close()
            raise
    return env


def prewarm_simulation_control(
    runtime: SimpleNamespace,
    bddl_path: Path,
    init_state: Any,
    config: EvalConfig,
) -> None:
    """Initialize lazy controller kernels in a disposable environment.

    robosuite performs one-time controller setup during the first env.step(),
    which can take several seconds. Warming a separate environment keeps that
    delay outside the measured rollout without mutating the real episode's
    MuJoCo or controller state.
    """
    global _CONTROL_STACK_WARMED
    if _CONTROL_STACK_WARMED:
        return
    LOGGER.info("Pre-warming the LIBERO control stack in a disposable environment")
    env = make_env(
        runtime,
        bddl_path,
        config.env_resolution,
        2,
        config.control_hz,
        config.seed,
        config.video_camera,
        config.video_width,
        config.video_height,
        True,
    )
    started_at = time.monotonic()
    try:
        restore_state(env, init_state)
        dummy_action = np.asarray([0.0] * 6 + [-1.0], dtype=np.float32)
        env.step(dummy_action.tolist())
    finally:
        close_env(env)
    _CONTROL_STACK_WARMED = True
    LOGGER.info(
        "LIBERO control-stack pre-warm completed in %.3f s",
        time.monotonic() - started_at,
    )


def capture_camera_observations(env: Any, observation: dict[str, Any]) -> dict[str, Any]:
    """Render configured cameras on demand instead of inside every env.step()."""
    camera_dimensions = getattr(env, CAMERA_DIMENSIONS_ATTRIBUTE, None)
    if not isinstance(camera_dimensions, dict):
        raise RuntimeError("Environment has no configured camera dimensions")
    expected_keys = {f"{camera_name}_image" for camera_name in camera_dimensions}
    if expected_keys <= observation.keys():
        return observation
    wrapped_env = getattr(env, "env", env)
    sim = getattr(wrapped_env, "sim", None)
    if sim is None:
        raise AttributeError("LIBERO environment does not expose a MuJoCo sim renderer")
    result = dict(observation)
    for camera_name, (width, height) in camera_dimensions.items():
        result[f"{camera_name}_image"] = np.asarray(
            sim.render(width=width, height=height, camera_name=camera_name),
            dtype=np.uint8,
        )
    return result


def launch_native_mujoco_viewer(env: Any) -> Any:
    """Attach MuJoCo Simulate to the exact model/data used by LIBERO."""
    import mujoco
    from mujoco import viewer as mujoco_viewer

    wrapped_env = getattr(env, "env", env)
    sim = getattr(wrapped_env, "sim", None)
    raw_model = getattr(getattr(sim, "model", None), "_model", None)
    raw_data = getattr(getattr(sim, "data", None), "_data", None)
    if not isinstance(raw_model, mujoco.MjModel) or not isinstance(raw_data, mujoco.MjData):
        raise TypeError(
            "LIBERO/robosuite did not expose native mujoco.MjModel and mujoco.MjData"
        )
    viewer = mujoco_viewer.launch_passive(raw_model, raw_data)
    camera_id = mujoco.mj_name2id(
        raw_model,
        mujoco.mjtObj.mjOBJ_CAMERA,
        MAIN_VIEW_CAMERA,
    )
    if camera_id < 0:
        viewer.close()
        raise ValueError(f"MuJoCo model has no camera named {MAIN_VIEW_CAMERA!r}")
    with viewer.lock():
        # robosuite group 0 contains simplified collision bodies and group 1
        # contains the intended visual meshes. Showing both causes overlap and
        # z-fighting on the robot in MuJoCo's native viewer.
        viewer.opt.geomgroup[0] = 0
        viewer.opt.geomgroup[1] = 1
        viewer.cam.type = mujoco.mjtCamera.mjCAMERA_FIXED
        viewer.cam.fixedcamid = camera_id
    viewer.sync()
    setattr(env, NATIVE_VIEWER_ATTRIBUTE, viewer)
    setattr(env, NATIVE_VIEWER_CLOSED_ATTRIBUTE, False)
    LOGGER.info("Native MuJoCo viewer opened (passive mode)")
    return viewer


def render_live_window(env: Any) -> None:
    """Synchronize the optional native MuJoCo viewer with the current state."""
    viewer = getattr(env, NATIVE_VIEWER_ATTRIBUTE, None)
    if viewer is None:
        return
    if not viewer.is_running():
        if not getattr(env, NATIVE_VIEWER_CLOSED_ATTRIBUTE, False):
            LOGGER.info("Native MuJoCo viewer was closed; rollout will continue without a window")
            setattr(env, NATIVE_VIEWER_CLOSED_ATTRIBUTE, True)
        return
    viewer.sync()


def close_native_mujoco_viewer(env: Any) -> None:
    """Close the native viewer while keeping the offscreen environment alive."""
    viewer = getattr(env, NATIVE_VIEWER_ATTRIBUTE, None)
    if viewer is not None:
        try:
            viewer.close()
            deadline = time.monotonic() + 2.0
            while viewer.is_running() and time.monotonic() < deadline:
                time.sleep(0.01)
        except Exception:
            LOGGER.exception("Failed to close native MuJoCo viewer cleanly")
        finally:
            setattr(env, NATIVE_VIEWER_ATTRIBUTE, None)


def close_env(env: Any) -> None:
    """Close the native viewer before releasing its shared MuJoCo model/data."""
    close_native_mujoco_viewer(env)
    env.close()


class RealTimeControlLimiter:
    """Prevent control steps from running faster than the configured wall-clock rate."""

    def __init__(self, control_hz: float, enabled: bool) -> None:
        if control_hz <= 0:
            raise ValueError("control_hz must be positive")
        self.period_seconds = 1.0 / float(control_hz)
        self.enabled = enabled
        self._last_step_time: float | None = None

    def wait_before_step(self) -> float:
        now = time.monotonic()
        if self.enabled and self._last_step_time is not None:
            remaining = self._last_step_time + self.period_seconds - now
            if remaining > 0:
                time.sleep(remaining)
        self._last_step_time = time.monotonic()
        return self._last_step_time


def summarize_control_timing(step_times: list[float], control_hz: float) -> dict[str, float | None]:
    """Summarize actual wall-clock intervals between physics control steps."""
    summary: dict[str, float | None] = {
        "simulated_duration_seconds": len(step_times) / float(control_hz),
        "measured_control_hz": None,
        "control_interval_min_seconds": None,
        "control_interval_mean_seconds": None,
        "control_interval_max_seconds": None,
    }
    if len(step_times) < 2:
        return summary
    intervals = np.diff(np.asarray(step_times, dtype=np.float64))
    mean_interval = float(np.mean(intervals))
    summary.update(
        {
            "measured_control_hz": 1.0 / mean_interval,
            "control_interval_min_seconds": float(np.min(intervals)),
            "control_interval_mean_seconds": mean_interval,
            "control_interval_max_seconds": float(np.max(intervals)),
        }
    )
    return summary


def _validated_rgb_frame(frame: Any, source_name: str) -> np.ndarray:
    """Return a validated RGB array without changing its orientation."""
    frame = np.asarray(frame, dtype=np.uint8)
    if frame.ndim != 3 or frame.shape[2] != 3:
        raise ValueError(f"Camera {source_name!r} produced invalid RGB shape {frame.shape}")
    return frame


def _policy_oriented_rgb_frame(frame: Any, source_name: str) -> np.ndarray:
    """Apply VLA-Adapter's required 180-degree LIBERO training transform."""
    return _validated_rgb_frame(frame, source_name)[::-1, ::-1].copy()


def _display_oriented_rgb_frame(frame: Any, source_name: str) -> np.ndarray:
    """Convert the OpenGL framebuffer to a normal human-viewable image."""
    return _validated_rgb_frame(frame, source_name)[::-1].copy()


def _oriented_camera_frame(observation: dict[str, Any], camera_name: str) -> np.ndarray:
    """Extract one observation camera using LIBERO's training-time orientation."""
    observation_key = f"{camera_name}_image"
    if observation_key not in observation:
        raise KeyError(
            f"Video camera {camera_name!r} did not produce observation key {observation_key!r}; "
            f"available image keys={sorted(key for key in observation if key.endswith('_image'))}"
        )
    return _policy_oriented_rgb_frame(observation[observation_key], camera_name)


def _resize_video_frame(frame: np.ndarray, output_width: int, output_height: int) -> np.ndarray:
    """Resize only the replay frame; policy observations remain untouched."""
    if frame.shape[:2] == (output_height, output_width):
        return frame
    import cv2

    interpolation = (
        cv2.INTER_AREA
        if output_width < frame.shape[1] or output_height < frame.shape[0]
        else cv2.INTER_CUBIC
    )
    return cv2.resize(frame, (output_width, output_height), interpolation=interpolation)


def video_frame_from_observation(
    observation: dict[str, Any],
    camera_name: str,
    output_width: int,
    output_height: int,
) -> np.ndarray:
    """Build one replay frame, including the side-by-side VLA camera mode."""
    if camera_name == VLA_VIDEO_VIEW:
        primary = _oriented_camera_frame(observation, VLA_OBSERVATION_CAMERAS[0])
        wrist = _oriented_camera_frame(observation, VLA_OBSERVATION_CAMERAS[1])
        if primary.shape != wrist.shape:
            raise ValueError(
                "VLA camera shapes must match before concatenation: "
                f"agentview={primary.shape}, wrist={wrist.shape}"
            )
        frame = np.concatenate((primary, wrist), axis=1)
    else:
        observation_key = f"{camera_name}_image"
        if observation_key not in observation:
            raise KeyError(
                f"Video camera {camera_name!r} did not produce observation key "
                f"{observation_key!r}"
            )
        frame = _display_oriented_rgb_frame(observation[observation_key], camera_name)
    return _resize_video_frame(frame, output_width, output_height)


def render_high_resolution_camera_frame(
    env: Any,
    camera_name: str,
    output_width: int,
    output_height: int,
) -> np.ndarray:
    """Render one camera at replay resolution without changing policy observations."""
    wrapped_env = getattr(env, "env", env)
    sim = getattr(wrapped_env, "sim", None)
    if sim is None:
        raise AttributeError("LIBERO environment does not expose a MuJoCo sim renderer")
    frame = sim.render(
        width=output_width,
        height=output_height,
        camera_name=camera_name,
    )
    oriented = _display_oriented_rgb_frame(frame, camera_name)
    expected_shape = (output_height, output_width, 3)
    if oriented.shape != expected_shape:
        raise ValueError(
            f"High-resolution camera {camera_name!r} returned {oriented.shape}; "
            f"expected {expected_shape}"
        )
    return oriented


class PairedVideoWriter:
    """Stream the policy-camera mosaic and matching high-resolution agentview."""

    def __init__(
        self,
        runtime: SimpleNamespace,
        combined_path: Path,
        main_view_path: Path,
        fps: int,
        video_camera: str,
        video_width: int,
        video_height: int,
        main_view_width: int,
        main_view_height: int,
    ) -> None:
        self.runtime = runtime
        self.combined_path = combined_path
        self.main_view_path = main_view_path
        self.fps = fps
        self.video_camera = video_camera
        self.video_width = video_width
        self.video_height = video_height
        self.main_view_width = main_view_width
        self.main_view_height = main_view_height
        self.frame_count = 0
        self._combined_writer = None
        self._main_view_writer = None

    def _open(self) -> None:
        if self._combined_writer is not None:
            return
        self.combined_path.parent.mkdir(parents=True, exist_ok=True)
        self.main_view_path.parent.mkdir(parents=True, exist_ok=True)
        self.combined_path.unlink(missing_ok=True)
        self.main_view_path.unlink(missing_ok=True)
        self._combined_writer = self.runtime.imageio.get_writer(self.combined_path, fps=self.fps)
        try:
            self._main_view_writer = self.runtime.imageio.get_writer(
                self.main_view_path,
                fps=self.fps,
            )
        except Exception:
            self._combined_writer.close()
            self._combined_writer = None
            raise

    def append(self, env: Any, observation: dict[str, Any]) -> None:
        self._open()
        assert self._combined_writer is not None and self._main_view_writer is not None
        combined_frame = video_frame_from_observation(
            observation,
            self.video_camera,
            self.video_width,
            self.video_height,
        )
        main_view_frame = render_high_resolution_camera_frame(
            env,
            MAIN_VIEW_CAMERA,
            self.main_view_width,
            self.main_view_height,
        )
        self._combined_writer.append_data(combined_frame)
        self._main_view_writer.append_data(main_view_frame)
        self.frame_count += 1

    def close(self) -> None:
        close_error: Exception | None = None
        for writer in (self._combined_writer, self._main_view_writer):
            if writer is None:
                continue
            try:
                writer.close()
            except Exception as exc:  # Try to finalize both files before surfacing the error.
                close_error = close_error or exc
        self._combined_writer = None
        self._main_view_writer = None
        if close_error is not None:
            raise close_error


def postprocess_recorded_trajectory(
    runtime: SimpleNamespace,
    env: Any,
    recorder: TrajectoryRecorder,
    save_observations: bool,
    combined_path: Path | None,
    main_view_path: Path | None,
    fps: int,
    video_camera: str,
    video_width: int,
    video_height: int,
    main_view_width: int,
    main_view_height: int,
) -> int:
    """Rebuild images/videos from states after rollout, outside the control loop."""
    if (combined_path is None) != (main_view_path is None):
        raise ValueError("Both video paths must be provided together")
    writer = None
    if combined_path is not None and main_view_path is not None:
        writer = PairedVideoWriter(
            runtime=runtime,
            combined_path=combined_path,
            main_view_path=main_view_path,
            fps=fps,
            video_camera=video_camera,
            video_width=video_width,
            video_height=video_height,
            main_view_width=main_view_width,
            main_view_height=main_view_height,
        )
    agentview_images: list[np.ndarray] = []
    wrist_images: list[np.ndarray] = []
    try:
        for state_index, sim_state in enumerate(recorder.sim_states):
            observation = restore_state(env, sim_state)
            observation = capture_camera_observations(env, observation)
            if save_observations:
                agentview_images.append(
                    np.asarray(observation["agentview_image"], dtype=np.uint8).copy()
                )
                wrist_images.append(
                    np.asarray(observation["robot0_eye_in_hand_image"], dtype=np.uint8).copy()
                )
            if writer is not None and state_index < recorder.action_count:
                writer.append(env, observation)
    finally:
        if writer is not None:
            writer.close()
    if save_observations:
        recorder.agentview_images = agentview_images
        recorder.wrist_images = wrist_images
        recorder.capture_images = True
    return 0 if writer is None else writer.frame_count


def restore_state(env, state) -> dict:
    state_array = state.numpy() if hasattr(state, "numpy") else np.asarray(state)
    return env.regenerate_obs_from_state(state_array)


def proprio_from_obs(obs: dict) -> np.ndarray:
    return np.concatenate(
        (
            obs["robot0_eef_pos"],
            quaternion_to_axis_angle(obs["robot0_eef_quat"]),
            obs["robot0_gripper_qpos"],
        )
    ).astype(np.float32)


def validate_observation(obs: dict) -> None:
    required = (
        "agentview_image",
        "robot0_eye_in_hand_image",
        "robot0_eef_pos",
        "robot0_eef_quat",
        "robot0_gripper_qpos",
    )
    missing = [key for key in required if key not in obs]
    if missing:
        raise KeyError(f"Observation is missing keys: {missing}")
    proprio = proprio_from_obs(obs)
    if proprio.shape != (8,):
        raise ValueError(f"Expected 8-D proprio, got {proprio.shape}")
    LOGGER.info("agentview_image: %s", obs["agentview_image"].shape)
    LOGGER.info("robot0_eye_in_hand_image: %s", obs["robot0_eye_in_hand_image"].shape)
    LOGGER.info("proprio: %s", proprio.shape)


def build_model(runtime: SimpleNamespace, args: EvalConfig):
    cfg = runtime.GenerateConfig(
        model_family="openvla",
        pretrained_checkpoint=args.checkpoint,
        use_l1_regression=True,
        use_minivlm=True,
        use_film=False,
        num_images_in_input=2,
        use_proprio=True,
        center_crop=True,
        num_open_loop_steps=args.open_loop_steps,
        task_suite_name=args.stats_key,
        num_trials_per_task=args.trials,
        env_img_res=args.env_resolution,
        seed=args.seed,
        use_pro_version=(
            "Pro" in str(args.checkpoint)
            if args.use_pro_version is None
            else args.use_pro_version
        ),
        use_wandb=False,
    )
    runtime.set_seed_everywhere(args.seed)
    model, action_head, proprio_projector, noisy_action_projector, processor = runtime.initialize_model(cfg)
    resize_size = runtime.get_image_resize_size(cfg)
    components = SimpleNamespace(
        model=model,
        action_head=action_head,
        proprio_projector=proprio_projector,
        noisy_action_projector=noisy_action_projector,
        processor=processor,
        resize_size=resize_size,
    )
    return cfg, components


def mask_policy_camera_observations(
    obs: dict[str, Any], disabled_policy_cameras: tuple[str, ...] = ()
) -> dict[str, Any]:
    """Black out selected VLA inputs while preserving the two-image contract."""
    if not disabled_policy_cameras:
        return obs
    if len(disabled_policy_cameras) >= len(VLA_OBSERVATION_CAMERAS):
        raise ValueError("At least one VLA policy camera must remain enabled")
    masked = dict(obs)
    for camera in disabled_policy_cameras:
        if camera not in VLA_OBSERVATION_CAMERAS:
            raise ValueError(f"Unknown VLA policy camera: {camera}")
        key = f"{camera}_image"
        image = np.asarray(obs.get(key))
        if image.ndim != 3 or image.shape[-1] != 3:
            raise ValueError(f"Observation '{key}' must be an HxWx3 image")
        masked[key] = np.zeros_like(image)
    return masked


def checked_action_chunk(
    runtime: SimpleNamespace,
    cfg,
    components,
    obs: dict,
    prompt: str,
    disabled_policy_cameras: tuple[str, ...] = (),
) -> list[np.ndarray]:
    policy_input = mask_policy_camera_observations(obs, disabled_policy_cameras)
    policy_obs, _ = runtime.prepare_observation(policy_input, components.resize_size)
    actions = runtime.get_action(
        cfg,
        components.model,
        policy_obs,
        prompt,
        processor=components.processor,
        action_head=components.action_head,
        proprio_projector=components.proprio_projector,
        noisy_action_projector=components.noisy_action_projector,
        use_film=cfg.use_film,
        use_minivlm=cfg.use_minivlm,
    )
    actions = [np.asarray(action, dtype=np.float32) for action in actions]
    if not actions:
        raise RuntimeError("Policy returned an empty action chunk")
    for index, action in enumerate(actions):
        if action.shape != (7,):
            raise ValueError(f"Action {index} has shape {action.shape}; expected (7,)")
        if not np.isfinite(action).all():
            raise ValueError(f"Action {index} contains NaN or Inf")
    return actions


def run_episode(
    runtime: SimpleNamespace,
    args: EvalConfig,
    cfg,
    components,
    bddl_path,
    init_state,
    prompt,
    episode_id,
):
    env = make_env(
        runtime,
        bddl_path,
        args.env_resolution,
        args.max_steps + 1,
        args.control_hz,
        args.seed + episode_id,
        args.video_camera,
        args.video_width,
        args.video_height,
        args.headless,
    )
    control_hz = float(getattr(env.env, "control_freq", args.control_hz))
    recorder = TrajectoryRecorder(
        control_hz=control_hz,
        # Camera images are reconstructed from saved states after the realtime
        # loop so they cannot consume the 50 ms control budget.
        capture_images=False,
    )
    rate_limiter = RealTimeControlLimiter(control_hz, args.realtime_control)
    combined_pending_path = args.output / f".episode_{episode_id:03d}_vla_views.pending.mp4"
    main_view_pending_path = args.output / f".episode_{episode_id:03d}_agentview_hd.pending.mp4"
    video_frame_count = 0
    success = False
    error = None
    steps = 0
    queries = 0
    control_step_times: list[float] = []
    try:
        obs = restore_state(env, init_state)
        obs = capture_camera_observations(env, obs)
        validate_observation(obs)
        recorder.record_initial(env, obs)
        render_live_window(env)
        def query_policy(observation: dict[str, Any], _step: int):
            observation = capture_camera_observations(env, observation)
            return observation, checked_action_chunk(
                runtime,
                cfg,
                components,
                observation,
                prompt,
                args.disabled_policy_cameras,
            )

        loop_result = run_control_loop(
            env=env,
            recorder=recorder,
            initial_observation=obs,
            target_action_count=args.max_steps,
            rate_limiter=rate_limiter,
            action_source="policy",
            open_loop_steps=args.open_loop_steps,
            policy_query=query_policy,
            policy_action_transform=lambda action: runtime.process_action(
                action,
                cfg.model_family,
            ),
            on_transition=lambda _observation, _step, _success: render_live_window(env),
            stop_on_success=False,
            horizon_reason="max_steps",
        )
        success = loop_result.success
        steps = loop_result.executed_steps
        queries = loop_result.policy_queries
        control_step_times = loop_result.control_step_times
    except Exception as exc:  # Keep later trials runnable and record the exact failure.
        error = f"{type(exc).__name__}: {exc}"
        LOGGER.exception("Episode %d failed", episode_id)
    finally:
        close_native_mujoco_viewer(env)
        should_postprocess = (
            recorder.state_count
            and (
                not args.no_video
                or (args.save_trajectory and args.save_observation_images)
            )
        )
        if should_postprocess:
            try:
                LOGGER.info(
                    "Post-processing %d recorded states (video=%s observations=%s)",
                    recorder.state_count,
                    not args.no_video,
                    args.save_trajectory and args.save_observation_images,
                )
                video_frame_count = postprocess_recorded_trajectory(
                    runtime=runtime,
                    env=env,
                    recorder=recorder,
                    save_observations=args.save_trajectory and args.save_observation_images,
                    combined_path=None if args.no_video else combined_pending_path,
                    main_view_path=None if args.no_video else main_view_pending_path,
                    fps=args.video_fps,
                    video_camera=args.video_camera,
                    video_width=args.video_width,
                    video_height=args.video_height,
                    main_view_width=args.main_view_video_width,
                    main_view_height=args.main_view_video_height,
                )
            except Exception as exc:
                if error is None:
                    error = f"{type(exc).__name__}: failed to render videos: {exc}"
                LOGGER.exception("Episode %d post-rollout video rendering failed", episode_id)
        close_env(env)
    timing = summarize_control_timing(control_step_times, control_hz)
    LOGGER.info("episode=%d control timing: %s", episode_id, timing)
    maximum_interval = timing["control_interval_max_seconds"]
    if maximum_interval is not None and maximum_interval > (1.1 / control_hz):
        LOGGER.warning(
            "Episode %d missed the %.1f Hz soft-real-time deadline; maximum control "
            "interval was %.4f s. Policy inference or system load exceeded the %.4f s budget.",
            episode_id,
            control_hz,
            maximum_interval,
            1.0 / control_hz,
        )
    return success, steps, queries, error, recorder, video_frame_count, timing


def write_jsonl(path: Path, record: dict) -> None:
    with path.open("a", encoding="utf-8") as handle:
        handle.write(json.dumps(record, ensure_ascii=False) + "\n")


def evaluation_runtime_metadata(config: EvalConfig) -> dict[str, Any]:
    """Fields shared by per-episode, trajectory, and aggregate metadata."""
    return {
        "control_hz": config.control_hz,
        "realtime_control": config.realtime_control,
        "headless": config.headless,
        "viewer_backend": None if config.headless else "mujoco_native_passive",
        "video_camera": MAIN_VIEW_CAMERA,
        "video_width": config.main_view_video_width,
        "video_height": config.main_view_video_height,
        "main_view_video_camera": MAIN_VIEW_CAMERA,
        "main_view_video_width": config.main_view_video_width,
        "main_view_video_height": config.main_view_video_height,
        "vla_views_video_camera": config.video_camera,
        "vla_views_video_width": config.video_width,
        "vla_views_video_height": config.video_height,
        "video_fps": config.video_fps,
        "disabled_policy_cameras": list(config.disabled_policy_cameras),
    }


def main() -> int:
    logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
    validate_command_line()
    config_path = DEFAULT_CONFIG_PATH.resolve()
    args = load_config(config_path)
    apply_runtime_environment(args)
    LOGGER.info("config: %s", config_path)
    LOGGER.info(
        "runtime environment: CUDA_VISIBLE_DEVICES=%s MUJOCO_GL=%s",
        os.environ.get("CUDA_VISIBLE_DEVICES", "<unchanged>"),
        os.environ.get("MUJOCO_GL", "<unchanged>"),
    )
    _, liberox_root = add_repo_paths(args.vla_root, args.liberox_root)
    runtime = load_runtime()
    bddl_path, init_path = resolve_task(liberox_root, args.level, args.task_name)
    task_description = str(runtime.parse_bddl_file(str(bddl_path))["language"])
    states = load_initial_states(runtime, init_path)
    LOGGER.info("task: %s", task_description)
    LOGGER.info("bddl: %s", bddl_path)
    LOGGER.info("available initial states: %d", len(states))

    if args.env_only:
        env = make_env(
            runtime,
            bddl_path,
            args.env_resolution,
            20,
            args.control_hz,
            args.seed,
            args.video_camera,
            args.video_width,
            args.video_height,
            args.headless,
        )
        try:
            obs = restore_state(env, states[0])
            obs = capture_camera_observations(env, obs)
            validate_observation(obs)
            render_live_window(env)
            LOGGER.info(
                "video_frame[%s]: %s",
                args.video_camera,
                video_frame_from_observation(
                    obs,
                    args.video_camera,
                    args.video_width,
                    args.video_height,
                ).shape,
            )
            dummy_action = np.asarray([0.0] * 6 + [-1.0], dtype=np.float32)
            obs, _, _, _ = env.step(dummy_action.tolist())
            obs = capture_camera_observations(env, obs)
            validate_observation(obs)
            render_live_window(env)
            LOGGER.info("dummy_action: %s", dummy_action.shape)
            LOGGER.info("Environment-only smoke test passed")
        finally:
            close_env(env)
        return 0

    if args.trials > len(states):
        raise ValueError(
            f"Requested {args.trials} trials, but only {len(states)} initial states are available"
        )

    runtime = load_policy_runtime(runtime)
    args.output.mkdir(parents=True, exist_ok=True)
    results_path = args.output / "results.jsonl"

    cfg, components = build_model(runtime, args)
    prewarm_simulation_control(runtime, bddl_path, states[0], args)
    results_path.unlink(missing_ok=True)
    successes = 0
    episode_errors = 0
    episode_timings: list[dict[str, float | None]] = []
    runtime_metadata = evaluation_runtime_metadata(args)
    for episode_id in range(args.trials):
        success, steps, queries, error, recorder, video_frame_count, timing = run_episode(
            runtime,
            args,
            cfg,
            components,
            bddl_path,
            states[episode_id],
            task_description,
            episode_id,
        )
        episode_timings.append(timing)
        successes += int(success)
        episode_errors += int(error is not None)
        status = "success" if success else "failure"
        # Keep the conventional episode filename on the useful external policy
        # camera. The raw two-camera policy-input mosaic is a supplementary
        # artifact with an explicit suffix.
        main_view_video_path = args.output / f"episode_{episode_id:03d}_{status}.mp4"
        vla_views_video_path = args.output / f"episode_{episode_id:03d}_{status}_vla_views.mp4"
        if video_frame_count and not args.no_video:
            combined_pending_path = args.output / f".episode_{episode_id:03d}_vla_views.pending.mp4"
            main_view_pending_path = args.output / f".episode_{episode_id:03d}_agentview_hd.pending.mp4"
            combined_pending_path.replace(vla_views_video_path)
            main_view_pending_path.replace(main_view_video_path)
        video_value = "" if args.no_video or not video_frame_count else str(main_view_video_path)
        main_view_video_value = (
            "" if args.no_video or not video_frame_count else str(main_view_video_path)
        )
        vla_views_video_value = (
            "" if args.no_video or not video_frame_count else str(vla_views_video_path)
        )
        trajectory_paths: dict[str, str] = {}
        if args.save_trajectory and recorder.state_count:
            trajectory_metadata = {
                "config": str(config_path),
                "episode": episode_id,
                "seed": args.seed + episode_id,
                "task": task_description,
                "task_name": args.task_name,
                "level": args.level,
                "bddl": str(bddl_path),
                "checkpoint": str(args.checkpoint),
                "requested_stats_key": args.stats_key,
                "resolved_stats_key": str(cfg.unnorm_key),
                "success": success,
                "steps": steps,
                "target_total_steps": args.max_steps,
                "policy_queries": queries,
                "open_loop_steps": args.open_loop_steps,
                **runtime_metadata,
                "video_frames": video_frame_count,
                "error": error,
                "control_mode": "policy",
                "video": video_value,
                "main_view_video": main_view_video_value,
                "vla_views_video": vla_views_video_value,
                **timing,
            }
            trajectory_paths = save_trajectory_bundle(
                recorder,
                args.output / f"trajectory_{episode_id:03d}",
                trajectory_metadata,
                save_observations=args.save_observation_images,
                create_plot=args.trajectory_plot,
            )
        record = {
            "episode": episode_id,
            "success": success,
            "steps": steps,
            "policy_queries": queries,
            "open_loop_steps": args.open_loop_steps,
            **runtime_metadata,
            "video_frames": video_frame_count,
            "task": task_description,
            "level": args.level,
            "bddl": str(bddl_path),
            "checkpoint": str(args.checkpoint),
            "stats_key": str(cfg.unnorm_key),
            "seed": args.seed + episode_id,
            "video": video_value,
            "main_view_video": main_view_video_value,
            "vla_views_video": vla_views_video_value,
            "error": error,
            **timing,
            **trajectory_paths,
        }
        write_jsonl(results_path, record)
        LOGGER.info("episode=%d success=%s steps=%d queries=%d", episode_id, success, steps, queries)

    measured_rates = [
        value
        for timing in episode_timings
        if (value := timing["measured_control_hz"]) is not None
    ]
    maximum_intervals = [
        value
        for timing in episode_timings
        if (value := timing["control_interval_max_seconds"]) is not None
    ]
    summary = {
        "config": str(config_path),
        "successes": successes,
        "episode_errors": episode_errors,
        "episodes": args.trials,
        "success_rate": successes / args.trials,
        "task": task_description,
        "level": args.level,
        "checkpoint": str(args.checkpoint),
        "requested_stats_key": args.stats_key,
        "resolved_stats_key": str(cfg.unnorm_key),
        "open_loop_steps": args.open_loop_steps,
        **runtime_metadata,
        "measured_control_hz_mean": (
            None if not measured_rates else float(np.mean(measured_rates))
        ),
        "control_interval_max_seconds": (
            None if not maximum_intervals else float(max(maximum_intervals))
        ),
        "episodes_with_deadline_miss": sum(
            interval > (1.1 / args.control_hz) for interval in maximum_intervals
        ),
        "trajectories_saved": args.save_trajectory,
        "observation_images_saved": args.save_trajectory and args.save_observation_images,
    }
    (args.output / "summary.json").write_text(
        json.dumps(summary, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    LOGGER.info("summary: %s", summary)
    return 1 if episode_errors else 0


if __name__ == "__main__":
    raise SystemExit(main())
