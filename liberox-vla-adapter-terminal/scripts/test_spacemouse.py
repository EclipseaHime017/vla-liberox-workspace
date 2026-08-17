#!/usr/bin/env python3
"""Test SpaceMouse input and optional 20 Hz LIBERO-X manual control."""

from __future__ import annotations

import argparse
import csv
import json
import logging
import os
import sys
import tempfile
import time
import uuid
from datetime import datetime
from pathlib import Path
from typing import Any, Iterable

import numpy as np

import eval_pickplace_direct as direct

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from backend.app.devices.spacemouse import (
    AXIS_NAMES,
    DEFAULT_SPACEMOUSE_CONFIG,
    SpaceMouseInput,
    SpaceMouseSnapshot,
    SpaceMouseTestConfig,
    load_spacemouse_config,
)
from simulation_core import run_control_loop
from trajectory_utils import TrajectoryRecorder, save_trajectory_bundle


LOGGER = logging.getLogger("spacemouse_test")
EVAL_CONFIG_PATH = PROJECT_ROOT / "config.yaml"


def validate_command_line() -> None:
    parser = argparse.ArgumentParser(
        description=__doc__,
        epilog=f"Configuration is always loaded from: {DEFAULT_SPACEMOUSE_CONFIG}",
    )
    parser.parse_args()


def _atomic_write_json(path: Path, value: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    handle = tempfile.NamedTemporaryFile(
        mode="w",
        encoding="utf-8",
        dir=path.parent,
        prefix=f".{path.name}.",
        suffix=".tmp",
        delete=False,
    )
    temporary = Path(handle.name)
    try:
        with handle:
            json.dump(value, handle, ensure_ascii=False, indent=2)
            handle.write("\n")
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temporary, path)
    except Exception:
        temporary.unlink(missing_ok=True)
        raise


def _atomic_write_csv(path: Path, fieldnames: list[str], rows: Iterable[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    handle = tempfile.NamedTemporaryFile(
        mode="w",
        encoding="utf-8",
        newline="",
        dir=path.parent,
        prefix=f".{path.name}.",
        suffix=".tmp",
        delete=False,
    )
    temporary = Path(handle.name)
    try:
        with handle:
            writer = csv.DictWriter(handle, fieldnames=fieldnames)
            writer.writeheader()
            writer.writerows(rows)
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temporary, path)
    except Exception:
        temporary.unlink(missing_ok=True)
        raise


def _distribution(values: Iterable[float]) -> dict[str, float | int | None]:
    array = np.asarray(list(values), dtype=np.float64)
    array = array[np.isfinite(array)]
    if not len(array):
        return {
            "count": 0,
            "min": None,
            "mean": None,
            "p50": None,
            "p95": None,
            "p99": None,
            "max": None,
        }
    return {
        "count": int(len(array)),
        "min": float(np.min(array)),
        "mean": float(np.mean(array)),
        "p50": float(np.percentile(array, 50)),
        "p95": float(np.percentile(array, 95)),
        "p99": float(np.percentile(array, 99)),
        "max": float(np.max(array)),
    }


def _create_run_dir(config: SpaceMouseTestConfig) -> Path:
    stamp = datetime.now().strftime("%Y%m%d_%H%M%S_%f")
    run_dir = config.output_root / f"{config.mode}_{stamp}_{uuid.uuid4().hex[:8]}"
    run_dir.mkdir(parents=True, exist_ok=False)
    return run_dir


def _config_record(config: SpaceMouseTestConfig) -> dict[str, Any]:
    return {
        "mode": config.mode,
        "device_name": config.device_name,
        "device_index": config.device_index,
        "device_path": config.device_path,
        "expected_vendor_id": f"{config.expected_vendor_id:04x}",
        "expected_product_id": f"{config.expected_product_id:04x}",
        "axis_convention": config.axis_convention,
        "axis_order": list(config.axis_order),
        "axis_signs": list(config.axis_signs),
        "translation_gain": config.translation_gain,
        "rotation_gain": config.rotation_gain,
        "deadzone": config.deadzone,
        "smoothing_alpha": config.smoothing_alpha,
        "neutral_calibration_seconds": config.neutral_calibration_seconds,
        "neutral_max_abs": config.neutral_max_abs,
        "poll_interval_ms": config.poll_interval_ms,
        "stale_timeout_ms": config.stale_timeout_ms,
        "test_duration_seconds": config.test_duration_seconds,
        "max_steps": config.max_steps,
        "countdown_seconds": config.countdown_seconds,
        "save_video": config.save_video,
        "trajectory_plot": config.trajectory_plot,
        "output_root": str(config.output_root),
    }


def _snapshot_row(snapshot: SpaceMouseSnapshot, start_time: float, index: int) -> dict[str, Any]:
    row: dict[str, Any] = {
        "sample_index": index,
        "wall_time_seconds": snapshot.captured_monotonic - start_time,
        "sequence": snapshot.sequence,
        "device_timestamp": snapshot.device_timestamp,
        "sample_age_ms": None
        if snapshot.sample_age_seconds is None
        else 1000.0 * snapshot.sample_age_seconds,
        "connected": snapshot.connected,
        "stale": snapshot.stale,
        "button_left": snapshot.buttons[0] if len(snapshot.buttons) > 0 else 0,
        "button_right": snapshot.buttons[1] if len(snapshot.buttons) > 1 else 0,
        "gripper_action": snapshot.action[6],
        "error": snapshot.error or "",
    }
    for prefix, values in (
        ("raw", snapshot.raw_axes),
        ("corrected", snapshot.corrected_axes),
        ("command", snapshot.command_axes),
    ):
        for name, value in zip(AXIS_NAMES, values):
            row[f"{prefix}_{name}"] = value
    return row


SAMPLE_FIELDS = [
    "sample_index",
    "wall_time_seconds",
    "sequence",
    "device_timestamp",
    "sample_age_ms",
    "connected",
    "stale",
    *(f"raw_{name}" for name in AXIS_NAMES),
    *(f"corrected_{name}" for name in AXIS_NAMES),
    *(f"command_{name}" for name in AXIS_NAMES),
    "button_left",
    "button_right",
    "gripper_action",
    "error",
]


def _countdown(seconds: int) -> None:
    for remaining in range(seconds, 0, -1):
        LOGGER.info("Manual control starts in %d... keep the cap neutral", remaining)
        time.sleep(1.0)


def _axis_and_button_coverage(samples: list[SpaceMouseSnapshot]) -> dict[str, Any]:
    if samples:
        raw = np.asarray([sample.raw_axes for sample in samples], dtype=np.float64)
        minimum = np.min(raw, axis=0)
        maximum = np.max(raw, axis=0)
    else:
        minimum = np.zeros(6)
        maximum = np.zeros(6)
    axes = {
        name: {
            "min": float(minimum[index]),
            "max": float(maximum[index]),
            "both_directions_observed": bool(minimum[index] < -0.1 and maximum[index] > 0.1),
        }
        for index, name in enumerate(AXIS_NAMES)
    }
    return {
        "axes": axes,
        "left_button_observed": any(len(sample.buttons) > 0 and sample.buttons[0] for sample in samples),
        "right_button_observed": any(len(sample.buttons) > 1 and sample.buttons[1] for sample in samples),
        "all_axes_both_directions": all(value["both_directions_observed"] for value in axes.values()),
    }


def _device_statistics(
    controller: SpaceMouseInput,
    samples: list[SpaceMouseSnapshot],
) -> dict[str, Any]:
    diagnostics = controller.diagnostics()
    event_times = np.asarray(diagnostics.pop("event_times"), dtype=np.float64)
    event_intervals_ms = np.diff(event_times) * 1000.0 if len(event_times) > 1 else []
    sample_ages_ms = [
        1000.0 * sample.sample_age_seconds
        for sample in samples
        if sample.sample_age_seconds is not None and not sample.stale
    ]
    diagnostics.update(
        {
            "event_interval_ms": _distribution(event_intervals_ms),
            "sample_age_ms": _distribution(sample_ages_ms),
            "coverage": _axis_and_button_coverage(samples),
        }
    )
    mean_interval = diagnostics["event_interval_ms"]["mean"]
    diagnostics["measured_event_hz"] = (
        None if mean_interval in {None, 0.0} else 1000.0 / float(mean_interval)
    )
    return diagnostics


def run_device_test(
    controller: SpaceMouseInput,
    config: SpaceMouseTestConfig,
) -> tuple[list[SpaceMouseSnapshot], dict[str, Any]]:
    LOGGER.info(
        "Device test running for %.1f s. Move every axis in both directions and press both buttons.",
        config.test_duration_seconds,
    )
    started = time.monotonic()
    deadline = started + config.test_duration_seconds
    samples: list[SpaceMouseSnapshot] = []
    interrupted = False
    try:
        while time.monotonic() < deadline:
            snapshot = controller.latest_snapshot()
            samples.append(snapshot)
            raw = " ".join(f"{name}={value:+.3f}" for name, value in zip(AXIS_NAMES, snapshot.raw_axes))
            command = " ".join(
                f"{name}={value:+.3f}" for name, value in zip(AXIS_NAMES, snapshot.command_axes)
            )
            print(
                f"\rRAW {raw} | CMD {command} | buttons={snapshot.buttons} "
                f"gripper={snapshot.action[6]:+.0f} stale={snapshot.stale}   ",
                end="",
                flush=True,
            )
            if snapshot.error is not None:
                raise RuntimeError(snapshot.error)
            time.sleep(0.05)
    except KeyboardInterrupt:
        interrupted = True
    finally:
        print()
    coverage = _axis_and_button_coverage(samples)
    result = {
        "mode": "device",
        "duration_seconds": time.monotonic() - started,
        "sample_count": len(samples),
        "interrupted": interrupted,
        "coverage": coverage,
        "functional_check_complete": bool(
            coverage["all_axes_both_directions"]
            and coverage["left_button_observed"]
            and coverage["right_button_observed"]
        ),
    }
    if not result["functional_check_complete"]:
        LOGGER.warning("Device test completed, but not every axis direction/button was observed")
    return samples, result


class TimedEnvironment:
    """Delegate environment calls while measuring every actual env.step."""

    def __init__(self, environment: Any):
        self.environment = environment
        self.step_started: list[float] = []
        self.step_durations: list[float] = []

    def __getattr__(self, name: str) -> Any:
        return getattr(self.environment, name)

    def step(self, action: Any):
        started = time.monotonic()
        self.step_started.append(started)
        try:
            return self.environment.step(action)
        finally:
            self.step_durations.append(time.monotonic() - started)


def _viewer_is_closed(env: Any) -> bool:
    viewer = getattr(env, direct.NATIVE_VIEWER_ATTRIBUTE, None)
    return viewer is not None and not viewer.is_running()


def _render_agentview_video(
    runtime: Any,
    env: Any,
    recorder: TrajectoryRecorder,
    path: Path,
    fps: int,
    width: int,
    height: int,
) -> int:
    if recorder.action_count == 0:
        return 0
    pending = path.with_name(f".{path.name}.pending.mp4")
    writer = runtime.imageio.get_writer(pending, fps=fps)
    render_error: Exception | None = None
    try:
        for sim_state in recorder.sim_states[: recorder.action_count]:
            direct.restore_state(env, sim_state)
            writer.append_data(
                direct.render_high_resolution_camera_frame(
                    env,
                    direct.MAIN_VIEW_CAMERA,
                    width,
                    height,
                )
            )
    except Exception as exc:
        render_error = exc
    finally:
        writer.close()
    if render_error is not None:
        pending.unlink(missing_ok=True)
        raise render_error
    os.replace(pending, path)
    return recorder.action_count


def _write_control_timing(
    path: Path,
    control_hz: float,
    step_times: list[float],
    step_durations: list[float],
    viewer_durations: list[float],
    samples: list[SpaceMouseSnapshot],
) -> None:
    rows: list[dict[str, Any]] = []
    deadline_seconds = 1.1 / control_hz
    for index in range(min(len(step_times), len(step_durations), len(samples))):
        interval = None if index == 0 else step_times[index] - step_times[index - 1]
        sample_age = samples[index].sample_age_seconds
        rows.append(
            {
                "step": index,
                "step_monotonic": step_times[index],
                "control_interval_ms": None if interval is None else 1000.0 * interval,
                "sample_age_ms": None if sample_age is None else 1000.0 * sample_age,
                "env_step_ms": 1000.0 * step_durations[index],
                "viewer_sync_ms": 1000.0 * viewer_durations[index]
                if index < len(viewer_durations)
                else None,
                "deadline_miss": False if interval is None else interval > deadline_seconds,
                "stale": samples[index].stale,
            }
        )
    _atomic_write_csv(
        path,
        [
            "step",
            "step_monotonic",
            "control_interval_ms",
            "sample_age_ms",
            "env_step_ms",
            "viewer_sync_ms",
            "deadline_miss",
            "stale",
        ],
        rows,
    )


def _simulation_timing_summary(
    control_hz: float,
    step_times: list[float],
    step_durations: list[float],
    viewer_durations: list[float],
    samples: list[SpaceMouseSnapshot],
) -> dict[str, Any]:
    intervals = np.diff(np.asarray(step_times, dtype=np.float64)) if len(step_times) > 1 else []
    interval_ms = np.asarray(intervals) * 1000.0
    sample_age_ms = [
        1000.0 * sample.sample_age_seconds
        for sample in samples[: len(step_times)]
        if sample.sample_age_seconds is not None and not sample.stale
    ]
    deadline_seconds = 1.1 / control_hz
    misses = int(np.count_nonzero(np.asarray(intervals) > deadline_seconds))
    denominator = max(1, len(intervals))
    timing = {
        **direct.summarize_control_timing(step_times, control_hz),
        "control_interval_ms": _distribution(interval_ms),
        "input_sample_age_ms": _distribution(sample_age_ms),
        "env_step_ms": _distribution(1000.0 * np.asarray(step_durations)),
        "viewer_sync_ms": _distribution(1000.0 * np.asarray(viewer_durations)),
        "deadline_miss_count": misses,
        "deadline_miss_rate": misses / denominator,
    }
    bottlenecks: list[str] = []
    if (timing["input_sample_age_ms"]["p95"] or 0.0) > 50.0:
        bottlenecks.append("spacemouse_input")
    if (timing["env_step_ms"]["p95"] or 0.0) > 45.0:
        bottlenecks.append("mujoco_env_step")
    if (timing["viewer_sync_ms"]["p95"] or 0.0) > 5.0:
        bottlenecks.append("mujoco_viewer_sync")
    if timing["deadline_miss_rate"] > 0.01 and not bottlenecks:
        bottlenecks.append("scheduler_or_system_load")
    timing["suspected_bottlenecks"] = bottlenecks
    sample_age_p95 = timing["input_sample_age_ms"]["p95"]
    timing["acceptance"] = {
        "sample_age_p95_le_50ms": sample_age_p95 is not None and sample_age_p95 <= 50.0,
        "deadline_miss_rate_le_1pct": timing["deadline_miss_rate"] <= 0.01,
    }
    return timing


def run_simulation_test(
    controller: SpaceMouseInput,
    test_config: SpaceMouseTestConfig,
    eval_config: direct.EvalConfig,
    run_dir: Path,
) -> tuple[list[SpaceMouseSnapshot], dict[str, Any]]:
    if eval_config.control_hz != 20:
        raise ValueError(
            f"SpaceMouse test requires config.yaml control_hz=20, got {eval_config.control_hz}"
        )
    # Simulation-only testing must not depend on, import, or validate the VLA
    # checkout. Only LIBERO-X is needed to construct and step the environment.
    liberox_root = eval_config.liberox_root.expanduser().resolve()
    if not (liberox_root / "libero").is_dir():
        raise FileNotFoundError(
            f"LIBERO-X root looks invalid: {liberox_root} (missing libero/)"
        )
    if str(liberox_root) not in sys.path:
        sys.path.insert(0, str(liberox_root))
    runtime = direct.load_runtime()
    bddl_path, init_path = direct.resolve_task(
        liberox_root,
        eval_config.level,
        eval_config.task_name,
    )
    states = direct.load_initial_states(runtime, init_path)
    initial_state = states[0]
    task_description = str(runtime.parse_bddl_file(str(bddl_path))["language"])
    direct.prewarm_simulation_control(runtime, bddl_path, initial_state, eval_config)

    env = direct.make_env(
        runtime,
        bddl_path,
        eval_config.env_resolution,
        test_config.max_steps + 1,
        eval_config.control_hz,
        eval_config.seed,
        eval_config.video_camera,
        eval_config.video_width,
        eval_config.video_height,
        eval_config.headless,
    )
    timed_env = TimedEnvironment(env)
    recorder = TrajectoryRecorder(control_hz=20.0, capture_images=False)
    rate_limiter = direct.RealTimeControlLimiter(20.0, True)
    samples: list[SpaceMouseSnapshot] = []
    viewer_durations: list[float] = []
    error: str | None = None
    stopped_reason = "max_steps"
    success = False
    video_frames = 0
    trajectory_paths: dict[str, str] = {}
    started = time.monotonic()

    try:
        observation = direct.restore_state(env, initial_state)
        recorder.record_initial(timed_env, observation)
        direct.render_live_window(env)
        _countdown(test_config.countdown_seconds)
        LOGGER.info(
            "SpaceMouse control active: left=open, right=close, Ctrl+C or closing Viewer stops"
        )

        def manual_query(_step: int) -> np.ndarray:
            snapshot = controller.latest_snapshot()
            samples.append(snapshot)
            if snapshot.error is not None:
                raise RuntimeError(f"SpaceMouse reader failed: {snapshot.error}")
            return np.asarray(snapshot.action, dtype=np.float32)

        def after_transition(_observation: dict[str, Any], _step: int, _success: bool) -> None:
            viewer_started = time.monotonic()
            direct.render_live_window(env)
            viewer_durations.append(time.monotonic() - viewer_started)

        loop_result = run_control_loop(
            env=timed_env,
            recorder=recorder,
            initial_observation=observation,
            target_action_count=test_config.max_steps,
            rate_limiter=rate_limiter,
            action_source="spacemouse",
            manual_query=manual_query,
            stop_requested=lambda: not controller.latest_snapshot().connected
            or _viewer_is_closed(env),
            on_transition=after_transition,
            stop_on_success=True,
            horizon_reason="max_steps",
        )
        success = loop_result.success
        stopped_reason = loop_result.stopped_reason
    except KeyboardInterrupt:
        stopped_reason = "keyboard_interrupt"
        LOGGER.info("Keyboard interrupt received; saving the partial trajectory")
    except Exception as exc:
        error = f"{type(exc).__name__}: {exc}"
        stopped_reason = "error"
        LOGGER.exception("SpaceMouse simulation test failed; saving available diagnostics")
    finally:
        direct.close_native_mujoco_viewer(env)
        if test_config.save_video and recorder.action_count:
            try:
                video_frames = _render_agentview_video(
                    runtime,
                    env,
                    recorder,
                    run_dir / "spacemouse_agentview.mp4",
                    eval_config.video_fps,
                    eval_config.main_view_video_width,
                    eval_config.main_view_video_height,
                )
            except Exception as exc:
                LOGGER.exception("Failed to render the post-run SpaceMouse video")
                if error is None:
                    error = f"{type(exc).__name__}: failed to render video: {exc}"
                    stopped_reason = "error"
        direct.close_env(env)

    timing = _simulation_timing_summary(
        20.0,
        timed_env.step_started,
        timed_env.step_durations,
        viewer_durations,
        samples,
    )
    _write_control_timing(
        run_dir / "control_timing.csv",
        20.0,
        timed_env.step_started,
        timed_env.step_durations,
        viewer_durations,
        samples,
    )
    if recorder.state_count:
        metadata = {
            "kind": "spacemouse_simulation_test",
            "task": task_description,
            "level": eval_config.level,
            "task_name": eval_config.task_name,
            "seed": eval_config.seed,
            "success": success,
            "stopped_reason": stopped_reason,
            "error": error,
            "device_name": test_config.device_name,
            "axis_convention": test_config.axis_convention,
            "axis_order": list(test_config.axis_order),
            "axis_signs": list(test_config.axis_signs),
            "timing": timing,
        }
        trajectory_paths = save_trajectory_bundle(
            recorder,
            run_dir / "spacemouse_trajectory",
            metadata,
            save_observations=False,
            create_plot=test_config.trajectory_plot,
        )

    result = {
        "mode": "simulation",
        "task": task_description,
        "level": eval_config.level,
        "task_name": eval_config.task_name,
        "seed": eval_config.seed,
        "control_hz": 20,
        "success": success,
        "stopped_reason": stopped_reason,
        "error": error,
        "action_count": recorder.action_count,
        "state_count": recorder.state_count,
        "video_frame_count": video_frames,
        "wall_duration_seconds": time.monotonic() - started,
        "timing": timing,
        "artifacts": {
            **trajectory_paths,
            "agentview_video": str(run_dir / "spacemouse_agentview.mp4")
            if video_frames
            else None,
            "control_timing_csv": str(run_dir / "control_timing.csv"),
        },
    }
    return samples, result


def main() -> int:
    logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
    validate_command_line()
    test_config = load_spacemouse_config()
    eval_config = None
    if test_config.mode == "simulation":
        eval_config = direct.load_config(EVAL_CONFIG_PATH)
        direct.apply_runtime_environment(eval_config)
    run_dir = _create_run_dir(test_config)
    LOGGER.info("SpaceMouse config: %s", DEFAULT_SPACEMOUSE_CONFIG)
    LOGGER.info("output: %s", run_dir)

    controller = SpaceMouseInput(test_config)
    samples: list[SpaceMouseSnapshot] = []
    result: dict[str, Any] = {"mode": test_config.mode}
    calibration: dict[str, Any] | None = None
    error: str | None = None
    sample_started = time.monotonic()
    try:
        controller.start()
        LOGGER.info(
            "SpaceMouse opened; keep the cap untouched for %.1f seconds",
            test_config.neutral_calibration_seconds,
        )
        calibration = controller.calibrate_neutral()
        LOGGER.info("Neutral calibration complete: bias=%s", calibration["bias"])
        if test_config.mode == "device":
            _countdown(test_config.countdown_seconds)
            samples, result = run_device_test(controller, test_config)
        else:
            assert eval_config is not None
            samples, result = run_simulation_test(
                controller,
                test_config,
                eval_config,
                run_dir,
            )
    except KeyboardInterrupt:
        result = {"mode": test_config.mode, "stopped_reason": "keyboard_interrupt"}
    except Exception as exc:
        error = f"{type(exc).__name__}: {exc}"
        result = {"mode": test_config.mode, "stopped_reason": "error", "error": error}
        LOGGER.exception("SpaceMouse test failed")
    finally:
        controller.stop()

    device_statistics = _device_statistics(controller, samples)
    if test_config.mode == "device" and not result.get("error"):
        sample_age_p95 = device_statistics["sample_age_ms"]["p95"]
        result["acceptance"] = {
            "all_axes_and_buttons_observed": bool(result["functional_check_complete"]),
            "sample_age_p95_le_50ms": sample_age_p95 is not None and sample_age_p95 <= 50.0,
        }
        result["acceptance_passed"] = all(result["acceptance"].values())
    elif test_config.mode == "simulation" and not result.get("error"):
        result["acceptance_passed"] = all(result["timing"]["acceptance"].values())
    device_summary = {
        "config": _config_record(test_config),
        "calibration": calibration,
        "device": device_statistics,
        "error": error or result.get("error"),
    }
    _atomic_write_json(run_dir / "device_summary.json", device_summary)
    _atomic_write_csv(
        run_dir / "spacemouse_samples.csv",
        SAMPLE_FIELDS,
        (_snapshot_row(sample, sample_started, index) for index, sample in enumerate(samples)),
    )
    summary = {
        **result,
        "config_path": str(DEFAULT_SPACEMOUSE_CONFIG),
        "eval_config_path": str(EVAL_CONFIG_PATH),
        "output_dir": str(run_dir),
        "device_summary": str(run_dir / "device_summary.json"),
        "spacemouse_samples": str(run_dir / "spacemouse_samples.csv"),
    }
    _atomic_write_json(run_dir / "summary.json", summary)
    LOGGER.info("summary: %s", run_dir / "summary.json")
    if summary.get("error"):
        return 1
    return 0 if summary.get("acceptance_passed", False) else 2


if __name__ == "__main__":
    raise SystemExit(main())
