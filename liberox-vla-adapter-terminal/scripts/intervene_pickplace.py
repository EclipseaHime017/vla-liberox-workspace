#!/usr/bin/env python3
"""Branch a recorded LIBERO-X rollout and continue with VLA inference or human control."""

from __future__ import annotations

import argparse
import json
import logging
import math
import socket
from collections import deque
from dataclasses import dataclass, fields, replace
from datetime import datetime
from pathlib import Path
from typing import Any

import numpy as np
import yaml

import eval_pickplace_direct as direct
from simulation_core import run_control_loop
from trajectory_utils import (
    TrajectoryRecorder,
    load_trajectory,
    plot_action_comparison,
    save_trajectory_bundle,
)


WORKSPACE_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_INTERVENTION_CONFIG_PATH = WORKSPACE_ROOT / "configs" / "intervention_config.yaml"
VALID_CONTROL_MODES = frozenset({"policy", "manual_stdin", "manual_jsonl", "manual_udp"})
LOGGER = logging.getLogger("liberox_intervention")


@dataclass(frozen=True)
class InterventionConfig:
    source_trajectory: Path
    resume_step: int | None
    resume_time_seconds: float | None
    control_mode: str
    open_loop_steps: int
    manual_action_file: Path | None
    udp_bind_host: str
    udp_bind_port: int
    controller_timeout_seconds: float
    output_root: Path
    save_video: bool
    save_latest_frame: bool


def validate_command_line() -> None:
    parser = argparse.ArgumentParser(
        description=__doc__,
        epilog=f"Configuration is always loaded from: {DEFAULT_INTERVENTION_CONFIG_PATH}",
    )
    parser.parse_args()


def _require_string(config: dict[str, Any], key: str) -> str:
    value = config[key]
    if not isinstance(value, str) or not value.strip():
        raise TypeError(f"Intervention key '{key}' must be a non-empty string")
    return value


def _require_int(config: dict[str, Any], key: str) -> int:
    value = config[key]
    if isinstance(value, bool) or not isinstance(value, int):
        raise TypeError(f"Intervention key '{key}' must be an integer")
    return value


def _require_bool(config: dict[str, Any], key: str) -> bool:
    value = config[key]
    if not isinstance(value, bool):
        raise TypeError(f"Intervention key '{key}' must be true or false")
    return value


def _resolve_path(value: str, config_dir: Path) -> Path:
    path = Path(value).expanduser()
    if not path.is_absolute():
        path = config_dir / path
    return path.resolve()


def load_intervention_config(config_path: Path) -> InterventionConfig:
    if not config_path.is_file():
        raise FileNotFoundError(f"Intervention configuration not found: {config_path}")
    try:
        raw = yaml.load(config_path.read_text(encoding="utf-8"), Loader=direct.UniqueKeyLoader)
    except yaml.YAMLError as exc:
        raise ValueError(f"Invalid YAML in {config_path}: {exc}") from exc
    if not isinstance(raw, dict) or any(not isinstance(key, str) for key in raw):
        raise TypeError("Intervention configuration root must be a string-keyed mapping")

    expected = {field.name for field in fields(InterventionConfig)}
    actual = set(raw)
    missing = sorted(expected - actual)
    unknown = sorted(actual - expected)
    if missing:
        raise ValueError(f"Missing intervention configuration keys: {missing}")
    if unknown:
        raise ValueError(f"Unknown intervention configuration keys: {unknown}")

    resume_step = raw["resume_step"]
    if resume_step is not None and (isinstance(resume_step, bool) or not isinstance(resume_step, int)):
        raise TypeError("Intervention key 'resume_step' must be an integer or null")
    resume_time = raw["resume_time_seconds"]
    if resume_time is not None and (
        isinstance(resume_time, bool) or not isinstance(resume_time, (int, float))
    ):
        raise TypeError("Intervention key 'resume_time_seconds' must be a number or null")
    if (resume_step is None) == (resume_time is None):
        raise ValueError("Set exactly one of resume_step or resume_time_seconds")

    manual_action_file = raw["manual_action_file"]
    if manual_action_file is not None and (
        not isinstance(manual_action_file, str) or not manual_action_file.strip()
    ):
        raise TypeError("Intervention key 'manual_action_file' must be a non-empty string or null")

    timeout = raw["controller_timeout_seconds"]
    if isinstance(timeout, bool) or not isinstance(timeout, (int, float)):
        raise TypeError("Intervention key 'controller_timeout_seconds' must be a number")

    config_dir = config_path.parent
    config = InterventionConfig(
        source_trajectory=_resolve_path(_require_string(raw, "source_trajectory"), config_dir),
        resume_step=resume_step,
        resume_time_seconds=None if resume_time is None else float(resume_time),
        control_mode=_require_string(raw, "control_mode"),
        open_loop_steps=_require_int(raw, "open_loop_steps"),
        manual_action_file=None
        if manual_action_file is None
        else _resolve_path(manual_action_file, config_dir),
        udp_bind_host=_require_string(raw, "udp_bind_host"),
        udp_bind_port=_require_int(raw, "udp_bind_port"),
        controller_timeout_seconds=float(timeout),
        output_root=_resolve_path(_require_string(raw, "output_root"), config_dir),
        save_video=_require_bool(raw, "save_video"),
        save_latest_frame=_require_bool(raw, "save_latest_frame"),
    )

    if config.control_mode not in VALID_CONTROL_MODES:
        raise ValueError(f"control_mode must be one of {sorted(VALID_CONTROL_MODES)}")
    if config.resume_step is not None and config.resume_step < 0:
        raise ValueError("resume_step must be >= 0")
    if config.resume_time_seconds is not None and config.resume_time_seconds < 0:
        raise ValueError("resume_time_seconds must be >= 0")
    if not 1 <= config.open_loop_steps <= 8:
        raise ValueError("open_loop_steps must be in [1, 8]")
    if not 1 <= config.udp_bind_port <= 65535:
        raise ValueError("udp_bind_port must be in [1, 65535]")
    if config.controller_timeout_seconds <= 0:
        raise ValueError("controller_timeout_seconds must be > 0")
    if config.control_mode == "manual_jsonl":
        if config.manual_action_file is None:
            raise ValueError("manual_jsonl mode requires manual_action_file")
        if not config.manual_action_file.is_file():
            raise FileNotFoundError(f"Manual action file not found: {config.manual_action_file}")
    return config


def _validated_manual_action(value: Any) -> np.ndarray:
    action = np.asarray(value, dtype=np.float32)
    if action.shape != (7,):
        raise ValueError(f"Manual action must contain 7 values, got shape {action.shape}")
    if not np.isfinite(action).all():
        raise ValueError("Manual action contains NaN or Inf")
    if np.any(action < -1.0) or np.any(action > 1.0):
        raise ValueError("Manual actions use normalized OSC_POSE values and must stay within [-1, 1]")
    return action


def _decode_action_message(message: str) -> tuple[np.ndarray | None, int]:
    text = message.strip()
    if not text:
        raise ValueError("Empty controller message")
    if text.lower() in {"stop", "quit", "exit"}:
        return None, 1
    try:
        value = json.loads(text)
    except json.JSONDecodeError:
        value = [float(token) for token in text.replace(",", " ").split()]
    repeat = 1
    if isinstance(value, dict):
        if str(value.get("command", "")).lower() in {"stop", "quit", "exit"}:
            return None, 1
        repeat = value.get("repeat", 1)
        if isinstance(repeat, bool) or not isinstance(repeat, int) or not 1 <= repeat <= 10000:
            raise ValueError("Controller repeat must be an integer in [1, 10000]")
        if "action" not in value:
            raise ValueError("Controller object must contain an 'action' array")
        value = value["action"]
    return _validated_manual_action(value), repeat


class ManualController:
    def next_action(self, state: dict[str, Any]) -> np.ndarray | None:
        raise NotImplementedError

    def after_step(self, state: dict[str, Any]) -> None:
        del state

    def close(self) -> None:
        pass


class JsonlController(ManualController):
    def __init__(self, path: Path):
        self.actions: deque[np.ndarray] = deque()
        for line_number, line in enumerate(path.read_text(encoding="utf-8").splitlines(), start=1):
            if not line.strip():
                continue
            try:
                action, repeat = _decode_action_message(line)
            except Exception as exc:
                raise ValueError(f"Invalid manual action at {path}:{line_number}: {exc}") from exc
            if action is None:
                break
            for _ in range(repeat):
                self.actions.append(action.copy())
        if not self.actions:
            raise ValueError(f"Manual action file contains no actions: {path}")

    def next_action(self, state: dict[str, Any]) -> np.ndarray | None:
        del state
        return self.actions.popleft() if self.actions else None


class StdinController(ManualController):
    def __init__(self):
        self.pending: deque[np.ndarray] = deque()
        LOGGER.info(
            "stdin controller: enter 7 normalized values, JSON {'action': [...], 'repeat': N}, or 'stop'"
        )

    def next_action(self, state: dict[str, Any]) -> np.ndarray | None:
        if self.pending:
            return self.pending.popleft()
        LOGGER.info("manual state: %s", json.dumps(state, ensure_ascii=False))
        try:
            message = input("manual action> ")
        except EOFError:
            return None
        action, repeat = _decode_action_message(message)
        if action is None:
            return None
        for _ in range(repeat - 1):
            self.pending.append(action.copy())
        return action


class UdpController(ManualController):
    def __init__(self, host: str, port: int, timeout_seconds: float):
        self.socket = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
        self.socket.bind((host, port))
        self.socket.settimeout(timeout_seconds)
        self.pending: deque[np.ndarray] = deque()
        self.peer: tuple[str, int] | None = None
        LOGGER.info("UDP controller listening on %s:%d", host, port)

    def next_action(self, state: dict[str, Any]) -> np.ndarray | None:
        if self.pending:
            return self.pending.popleft()
        while True:
            try:
                payload, self.peer = self.socket.recvfrom(65535)
            except socket.timeout as exc:
                raise TimeoutError("Timed out waiting for a UDP controller action") from exc
            try:
                action, repeat = _decode_action_message(payload.decode("utf-8"))
            except Exception as exc:
                if self.peer is not None:
                    self.socket.sendto(
                        json.dumps({"ok": False, "error": str(exc)}).encode("utf-8"), self.peer
                    )
                continue
            if action is None:
                return None
            for _ in range(repeat - 1):
                self.pending.append(action.copy())
            return action

    def after_step(self, state: dict[str, Any]) -> None:
        if self.peer is not None:
            self.socket.sendto(
                json.dumps({"ok": True, **state}, ensure_ascii=False).encode("utf-8"), self.peer
            )

    def close(self) -> None:
        self.socket.close()


def make_manual_controller(config: InterventionConfig) -> ManualController:
    if config.control_mode == "manual_stdin":
        return StdinController()
    if config.control_mode == "manual_jsonl":
        assert config.manual_action_file is not None
        return JsonlController(config.manual_action_file)
    if config.control_mode == "manual_udp":
        return UdpController(
            config.udp_bind_host,
            config.udp_bind_port,
            config.controller_timeout_seconds,
        )
    raise ValueError(f"No manual controller for mode {config.control_mode}")


def resolve_resume_step(
    config: InterventionConfig, trajectory: dict[str, np.ndarray], metadata: dict[str, Any]
) -> int:
    action_count = len(trajectory["env_action"])
    if config.resume_step is not None:
        step = config.resume_step
    else:
        control_hz = float(metadata["control_hz"])
        assert config.resume_time_seconds is not None
        step = int(round(config.resume_time_seconds * control_hz))
        LOGGER.info(
            "resume_time_seconds=%.6f maps to nearest state step %d (%.6f s at %.3f Hz)",
            config.resume_time_seconds,
            step,
            step / control_hz,
            control_hz,
        )
    if not 0 <= step < action_count:
        raise ValueError(f"Resume step {step} is outside source action range [0, {action_count - 1}]")
    return step


def state_payload(
    recorder: TrajectoryRecorder, latest_frame: Path | None, success: bool
) -> dict[str, Any]:
    step = recorder.action_count
    return {
        "step": step,
        "time_seconds": step / recorder.control_hz,
        "eef_6d": np.concatenate(
            (recorder.eef_positions[-1], recorder.eef_axis_angles[-1]), axis=0
        ).tolist(),
        "gripper_qpos": recorder.gripper_qpos[-1].tolist(),
        "success": success,
        "latest_frame": "" if latest_frame is None else str(latest_frame),
    }


def _make_run_directory(config: InterventionConfig, resume_step: int) -> Path:
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S_%f")
    run_dir = config.output_root / (
        f"{config.source_trajectory.stem}_step_{resume_step:04d}_{config.control_mode}_{timestamp}"
    )
    run_dir.mkdir(parents=True, exist_ok=False)
    return run_dir


def _validate_source_against_eval(metadata: dict[str, Any], eval_config: direct.EvalConfig) -> None:
    for key, current_value in (("task_name", eval_config.task_name), ("level", eval_config.level)):
        source_value = metadata.get(key)
        if source_value != current_value:
            raise ValueError(
                f"Source trajectory {key}={source_value!r} does not match config.yaml value {current_value!r}"
            )
    source_checkpoint = metadata.get("checkpoint")
    if source_checkpoint != str(eval_config.checkpoint):
        LOGGER.warning(
            "Branch policy checkpoint differs from source: source=%s current=%s",
            source_checkpoint,
            eval_config.checkpoint,
        )


def main() -> int:
    logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
    validate_command_line()
    intervention_path = DEFAULT_INTERVENTION_CONFIG_PATH.resolve()
    intervention = load_intervention_config(intervention_path)
    eval_config = direct.load_config(direct.DEFAULT_CONFIG_PATH.resolve())
    direct.apply_runtime_environment(eval_config)

    observation_path = intervention.source_trajectory.with_name(
        f"{intervention.source_trajectory.stem}_observations.npz"
    )
    source_has_observations = observation_path.is_file()
    trajectory, source_metadata = load_trajectory(intervention.source_trajectory)
    _validate_source_against_eval(source_metadata, eval_config)
    resume_step = resolve_resume_step(intervention, trajectory, source_metadata)
    source_action_count = len(trajectory["env_action"])
    target_total_steps = int(
        source_metadata.get("target_total_steps", source_action_count)
    )
    if target_total_steps <= resume_step:
        raise ValueError(
            "Source target_total_steps must be greater than resume_step: "
            f"target_total_steps={target_total_steps}, resume_step={resume_step}"
        )
    remaining_steps = target_total_steps - resume_step
    control_hz = float(source_metadata["control_hz"])
    recorder = TrajectoryRecorder.from_prefix(
        trajectory,
        resume_step=resume_step,
        control_hz=control_hz,
    )
    run_dir = _make_run_directory(intervention, resume_step)
    latest_frame_path = (
        run_dir / "latest_frame.png"
        if intervention.save_latest_frame and intervention.control_mode != "policy"
        else None
    )

    _, liberox_root = direct.add_repo_paths(eval_config.vla_root, eval_config.liberox_root)
    runtime = direct.load_runtime()
    bddl_path, _ = direct.resolve_task(liberox_root, eval_config.level, eval_config.task_name)
    task_description = str(runtime.parse_bddl_file(str(bddl_path))["language"])
    source_seed = int(source_metadata.get("seed", eval_config.seed))
    direct.prewarm_simulation_control(
        runtime,
        bddl_path,
        trajectory["sim_state"][resume_step],
        eval_config,
    )
    env = direct.make_env(
        runtime,
        bddl_path,
        eval_config.env_resolution,
        remaining_steps + 1,
        eval_config.control_hz,
        source_seed,
        eval_config.video_camera,
        eval_config.video_width,
        eval_config.video_height,
        eval_config.headless,
    )
    env_control_hz = float(getattr(env.env, "control_freq", eval_config.control_hz))
    if not math.isclose(env_control_hz, control_hz):
        direct.close_env(env)
        raise ValueError(
            f"Source control_hz={control_hz} does not match replay environment {env_control_hz}"
        )

    policy_cfg = None
    components = None
    controller: ManualController | None = None
    video_path = run_dir / "intervention.mp4"
    main_view_video_path = run_dir / "intervention_agentview_hd.mp4"
    rate_limiter = direct.RealTimeControlLimiter(
        env_control_hz,
        eval_config.realtime_control,
    )
    video_frame_count = 0

    success = False
    error: str | None = None
    stopped_reason = "max_steps"
    policy_queries = 0
    branch_steps = 0
    control_step_times: list[float] = []
    try:
        observation = direct.restore_state(env, trajectory["sim_state"][resume_step])
        observation = direct.capture_camera_observations(env, observation)
        direct.validate_observation(observation)
        direct.render_live_window(env)
        restored_state = np.asarray(env.get_sim_state())
        state_error = float(np.max(np.abs(restored_state - trajectory["sim_state"][resume_step])))
        if state_error > 1e-9:
            raise RuntimeError(f"MuJoCo state restoration mismatch: max_abs_error={state_error}")
        LOGGER.info(
            "restored source=%s step=%d time=%.3fs max_state_error=%.3g",
            intervention.source_trajectory,
            resume_step,
            resume_step / control_hz,
            state_error,
        )

        if intervention.control_mode == "policy":
            runtime = direct.load_policy_runtime(runtime)
            branch_eval_config = replace(
                eval_config,
                open_loop_steps=intervention.open_loop_steps,
                trials=1,
            )
            policy_cfg, components = direct.build_model(runtime, branch_eval_config)
        else:
            controller = make_manual_controller(intervention)

        if latest_frame_path is not None:
            observation = direct.capture_camera_observations(env, observation)
            runtime.imageio.imwrite(
                latest_frame_path,
                direct.video_frame_from_observation(
                    observation,
                    eval_config.video_camera,
                    eval_config.video_width,
                    eval_config.video_height,
                ),
            )

        def after_transition(
            current_observation: dict[str, Any],
            _step: int,
            current_success: bool,
        ) -> None:
            direct.render_live_window(env)
            if latest_frame_path is not None:
                current_observation = direct.capture_camera_observations(
                    env,
                    current_observation,
                )
                runtime.imageio.imwrite(
                    latest_frame_path,
                    direct.video_frame_from_observation(
                        current_observation,
                        eval_config.video_camera,
                        eval_config.video_width,
                        eval_config.video_height,
                    ),
                )
            if controller is not None:
                controller.after_step(
                    state_payload(recorder, latest_frame_path, current_success)
                )

        if intervention.control_mode == "policy":
            assert policy_cfg is not None and components is not None

            def query_policy(current_observation: dict[str, Any], _step: int):
                current_observation = direct.capture_camera_observations(
                    env,
                    current_observation,
                )
                return current_observation, direct.checked_action_chunk(
                    runtime,
                    policy_cfg,
                    components,
                    current_observation,
                    task_description,
                    eval_config.disabled_policy_cameras,
                )

            loop_result = run_control_loop(
                env=env,
                recorder=recorder,
                initial_observation=observation,
                target_action_count=target_total_steps,
                rate_limiter=rate_limiter,
                action_source="policy_requery",
                open_loop_steps=intervention.open_loop_steps,
                policy_query=query_policy,
                policy_action_transform=lambda action: runtime.process_action(
                    action,
                    policy_cfg.model_family,
                ),
                on_transition=after_transition,
                stop_on_success=False,
                horizon_reason="max_steps",
            )
        else:
            assert controller is not None
            loop_result = run_control_loop(
                env=env,
                recorder=recorder,
                initial_observation=observation,
                target_action_count=target_total_steps,
                rate_limiter=rate_limiter,
                action_source="human",
                manual_query=lambda _step: controller.next_action(
                    state_payload(
                        recorder,
                        latest_frame_path,
                        any(recorder.dones),
                    )
                ),
                on_transition=after_transition,
                stop_on_success=False,
                horizon_reason="max_steps",
            )
        success = loop_result.success
        branch_steps = loop_result.executed_steps
        policy_queries = loop_result.policy_queries
        stopped_reason = loop_result.stopped_reason
        control_step_times = loop_result.control_step_times
    except Exception as exc:
        error = f"{type(exc).__name__}: {exc}"
        stopped_reason = "error"
        LOGGER.exception("Intervention branch failed")
    finally:
        if controller is not None:
            controller.close()
        direct.close_native_mujoco_viewer(env)
        should_postprocess = recorder.state_count and (
            intervention.save_video or source_has_observations
        )
        if should_postprocess:
            try:
                LOGGER.info(
                    "Post-processing %d intervention states (video=%s observations=%s)",
                    recorder.state_count,
                    intervention.save_video,
                    source_has_observations,
                )
                video_frame_count = direct.postprocess_recorded_trajectory(
                    runtime=runtime,
                    env=env,
                    recorder=recorder,
                    save_observations=source_has_observations,
                    combined_path=video_path if intervention.save_video else None,
                    main_view_path=main_view_video_path if intervention.save_video else None,
                    fps=eval_config.video_fps,
                    video_camera=eval_config.video_camera,
                    video_width=eval_config.video_width,
                    video_height=eval_config.video_height,
                    main_view_width=eval_config.main_view_video_width,
                    main_view_height=eval_config.main_view_video_height,
                )
            except Exception as exc:
                if error is None:
                    error = f"{type(exc).__name__}: failed to render videos: {exc}"
                    stopped_reason = "error"
                LOGGER.exception("Intervention post-control video rendering failed")
        direct.close_env(env)
        if latest_frame_path is not None:
            latest_frame_path.unlink(missing_ok=True)

    video_value = str(video_path) if video_frame_count else ""
    main_view_video_value = str(main_view_video_path) if video_frame_count else ""
    timing = direct.summarize_control_timing(control_step_times, env_control_hz)
    maximum_interval = timing["control_interval_max_seconds"]
    if maximum_interval is not None and maximum_interval > (1.1 / env_control_hz):
        LOGGER.warning(
            "Intervention missed the %.1f Hz soft-real-time deadline; maximum control "
            "interval was %.4f s. Inference, the manual controller, or system load "
            "exceeded the %.4f s budget.",
            env_control_hz,
            maximum_interval,
            1.0 / env_control_hz,
        )

    branch_metadata = {
        "intervention_config": str(intervention_path),
        "source_trajectory": str(intervention.source_trajectory),
        "source_episode": source_metadata.get("episode"),
        "source_success": source_metadata.get("success"),
        "source_resume_step": resume_step,
        "source_resume_time_seconds": resume_step / control_hz,
        "source_action_count": source_action_count,
        "target_total_steps": target_total_steps,
        "planned_branch_steps": remaining_steps,
        "control_mode": intervention.control_mode,
        "open_loop_steps": intervention.open_loop_steps,
        "control_hz": eval_config.control_hz,
        "realtime_control": eval_config.realtime_control,
        "headless": eval_config.headless,
        "viewer_backend": None
        if eval_config.headless
        else "mujoco_native_passive",
        "video_camera": eval_config.video_camera,
        "video_width": eval_config.video_width,
        "video_height": eval_config.video_height,
        "main_view_video_camera": direct.MAIN_VIEW_CAMERA,
        "main_view_video_width": eval_config.main_view_video_width,
        "main_view_video_height": eval_config.main_view_video_height,
        "video_fps": eval_config.video_fps,
        "video_frames": video_frame_count,
        "task": task_description,
        "task_name": eval_config.task_name,
        "level": eval_config.level,
        "bddl": str(bddl_path),
        "checkpoint": str(eval_config.checkpoint) if intervention.control_mode == "policy" else None,
        "resolved_stats_key": None
        if policy_cfg is None
        else str(policy_cfg.unnorm_key),
        "branch_steps": branch_steps,
        "total_steps": recorder.action_count,
        "total_length_preserved": recorder.action_count == source_action_count,
        "policy_queries": policy_queries,
        "success": success,
        "stopped_reason": stopped_reason,
        "error": error,
        "video": video_value,
        "main_view_video": main_view_video_value,
        **timing,
    }
    comparison_paths: dict[str, str] = {}
    if recorder.action_count > resume_step:
        comparison_paths = plot_action_comparison(
            trajectory,
            recorder.arrays(),
            resume_step,
            control_hz,
            run_dir / "trajectory_action_comparison",
            branch_label="re-inference"
            if intervention.control_mode == "policy"
            else "human takeover",
        )
        branch_metadata.update(comparison_paths)
    trajectory_paths = save_trajectory_bundle(
        recorder,
        run_dir / "trajectory",
        branch_metadata,
        save_observations=source_has_observations,
        create_plot=False,
        intervention_step=resume_step,
    )
    trajectory_paths.update(comparison_paths)
    summary = {**branch_metadata, **trajectory_paths, "output": str(run_dir)}
    (run_dir / "summary.json").write_text(
        json.dumps(summary, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    LOGGER.info("intervention summary: %s", summary)
    return 1 if error is not None else 0


if __name__ == "__main__":
    raise SystemExit(main())
