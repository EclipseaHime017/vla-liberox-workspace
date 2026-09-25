#!/usr/bin/env python3
"""Interactive FACTR calibration, explicitly enabled gravity and no-VLA simulation.

The fixed config lives in workspace/configs/factr_test_config.yaml. Device mode
does not import the simulator or policy stack. Startup verifies/retains torque OFF;
calibration is read-only. Only explicit g enables physical compensation. No test recordings are written.
"""
from __future__ import annotations

import argparse
import faulthandler
import json
import logging
import select
from pathlib import Path
import sys
import time

import numpy as np

PROJECT_ROOT = Path(__file__).resolve().parents[1]
WORKSPACE_ROOT = PROJECT_ROOT.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from backend.app.devices.factr import DEFAULT_FACTR_CONFIG, load_factr_config

LOGGER = logging.getLogger("factr_test")
OWNER = "factr-runtime-test"


class CalibrationCancelled(Exception):
    pass


class ControllerUnavailable(Exception):
    pass


def _check_snapshot(snapshot) -> None:
    if snapshot.error:
        raise ControllerUnavailable(f"FACTR read error: {snapshot.error}")
    if not snapshot.connected:
        raise ControllerUnavailable("FACTR disconnected; stopping manual control")
    if snapshot.stale:
        age = snapshot.sample_age_seconds
        label = "unknown" if age is None else f"{age*1000:.1f} ms"
        raise ControllerUnavailable(f"FACTR sample is stale (age={label}, sequence={snapshot.sequence}); stopping manual control")


def _check_menu_snapshot(service, *, timeout=1., clock=time.monotonic, sleep=time.sleep):
    """Bounded resampling in passive standby only; motion still fails immediately."""
    deadline = clock()+timeout
    sample = service.latest_snapshot()
    if sample.connected and sample.stale and not sample.error:
        status = service.status()
        if not status.get("gravity_enabled") and status.get("state") != "ARMED":
            LOGGER.info("Waiting for a fresh FACTR sample before returning to the menu")
            while sample.connected and sample.stale and not sample.error and clock() < deadline:
                sleep(.01)
                sample = service.latest_snapshot()
    _check_snapshot(sample)


def _read_terminal_input(prompt, service, *, stream=None, select_fn=None):
    """Keep observing faults while waiting for keyboard input; never touches IO bus."""
    stream = sys.stdin if stream is None else stream
    select_fn = select.select if select_fn is None else select_fn
    print(prompt, end="", flush=True)
    while True:
        _check_menu_snapshot(service)
        readable, _, _ = select_fn([stream], [], [], .1)
        if readable:
            line = stream.readline()
            if line == "":
                raise EOFError("FACTR terminal input closed (EOF)")
            # Check once more before dispatching an operator command.
            _check_menu_snapshot(service)
            return line.rstrip("\r\n")




def calibrate_interactively(service, config, *, input_fn=input, **_kwargs):
    if service.status().get("gravity_enabled"):
        raise RuntimeError("Support the arm and disable gravity before calibrating")
    if input_fn("Place the WHOLE ARM in the official Figure 1 reference and RELEASE the trigger. "
                "Enter=calibrate once, q=cancel: ").strip().lower() not in {"", "y", "yes"}:
        raise CalibrationCancelled("Calibration cancelled")
    return service.calibrate()


def offer_gravity(service):
    # Reached only after an explicit g, never as a side effect of preparing/arming.
    LOGGER.warning("Official compensation ON requested. Support the leader; faults or exit remove motor support.")
    service.enable_gravity()
    LOGGER.warning("Official compensation ON (gravity + friction + null-space + limit barrier); inspect i for timing/settings")


def _countdown(seconds: int) -> None:
    for remaining in range(seconds, 0, -1):
        LOGGER.info("Joint following starts in %d; simulation remains paused", remaining)
        time.sleep(1)


def run_device_test(service, config, *, clock=time.monotonic,
                    sleep=time.sleep) -> dict:
    started = clock()
    sample_count = 0
    service.arm(OWNER, config.translation_gain, config.rotation_gain, gripper=-1)
    LOGGER.info("Device-only observation: move each joint/trigger; gravity_enabled=%s",
                service.status().get("gravity_enabled", False))
    while clock()-started < config.test_duration_seconds:
        snapshot = service.snapshot(OWNER)
        sample_count += 1
        _check_snapshot(snapshot)
        print("\r" + " ".join(f"q{i+1}={q:+.3f}" for i, q in enumerate(snapshot.joint_positions))
              + f" gripper={snapshot.gripper_fraction} age={snapshot.sample_age_seconds*1000:.1f} ms    ",
              end="", flush=True)
        sleep(0.05)
    print()
    return {"mode": "device", "stopped_reason": "duration", "sample_count": sample_count,
            "wall_duration_seconds": clock()-started,
            "physical_acceptance": "Operator must verify joint directions and gripper behavior"}


class _StepCounter:
    """Control-loop adapter: count steps only; no states, images or actions stored."""
    def __init__(self):
        self.action_count = 0
        self.state_count = 0
        self.success = False
        self.dones = ()

    def record_initial(self, _env, _observation):
        self.state_count = 1

    def record_transition(self, _env, _observation, _raw, _action, _reward, done, **_kwargs):
        self.action_count += 1
        self.state_count += 1
        self.success = self.success or bool(done)
        self.dones = (self.success,)


def run_simulation_test(service, config, *, on_prepared=None) -> dict:
    # Keep all simulator imports behind the explicit mode boundary.
    import eval_pickplace_direct as direct
    from backend.app.devices.factr_joint_control import JointFollower, align_for_takeover

    eval_config = direct.load_config(WORKSPACE_ROOT / "configs/config.yaml")
    if eval_config.control_hz != 20:
        raise ValueError("FACTR simulation requires config.yaml control_hz=20")
    direct.apply_runtime_environment(eval_config)
    liberox_root = eval_config.liberox_root.expanduser().resolve()
    if not (liberox_root / "libero").is_dir():
        raise FileNotFoundError(f"LIBERO-X root is invalid: {liberox_root}")
    if str(liberox_root) not in sys.path:
        sys.path.insert(0, str(liberox_root))
    runtime = direct.load_runtime()  # Not load_policy_runtime / build_model.
    bddl, init_path = direct.resolve_task(liberox_root, eval_config.level, eval_config.task_name)
    initial_state = direct.load_initial_states(runtime, init_path)[0]
    task = str(runtime.parse_bddl_file(str(bddl))["language"])
    LOGGER.info("Prewarming current task controller (no VLA): %s", task)
    direct.prewarm_simulation_control(runtime, bddl, initial_state, eval_config)
    env = direct.make_env(runtime, bddl, eval_config.env_resolution, config.max_steps+1,
                         20, eval_config.seed, eval_config.video_camera,
                         eval_config.video_width, eval_config.video_height, eval_config.headless)
    recorder = _StepCounter()
    started = time.monotonic()
    result = {"mode": "simulation", "task": task, "level": eval_config.level,
              "task_name": eval_config.task_name, "seed": eval_config.seed,
              "init_state_index": 0, "control_hz": 20, "manual_source": "factr",
              "stopped_reason": "max_steps", "success": False, "error": None}
    try:
        observation = direct.restore_state(env, initial_state)
        recorder.record_initial(env, observation)
        direct.render_live_window(env)
        follower = JointFollower(env)
        if on_prepared is not None:
            on_prepared()
        if not service.status().get("gravity_enabled"):
            raise ControllerUnavailable("Enable gravity compensation before joint alignment")
        LOGGER.warning("Simulation is paused. Leader will slowly align to follower; keep the workspace clear.")
        def viewer_closed():
            viewer = getattr(env, direct.NATIVE_VIEWER_ATTRIBUTE, None)
            return viewer is not None and not viewer.is_running()
        if not align_for_takeover(service, OWNER, follower.joints(), viewer_closed,
                                  lambda _: direct.render_live_window(env)):
            result["stopped_reason"] = "alignment_cancelled"
            return result
        _countdown(config.countdown_seconds)
        service.finish_alignment(OWNER)
        sample = service.arm(OWNER, config.translation_gain, config.rotation_gain, gripper=-1)
        _check_snapshot(sample)
        follower.check_aligned(sample.joint_positions)
        LOGGER.info("FACTR control active: trigger=open/close; Ctrl+C or closing Viewer stops")

        deadline = time.monotonic()
        while recorder.action_count < config.max_steps and not viewer_closed():
            time.sleep(max(0., deadline-time.monotonic()))
            snapshot = service.snapshot(OWNER)
            _check_snapshot(snapshot)
            observation, reward, done, _ = follower.step(snapshot.joint_positions, snapshot.gripper_command)
            recorder.record_transition(env, observation, None, None, reward, done)
            direct.render_live_window(env)
            deadline = max(deadline+.05, time.monotonic())
        result.update(success=recorder.success, stopped_reason="viewer_closed" if viewer_closed() else "max_steps")
    except KeyboardInterrupt:
        result["stopped_reason"] = "keyboard_interrupt"
        raise  # Exit the entire CLI, not just the current simulation.
    except CalibrationCancelled:
        raise
    except ControllerUnavailable as exc:
        result.update(stopped_reason="controller_unavailable", error=str(exc))
    except Exception as exc:
        LOGGER.exception("FACTR simulation stopped (no recording)")
        result.update(stopped_reason="error", error=f"{type(exc).__name__}: {exc}")
    finally:
        if result.get("error") or result.get("cleanup_error") or result["stopped_reason"] == "keyboard_interrupt":
            try:
                service.close()  # Remove physical support before viewer cleanup.
            except Exception as exc:
                result["cleanup_error"] = f"Motor shutdown not verified: {exc}"
        try:
            if not result.get("error") and result["stopped_reason"] != "keyboard_interrupt":
                service.finish_alignment(OWNER)
            service.disarm(OWNER)
        except Exception as exc:
            result["cleanup_error"] = f"Disarm failed: {exc}"
        direct.close_native_mujoco_viewer(env)
        result.update(action_count=recorder.action_count, state_count=recorder.state_count,
                      success=bool(any(recorder.dones)), wall_duration_seconds=time.monotonic()-started)
        direct.close_env(env)
    return result


def run_test(config, *, service_factory=None, input_fn=None) -> int:
    if service_factory is None:
        from backend.app.devices.factr_client import FactrClient
        service_factory = FactrClient
    from backend.app.devices.factr_calibration import load_profile, save_profile
    from backend.app.devices.factr_runtime_config import parse_runtime_options
    options = parse_runtime_options(config.runtime, WORKSPACE_ROOT/"configs")
    profile_path = Path(options.calibration_file)
    service, code = None, 0
    LOGGER.info("FACTR live test: no test directories, CSV, trajectory or video recording. "
                "Only successful calibration is saved.")
    try:
        service = service_factory(config)
        if input_fn is None:
            input_fn = lambda prompt: _read_terminal_input(prompt, service)
        LOGGER.warning("Commands: c=calibrate, s=test, g=ENABLE MOTOR GRAVITY DIRECTLY, "
                       "d=disable, i=status, q=quit. Prepare physical support before g.")
        if profile_path.is_file():
            try:
                service.set_profile(load_profile(profile_path, config, service.fingerprint))
                LOGGER.info("Calibration loaded: %s", profile_path)
            except (OSError, ValueError, RuntimeError) as exc:
                LOGGER.warning("Calibration not usable: %s. Run c.", exc)
        while True:
            _check_menu_snapshot(service)
            command = input_fn("\n[c] calibrate  [s] test  [g] gravity ON  [d] gravity OFF  [i] status  [q] quit\nFACTR> ").strip().lower()
            try:
                if command == "q":
                    break  # finally closes the runtime, which disables before exit.
                elif command == "i":
                    print(json.dumps(service.diagnostics(), ensure_ascii=False, indent=2), flush=True)
                elif command == "c":
                    profile = calibrate_interactively(service, config, input_fn=input_fn)
                    save_profile(profile_path, profile)
                    LOGGER.info("Whole-arm calibration saved: %s; motor output remains OFF", profile_path)
                    if service.status().get("physical_error"):
                        LOGGER.warning("Calibration succeeded, but CURRENT pose is outside the operating range. "
                                       "Support and reposition before g/s; do not recalibrate. %s",
                                       service.status()["physical_error"])
                elif command == "g":
                    offer_gravity(service)
                elif command == "d":
                    service.disable_gravity()
                elif command == "s":
                    if not service.status().get("calibrated"):
                        raise RuntimeError("Complete whole-arm reference calibration (c) before starting")
                    try:
                        if config.mode == "device":
                            result = run_device_test(service, config)
                        else:
                            def on_prepared():
                                if not service.status().get("gravity_enabled"):
                                    choice = input_fn("Scene ready. g=gravity ON then follow, Enter=passive follow: ").strip().lower()
                                    if choice == "g":
                                        offer_gravity(service)
                                    elif choice:
                                        raise CalibrationCancelled("Simulation start cancelled")
                            result = run_simulation_test(service, config, on_prepared=on_prepared)
                        LOGGER.info("Test finished: %s", json.dumps(result, ensure_ascii=False))
                        if result.get("error") or result.get("cleanup_error"):
                            raise ControllerUnavailable(result.get("error") or result["cleanup_error"])
                    except CalibrationCancelled:
                        raise
                    except BaseException:
                        service.close()  # Also covers preparation failures before the control loop.
                        raise
                    finally:
                        service.disarm(OWNER)
                elif command:
                    LOGGER.info("Unknown command %r; use c/s/g/d/i/q", command)
            except (CalibrationCancelled, ValueError, RuntimeError, OSError, TimeoutError) as exc:
                LOGGER.warning("Command not completed: %s", exc)
    except KeyboardInterrupt:
        code = 130
        LOGGER.warning("Keyboard interrupt: stopping motor output now; support the physical leader")
    except EOFError:
        code = 1
        LOGGER.error("Terminal input closed (EOF): stopping motor output; support the physical leader")
    except ControllerUnavailable as exc:
        code = 1
        LOGGER.error("%s; closing FACTR and stopping motor output", exc)
    except Exception:
        code = 1
        LOGGER.exception("FACTR test failed")
    finally:
        if service is not None:
            try:
                service.close()
            except Exception:
                code = 1
                LOGGER.exception("Motor shutdown not verified; support the leader and inspect physical power")
    LOGGER.info("FACTR test exited: code=%d", code)
    return code


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__,
                                     epilog=f"Fixed configuration: {DEFAULT_FACTR_CONFIG}")
    parser.parse_args()
    logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
    faulthandler.enable(all_threads=True)  # Fatal native errors to stderr, no log file.
    if not sys.stdin.isatty():
        parser.error("FACTR test requires an interactive terminal for calibration and g/d/q commands")
    return run_test(load_factr_config())


if __name__ == "__main__":
    raise SystemExit(main())
