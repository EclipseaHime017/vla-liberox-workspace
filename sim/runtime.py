"""Headless LIBERO-X rollout, independent of the platform and training loops."""
from __future__ import annotations

from collections import deque
import importlib
import importlib.metadata
import json
import math
import os
from pathlib import Path
import random
import signal
import statistics
import subprocess
import sys
import threading
import time
import traceback

import numpy as np
import yaml

from core.config import LoadedConfig
from core.config_sources import UniqueKeyLoader
from core.registry import family_adapter, model_contract
from sim.metrics import ConsecutiveSuccess, aggregate_trials, _apply_close_error
from core.io import atomic_json, sha256_file
from core.artifacts import decode_child, validate_policy
from core.upstream import upstream_lock


def proprio_from_observation(obs):
    quat = np.asarray(obs["robot0_eef_quat"], dtype=np.float64).copy()
    if quat.shape != (4,) or not np.isfinite(quat).all():
        raise ValueError("Expected a finite xyzw quaternion")
    quat[3] = np.clip(quat[3], -1, 1)
    denominator = math.sqrt(max(0, 1 - quat[3] ** 2))
    angle = np.zeros(3) if math.isclose(denominator, 0) else quat[:3] * (2 * math.acos(quat[3]) / denominator)
    proprio = np.concatenate([obs["robot0_eef_pos"], angle, obs["robot0_gripper_qpos"]]).astype(np.float32)
    if proprio.shape != (8,) or not np.isfinite(proprio).all():
        raise ValueError("Invalid robot proprioception")
    return proprio


def initialize_simulation(request, directory):
    """Configure paths before imports; never create ~/.libero or require a display."""
    root = Path(request["liberox_root"])
    lock = upstream_lock("libero_x")
    versions = {name: importlib.metadata.version(name) for name in lock["packages"]}
    for name, expected in lock["packages"].items():
        if versions[name] != expected:
            raise RuntimeError(f"Evaluation requires {name}=={expected}; found {versions[name]}")
    revision = subprocess.check_output(["git", "-C", str(root), "rev-parse", "HEAD"], text=True).strip()
    if revision != lock["commit"]:
        raise ValueError(f"LIBERO-X revision mismatch: expected {lock['commit']}, found {revision}")
    subprocess.run(["git", "-C", str(root), "diff", "--quiet", "HEAD", "--", "libero"], check=True)
    os.environ["MUJOCO_GL"] = request["render_backend"]
    os.environ["PYOPENGL_PLATFORM"] = request["render_backend"]
    os.environ.setdefault("JAX_PLATFORMS", "cpu")
    os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")
    config_dir = directory / "libero-config"
    config_dir.mkdir(exist_ok=True)
    benchmark = root / "libero/libero"
    (config_dir / "config.yaml").write_text(yaml.safe_dump({
        "benchmark_root": str(benchmark), "bddl_files": str(root / "libero/libero_x/bddl"),
        "init_states": str(root / "libero/libero_x/init"), "datasets": str(root / "datasets"),
        "assets": str(benchmark / "assets"),
    }))
    os.environ["LIBERO_CONFIG_PATH"] = str(config_dir)
    os.environ["MPLCONFIGDIR"] = str(directory / "matplotlib-cache")
    sys.path.insert(0, str(root))
    from libero.libero.envs import OffScreenRenderEnv
    return OffScreenRenderEnv, {"packages": versions, "libero_x_commit": revision,
                               "backend": request["render_backend"],
                               "cuda_visible_devices": os.environ.get("CUDA_VISIBLE_DEVICES"),
                               "egl_device_id": os.environ.get("MUJOCO_EGL_DEVICE_ID")}


def create_env(factory, task, seed, control_hz):
    env = factory(bddl_file_name=task["bddl_path"], use_camera_obs=False,
                  camera_names=["agentview", "robot0_eye_in_hand"],
                  camera_widths=[256, 256], camera_heights=[256, 256],
                  horizon=task["max_steps"] + 1, control_freq=control_hz, hard_reset=False)
    try:
        env.seed(seed)
        env.reset()
    except BaseException:
        env.close()
        raise
    return env


def capture(env, observation):
    result = dict(observation)
    for camera in ("agentview", "robot0_eye_in_hand"):
        frame = np.asarray(env.env.sim.render(width=256, height=256, camera_name=camera))
        if frame.shape != (256, 256, 3) or frame.dtype != np.uint8:
            raise ValueError(f"Invalid camera frame for {camera}: {frame.shape}, {frame.dtype}")
        result[f"{camera}_image"] = frame
    proprio_from_observation(result)
    return result


def load_states(task):
    import torch
    # LIBERO .init files contain trusted upstream NumPy state arrays.
    states = torch.load(task["init_path"], map_location="cpu", weights_only=False)
    if not len(states) or max(task["init_state_indices"]) >= len(states):
        raise ValueError("Initial-state indices exceed the selected task's state count")
    return states


def render_preflight(factory, task, backend, control_hz):
    from OpenGL import GL
    states = load_states(task)
    env = create_env(factory, task, 0, control_hz)
    try:
        obs = env.regenerate_obs_from_state(np.asarray(states[task["init_state_indices"][0]]))
        before = capture(env, obs)
        info = {name: GL.glGetString(value).decode() for name, value in (
            ("vendor", GL.GL_VENDOR), ("renderer", GL.GL_RENDERER), ("version", GL.GL_VERSION))}
        if backend == "egl" and "NVIDIA" not in info["vendor"]:
            raise RuntimeError(f"EGL is not using NVIDIA hardware: {info}")
        env.step([0.] * 6 + [-1.])
        for camera in ("agentview_image", "robot0_eye_in_hand_image"):
            if before[camera].std() < 1:
                raise RuntimeError(f"Render preflight produced a blank image: {camera}")
        info["cameras"] = {name: {"shape": list(before[name].shape), "std": float(before[name].std())}
                           for name in ("agentview_image", "robot0_eye_in_hand_image")}
        return info
    finally:
        env.close()


def episode(env, state, predict, process_action, prompt, schedule, *, steps, open_loop_steps,
            native_horizon, success_streak, stopping):
    started = time.monotonic()
    row = {**schedule, "success": False, "error": None, "steps": 0, "first_success_step": None,
           "max_done_streak": 0, "final_done": False, "policy_queries": 0,
           "inference_latency_ms": None, "measured_control_hz": None, "elapsed_seconds": 0.}
    tracker = ConsecutiveSuccess(success_streak)
    latencies = []
    try:
        observation = env.regenerate_obs_from_state(np.asarray(state))
        actions = deque()
        for step in range(steps):
            if stopping():
                raise InterruptedError("Evaluation canceled")
            if not actions:
                observation = capture(env, observation)
                query_started = time.monotonic()
                chunk = np.asarray(predict(observation, prompt), dtype=np.float32)
                latencies.append(1000 * (time.monotonic() - query_started))
                if chunk.shape != (native_horizon, 7) or not np.isfinite(chunk).all():
                    raise ValueError(f"Invalid policy chunk: {chunk.shape}; expected {(native_horizon, 7)}")
                actions.extend(chunk[:open_loop_steps])
                row["policy_queries"] += 1
            if stopping():
                raise InterruptedError("Evaluation canceled after inference")
            action = np.asarray(process_action(actions.popleft()), dtype=np.float32)
            if action.shape != (7,) or not np.isfinite(action).all():
                raise ValueError("Invalid environment action")
            observation, _, done, _ = env.step(action.tolist())
            tracker.observe(bool(done), step + 1)
            row.update(steps=step + 1, final_done=bool(done))
        row["success"] = tracker.success
    except InterruptedError:
        raise
    except Exception as exc:
        row["error"] = f"{type(exc).__name__}: {exc}"
    row.update(first_success_step=tracker.first_confirmed_step, max_done_streak=tracker.maximum,
               inference_latency_ms=statistics.fmean(latencies) if latencies else None,
               elapsed_seconds=time.monotonic() - started)
    row["measured_control_hz"] = row["steps"] / row["elapsed_seconds"] if row["elapsed_seconds"] else None
    return row


def run_worker(path):
    request = json.loads(path.read_text())
    directory = path.parent
    record = {"status": "STARTING", "request": request, "tasks": [], "error": None}
    result_path = directory / "result.json"
    atomic_json(result_path, record)
    stopped = threading.Event()
    for sig in (signal.SIGTERM, signal.SIGINT):
        signal.signal(sig, lambda *_: stopped.set())
    from core.resources import process_alive

    def watch_owner():
        while not stopped.wait(1):
            if not process_alive(request["launcher_identity"]):
                os._exit(1)

    threading.Thread(target=watch_owner, daemon=True).start()
    try:
        from core.resources import register_worker
        register_worker(request["lease_token"])
        for task in request["tasks"]:
            for key in ("bddl", "init"):
                if sha256_file(Path(task[f"{key}_path"])) != task[f"{key}_sha256"]:
                    raise ValueError(f"Task asset changed: {task[f'{key}_path']}")
        factory, record["runtime"] = initialize_simulation(request, directory)
        record["runtime"]["render"] = render_preflight(factory, request["tasks"][0], request["render_backend"], 20)
        atomic_json(result_path, record)
        if request["check_render"]:
            record["status"] = "COMPLETED"
            return 0
        manifest = Path(request["policy_path"])
        if sha256_file(manifest) != request["policy_sha256"]:
            raise ValueError("Policy manifest changed after evaluation submission")
        payload = yaml.load(manifest.read_text(), Loader=UniqueKeyLoader)
        validate_policy(manifest, payload)
        native, parent = decode_child(payload)
        settings = native["model_config"]
        adapter = family_adapter(settings["family"])
        revision = adapter.resolve_revision(settings, hash_file=sha256_file, parent=parent)
        if revision != parent["revision"]:
            raise ValueError("The actual base model differs from the trained policy's parent")
        import torch
        if not torch.cuda.is_available():
            raise RuntimeError("CUDA is required for checkpoint evaluation")
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False
        record["runtime"].update(torch=torch.__version__, cuda=torch.version.cuda,
                                  policy_gpu=torch.cuda.get_device_name(0))
        contract = model_contract(settings)
        if contract["io"]["control_hz"] != 20:
            raise ValueError("This LIBERO-X evaluator requires the 20 Hz model contract")
        config = LoadedConfig(manifest, {"model": settings,
            "training": {"device": "cuda:0", "dtype": "bfloat16", "seed": request["base_seed"]},
            "paths": {"vla_adapter_root": request["vla_adapter_root"]}})
        backend = importlib.import_module(adapter.BACKEND)
        predict, action = backend.evaluation_policy(config, manifest, parent)
        # Recheck artifacts across the load boundary, just like the main provider.
        validate_policy(manifest, payload)
        if sha256_file(manifest) != request["policy_sha256"]:
            raise ValueError("Policy manifest changed during loading")
        if adapter.resolve_revision(settings, hash_file=sha256_file, parent=parent) != revision:
            raise ValueError("Base model changed during loading")
        record.update(status="RUNNING", policy=payload)
        for task in request["tasks"]:
            task_result = {"task": task, "trials": [], "aggregate": aggregate_trials([], scheduled_trials=len(task["schedule"]))}
            record["tasks"].append(task_result)
            states = load_states(task)
            env = None
            errors = 0
            try:
                for item in task["schedule"]:
                    if stopped.is_set():
                        raise InterruptedError("Evaluation canceled")
                    seed = item["seed"]
                    random.seed(seed)
                    np.random.seed(seed)
                    torch.manual_seed(seed)
                    torch.cuda.manual_seed_all(seed)
                    trial_started = time.monotonic()
                    try:
                        if env is None:
                            env = create_env(factory, task, seed, 20)
                        else:
                            env.seed(seed)
                            env.reset()
                        task_result["prompt"] = env.language_instruction
                        row = episode(env, states[item["init_state_index"]], predict, action,
                                      env.language_instruction, item, steps=task["max_steps"],
                                      open_loop_steps=request["open_loop_steps"],
                                      native_horizon=contract["io"]["native_action_horizon"],
                                      success_streak=request["success_streak"], stopping=stopped.is_set)
                    except InterruptedError:
                        raise
                    except Exception as exc:
                        row = {**item, "success": False, "error": f"{type(exc).__name__}: {exc}",
                               "steps": 0, "first_success_step": None, "policy_queries": 0,
                               "elapsed_seconds": time.monotonic() - trial_started}
                    if row["error"] and env is not None:
                        old, env = env, None
                        try:
                            old.close()
                        except Exception as exc:
                            _apply_close_error(row, exc)
                    task_result["trials"].append(row)
                    task_result["aggregate"] = aggregate_trials(task_result["trials"], scheduled_trials=len(task["schedule"]))
                    atomic_json(result_path, record)
                    print(json.dumps({"task": task["task_name"], "trial": item["trial_index"],
                                      "success": row["success"], "error": row["error"],
                                      "success_rate": task_result["aggregate"]["success_rate"]}), flush=True)
                    errors = errors + 1 if row["error"] else 0
                    if errors >= request["consecutive_error_limit"]:
                        raise RuntimeError("Consecutive trial errors exceeded the configured limit")
            finally:
                if env is not None:
                    try:
                        env.close()
                    except Exception as exc:
                        if task_result["trials"]:
                            _apply_close_error(task_result["trials"][-1], exc)
                            task_result["aggregate"] = aggregate_trials(task_result["trials"], scheduled_trials=len(task["schedule"]))
                        raise
        record["status"] = "FAILED" if any(t["aggregate"]["errors"] for t in record["tasks"]) else "COMPLETED"
        return 0 if record["status"] == "COMPLETED" else 1
    except InterruptedError as exc:
        record.update(status="CANCELED", error=str(exc))
        return 130
    except Exception as exc:
        record.update(status="FAILED", error=f"{type(exc).__name__}: {exc}", traceback=traceback.format_exc())
        print(record["traceback"], file=sys.stderr, flush=True)
        return 1
    finally:
        atomic_json(result_path, record)
        stopped.set()
