"""Frozen, sequential checkpoint evaluation in each registered model environment."""
from __future__ import annotations

from datetime import datetime, timezone
import json
import os
from pathlib import Path
import signal
import subprocess
import threading
import uuid

import yaml

from .base_models import execution_config, validate_open_loop_steps
from .config_sources import UniqueKeyLoader
from .evaluation_metrics import build_schedule
from .io import atomic_json, sha256_file, stable_hash
from .model_artifacts import decode_child
from .server_resources import TrainingLease, process_identity

KEYS = {"schema_version", "liberox_root", "vla_adapter_root", "output_root", "render_backend",
        "check_environment", "policies", "tasks", "trials", "base_seed", "seed_count", "schedule_seed",
        "success_streak", "consecutive_error_limit", "open_loop_steps"}


def _mapping(value, keys, name):
    if not isinstance(value, dict) or set(value) != keys:
        raise ValueError(f"{name} requires exactly {sorted(keys)}")


def _path(value, base):
    if not isinstance(value, str) or not value.strip():
        raise ValueError("Expected a non-empty path")
    path = Path(value).expanduser()
    return str((base / path).resolve())


def load_evaluation(path):
    path = path.expanduser().resolve()
    raw = yaml.load(path.read_text(), Loader=UniqueKeyLoader)
    _mapping(raw, KEYS, "server evaluation")
    if type(raw["schema_version"]) is not int or raw["schema_version"] != 1:
        raise ValueError("Unsupported evaluation schema_version")
    if raw["render_backend"] not in ("egl", "osmesa"):
        raise ValueError("render_backend must be egl or osmesa")
    if not isinstance(raw["check_environment"], str) or not raw["check_environment"].strip():
        raise ValueError("check_environment must be a Conda environment name")
    for key in ("liberox_root", "vla_adapter_root", "output_root"):
        raw[key] = _path(raw[key], path.parent)
    if not isinstance(raw["policies"], list) or not raw["policies"]:
        raise ValueError("policies must list at least one exported policy.yaml")
    raw["policies"] = [_path(value, path.parent) for value in raw["policies"]]
    if len(set(raw["policies"])) != len(raw["policies"]):
        raise ValueError("Duplicate checkpoint in policies")
    for key in ("success_streak", "consecutive_error_limit"):
        if type(raw[key]) is not int or raw[key] < 1:
            raise ValueError(f"{key} must be a positive integer")
    if raw["open_loop_steps"] is not None and (type(raw["open_loop_steps"]) is not int or raw["open_loop_steps"] < 1):
        raise ValueError("open_loop_steps must be null or a positive integer")
    if not isinstance(raw["tasks"], list) or not raw["tasks"]:
        raise ValueError("tasks must be a non-empty list")
    seen = set()
    for task in raw["tasks"]:
        _mapping(task, {"level", "task_name", "max_steps", "init_state_indices"}, "task")
        for key in ("level", "task_name"):
            name = task[key]
            if not isinstance(name, str) or not name or name in (".", "..") or Path(name).name != name:
                raise ValueError(f"Unsafe task {key}")
        if type(task["max_steps"]) is not int or task["max_steps"] < raw["success_streak"]:
            raise ValueError("max_steps must be an integer >= success_streak")
        identity = (task["level"], task["task_name"])
        if identity in seen:
            raise ValueError("Duplicate evaluation task")
        seen.add(identity)
        for name in ("bddl", "init"):
            file = Path(raw["liberox_root"]) / "libero/libero_x" / name / task["level"] / f"{task['task_name']}.{name}"
            task[f"{name}_path"] = str(file)
            task[f"{name}_sha256"] = sha256_file(file)
        task["schedule"] = build_schedule(trials=raw["trials"], init_state_indices=task["init_state_indices"],
                                          base_seed=raw["base_seed"], seed_count=raw["seed_count"],
                                          schedule_seed=raw["schedule_seed"])
        task["schedule_sha256"] = stable_hash(task["schedule"])
    return raw


def policy_request(path, config):
    path = Path(path)
    payload = yaml.load(path.read_text(), Loader=UniqueKeyLoader)
    native, parent = decode_child(payload)
    if parent is None:
        raise ValueError("Use an exported schema 5 policy.yaml, not a training state.pt checkpoint")
    settings = native["model_config"]
    execution = execution_config(settings, trained=True)
    steps = execution["open_loop_steps"] if config["open_loop_steps"] is None else config["open_loop_steps"]
    validate_open_loop_steps(steps, execution["max_open_loop_steps"])
    return {"policy_path": str(path), "policy_sha256": sha256_file(path), "open_loop_steps": steps,
            "environment": settings["environment"], "label": payload["label"]}


def _stop_worker(child):
    if child.poll() is None:
        try:
            os.killpg(child.pid, signal.SIGTERM)
        except ProcessLookupError:
            pass
        try:
            child.wait(timeout=15)
        except subprocess.TimeoutExpired:
            try:
                os.killpg(child.pid, signal.SIGKILL)
            except ProcessLookupError:
                pass
    child.wait()


def run_evaluations(config, *, check_render=False):
    plans = [{"environment": config["check_environment"], "label": "render-preflight"}] if check_render else [
        policy_request(path, config) for path in config["policies"]]
    stopping = threading.Event()
    previous = {sig: signal.signal(sig, lambda *_: stopping.set()) for sig in (signal.SIGTERM, signal.SIGINT)}
    child = None
    summary = {"status": "STARTING", "evaluations": []}
    directory = Path(config["output_root"]) / (datetime.now(timezone.utc).strftime("eval_%Y%m%d_%H%M%S_") + uuid.uuid4().hex[:8])
    try:
        # Same lease as direct/queued training, including PBS GPU-scoped leases.
        with TrainingLease() as lease:
            directory.mkdir(parents=True, exist_ok=False)
            atomic_json(directory / "config.json", config)
            atomic_json(directory / "summary.json", summary)
            print(f"Evaluation results: {directory}", flush=True)
            for index, plan in enumerate(plans):
                if stopping.is_set():
                    break
                target = directory / f"{index:03d}"
                target.mkdir()
                request = {**config, **plan, "check_render": check_render, "lease_token": lease.token,
                           "launcher_identity": process_identity(os.getpid())}
                atomic_json(target / "request.json", request)
                item = {"label": plan["label"], "status": "RUNNING", "result": str(target / "result.json"),
                        "log": str(target / "worker.log")}
                summary["evaluations"].append(item)
                summary["status"] = "RUNNING"
                atomic_json(directory / "summary.json", summary)
                script = Path(__file__).resolve().parents[2] / "scripts/evaluate_server.py"
                command = ["conda", "run", "--no-capture-output", "-n", plan["environment"],
                           "python", "-u", str(script), "--worker", str(target / "request.json")]
                print(f"Testing {plan['label']} — {target / 'worker.log'}", flush=True)
                with (target / "worker.log").open("w") as log:
                    try:
                        child = subprocess.Popen(command, stdin=subprocess.DEVNULL, stdout=log, stderr=subprocess.STDOUT,
                                                 start_new_session=True, env={**os.environ, "PYTHONDONTWRITEBYTECODE": "1"})
                        while child.poll() is None and not stopping.wait(.2):
                            pass
                        if stopping.is_set() and child.poll() is None:
                            _stop_worker(child)
                        code = child.wait()
                        child = None
                        result_file = target / "result.json"
                        result = json.loads(result_file.read_text()) if result_file.is_file() else {}
                        item["status"] = "CANCELED" if stopping.is_set() else (
                            "COMPLETED" if code == 0 and result.get("status") == "COMPLETED" else "FAILED")
                        item["error"] = result.get("error") or (f"Worker exited {code}; see {target / 'worker.log'}" if item["status"] != "COMPLETED" else None)
                        item["tasks"] = [{"task": task["task"]["task_name"], **task["aggregate"]} for task in result.get("tasks", [])]
                    except Exception as exc:
                        item.update(status="FAILED", error=f"{type(exc).__name__}: {exc}")
                    finally:
                        if child is not None:
                            _stop_worker(child)
                            child = None
                        atomic_json(directory / "summary.json", summary)
                print(f"{plan['label']}: {item['status']}" + (f" — {item['error']}" if item.get("error") else ""), flush=True)
            summary["status"] = "CANCELED" if stopping.is_set() else (
                "COMPLETED" if all(item["status"] == "COMPLETED" for item in summary["evaluations"]) else "FAILED")
            atomic_json(directory / "summary.json", summary)
            return 0 if summary["status"] == "COMPLETED" else 130 if stopping.is_set() else 1
    finally:
        if child is not None:
            _stop_worker(child)
        for sig, handler in previous.items():
            signal.signal(sig, handler)
