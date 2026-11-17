"""Parallel local LIBERO-X environments connected to remote inference replicas."""
from __future__ import annotations

from concurrent.futures import ProcessPoolExecutor, as_completed
from contextlib import contextmanager
from datetime import datetime, timezone
import hashlib
import json
import multiprocessing
import os
from pathlib import Path
import random
import signal
import threading
import time
import traceback
import uuid

import numpy as np

from core.registry import validate_open_loop_steps
from sim.metrics import aggregate_trials
from core.io import atomic_json, sha256_file
from cluster.remote_inference import Client, read_token
from core.resources import TrainingLease, process_identity, process_alive, register_worker, signal_worker


@contextmanager
def cancel_on_signal(lease, record):
    """Stop only this evaluation's registered workers and reject late joins."""
    def cancel(*_):
        record["canceled"] = True
        lease.retire()
        for worker in lease.workers():
            signal_worker(worker, signal.SIGTERM)
    previous = {sig: signal.signal(sig, cancel) for sig in (signal.SIGINT, signal.SIGTERM)}
    try:
        yield
    finally:
        for sig, handler in previous.items():
            signal.signal(sig, handler)


def simulation_worker(request):
    from sim.runtime import initialize_simulation, create_env, load_states, render_preflight, episode
    import torch
    torch.set_num_threads(1)
    directory = Path(request["directory"])
    directory.mkdir()
    result = {"status": "RUNNING", "trials": [], "requests": [], "error": None}
    stopped = threading.Event()
    def watch_owner():
        while not stopped.wait(1):
            if not process_alive(request["launcher_identity"]):
                os._exit(1)
    threading.Thread(target=watch_owner, daemon=True).start()
    client, env = None, None
    started = time.monotonic()
    try:
        register_worker(request["lease_token"])
        for assignment in request["assignments"]:
            for key in ("bddl", "init"):
                task = assignment["task"]
                if sha256_file(Path(task[f"{key}_path"])) != task[f"{key}_sha256"]:
                    raise ValueError("Task changed after evaluation submission")
        client = Client(request["endpoint"], read_token(request["token_file"]), expected_sha256=request["policy_sha256"])
        if client.metadata != request["metadata"]:
            raise ValueError("Remote inference replica changed before rollout")
        factory, result["runtime"] = initialize_simulation(request["config"], directory)
        first = request["assignments"][0]["task"]
        result["runtime"]["render"] = render_preflight(factory, first, request["config"]["render_backend"], 20)
        active_task, states = None, None
        for assignment in request["assignments"]:
            task, schedule = assignment["task"], assignment["schedule"]
            task_identity = (task["level"], task["task_name"])
            if active_task != task_identity:
                if env is not None:
                    env.close()
                env = None
                states = load_states(task)
                active_task = task_identity
            client.reset(schedule["seed"])
            random.seed(schedule["seed"])
            np.random.seed(schedule["seed"])
            torch.manual_seed(schedule["seed"])
            if env is None:
                env = create_env(factory, task, schedule["seed"], 20)
            else:
                env.seed(schedule["seed"])
                env.reset()
            action_digest = hashlib.sha256()
            def predict(observation, prompt):
                actions = client.predict(observation, prompt)
                action_digest.update(actions.astype("<f4").tobytes())
                return actions
            trial = episode(env, states[schedule["init_state_index"]], predict, lambda a: a,
                            env.language_instruction, schedule, steps=task["max_steps"],
                            open_loop_steps=request["open_loop_steps"],
                            native_horizon=client.metadata["contract"]["io"]["native_action_horizon"],
                            success_streak=request["config"]["success_streak"], stopping=lambda: False)
            trial["action_sha256"] = action_digest.hexdigest()
            trial["final_state_sha256"] = hashlib.sha256(np.asarray(env.env.sim.get_state().flatten(), dtype="<f8").tobytes()).hexdigest()
            result["trials"].append({"task": task["task_name"], "level": task["level"], **trial})
            result["requests"] = list(client.metrics)
            atomic_json(directory / "result.json", result)
            if trial["error"]:
                raise RuntimeError(trial["error"])
        result["status"] = "COMPLETED"
    except Exception as exc:
        result.update(status="FAILED", error=f"{type(exc).__name__}: {exc}", traceback=traceback.format_exc())
    finally:
        for close in ([client.close] if client else []) + ([env.close] if env else []):
            try:
                close()
            except Exception as exc:
                result.update(status="FAILED", error=f"Cleanup: {type(exc).__name__}: {exc}")
        result["wall_seconds"] = time.monotonic() - started
        atomic_json(directory / "result.json", result)
        stopped.set()
    return result


def run_remote(config, endpoints, token_file, policy_sha256, concurrency):
    if type(concurrency) is not int or not 1 <= concurrency <= 32 or not endpoints:
        raise ValueError("Use 1–32 concurrent simulation workers and at least one endpoint")
    metadata = []
    for url in endpoints:
        client = Client(url, read_token(token_file), expected_sha256=policy_sha256)
        try:
            metadata.append(client.metadata)
        finally:
            client.close()
    for replica in metadata:
        if replica["contract"]["io"]["control_hz"] != 20:
            raise ValueError("LIBERO-X evaluation requires the 20 Hz model contract")
        for key in ("contract", "policy", "execution"):
            if replica[key] != metadata[0][key]:
                raise ValueError("Inference replicas must serve the same checkpoint and model contract")
    steps = config["open_loop_steps"]
    if steps is None:
        steps = metadata[0]["execution"]["open_loop_steps"]
    validate_open_loop_steps(steps, metadata[0]["execution"]["max_open_loop_steps"])
    assignments = [{"task": task, "schedule": schedule} for task in config["tasks"] for schedule in task["schedule"]]
    if concurrency > len(assignments):
        raise ValueError("Concurrency exceeds the number of scheduled trials")
    directory = Path(config["output_root"]) / (datetime.now(timezone.utc).strftime("remote_%Y%m%d_%H%M%S_") + uuid.uuid4().hex[:8])
    record = {"status": "RUNNING", "config": config, "replicas": metadata, "concurrency": concurrency,
              "open_loop_steps": steps, "workers": [], "tasks": [], "canceled": False}
    with TrainingLease() as lease, cancel_on_signal(lease, record):
        directory.mkdir(parents=True)
        atomic_json(directory / "summary.json", record)
        print(f"Remote evaluation results: {directory}", flush=True)
        started = time.monotonic()
        with ProcessPoolExecutor(max_workers=concurrency, mp_context=multiprocessing.get_context("spawn")) as pool:
            futures = []
            for index in range(concurrency):
                if record["canceled"]:
                    break
                replica = index % len(endpoints)
                request = {"config": config, "directory": str(directory / f"{index:03d}"),
                           "lease_token": lease.token, "launcher_identity": process_identity(os.getpid()),
                           "endpoint": endpoints[replica], "metadata": metadata[replica],
                           "token_file": str(Path(token_file).resolve()), "policy_sha256": policy_sha256,
                           "open_loop_steps": steps, "assignments": assignments[index::concurrency]}
                try:
                    futures.append(pool.submit(simulation_worker, request))
                except Exception:
                    if not record["canceled"]:
                        raise
                    break
            for future in as_completed(futures):
                try:
                    result = future.result()
                except Exception as exc:
                    result = {"status": "FAILED", "error": f"{type(exc).__name__}: {exc}", "trials": [], "requests": []}
                record["workers"].append(result)
                print(json.dumps({"worker_status": result["status"], "error": result["error"], "trials": len(result["trials"])}), flush=True)
                atomic_json(directory / "summary.json", record)
        duration = time.monotonic() - started
        trials = [trial for worker in record["workers"] for trial in worker["trials"]]
        requests = [value for worker in record["workers"] for value in worker["requests"]]
        for task in config["tasks"]:
            selected = [trial for trial in trials if (trial["level"], trial["task"]) == (task["level"], task["task_name"])]
            record["tasks"].append({"task": task["task_name"], **aggregate_trials(selected, scheduled_trials=len(task["schedule"]))})
        stats = {}
        for key in ("roundtrip_ms", "inference_ms", "queue_ms", "request_bytes", "allocated_mib", "reserved_mib"):
            values = [value[key] for value in requests]
            stats[key] = {"mean": float(np.mean(values)), "p95": float(np.percentile(values, 95)), "max": float(max(values))} if values else None
        completed = all(w["status"] == "COMPLETED" for w in record["workers"]) and len(trials) == len(assignments)
        record.update(status="CANCELED" if record["canceled"] else "COMPLETED" if completed else "FAILED",
                      wall_seconds=duration, total_control_steps=sum(t["steps"] for t in trials),
                      aggregate_steps_per_second=sum(t["steps"] for t in trials) / duration, inference_requests=len(requests), metrics=stats)
        atomic_json(directory / "summary.json", record)
        print(json.dumps({key: record[key] for key in ("status", "wall_seconds", "total_control_steps", "aggregate_steps_per_second", "inference_requests", "metrics")}), flush=True)
    return 0 if record["status"] == "COMPLETED" else 130 if record["canceled"] else 1
