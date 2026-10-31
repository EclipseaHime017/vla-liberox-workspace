"""PBS allocation launcher; experiments keep independent server configurations."""
from __future__ import annotations

from dataclasses import asdict, replace
from datetime import datetime, timezone
import fcntl
import json
import os
from pathlib import Path
import re
import shlex
import shutil
import signal
import subprocess
import time
import uuid

import yaml

from vla_rynn_iql.config import load_train_config
from vla_rynn_iql.config_sources import UniqueKeyLoader, merge_config, resolve_path, resolve_source_paths, strict_keys
from vla_rynn_iql.io import atomic_json, sha256_file, stable_hash
from vla_rynn_iql.server_config import configure_server, distributed_config, load_server_config, validate_global_batch
from vla_rynn_iql.server_inputs import choose_task, discover_tasks, open_run_source
from vla_rynn_iql.server_pipeline import ServerRun, training_settings
from vla_rynn_iql.server_queue import implementation_fingerprint as training_fingerprint

HPC_ROOT = Path(__file__).resolve().parent
PROJECT = HPC_ROOT.parent
GPU_UUID = re.compile(r"GPU-[0-9a-fA-F]{8}(?:-[0-9a-fA-F]{4}){3}-[0-9a-fA-F]{12}")


def implementation_fingerprint():
    return stable_hash({"training": training_fingerprint(), "hpc": {
        path.name: sha256_file(path) for path in sorted(HPC_ROOT.glob("*.py"))}})


def walltime_seconds(value):
    if not isinstance(value, str) or not re.fullmatch(r"\d{2}:\d{2}:\d{2}", value):
        raise ValueError("PBS walltime must be HH:MM:SS")
    hours, minutes, seconds = map(int, value.split(":"))
    total = hours * 3600 + minutes * 60 + seconds
    if minutes >= 60 or seconds >= 60 or not 600 <= total <= 86400:
        raise ValueError("PBS walltime must be between 00:10:00 and 24:00:00")
    return total


def load_pbs_config(path):
    path = Path(path).expanduser().resolve()
    raw = yaml.load(path.read_text(), Loader=UniqueKeyLoader)
    strict_keys(raw, {"schema_version", "job_name", "project", "queue", "walltime", "gpus",
                      "launcher_environment", "output_root", "experiments"}, "PBS")
    if type(raw["schema_version"]) is not int or raw["schema_version"] != 1:
        raise ValueError("Use PBS schema_version=1")
    for key in ("job_name", "project", "launcher_environment"):
        if not isinstance(raw[key], str) or not re.fullmatch(r"[A-Za-z0-9][\w.-]*", raw[key], flags=re.ASCII):
            raise ValueError(f"Invalid PBS {key}")
    route, separator, server = str(raw["queue"]).partition("@")
    if (route not in {"ai", "normal"} or
            (separator and not re.fullmatch(r"[A-Za-z0-9][A-Za-z0-9.-]*", server))):
        raise ValueError("Submit through the ai or normal routing queue, not an internal execution queue")
    if type(raw["gpus"]) is not int or not 1 <= raw["gpus"] <= 8:
        raise ValueError("PBS gpus must be 1–8 on one node")
    walltime_seconds(raw["walltime"])
    experiments = raw["experiments"]
    if not isinstance(experiments, list) or not 1 <= len(experiments) <= raw["gpus"]:
        raise ValueError("PBS experiments must be a non-empty list, with at least one GPU per experiment")
    for entry in experiments:
        if (not isinstance(entry, dict) or {"name", "config", "gpu_ids"} - entry.keys()
                or entry.keys() - {"name", "config", "gpu_ids", "model", "method", "runs_root", "task",
                                   "reward_source", "overrides"}):
            raise ValueError("Invalid PBS experiment keys")
        name = entry["name"]
        if not isinstance(name, str) or not re.fullmatch(r"[A-Za-z0-9][\w-]*", name, flags=re.ASCII):
            raise ValueError("PBS experiment names must be unique safe directory names")
    validate_gpu_groups(experiments, raw["gpus"])
    raw["output_root"] = str(resolve_path(raw["output_root"], path.parent, "PBS output_root"))
    return path, raw


def validate_gpu_groups(entries, count):
    used, names = set(), set()
    for entry in entries:
        ids = distributed_config({"gpu_ids": entry["gpu_ids"]}).gpu_ids
        if entry["name"] in names or max(ids) >= count or used.intersection(ids):
            raise ValueError("PBS experiment GPU groups overlap or exceed the allocation")
        names.add(entry["name"])
        used.update(ids)
    if used != set(range(count)):
        raise ValueError("PBS GPU groups must cover exactly the allocated devices")


def stage_submission(path):
    """Snapshot configurations and render a job script without submitting or loading a model."""
    path, settings = load_pbs_config(path)
    conda = shutil.which("conda")
    if conda is None:
        raise ValueError("Conda must be on PATH when preparing a PBS submission")
    entries = []
    for entry in settings["experiments"]:
        runs = resolve_path(entry["runs_root"], path.parent, "runs_root") if entry.get("runs_root") else None
        server = configure_server(resolve_path(entry["config"], path.parent, "config"),
                                  model=entry.get("model"), method=entry.get("method"),
                                  task=entry.get("task"), runs_root=runs)
        changes = resolve_source_paths(entry.get("overrides", {}), path)
        overrides = merge_config(resolve_source_paths(server.overrides, server.path), changes)
        server = replace(server, overrides=overrides,
                         reward_source=entry.get("reward_source", server.reward_source),
                         distributed=replace(server.distributed, gpu_ids=tuple(entry["gpu_ids"])))
        entries.append((entry, server))
    name = f"pbs_{datetime.now(timezone.utc):%Y%m%d_%H%M%S}_{uuid.uuid4().hex[:8]}"
    directory = Path(settings["output_root"]) / name
    directory.mkdir(parents=True, exist_ok=False)
    plans = []
    try:
        for entry, server in entries:
            training_path = directory / f"{entry['name']}.training.yaml"
            changes = merge_config(server.model_profiles.get(overrides_family(server), {}), server.overrides)
            resolved = load_train_config(server.training_config, overrides=changes, overrides_path=server.path).raw
            training_path.write_text(yaml.safe_dump(resolved, sort_keys=False))
            document = asdict(server)
            document.pop("path")
            document.update(schema_version=4, training_config=str(training_path))
            for key in ("runs_root", "output_root", "cache_root"):
                document[key] = str(document[key])
            config_path = directory / f"{entry['name']}.server.yaml"
            config_path.write_text(yaml.safe_dump(document, sort_keys=False))
            frozen = load_server_config(config_path)
            changes = merge_config(frozen.model_profiles.get(overrides_family(frozen), {}), frozen.overrides)
            raw = load_train_config(frozen.training_config, overrides=changes, overrides_path=frozen.path).raw
            validate_global_batch(raw, frozen.distributed)
            plans.append({"name": entry["name"], "config": str(config_path), "gpu_ids": entry["gpu_ids"],
                          "config_sha256": sha256_file(config_path),
                          "training_config": str(training_path), "training_sha256": sha256_file(training_path),
                          "model": raw["model"]["base_id"], "method": raw["training"]["method"],
                          "global_batch": raw["training"]["micro_batch_size"],
                          "steps": raw["training"]["train_steps"]})
        manifest = {"schema_version": 1, "settings": settings, "experiments": plans,
                    "implementation": implementation_fingerprint()}
        atomic_json(directory / "request.json", manifest)
        script = HPC_ROOT / "train_pbs.py"
        command = shlex.join([conda, "run", "--no-capture-output", "-n", settings["launcher_environment"],
                              "python", "-u", str(script), "run", str(directory)])
        job = ("#!/bin/bash\n"
               f"#PBS -N {settings['job_name']}\n#PBS -q {settings['queue']}\n"
               f"#PBS -P {settings['project']}\n#PBS -l select=1:ngpus={settings['gpus']}\n"
               f"#PBS -l walltime={settings['walltime']}\n#PBS -j oe\n"
               "set -euo pipefail\n"
               f"export PATH={shlex.quote(str(Path(conda).parent))}:\"$PATH\"\n"
               f"cd {shlex.quote(str(PROJECT.parent))}\nexec {command}\n")
        (directory / "job.pbs").write_text(job)
    except BaseException as exc:
        atomic_json(directory / "result.json", {"status": "INVALID", "error": f"{type(exc).__name__}: {exc}"})
        raise
    return directory, manifest


def overrides_family(server):
    return server.overrides["model"].get("family") or load_train_config(server.training_config).raw["model"]["family"]


def verified_request(directory):
    manifest = json.loads((directory / "request.json").read_text())
    if manifest.get("schema_version") != 1 or manifest.get("implementation") != implementation_fingerprint():
        raise ValueError("PBS training implementation changed; prepare a new request")
    validate_gpu_groups(manifest["experiments"], manifest["settings"]["gpus"])
    for entry in manifest["experiments"]:
        if (sha256_file(Path(entry["config"])) != entry["config_sha256"]
                or sha256_file(Path(entry["training_config"])) != entry["training_sha256"]):
            raise ValueError("PBS configuration snapshot changed; prepare a new request")
    return manifest


def allocated_devices(expected, *, environ=None, query=None):
    """Resolve only PBS-assigned devices; never infer an allocation from idle GPUs."""
    environ = os.environ if environ is None else environ
    if not environ.get("PBS_JOBID") or not environ.get("PBS_NODEFILE"):
        raise ValueError("Run this entrypoint inside a PBS GPU job")
    nodes = Path(environ["PBS_NODEFILE"]).read_text().split()
    if not nodes or len(set(nodes)) != 1:
        raise ValueError("Server training requires a single-node PBS allocation")
    mask = environ.get("CUDA_VISIBLE_DEVICES", "")
    tokens = [item.strip() for item in mask.split(",")]
    if len(tokens) != expected or any(not GPU_UUID.fullmatch(item) for item in tokens):
        raise ValueError("PBS must provide CUDA_VISIBLE_DEVICES as full GPU UUIDs matching the requested count; "
                         "numeric, abbreviated and MIG identifiers are not accepted")
    if query is None:
        query = subprocess.check_output(["nvidia-smi", "--query-gpu=uuid", "--format=csv,noheader"], text=True)
    devices = {line.strip() for line in query.splitlines()}
    if any(token not in devices for token in tokens):
        raise ValueError("PBS GPU allocation contains unknown, abbreviated or MIG devices")
    result = tuple(tokens)
    if len(set(result)) != expected:
        raise ValueError("PBS allocation contains duplicate physical GPUs")
    return result


def run_allocation(directory):
    """Foreground only: two independent runs, one allocation, no detached queue worker."""
    from contextlib import ExitStack

    directory = Path(directory).resolve()
    with (directory / "runner.lock").open("a") as lock:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        if (directory / "result.json").exists():
            raise ValueError("This PBS request already ran; prepare a new request to avoid duplicate training")
        manifest = verified_request(directory)
        settings = manifest["settings"]
        devices = allocated_devices(settings["gpus"])
        stop = False

        def request_stop(*_):
            nonlocal stop
            stop = True

        handlers = {sig: signal.signal(sig, request_stop) for sig in (signal.SIGINT, signal.SIGTERM)}
        started = time.monotonic()
        deadline = started + walltime_seconds(settings["walltime"]) - 300
        runs, error, preparation = [], None, []
        stack = ExitStack()
        try:
            for entry in manifest["experiments"]:
                server = load_server_config(Path(entry["config"]))
                if tuple(entry["gpu_ids"]) != server.distributed.gpu_ids:
                    raise ValueError("PBS GPU assignment differs from the frozen experiment")
                root = stack.enter_context(open_run_source(server.runs_root))
                server = replace(server, runs_root=root)
                project = load_train_config(server.training_config).raw["data"]["project_id"]
                task = choose_task(discover_tasks(root, project), server.task_id)
                raw = training_settings(server, task)
                scope = tuple(devices[index] for index in entry["gpu_ids"])
                run = ServerRun(server, task, raw, server.distributed, server.reward_source,
                                directory=directory / "runs" / entry["name"], devices=scope)
                preparation.append((entry, run, scope))
            for entry, run, scope in preparation:
                if stop or time.monotonic() >= deadline:
                    stop = True
                    break
                runs.append((entry, run))
                run.start()
                print(json.dumps({"experiment": entry["name"], "devices": scope,
                                  "directory": str(run.directory)}, ensure_ascii=False), flush=True)
            while True:
                codes = [run.poll() for _, run in runs]
                if all(code is not None for code in codes):
                    break
                if stop or time.monotonic() >= deadline:
                    stop = True
                    for _, run in runs:
                        run.cancel()
                atomic_json(directory / "progress.json", {entry["name"]: run.progress() for entry, run in runs})
                time.sleep(1)
        except BaseException as exc:
            error = f"{type(exc).__name__}: {exc}"
        finally:
            for _, run in runs:
                try:
                    run.shutdown()
                except Exception as exc:
                    error = error or f"{type(exc).__name__}: {exc}"
            for sig, handler in handlers.items():
                signal.signal(sig, handler)
            stack.close()
        states = {entry["name"]: run.state for entry, run in runs}
        complete = len(runs) == len(manifest["experiments"]) and all(
            run.state.get("status") == "COMPLETED" for _, run in runs)
        status = "FAILED" if error else "INTERRUPTED" if stop else "COMPLETED" if complete else "FAILED"
        atomic_json(directory / "result.json", {"status": status, "error": error,
                    "pbs_job_id": os.environ["PBS_JOBID"], "experiments": states,
                    "elapsed_seconds": time.monotonic() - started})
        return 0 if status == "COMPLETED" else 130 if status == "INTERRUPTED" else 1


def submit(directory, *, afterok=None):
    """Only this explicit operation reserves billable PBS resources."""
    directory = Path(directory).resolve()
    if afterok is not None and (not isinstance(afterok, str) or not re.fullmatch(r"\d+\.[A-Za-z0-9][A-Za-z0-9._-]*", afterok)):
        raise ValueError("afterok must be the complete PBS job ID returned by qsub")
    with (directory / "submission.lock").open("a") as lock:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        if (directory / "submission.json").exists():
            raise ValueError("This PBS request was already submitted or attempted; check qstat before trying again")
        manifest = verified_request(directory)
        environments = {manifest["settings"]["launcher_environment"]}
        for entry in manifest["experiments"]:
            server = load_server_config(Path(entry["config"]))
            if not server.runs_root.exists():
                raise ValueError(f"PBS input is missing: {server.runs_root}")
            environments.add(load_train_config(server.training_config).raw["model"]["environment"])
        installed = json.loads(subprocess.check_output(["conda", "env", "list", "--json"], text=True))
        missing = environments - {Path(value).name for value in installed["envs"]}
        if missing:
            raise ValueError(f"Install the required Conda environments before PBS submission: {sorted(missing)}")
        atomic_json(directory / "submission.json", {"status": "SUBMITTING", "afterok": afterok})
        dependency = ["-W", f"depend=afterok:{afterok}"] if afterok else []
        try:
            result = subprocess.check_output(["qsub", "-o", str(directory / "pbs.log"),
                                              *dependency, str(directory / "job.pbs")], text=True, timeout=30).strip()
        except subprocess.CalledProcessError as exc:
            atomic_json(directory / "submission.json", {"status": "REJECTED", "exit_code": exc.returncode})
            raise RuntimeError(f"qsub rejected the request (exit {exc.returncode})") from exc
        except subprocess.TimeoutExpired as exc:
            raise RuntimeError("qsub timed out; submission outcome is unknown, check qstat before retrying") from exc
        atomic_json(directory / "submission.json", {"status": "SUBMITTED", "job_id": result, "afterok": afterok})
        return result
