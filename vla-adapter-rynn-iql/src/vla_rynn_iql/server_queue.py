"""Persistent single-machine FIFO. Status operations import no training runtime."""
from __future__ import annotations

from contextlib import contextmanager
from dataclasses import asdict, replace
from datetime import datetime, timezone
import copy
import fcntl
import json
import os
from pathlib import Path
import signal
import socket
import subprocess
import sys
import time
import uuid

from .io import atomic_json, sha256_file, stable_hash
from .server_resources import TrainingBusy, TrainingCleanupError, TrainingLease, process_alive, process_identity

DEFAULT_QUEUE = Path(__file__).resolve().parents[2] / "outputs/server-queue"
TERMINAL = {"COMPLETED", "FAILED", "CANCELLED", "INTERRUPTED"}


def timestamp():
    return datetime.now(timezone.utc).isoformat()


def implementation_fingerprint():
    root = Path(__file__).parent
    return stable_hash({str(path.relative_to(root)): sha256_file(path) for path in sorted(root.rglob("*.py"))})


def input_fingerprint(task, raw, source):
    from .methods import training_method
    from .server_inputs import saved_reward
    needs_rewards = training_method(raw).requires_rewards
    selected = task.selected(source if needs_rewards else None)
    files = {}
    for run in selected:
        paths = [run.path, run.episode_dir / "trajectory.npz", run.episode_dir / "trajectory_observations.npz"]
        if needs_rewards and source != "sparse":
            paths.append(run.rewards[source])
            if paths[-1].name != "stage_annotation.json":
                paths.append(saved_reward(run, source)[1])
        files[run.run_id] = {str(path.relative_to(run.path.parent)): sha256_file(path) for path in paths}
    if task.export_path:
        files["export"] = {name: sha256_file(task.export_path.parent / name)
                           for name in ("export.json", "dataset.json")}
    return stable_hash({"task": task.task_id, "files": files})


def resume_fingerprint(raw):
    value = raw["training"].get("resume_checkpoint")
    if value is None:
        return None
    root = Path(value)
    paths = sorted(root.rglob("*"))
    if not root.is_dir() or not paths or any(path.is_symlink() for path in paths):
        raise ValueError("Queued resume checkpoint must be a complete local directory without symlinks")
    return stable_hash({str(path.relative_to(root)): sha256_file(path) for path in paths if path.is_file()})


class TrainingQueue:
    def __init__(self, root=DEFAULT_QUEUE):
        self.root = Path(root).expanduser().resolve()

    @contextmanager
    def state(self, *, write=True):
        self.root.mkdir(parents=True, exist_ok=True)
        with (self.root / "state.lock").open("a") as lock:
            fcntl.flock(lock, fcntl.LOCK_EX if write else fcntl.LOCK_SH)
            path = self.root / "queue.json"
            state = json.loads(path.read_text()) if path.exists() else {
                "schema_version": 1, "host": socket.gethostname(), "jobs": [],
                "worker": None, "worker_ready": False, "stop": False}
            if state.get("schema_version") != 1 or state.get("host") != socket.gethostname():
                raise ValueError("Queue must use its original host and supported schema")
            yield state
            if write:
                atomic_json(path, state)

    def list(self):
        with self.state(write=False) as state:
            return copy.deepcopy(state)

    @staticmethod
    def _job(state, identifier):
        for job in state["jobs"]:
            if job["id"] == identifier:
                return job
        raise ValueError(f"Queue job not found: {identifier}")

    def update(self, identifier, **changes):
        with self.state() as state:
            self._job(state, identifier).update(changes, updated_at=timestamp())

    def add(self, server, *, name=None):
        from .config import load_train_config
        from .model_assets import pin_model
        from .server_inputs import choose_task, discover_tasks, open_run_source
        from .server_pipeline import training_settings, execution_plan
        project = load_train_config(server.training_config).section("data")["project_id"]
        with open_run_source(server.runs_root) as root:
            task = choose_task(discover_tasks(root, project), server.task_id)
            raw = training_settings(server, task)
            raw["model"] = pin_model(raw["model"])
            plan = execution_plan(replace(server, runs_root=root), task, raw,
                                  server.distributed, server.reward_source)
            fingerprint = input_fingerprint(task, raw, server.reward_source)
        request = {"schema_version": 1, "raw": raw, "input_fingerprint": fingerprint,
                   "implementation": implementation_fingerprint(),
                   "resume_fingerprint": resume_fingerprint(raw),
                   "server": {key: str(value) if isinstance(value, Path) else value
                              for key, value in asdict(replace(server, task_id=task.task_id)).items()},
                   "cuda_visible_devices": plan["cuda_visible_devices"]}
        return self._enqueue(request, name or f"{raw['model']['base_id']} / {raw['training']['method']}")

    def _enqueue(self, request, name):
        identifier = "q_" + uuid.uuid4().hex[:12]
        job_dir = self.root / "jobs" / identifier
        job_dir.mkdir(parents=True, exist_ok=False)
        atomic_json(job_dir / "request.json", request)
        job = {"id": identifier, "name": name, "status": "QUEUED", "created_at": timestamp(),
               "request_sha256": stable_hash(request),
               "run_directory": str(Path(request["server"]["output_root"]) / f"server_queue_{identifier}")}
        with self.state() as state:
            state["jobs"].append(job)
        return job

    def request(self, job):
        value = json.loads((self.root / "jobs" / job["id"] / "request.json").read_text())
        if stable_hash(value) != job["request_sha256"]:
            raise ValueError("Queued configuration was modified; submit a new job")
        return value

    def cancel(self, identifier):
        with self.state() as state:
            job = self._job(state, identifier)
            if job["status"] == "QUEUED":
                job["status"] = "CANCELLED"
            elif job["status"] == "RUNNING":
                job["cancel_requested"] = True
            else:
                raise ValueError("Only queued or running jobs can be cancelled")

    def stop(self):
        with self.state() as state:
            state["stop"] = True

    def retry(self, identifier):
        job = self._job(self.list(), identifier)
        if job["status"] not in {"FAILED", "CANCELLED", "INTERRUPTED"}:
            raise ValueError("Only stopped jobs can be retried; active jobs are never duplicated")
        return self._enqueue(self.request(job), job["name"])

    def start(self):
        with self.state() as state:
            if process_alive(state["worker"]):
                raise ValueError("Queue worker is already running")
            script = Path(__file__).resolve().parents[2] / "scripts/train_queue.py"
            with (self.root / "worker.log").open("a") as log:
                process = subprocess.Popen([sys.executable, "-B", "-u", str(script), "--queue", str(self.root), "run"],
                    stdin=subprocess.DEVNULL, stdout=log, stderr=subprocess.STDOUT, start_new_session=True)
            state["worker"] = process_identity(process.pid)
            state["worker_ready"] = False
        for _ in range(100):
            state = self.list()
            if state.get("worker_ready") and process_alive(state["worker"]):
                return state["worker"]
            if process.poll() is not None:
                raise RuntimeError(f"Queue worker failed; see {self.root / 'worker.log'}")
            time.sleep(.05)
        self.stop()
        raise RuntimeError(f"Queue startup timed out; stop requested; see {self.root / 'worker.log'}")

    def reconcile(self):
        """Never requeue a possibly executed job after supervisor failure."""
        for job in self.list()["jobs"]:
            if job["status"] != "RUNNING":
                continue
            directory = Path(job["run_directory"])
            if job.get("cancel_requested") and directory.is_dir():
                (directory / "cancel.request").touch()
            path = directory / "pipeline.json"
            pipeline = json.loads(path.read_text()) if path.is_file() else {}
            if process_alive(pipeline.get("process")):
                return False
            try:
                with TrainingLease():
                    pass
            except TrainingBusy:
                return False
            summary = directory / "summary.json"
            result = json.loads(summary.read_text()) if summary.is_file() else {}
            status = result.get("status", pipeline.get("status"))
            if status not in TERMINAL:
                status = "INTERRUPTED"
            if job.get("cancel_requested") and status == "INTERRUPTED":
                status = "CANCELLED"
            self.update(job["id"], status=status, result=result,
                        error=None if status == "COMPLETED" else pipeline.get("error", "Previous worker exited; explicit retry required"))
        return True

    def run(self, *, once=False):
        self.root.mkdir(parents=True, exist_ok=True)
        with (self.root / "runner.lock").open("a") as lock:
            try:
                fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
            except BlockingIOError:
                raise ValueError("Queue worker is already running") from None
            with self.state() as state:
                if process_alive(state["worker"]) and state["worker"]["pid"] != os.getpid():
                    raise ValueError("Queue worker is already starting")
                state.update(worker=process_identity(os.getpid()), worker_ready=True, stop=False)
            stopping = False

            def request_stop(*_):
                nonlocal stopping
                stopping = True

            handlers = {sig: signal.signal(sig, request_stop) for sig in (signal.SIGINT, signal.SIGTERM)}
            implementation = implementation_fingerprint()
            try:
                while True:
                    state = self.list()
                    if stopping or state["stop"]:
                        break
                    if not self.reconcile():
                        time.sleep(.5)
                        continue
                    if implementation_fingerprint() != implementation:
                        raise RuntimeError("Server code changed; restart the queue worker before training")
                    job = next((job for job in self.list()["jobs"] if job["status"] == "QUEUED"), None)
                    if job is None:
                        if once:
                            break
                        time.sleep(.5)
                        continue
                    try:
                        with TrainingLease():
                            pass
                    except TrainingBusy:
                        time.sleep(.5)
                        continue
                    with self.state() as state:
                        current = self._job(state, job["id"])
                        if current["status"] != "QUEUED" or state["stop"]:
                            continue
                        current.update(status="RUNNING", started_at=timestamp())
                    try:
                        self._execute(job, lambda: stopping)
                    except TrainingBusy:
                        self.update(job["id"], status="QUEUED")
                        time.sleep(.5)
                    except TrainingCleanupError:
                        raise
                    except Exception as exc:
                        self.update(job["id"], status="FAILED", error=f"{type(exc).__name__}: {exc}")
            finally:
                with self.state() as state:
                    state["worker"] = None
                    state["worker_ready"] = False
                for sig, handler in handlers.items():
                    signal.signal(sig, handler)

    def _execute(self, job, stopping):
        from .server_config import ServerConfig, distributed_config
        from .server_inputs import choose_task, discover_tasks, open_run_source
        from .server_pipeline import ServerRun
        from .config import validate_train_config
        request = self.request(job)
        if request["implementation"] != implementation_fingerprint():
            raise ValueError("Queued implementation changed; submit a new job with the updated code")
        values = dict(request["server"])
        for key in ("path", "training_config", "runs_root", "output_root", "cache_root"):
            values[key] = Path(values[key])
        values["distributed"] = distributed_config(values["distributed"])
        server = ServerConfig(**values)
        raw = validate_train_config(request["raw"], server.path).raw
        if resume_fingerprint(raw) != request["resume_fingerprint"]:
            raise ValueError("Queued resume checkpoint changed; submit a new job")
        original = os.environ.get("CUDA_VISIBLE_DEVICES")
        devices = request["cuda_visible_devices"]
        if original is not None and not set(devices.split(",")) <= set(original.split(",")):
            raise ValueError("Queued GPUs are outside this worker's CUDA_VISIBLE_DEVICES allocation")
        try:
            os.environ["CUDA_VISIBLE_DEVICES"] = devices
            with open_run_source(server.runs_root) as root:
                task = choose_task(discover_tasks(root, raw["data"]["project_id"]), server.task_id)
                if input_fingerprint(task, raw, server.reward_source) != request["input_fingerprint"]:
                    raise ValueError("Queued dataset or rewards changed; submit a new job")
                if self._job(self.list(), job["id"]).get("cancel_requested") or stopping():
                    self.update(job["id"], status="CANCELLED")
                    return
                distributed = replace(server.distributed, gpu_ids=tuple(range(server.distributed.world_size)))
                run = ServerRun(replace(server, runs_root=root), task, raw, distributed,
                                server.reward_source, directory=Path(job["run_directory"]))
                try:
                    run.start()
                    while run.poll() is None:
                        current = self._job(self.list(), job["id"])
                        if (current.get("cancel_requested") or stopping()) and run.cancel_at is None:
                            run.cancel()
                        time.sleep(.5)
                    status = run.state["status"]
                    if run.cancel_at is not None and status == "INTERRUPTED":
                        status = "CANCELLED"
                    self.update(job["id"], status=status, result=run.state.get("result"),
                                error=run.state.get("error"), finished_at=timestamp())
                finally:
                    run.shutdown()
        finally:
            if original is None:
                os.environ.pop("CUDA_VISIBLE_DEVICES", None)
            else:
                os.environ["CUDA_VISIBLE_DEVICES"] = original
