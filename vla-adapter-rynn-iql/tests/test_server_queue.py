import copy
import fcntl
import json
import os
from pathlib import Path
import signal
import subprocess
import sys
import time
from types import SimpleNamespace

import pytest

from vla_rynn_iql import model_assets, server_resources
from vla_rynn_iql.base_models import model_contract
from vla_rynn_iql.server_queue import TrainingQueue, resume_fingerprint
from vla_rynn_iql.server_resources import TrainingBusy, TrainingCleanupError, TrainingLease, process_identity, process_alive
from test_server_inputs import global_runs
from test_server_pipeline import configuration


@pytest.fixture
def queued(configured, tmp_path, monkeypatch):
    task, _, _ = global_runs(configured)
    server = configuration(tmp_path, task, configured)
    monkeypatch.setattr(server_resources, "LEASE_ROOT", tmp_path / "leases")
    monkeypatch.setattr(model_assets, "pin_model", lambda settings: {
        **settings, "contract": model_contract(settings), "base_revision": "a" * 64})
    return TrainingQueue(tmp_path / "queue"), server


def test_queue_freezes_configuration_and_variant_observation(queued, monkeypatch):
    from dataclasses import replace
    from vla_rynn_iql import base_models
    queue, server = queued
    job = queue.add(server)
    original = copy.deepcopy(queue.request(job))
    server.path.write_text("not: a valid configuration anymore\n")
    base = base_models.BASE_MODELS[original["raw"]["model"]["base_id"]]
    contract = copy.deepcopy(base.contract)
    contract["observation"]["image_flip_vertical"] = False
    monkeypatch.setitem(base_models.BASE_MODELS, base.id, replace(base, contract=contract))
    seen = []

    class Run:
        process = None
        cancel_at = None
        state = {"status": "COMPLETED"}

        def __init__(self, server, task, raw, *args, **kwargs):
            seen.append(raw)

        def start(self):
            pass

        def poll(self):
            return 0

        def shutdown(self):
            pass

    monkeypatch.setattr("vla_rynn_iql.server_pipeline.ServerRun", Run)
    queue.run(once=True)
    assert queue.list()["jobs"][0]["status"] == "COMPLETED"
    assert seen[0]["training"] == original["raw"]["training"]
    assert seen[0]["model"]["contract"] == original["raw"]["model"]["contract"]
    assert seen[0]["model"]["base_revision"] == "a" * 64


@pytest.mark.parametrize("model,batch,mode", [("base", 64, "lora"), ("pi05-libero-base", 32, "full")])
def test_queue_freezes_selected_model_training_profile(queued, model, batch, mode):
    from vla_rynn_iql.server_config import configure_server
    queue, server = queued
    selected = configure_server(server.path, model=model, method="bc")
    job = queue.add(selected)
    request = queue.request(job)
    assert request["raw"]["training"]["micro_batch_size"] == batch
    assert request["raw"]["training"]["gradient_accumulation_steps"] == 1
    assert request["raw"]["model"]["backbone"] == mode
    family = request["raw"]["model"]["family"]
    selected.model_profiles[family]["training"]["micro_batch_size"] = 8
    assert queue.request(job)["raw"] == request["raw"]
    assert queue.request(job)["server"]["model_profiles"][family]["training"]["micro_batch_size"] == batch


def test_fifo_continues_after_failure_and_retry_is_explicit(queued, monkeypatch):
    queue, server = queued
    first, second = queue.add(server), queue.add(server)
    order = []

    def execute(job, stopping):
        order.append(job["id"])
        if job["id"] == first["id"]:
            raise ValueError("bad model")
        queue.update(job["id"], status="COMPLETED")

    monkeypatch.setattr(queue, "_execute", execute)
    queue.run(once=True)
    assert order == [first["id"], second["id"]]
    assert [job["status"] for job in queue.list()["jobs"]] == ["FAILED", "COMPLETED"]
    new = queue.retry(first["id"])
    assert new["id"] != first["id"] and new["status"] == "QUEUED"
    assert new["request_sha256"] == first["request_sha256"]
    with pytest.raises(ValueError, match="stopped"):
        queue.retry(second["id"])


def test_pending_cancel_and_stop_keep_other_jobs(queued, monkeypatch):
    queue, server = queued
    first, second, third = (queue.add(server) for _ in range(3))
    queue.cancel(first["id"])

    def execute(job, stopping):
        assert job["id"] == second["id"]
        queue.stop()
        queue.update(job["id"], status="COMPLETED")

    monkeypatch.setattr(queue, "_execute", execute)
    queue.run(once=True)
    assert [job["status"] for job in queue.list()["jobs"]] == ["CANCELLED", "COMPLETED", "QUEUED"]
    assert queue.list()["jobs"][2]["id"] == third["id"]


def test_signal_handler_does_not_reenter_state_lock(queued, monkeypatch):
    queue, server = queued
    queue.add(server)

    def execute(job, stopping):
        with queue.state():
            signal.raise_signal(signal.SIGTERM)
            assert stopping()
        queue.update(job["id"], status="INTERRUPTED")

    previous = signal.getsignal(signal.SIGTERM)
    monkeypatch.setattr(queue, "_execute", execute)
    queue.run(once=True)
    assert signal.getsignal(signal.SIGTERM) == previous


def test_changed_dataset_fails_before_launch(queued, monkeypatch):
    queue, server = queued
    job = queue.add(server)
    path = next(server.runs_root.rglob("trajectory.npz"))
    path.write_bytes(path.read_bytes() + b"changed")
    monkeypatch.setattr("vla_rynn_iql.server_pipeline.ServerRun", lambda *_a, **_kw: pytest.fail("must not launch"))
    queue.run(once=True)
    record = queue.list()["jobs"][0]
    assert record["status"] == "FAILED"
    assert "dataset or rewards changed" in record["error"]


def test_modified_request_is_rejected(queued):
    queue, server = queued
    job = queue.add(server)
    path = queue.root / "jobs" / job["id"] / "request.json"
    changed = queue.request(job)
    changed["raw"]["training"]["train_steps"] += 1
    path.write_text(json.dumps(changed))
    with pytest.raises(ValueError, match="configuration was modified"):
        queue.request(job)


def test_resume_checkpoint_is_content_bound(queued, tmp_path):
    from dataclasses import replace
    queue, server = queued
    checkpoint = tmp_path / "checkpoint"
    checkpoint.mkdir()
    weights = checkpoint / "actor.pt"
    weights.write_bytes(b"first")
    changes = copy.deepcopy(server.overrides)
    changes["training"]["resume_checkpoint"] = str(checkpoint)
    job = queue.add(replace(server, overrides=changes))
    request = queue.request(job)
    assert request["resume_fingerprint"] == resume_fingerprint(request["raw"])
    weights.write_bytes(b"different")
    queue.run(once=True)
    assert "resume checkpoint changed" in queue.list()["jobs"][0]["error"]


def test_reconcile_never_duplicates_orphan_and_handles_launch_boundary_cancel(queued):
    from vla_rynn_iql.io import atomic_json
    queue, server = queued
    job = queue.add(server)
    directory = Path(job["run_directory"])
    directory.mkdir(parents=True)
    queue.update(job["id"], status="RUNNING")
    queue.cancel(job["id"])
    atomic_json(directory / "pipeline.json", {"status": "PREPARING"})
    with TrainingLease():
        assert not queue.reconcile()
        assert (directory / "cancel.request").is_file()
        assert queue.list()["jobs"][0]["status"] == "RUNNING"
    assert queue.reconcile()
    assert queue.list()["jobs"][0]["status"] == "CANCELLED"


def test_reconcile_adopts_completed_summary(queued):
    from vla_rynn_iql.io import atomic_json
    queue, server = queued
    job = queue.add(server)
    directory = Path(job["run_directory"])
    directory.mkdir(parents=True)
    queue.update(job["id"], status="RUNNING")
    atomic_json(directory / "summary.json", {"status": "COMPLETED", "step": 10000})
    assert queue.reconcile()
    assert queue.list()["jobs"][0]["status"] == "COMPLETED"


def test_duplicate_worker_lock_is_rejected(queued):
    queue, _ = queued
    queue.root.mkdir()
    with (queue.root / "runner.lock").open("a") as lock:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        with pytest.raises(ValueError, match="already running"):
            queue.run(once=True)


def test_resource_lease_survives_monitor_close_until_supervisor_exits(tmp_path, monkeypatch):
    monkeypatch.setattr(server_resources, "LEASE_ROOT", tmp_path / "lease")
    lease = TrainingLease()
    child = subprocess.Popen([sys.executable, "-B", "-c", "import sys; print('ready', flush=True); sys.stdin.read()"],
        pass_fds=(lease.fd,), stdin=subprocess.PIPE, stdout=subprocess.PIPE, text=True)
    try:
        assert child.stdout.readline().strip() == "ready"
        lease.close()
        with pytest.raises(TrainingBusy):
            TrainingLease()
        child.communicate("", timeout=10)
        with TrainingLease():
            pass
    finally:
        lease.close()
        if child.poll() is None:
            child.kill()
        child.wait(timeout=10)


def test_pid_identity_rejects_reused_pid_or_old_boot():
    identity = process_identity(os.getpid())
    assert process_alive(identity)
    assert not process_alive({**identity, "start": "different"})
    assert not process_alive({**identity, "boot": "different"})


def test_background_worker_ready_stop_and_no_duplicate(queued):
    queue, _ = queued
    identity = queue.start()
    try:
        assert process_alive(identity)
        assert queue.list()["worker_ready"]
        with pytest.raises(ValueError, match="already running"):
            queue.start()
        queue.stop()
        deadline = time.monotonic() + 10
        while process_alive(identity) and time.monotonic() < deadline:
            time.sleep(.05)
        assert not process_alive(identity)
    finally:
        queue.stop()
        try:
            os.waitpid(identity["pid"], 0)
        except ChildProcessError:
            pass


def test_failed_child_start_is_not_acknowledged(queued, monkeypatch):
    queue, _ = queued
    monkeypatch.setattr("vla_rynn_iql.server_queue.subprocess.Popen",
                        lambda *_a, **_k: SimpleNamespace(pid=99999999, poll=lambda: 1))
    with pytest.raises(RuntimeError, match="worker failed"):
        queue.start()


def test_list_does_not_import_training_or_gpu_runtime(tmp_path):
    from vla_rynn_iql.config import PROJECT_ROOT
    program = """
import importlib.abc,runpy,sys
class Block(importlib.abc.MetaPathFinder):
    def find_spec(self, fullname, path=None, target=None):
        if fullname.split('.')[0] in {'torch', 'numpy', 'transformers', 'openpi'}:
            raise RuntimeError('Heavy import during queue status: '+fullname)
sys.meta_path.insert(0,Block())
sys.argv=sys.argv[1:]
runpy.run_path(sys.argv[0],run_name='__main__')
"""
    result = subprocess.run([sys.executable, "-B", "-c", program,
        str(PROJECT_ROOT / "scripts/train_queue.py"), "--queue", str(tmp_path / "empty-queue"), "list"],
        text=True, capture_output=True, timeout=10)
    assert result.returncode == 0, result.stderr
    assert json.loads(result.stdout)["jobs"] == []


def test_running_cancel_reuses_server_run_safe_stop(queued, monkeypatch):
    queue, server = queued
    job = queue.add(server)
    stopped = []

    class Run:
        cancel_at = None
        process = None
        state = {"status": "RUNNING"}

        def __init__(self, *_a, **_k):
            pass

        def start(self):
            self.process = self
            queue.cancel(job["id"])

        def poll(self):
            return 130 if self.cancel_at else None

        def cancel(self):
            stopped.append(True)
            self.cancel_at = 1
            self.state["status"] = "INTERRUPTED"

        def shutdown(self):
            pass

    monkeypatch.setattr("vla_rynn_iql.server_pipeline.ServerRun", Run)
    queue.run(once=True)
    assert stopped == [True]
    assert queue.list()["jobs"][0]["status"] == "CANCELLED"


def test_zip_inputs_live_until_training_finishes(configured, tmp_path, monkeypatch):
    import shutil
    from dataclasses import replace
    from test_server_inputs import portable_export
    monkeypatch.setattr(server_resources, "LEASE_ROOT", tmp_path / "leases")
    monkeypatch.setattr(model_assets, "pin_model", lambda settings: {
        **settings, "contract": model_contract(settings), "base_revision": "a" * 64})
    root, config, _ = portable_export(configured, tmp_path)
    from vla_rynn_iql.server_inputs import discover_tasks
    task, = discover_tasks(root, config.raw["data"]["project_id"])
    server = configuration(tmp_path, task, config)
    archive = Path(shutil.make_archive(str(tmp_path / "dataset-export"), "zip", root))
    queue = TrainingQueue(tmp_path / "queue")
    queue.add(replace(server, runs_root=archive))
    views = []

    class Run:
        process = None
        cancel_at = None
        state = {"status": "COMPLETED"}

        def __init__(self, server, *_a, **_k):
            views.append(server.runs_root)

        def start(self):
            assert list(views[0].rglob("trajectory_observations.npz"))

        def poll(self):
            assert views[0].is_dir()
            return 0

        def shutdown(self):
            pass

    monkeypatch.setattr("vla_rynn_iql.server_pipeline.ServerRun", Run)
    queue.run(once=True)
    assert queue.list()["jobs"][0]["status"] == "COMPLETED"
    assert not views[0].exists() and archive.is_file()


def test_registered_rank_blocks_new_jobs_even_without_supervisor_fd(tmp_path, monkeypatch):
    monkeypatch.setattr(server_resources, "LEASE_ROOT", tmp_path / "lease")
    lease = TrainingLease()
    program = """
import sys
from pathlib import Path
from vla_rynn_iql import server_resources as resources
resources.LEASE_ROOT=Path(sys.argv[1])
resources.register_worker(sys.argv[2])
print('ready',flush=True)
sys.stdin.read()
"""
    child = subprocess.Popen([sys.executable, "-B", "-c", program, str(server_resources.LEASE_ROOT), lease.token],
                             stdin=subprocess.PIPE, stdout=subprocess.PIPE, text=True)
    try:
        assert child.stdout.readline().strip() == "ready"
        lease.close()
        with pytest.raises(TrainingBusy, match="Previous server worker"):
            TrainingLease()
        child.communicate("", timeout=10)
        with TrainingLease():
            pass
    finally:
        lease.close()
        if child.poll() is None:
            child.kill()
        child.wait(timeout=10)


def test_stale_launcher_cannot_join_a_new_resource_lease(tmp_path, monkeypatch):
    monkeypatch.setattr(server_resources, "LEASE_ROOT", tmp_path / "lease")
    with TrainingLease() as previous:
        token = previous.token
    with pytest.raises(TrainingBusy, match="lease changed"):
        server_resources.register_worker(token)
    with TrainingLease():
        with pytest.raises(TrainingBusy, match="lease changed"):
            server_resources.register_worker(token)


def test_unsafe_cleanup_halts_queue_without_finalizing_or_advancing(queued, monkeypatch):
    queue, server = queued
    queue.add(server)
    queue.add(server)

    def execute(*_):
        raise TrainingCleanupError("rank still active")

    monkeypatch.setattr(queue, "_execute", execute)
    with pytest.raises(TrainingCleanupError):
        queue.run(once=True)
    assert [job["status"] for job in queue.list()["jobs"]] == ["RUNNING", "QUEUED"]
    assert queue.list()["worker"] is None


@pytest.mark.parametrize("failing", ["cancel", "poll"])
def test_shutdown_reaps_child_despite_status_io_failure(tmp_path, monkeypatch, failing):
    from vla_rynn_iql.server_pipeline import ServerRun
    monkeypatch.setattr(server_resources, "LEASE_ROOT", tmp_path / "lease")
    run = ServerRun.__new__(ServerRun)
    run.lease, run.log = TrainingLease(), None
    run.directory, run.cancel_at = tmp_path, None
    run.state = {}
    run.process = subprocess.Popen([sys.executable, "-B", "-c", "import sys; sys.stdin.read()"],
                                   stdin=subprocess.PIPE, start_new_session=True, pass_fds=(run.lease.fd,))

    def fail():
        raise OSError("output filesystem unavailable")

    monkeypatch.setattr(run, failing, fail)
    try:
        with pytest.raises(OSError, match="output filesystem"):
            run.shutdown()
        assert run.process.poll() is not None and run.lease is None
        with TrainingLease():
            pass
    finally:
        if run.process.poll() is None:
            run.process.kill()
        run.process.communicate(timeout=10)
        run.close()


def test_supervisor_exit_reaps_registered_rank_before_finalizing(tmp_path, monkeypatch):
    from vla_rynn_iql.server_pipeline import ServerRun
    monkeypatch.setattr(server_resources, "LEASE_ROOT", tmp_path / "lease")
    run = ServerRun.__new__(ServerRun)
    run.lease, run.log = TrainingLease(), None
    run.directory, run.cancel_at, run.state = tmp_path, None, {}
    program = """
import sys
from pathlib import Path
from vla_rynn_iql import server_resources as resources
resources.LEASE_ROOT=Path(sys.argv[1])
resources.register_worker(sys.argv[2])
print('ready',flush=True)
sys.stdin.read()
"""
    rank = subprocess.Popen([sys.executable, "-B", "-c", program, str(server_resources.LEASE_ROOT), run.lease.token],
                            stdin=subprocess.PIPE, stdout=subprocess.PIPE, text=True, start_new_session=True)
    try:
        assert rank.stdout.readline().strip() == "ready"
        run.process = subprocess.Popen([sys.executable, "-B", "-c", "pass"], start_new_session=True)
        run.process.wait(timeout=10)
        assert run.poll() != 0
        assert run.state["status"] == "FAILED" and "active ranks" in run.state["error"]
        rank.wait(timeout=10)
        with TrainingLease():
            pass
    finally:
        if rank.poll() is None:
            rank.kill()
        rank.communicate(timeout=10)
        run.close()
