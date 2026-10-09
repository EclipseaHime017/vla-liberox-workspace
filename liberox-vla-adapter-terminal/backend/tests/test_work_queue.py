from types import SimpleNamespace
import threading

import pytest

from backend.app.core.exceptions import ConflictError
from backend.app.services.offline_job_service import OfflineJobService
from backend.app.services.work_resources import ResourceLease
from backend.app.storage.repositories import OfflineJobRepository
from backend.app.domain.run import SimulationSession
from backend.app.workers.simulation_worker import SimulationManager


@pytest.fixture
def jobs(tmp_path):
    service = object.__new__(OfflineJobService)
    service.project_root = tmp_path
    service.jobs_root = tmp_path / "jobs"
    service.jobs_root.mkdir()
    service.gpu_lock_path = tmp_path / ".gpu-task.lock"
    service.repository = OfflineJobRepository(tmp_path / "catalog.sqlite", "test")
    service.lock = threading.RLock()
    service._local_tasks = {}
    service.launch_reserved = False
    service.manager = SimpleNamespace(active_session_id=None, draft=None, provider=None,
                                     controller_status=lambda: {"state": "DISCONNECTED"})
    yield service
    service.close()


def test_all_work_kinds_share_fifo_and_manual_cannot_queue(jobs):
    release, entered = threading.Event(), threading.Event()
    order = []

    def first():
        entered.set()
        assert release.wait(5)
        order.append("simulation")

    job, first_done = jobs.submit_local("simulation", first, cancel=release.set)
    second, second_done = jobs.submit_local("export", lambda: order.append("export"))
    with pytest.raises(ConflictError, match="人工接管"):
        jobs.immediate_lease()
    jobs._dispatch_training_queue()
    assert entered.wait(2)
    jobs._dispatch_training_queue()
    assert not second_done.done()
    assert jobs._job_queue()["jobs"][0]["id"] == job["id"]
    with pytest.raises(ConflictError):
        ResourceLease(jobs.gpu_lock_path)
    release.set()
    first_done.result(3)
    jobs._dispatch_training_queue()
    second_done.result(3)
    assert order == ["simulation", "export"]
    assert jobs.get(second["id"])["status"] == "COMPLETED"
    with jobs.immediate_lease():
        with pytest.raises(ConflictError):
            ResourceLease(jobs.gpu_lock_path)


def test_cancel_queued_calls_owner_and_never_executes(jobs):
    canceled = threading.Event()
    job, done = jobs.submit_local("simulation", lambda: pytest.fail("Canceled work ran"), cancel=canceled.set)
    jobs.stop(job["id"])
    jobs._dispatch_training_queue()
    assert done.cancelled() and canceled.is_set()
    assert jobs.get(job["id"])["status"] == "CANCELED"


def test_dataset_job_persists_exact_created_id_without_full_manifest(jobs):
    dataset = {"id": "new-dataset", "members": [{"run_id": "owned"}]}
    job, done = jobs.submit_local("dataset", lambda: dataset)
    jobs._dispatch_training_queue()
    assert done.result(3) == dataset
    assert jobs.get(job["id"])["result"] == {"dataset_id": "new-dataset"}


def test_failure_and_persistence_failure_release_slot(jobs, monkeypatch):
    def fail():
        raise ValueError("bad source")
    job, done = jobs.submit_local("dataset", fail)
    jobs._dispatch_training_queue()
    with pytest.raises(ValueError, match="bad source"):
        done.result(3)
    assert jobs.get(job["id"])["status"] == "FAILED"
    _, next_done = jobs.submit_local("publication", lambda: None)
    save = jobs._save_work
    def fail_terminal(payload):
        if payload["status"] == "COMPLETED":
            raise OSError("disk full")
        save(payload)
    monkeypatch.setattr(jobs, "_save_work", fail_terminal)
    jobs._dispatch_training_queue()
    with pytest.raises(OSError, match="disk full"):
        next_done.result(3)
    with ResourceLease(jobs.gpu_lock_path):
        pass
    assert not jobs._local_tasks


def test_draft_does_not_hold_entire_queue_and_close_drains(jobs):
    jobs.manager.draft = object()
    entered, release = threading.Event(), threading.Event()
    def execute():
        entered.set()
        assert release.wait(5)
    _, done = jobs.submit_local("simulation", execute, cancel=release.set)
    jobs._dispatch_training_queue()
    assert entered.wait(2)
    jobs.close()
    assert done.done()
    with ResourceLease(jobs.gpu_lock_path):
        pass


def test_close_waits_for_dispatched_callback_before_future_runs(jobs, monkeypatch):
    entered, release, closed = threading.Event(), threading.Event(), threading.Event()
    original = jobs._run_local
    def delayed(payload, lease):
        entered.set()
        assert release.wait(3)
        original(payload, lease)
    monkeypatch.setattr(jobs, "_run_local", delayed)
    _, done = jobs.submit_local("dataset", lambda: "saved")
    jobs._dispatch_training_queue()
    assert entered.wait(2) and not done.running()
    closer = threading.Thread(target=lambda: (jobs.close(), closed.set()))
    closer.start()
    try:
        assert not closed.wait(.05)
    finally:
        release.set()
        closer.join(3)
    assert closed.is_set() and done.result() == "saved"


def test_simulation_queues_but_manual_only_runs_immediately(jobs, tmp_path):
    manager = object.__new__(SimulationManager)
    manager.lock = threading.RLock()
    manager._frame_lock = threading.Lock()
    manager.work_queue = jobs
    manager.sessions = {}
    manager.active_session_id = None
    draft = manager.draft = SimpleNamespace(id="existing-draft")
    manager.controller_status = lambda *_: {"state": "READY"}
    manager._persist_manifest = lambda _: None
    manager._run_session = lambda record: setattr(record, "status", "COMPLETED")
    jobs.manager = manager
    simulation = SimulationSession(id="sim", kind="original", output_dir=tmp_path, max_steps=8, open_loop_steps=8)
    manual = SimulationSession(id="human", kind="branch", output_dir=tmp_path, max_steps=8,
                               open_loop_steps=8, control_mode="manual")
    assert manager._start_record(simulation)["status"] == "QUEUED"
    assert manager.active_session_id is None
    with pytest.raises(ConflictError, match="人工接管"):
        manager._start_record(manual)
    assert manual.id not in manager.sessions
    done = jobs._local_tasks[simulation.work_job_id][1]
    jobs._dispatch_training_queue()
    done.result(3)
    assert manager.draft is draft
    manager.draft = None
    manager._start_record(manual)
    manual.thread.join(3)
    assert manual.status == "COMPLETED" and manual.work_job_id is None
    assert len(jobs._job_queue()["jobs"]) == 1
