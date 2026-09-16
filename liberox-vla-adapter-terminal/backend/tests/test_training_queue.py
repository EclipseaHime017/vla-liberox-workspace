from __future__ import annotations

import json
import os
import threading
import subprocess
import sys
import time
from types import SimpleNamespace

import pytest
import yaml

from backend.app.core.exceptions import ConflictError
from backend.app.services.offline_job_service import OfflineJobService
from backend.app.storage.files import atomic_write_json
from backend.app.storage.repositories import OfflineJobRepository
from test_dataset_reward_versions import finish, setup_jobs


@pytest.fixture
def queued_jobs(tmp_path, monkeypatch):
    jobs, dataset = setup_jobs(tmp_path)
    version = finish(jobs, dataset, jobs.start_annotation(dataset["id"], source="sparse"))
    jobs.project_root = tmp_path
    jobs.gpu_lock_path = tmp_path / ".gpu-task.lock"
    jobs.repository = OfflineJobRepository(tmp_path / "jobs.sqlite", "test")
    jobs.manager = SimpleNamespace(active_session_id=None, draft=None, provider=None,
                                  controller_status=lambda: {"state": "DISCONNECTED"})
    jobs._new_job = OfflineJobService._new_job.__get__(jobs)
    jobs._prepare_launch = OfflineJobService._prepare_launch.__get__(jobs)
    launched = []

    def spawn(payload):
        launched.append(payload["id"])
        path, current = jobs._load_job(payload["id"])
        current.update(status="RUNNING", pid=os.getpid())
        atomic_write_json(path, current)
        jobs.repository.upsert(current, path)
        return jobs._public_job(current)

    monkeypatch.setattr(jobs, "_spawn_job", spawn)
    return jobs, dataset, version, launched


def set_status(jobs, identifier, status):
    path, payload = jobs._load_job(identifier)
    payload["status"] = status
    atomic_write_json(path, payload)  # Detached runner changes JSON, not the index.


@pytest.mark.parametrize("end_status", ["COMPLETED", "FAILED", "CANCELED"])
def test_fifo_runs_once_and_terminal_statuses_do_not_cancel_following(queued_jobs, end_status):
    jobs, dataset, _, launched = queued_jobs
    first = jobs.enqueue_training(dataset["id"], {"micro_batch_size": 1})
    second = jobs.enqueue_training(dataset["id"], {"micro_batch_size": 4})
    assert first["status"] == second["status"] == "QUEUED"
    assert launched == []
    jobs._dispatch_training_queue()
    jobs._dispatch_training_queue()
    assert launched == [first["id"]]
    set_status(jobs, first["id"], end_status)
    jobs._dispatch_training_queue()
    assert launched == [first["id"], second["id"]]
    jobs._dispatch_training_queue()
    assert len(launched) == 2


def test_enqueue_during_training_pins_config_and_never_unloads_gpu(queued_jobs):
    jobs, dataset, version, launched = queued_jobs
    first = jobs.enqueue_training(dataset["id"], {"micro_batch_size": 1})
    jobs._dispatch_training_queue()
    second = jobs.enqueue_training(dataset["id"], {"micro_batch_size": 4, "reward_gamma": .92})
    assert launched == [first["id"]]
    config = jobs.jobs_root / second["id"] / "effective_config.yaml"
    before = config.read_bytes()
    saved = yaml.safe_load(before)
    assert saved["reward"]["version_id"] == version["id"]
    assert saved["reward"]["gamma"] == .92
    assert saved["iql"]["micro_batch_size"] == 4
    assert saved["iql"]["resume_checkpoint"] is None
    # Change the current dataset reward; the pending task must keep its pin.
    new_job = jobs._new_job
    jobs._new_job = lambda **kwargs: kwargs
    finish(jobs, dataset, jobs.start_annotation(dataset["id"], source="sparse", gamma=.8))
    jobs._new_job = new_job
    assert config.read_bytes() == before
    set_status(jobs, first["id"], "COMPLETED")
    jobs._dispatch_training_queue()
    assert config.read_bytes() == before


def test_cancel_waiting_job_is_local_and_current_stop_does_not_touch_queue(queued_jobs, monkeypatch):
    jobs, dataset, _, launched = queued_jobs
    first = jobs.enqueue_training(dataset["id"], {})
    canceled = jobs.enqueue_training(dataset["id"], {})
    last = jobs.enqueue_training(dataset["id"], {})
    jobs._dispatch_training_queue()
    monkeypatch.setattr("os.kill", lambda *_: None)
    assert jobs.stop(canceled["id"])["status"] == "CANCELED"
    assert jobs.stop(first["id"])["status"] == "STOPPING"
    jobs._dispatch_training_queue()
    assert launched == [first["id"]]
    assert jobs.get(last["id"])["status"] == "QUEUED"
    set_status(jobs, first["id"], "CANCELED")
    jobs._dispatch_training_queue()
    assert launched == [first["id"], last["id"]]


@pytest.mark.parametrize("resource", ["simulation", "draft", "gpu", "controller"])
def test_resources_pause_dispatch_without_failing_job(queued_jobs, resource):
    jobs, dataset, _, launched = queued_jobs
    item = jobs.enqueue_training(dataset["id"], {})
    if resource == "simulation":
        jobs.manager.active_session_id = "sim"
    elif resource == "draft":
        jobs.manager.draft = {}
    elif resource == "gpu":
        jobs._external_gpu_lock = lambda: True
    else:
        jobs.manager.controller_status = lambda: {"state": "ARMED"}
    jobs._dispatch_training_queue()
    assert launched == []
    assert jobs.training_queue()["waiting_reason"]
    assert jobs.get(item["id"])["status"] == "QUEUED"


def test_queue_survives_service_restart_and_needs_no_browser_polling(queued_jobs):
    jobs, dataset, _, launched = queued_jobs
    item = jobs.enqueue_training(dataset["id"], {})
    restored = object.__new__(OfflineJobService)
    restored.__dict__.update(jobs.__dict__)
    restored.lock = threading.RLock()
    started = threading.Event()
    spawn = jobs._spawn_job
    restored._spawn_job = lambda payload: (spawn(payload), started.set())[0]
    restored.start_training_queue()
    try:
        assert started.wait(5)
        assert launched == [item["id"]]
    finally:
        restored.close_training_queue()
    assert jobs.get(item["id"])["status"] == "RUNNING"  # Closing backend doesn't stop the child.


def test_long_wait_uses_dispatch_timestamp_and_queued_dataset_cannot_be_deleted(queued_jobs):
    jobs, dataset, _, _ = queued_jobs
    item = jobs.enqueue_training(dataset["id"], {})
    path, payload = jobs._load_job(item["id"])
    payload["created_at"] = "2000-01-01T00:00:00+00:00"
    atomic_write_json(path, payload)
    with pytest.raises(ConflictError, match="active background job"):
        jobs.delete_dataset(dataset["id"], dataset["id"], force=True)
    jobs._spawn_job = lambda payload: payload  # Slow process start with no PID yet.
    jobs._dispatch_training_queue()
    assert jobs.get(item["id"])["status"] == "STARTING"


def test_queue_rejects_resume_and_invalid_settings_before_registration(queued_jobs):
    jobs, dataset, _, _ = queued_jobs
    for parameters in ({"resume_checkpoint": "/old.pt"}, {"micro_batch_size": 0}):
        with pytest.raises(ValueError):
            jobs.enqueue_training(dataset["id"], parameters)
    assert jobs.training_queue()["jobs"] == []


def test_spawn_failure_only_fails_selected_task(queued_jobs, monkeypatch):
    jobs, dataset, _, launched = queued_jobs
    first = jobs.enqueue_training(dataset["id"], {})
    second = jobs.enqueue_training(dataset["id"], {})
    spawn = jobs._spawn_job
    jobs._spawn_job = OfflineJobService._spawn_job.__get__(jobs)
    monkeypatch.setattr("backend.app.services.offline_job_service.subprocess.Popen",
                        lambda *_args, **_kwargs: (_ for _ in ()).throw(OSError("missing executable")))
    with pytest.raises(OSError, match="missing executable"):
        jobs._dispatch_training_queue()
    assert jobs.get(first["id"])["status"] == "FAILED"
    assert not jobs.launch_reserved
    jobs._spawn_job = spawn
    jobs._dispatch_training_queue()
    assert launched == [second["id"]]


def test_fast_runner_status_is_not_overwritten_by_launcher(queued_jobs, monkeypatch):
    jobs, dataset, _, _ = queued_jobs
    item = jobs.enqueue_training(dataset["id"], {})
    jobs._spawn_job = OfflineJobService._spawn_job.__get__(jobs)

    def fast_runner(*_args, **_kwargs):
        set_status(jobs, item["id"], "COMPLETED")
        return SimpleNamespace(pid=os.getpid())

    monkeypatch.setattr("backend.app.services.offline_job_service.subprocess.Popen", fast_runner)
    jobs._dispatch_training_queue()
    assert jobs.get(item["id"])["status"] == "COMPLETED"
    jobs._dispatch_training_queue()
    assert jobs.repository.training_queue_ids(pending_only=True) == []


@pytest.mark.parametrize("damage", ["missing", "corrupt", "array"])
def test_unreadable_job_does_not_block_later_tasks(queued_jobs, damage):
    jobs, dataset, _, launched = queued_jobs
    first = jobs.enqueue_training(dataset["id"], {})
    second = jobs.enqueue_training(dataset["id"], {})
    path = jobs._job_path(first["id"])
    if damage == "missing":
        path.unlink()
    else:
        path.write_text("[]" if damage == "array" else "not json")
    jobs._dispatch_training_queue()
    assert launched == [second["id"]]
    broken = next(item for item in jobs.training_queue()["jobs"] if item["id"] == first["id"])
    assert broken["status"] == "FAILED" and broken["error"]


def test_zombie_launcher_does_not_block_next_training(queued_jobs):
    jobs, dataset, _, launched = queued_jobs
    first = jobs.enqueue_training(dataset["id"], {})
    second = jobs.enqueue_training(dataset["id"], {})
    child = subprocess.Popen([sys.executable, "-c", "pass"])
    try:
        path, payload = jobs._load_job(first["id"])
        payload.update(status="RUNNING", pid=child.pid)
        atomic_write_json(path, payload)
        jobs.repository.upsert(payload, path)
        deadline = time.monotonic() + 5
        while jobs._pid_alive(child.pid) and time.monotonic() < deadline:
            time.sleep(.01)
        assert not jobs._pid_alive(child.pid)
        jobs._dispatch_training_queue()
        assert jobs.get(first["id"])["status"] == "FAILED"
        assert launched == [second["id"]]
    finally:
        child.wait(timeout=5)


def test_shutdown_requested_before_dispatch_does_not_start_waiting_task(queued_jobs):
    jobs, dataset, _, launched = queued_jobs
    item = jobs.enqueue_training(dataset["id"], {})
    jobs._queue_stop = threading.Event()
    jobs._queue_stop.set()
    jobs._dispatch_training_queue()
    assert launched == []
    assert jobs.get(item["id"])["status"] == "QUEUED"


def test_cancel_recovered_start_without_pid_does_not_stick_in_stopping(queued_jobs):
    jobs, dataset, _, launched = queued_jobs
    first = jobs.enqueue_training(dataset["id"], {})
    second = jobs.enqueue_training(dataset["id"], {})
    path, payload = jobs._load_job(first["id"])
    payload.update(status="STARTING", dispatched_at="2000-01-01T00:00:00+00:00")
    atomic_write_json(path, payload)
    jobs.repository.upsert(payload, path)
    assert jobs.stop(first["id"])["status"] == "STOPPING"
    jobs._dispatch_training_queue()
    assert jobs.get(first["id"])["status"] == "CANCELED"
    assert launched == [second["id"]]
