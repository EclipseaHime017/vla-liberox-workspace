from __future__ import annotations

import asyncio
import os
import threading
from types import SimpleNamespace

import pytest
from fastapi import FastAPI, Request
from pydantic import ValidationError

from backend.app.api import evaluations
from backend.app.api.models import EvaluationQueueRequest
from backend.app.core.exceptions import ConflictError
from backend.app.services.offline_job_service import OfflineJobService
from backend.app.services.policy_management_service import PolicyManagementService
from backend.app.storage.files import atomic_write_json
from test_evaluation_api import evaluation_request, make_service


def request_for(service, **changes):
    request = evaluation_request(**changes)
    request["schedule_sha256"] = service.preview_evaluation(request)["schedule_sha256"]
    return request


def finish(service, identifier, status):
    path, payload = service._load_job(identifier)
    payload["status"] = status
    atomic_write_json(path, payload)  # Runner updates the manifest, not SQLite.


@pytest.fixture
def queue(tmp_path, monkeypatch):
    service = make_service(tmp_path)
    launched = []

    def spawn(payload):
        launched.append(payload["id"])
        path, current = service._load_job(payload["id"])
        current.update(status="RUNNING", pid=os.getpid())
        atomic_write_json(path, current)
        service.repository.upsert(current, path)
        return service._public_job(current)

    monkeypatch.setattr(service, "_spawn_job", spawn)
    return service, launched


@pytest.mark.parametrize("status", ["COMPLETED", "FAILED", "CANCELED"])
def test_fifo_and_terminal_progression(queue, status):
    service, launched = queue
    first = service.enqueue_evaluation(request_for(service))
    second = service.enqueue_evaluation(request_for(service, trials=3))
    assert first["status"] == second["status"] == "QUEUED"
    assert service.list_evaluations(status="QUEUED")[0]["id"] == second["id"]
    assert launched == []
    service._dispatch_training_queue()
    service._dispatch_training_queue()
    assert launched == [first["id"]]
    assert service.get_evaluation(first["id"])["status"] == "STARTING"
    finish(service, first["id"], status)
    service._dispatch_training_queue()
    assert launched == [first["id"], second["id"]]
    assert service.get_evaluation(first["id"])["status"] == status
    assert len(service.evaluation_queue()["jobs"]) == 2
    assert service.training_queue()["jobs"] == []


def test_registration_during_active_job_does_not_unload_or_change_snapshot(queue):
    service, launched = queue
    first = service.enqueue_evaluation(request_for(service))
    service._dispatch_training_queue()
    service.manager.provider.unload = lambda: pytest.fail("registration must not unload the active model")
    request = request_for(service, trials=2, max_steps=123, base_seed=29)
    second = service.enqueue_evaluation(request)
    config_path = service.jobs_root / second["id"] / "effective_config.yaml"
    frozen = config_path.read_bytes()
    request.update(trials=99, base_seed=31)
    service.manager.eval_config.control_hz = 10
    assert config_path.read_bytes() == frozen
    assert second["parameters"]["trials"] == 2
    assert second["parameters"]["base_seed"] == 29
    assert launched == [first["id"]]
    assert service.evaluation_queue()["jobs"][1]["parameters"]["max_steps"] == 123


def test_cancel_and_stop_are_local_and_pending_delete_is_rejected(queue, monkeypatch):
    service, launched = queue
    first, second, third = [service.enqueue_evaluation(request_for(service)) for _ in range(3)]
    with pytest.raises(ConflictError, match="must be stopped"):
        service.delete_evaluation(second["id"], second["id"])
    service._dispatch_training_queue()
    signals = []
    monkeypatch.setattr("os.kill", lambda pid, sig: signals.append((pid, sig)))
    assert service.stop_evaluation(second["id"])["status"] == "CANCELED"
    assert not signals
    assert service.get_evaluation(second["id"])["status"] == "CANCELED"
    assert service.stop_evaluation(first["id"])["status"] == "STOPPING"
    service._dispatch_training_queue()
    assert launched == [first["id"]]
    assert service.get(third["id"])["status"] == "QUEUED"
    finish(service, first["id"], "CANCELED")
    service._dispatch_training_queue()
    assert launched == [first["id"], third["id"]]
    service.delete_evaluation(second["id"], second["id"])


def test_shared_training_and_evaluation_fifo(queue):
    service, launched = queue
    first = service.enqueue_evaluation(request_for(service))
    config = service.jobs_root / "train-middle" / "effective_config.yaml"
    config.parent.mkdir()
    config.write_text("{}")
    middle = service._new_job(kind="training", dataset_id="dataset", stages=[], config_path=config,
                              output_path=config.parent, parameters={}, queued=True)
    last = service.enqueue_evaluation(request_for(service))
    for item in (first, middle, last):
        service._dispatch_training_queue()
        assert launched[-1] == item["id"]
        service._dispatch_training_queue()
        assert launched[-1] == item["id"]
        finish(service, item["id"], "COMPLETED")
    assert launched == [first["id"], middle["id"], last["id"]]
    assert [item["id"] for item in service.training_queue()["jobs"]] == [middle["id"]]


@pytest.mark.parametrize("resource", ["simulation", "draft", "controller", "gpu"])
def test_busy_resources_wait_without_rejecting_registration(queue, resource):
    service, launched = queue
    if resource == "simulation":
        service.manager.active_session_id = "sim"
    elif resource == "draft":
        service.manager.draft = {}
    elif resource == "controller":
        service.manager.controller_status = lambda: {"state": "ARMED"}
    else:
        service._external_gpu_lock = lambda: True
    job = service.enqueue_evaluation(request_for(service))
    service._dispatch_training_queue()
    assert service.get(job["id"])["status"] == "QUEUED"
    assert launched == []
    assert service.evaluation_queue()["waiting_reason"]


def test_restart_dispatches_without_browser(queue):
    service, launched = queue
    item = service.enqueue_evaluation(request_for(service))
    restored = OfflineJobService(service.ui_config, service.manager, service.datasets)
    ready = threading.Event()
    restored._spawn_job = lambda payload: (service._spawn_job(payload), ready.set())[0]
    restored.start_training_queue()
    try:
        assert ready.wait(5)
        assert launched == [item["id"]]
    finally:
        restored.close()
    assert service.get(item["id"])["status"] == "RUNNING"


def test_launch_failure_is_visible_in_history_and_next_can_run(queue, monkeypatch):
    service, launched = queue
    first, second = [service.enqueue_evaluation(request_for(service)) for _ in range(2)]
    spawn = service._spawn_job
    service._spawn_job = OfflineJobService._spawn_job.__get__(service)
    monkeypatch.setattr("backend.app.services.offline_job_service.subprocess.Popen",
                        lambda *a, **k: (_ for _ in ()).throw(OSError("missing executable")))
    with pytest.raises(OSError, match="missing executable"):
        service._dispatch_training_queue()
    assert service.get_evaluation(first["id"])["status"] == "FAILED"
    service._spawn_job = spawn
    service._dispatch_training_queue()
    assert launched == [second["id"]]


def test_exception_after_spawn_does_not_overwrite_live_child(queue, monkeypatch):
    service, launched = queue
    first, second = [service.enqueue_evaluation(request_for(service)) for _ in range(2)]
    spawn = service._spawn_job

    def running_then_error(payload):
        spawn(payload)
        raise OSError("launcher record failed after spawn")

    service._spawn_job = running_then_error
    with pytest.raises(OSError, match="after spawn"):
        service._dispatch_training_queue()
    assert service.get(first["id"])["status"] == "RUNNING"
    service._dispatch_training_queue()
    assert launched == [first["id"]]
    monkeypatch.setattr("os.kill", lambda *_: None)
    assert service.stop_evaluation(first["id"])["status"] == "STOPPING"
    assert service.get(second["id"])["status"] == "QUEUED"


@pytest.mark.parametrize("damage", ["missing", "invalid"])
def test_corrupt_pending_manifest_does_not_block_following(queue, damage):
    service, launched = queue
    first, second = [service.enqueue_evaluation(request_for(service)) for _ in range(2)]
    path = service._job_path(first["id"])
    if damage == "missing":
        path.unlink()
    else:
        path.write_text("[]")
    service._dispatch_training_queue()
    assert launched == [second["id"]]
    assert service.get_evaluation(first["id"])["status"] == "FAILED"


def test_queue_request_validates_preview_hash_and_api(queue):
    service, launched = queue
    request = request_for(service)
    body = EvaluationQueueRequest.model_validate(request)
    with pytest.raises(ValidationError):
        EvaluationQueueRequest.model_validate(evaluation_request())
    with pytest.raises(ValueError, match="重新预览"):
        service.enqueue_evaluation({**request, "schedule_sha256": "0" * 64})
    assert service.evaluation_queue()["jobs"] == []
    app = FastAPI()
    app.state.offline_job_service = service
    api_request = Request({"type": "http", "app": app})

    async def exercise():
        job = await evaluations.enqueue(body, api_request)
        assert job["status"] == "QUEUED"
        assert (await evaluations.queue(api_request))["jobs"][0]["id"] == job["id"]
        assert (await evaluations.stop(job["id"], api_request))["status"] == "CANCELED"

    asyncio.run(exercise())
    assert launched == []


def test_queued_policy_is_protected():
    service = PolicyManagementService(SimpleNamespace(policy_catalog=None, active_session_id=None, draft=None),
        jobs=SimpleNamespace(list=lambda: [{"status": "QUEUED", "parameters": {"policy_id": "overlay"}}]))
    with pytest.raises(ConflictError, match="active job"):
        service._assert_idle("overlay")
    service._assert_idle("other")
