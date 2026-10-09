import asyncio
import json
import shutil
from pathlib import Path
from types import SimpleNamespace

import httpx
import pytest
from fastapi import FastAPI

from backend.app.api.annotation_lab import router
from backend.app.services.annotation_lab_service import AnnotationLabService
from backend.app.storage.files import atomic_write_json
from test_evaluation_api import make_service


@pytest.fixture
def lab(tmp_path, monkeypatch):
    queue = make_service(tmp_path)
    episode = queue.project_root / "runs/task/run/episodes/episode_000"
    episode.mkdir(parents=True)
    for name in ("trajectory.npz", "trajectory_observations.npz", "stage_annotation.json", "rynnvalue_evaluation.json", "robometer_evaluation.json"):
        (episode / name).write_bytes(b"read-only fixture")
    run = {"id": "one", "task_id": "task", "task": "pick the bowl", "status": "COMPLETED",
           "action_count": 10, "trajectory": str(episode / "trajectory.npz")}
    atomic_write_json(episode.parent.parent / "run.json", run)
    second = episode.parents[2] / "run_two" / "episodes" / "episode_000"
    shutil.copytree(episode, second)
    second_run = {**run, "id": "two", "trajectory": str(second / "trajectory.npz")}
    atomic_write_json(second.parent.parent / "run.json", second_run)
    def get(identifier):
        if identifier == "two": return second_run
        if identifier != "one": raise KeyError(identifier)
        return run
    service = AnnotationLabService(SimpleNamespace(get_run=get), queue.project_root, queue)
    monkeypatch.setattr(service, "defaults", lambda: {"model_available": True, "environment_available": True})
    return service, queue, episode


def test_queue_registration_is_independent_and_read_only(lab):
    service, queue, episode = lab
    before = {p.name: p.read_bytes() for p in episode.iterdir()}
    first = service.start(["one"], {"coarse_fps": 2.0})
    assert first["status"] == "QUEUED" and first["proposal_only"]
    _, payload = queue._load_job(first["id"])
    assert payload["kind"] == "annotation_lab" and payload["dataset_id"] is None
    assert payload["requires_gpu"] and payload["parameters"]["run_ids"] == ["one"]
    stage = payload["stages"][0]
    assert stage["environment"] == "keyframe-vlm" and "scripts/annotate.py" in stage["argv"][1]
    assert first["config"]["coarse_fps"] == 2.0 and first["config"]["mode"] == "localize" and first["config"]["window_seconds"] == 2
    assert service.source("one")["control_hz"] is None  # worker reads recording metadata
    assert service.list()[0]["id"] == first["id"]
    assert service.stop(first["id"])["status"] == "CANCELED"
    assert before == {p.name: p.read_bytes() for p in episode.iterdir()}


def test_bad_input_does_not_register_jobs(lab, monkeypatch):
    service, queue, _ = lab
    for ids, options in [(["one", "one"], {}), (["../one"], {}), (["one"], {"model_id": "substitute"}),
                         (["one"], {"cameras": ["other"]}), (["one"], {"coarse_fps": float("nan")}),
                         (["one"], {"coarse_fps": True}), (["one"], {"coarse_fps": 0})]:
        with pytest.raises(ValueError): service.start(ids, options)
    assert not queue.repository.queue_ids()
    monkeypatch.setattr(service, "defaults", lambda: {"model_available": False, "environment_available": True, "message": "not installed"})
    with pytest.raises(ValueError, match="not installed"): service.start(["one"], {})
    assert not queue.repository.queue_ids()


def test_results_evidence_and_api_validation(lab):
    service, queue, _ = lab
    app = FastAPI()
    app.state.annotation_lab_service = service
    app.include_router(router)
    async def check():
        async with httpx.AsyncClient(transport=httpx.ASGITransport(app=app), base_url="http://test") as client:
            assert (await client.post("/api/annotation-lab/experiments", json={"run_ids": ["one"], "options": {"command": "bad"}})).status_code == 422
            for fps in [0, -1, 11, True, "1"]:
                assert (await client.post("/api/annotation-lab/experiments", json={"run_ids": ["one"], "options": {"coarse_fps": fps}})).status_code == 422
            for window in [0, -1, 5, True, "2"]:
                assert (await client.post("/api/annotation-lab/experiments", json={"run_ids": ["one"], "options": {"window_seconds": window}})).status_code == 422
            for mode in ["invalid", True, 1]:
                assert (await client.post("/api/annotation-lab/experiments", json={"run_ids": ["one"], "options": {"mode": mode}})).status_code == 422
            response = await client.post("/api/annotation-lab/experiments", json={"run_ids": ["one"], "options": {"mode": "localize", "coarse_fps": 5, "window_seconds": 2}})
            assert response.status_code == 202
            assert response.json()["config"]["mode"] == "localize"
            assert response.json()["config"]["coarse_fps"] == 5
            assert response.json()["config"]["window_seconds"] == 2
            identifier = response.json()["id"]
            root = f"/api/annotation-lab/experiments/{identifier}"
            assert (await client.get(root + "/runs/one")).json() is None
            assert (await client.get(root + "/runs/other")).status_code == 422
            output = service.root / identifier
            atomic_write_json(output / "one/result.json", {"status": "COMPLETED", "proposals": []})
            atomic_write_json(output / "one/calls/plan_0.json", {"raw_response": "{}"})
            evidence = output / "one/evidence"
            evidence.mkdir()
            (evidence / "agentview_image_0.jpg").write_bytes(b"image")
            result = (await client.get(root + "/runs/one")).json()
            assert result["calls"][0]["raw_response"] == "{}"
            image = await client.get(root + "/runs/one/evidence/agentview_image/0")
            assert image.status_code == 200 and image.headers["content-type"] == "image/jpeg"
            assert (await client.get(root + "/runs/one/evidence/invalid/0")).status_code == 422
            assert (await client.post(root + "/stop")).json()["status"] == "CANCELED"
    asyncio.run(check())


def test_lab_and_other_jobs_share_fifo(lab, monkeypatch):
    service, queue, _ = lab
    launched = []
    def spawn(payload, **kwargs):
        launched.append(payload["id"])
        return payload
    monkeypatch.setattr(queue, "_spawn_job", spawn)
    first = service.start(["one"], {})
    second = service.start(["two"], {})
    queue._dispatch_training_queue()
    assert launched == [first["id"]]
    path, payload = queue._load_job(first["id"])
    payload.update(status="COMPLETED")
    atomic_write_json(path, payload)
    queue._dispatch_training_queue()
    assert launched == [first["id"], second["id"]]
    assert service.get(first["id"])["status"] == "COMPLETED"
    # No training or reward collaborator is installed in this fixture.
    assert vars(queue.datasets) == {}


def complete(queue, identifier):
    path, payload = queue._load_job(identifier)
    payload.update(status="COMPLETED")
    atomic_write_json(path, payload)
    queue.repository.upsert(payload, path)


def test_overwrite_is_per_trajectory_and_deletes_old_artifacts(lab):
    service, queue, episode = lab
    original = service.start(["one", "two"], {})
    for run_id in ["one", "two"]:
        atomic_write_json(service.root / original["id"] / run_id / "result.json", {"run_id": run_id})
    with pytest.raises(ValueError, match="未结束"): service.start(["one"], {})
    complete(queue, original["id"])
    atomic_write_json(service.root / original["id"] / "progress.json", {
        "status": "FAILED", "total_runs": 2, "completed_runs": 1, "failed_runs": 1, "error": "one failed",
        "runs": [{"run_id": "one", "status": "FAILED", "error": "one failed"},
                 {"run_id": "two", "status": "COMPLETED", "error": None}]})
    replacement = service.start(["one"], {})
    assert not (service.root / original["id"] / "one").exists()
    assert service.get(original["id"])["run_ids"] == ["two"]
    remaining = service.get(original["id"])
    assert remaining["status"] == "COMPLETED" and remaining["error"] is None
    assert remaining["progress"]["completed_runs"] == remaining["progress"]["total_runs"] == 1
    assert service.result(original["id"], "two")["run_id"] == "two"
    with pytest.raises(ValueError): service.result(original["id"], "one")
    service.start(["two"], {})
    assert not (service.root / original["id"]).exists()
    assert not (queue.jobs_root / original["id"]).exists()
    assert original["id"] not in queue.repository.queue_ids()
    assert (episode / "stage_annotation.json").read_bytes() == b"read-only fixture"
    service.stop(replacement["id"])
    assert service.current()["one"] == replacement["id"]  # cancellation cannot revive old results


def test_failed_enqueue_preserves_current_result(lab, monkeypatch):
    service, queue, _ = lab
    original = service.start(["one"], {})
    complete(queue, original["id"])
    atomic_write_json(service.root / original["id"] / "one/result.json", {"status": "COMPLETED"})
    def reject(**_): raise ValueError("rejected before registration")
    monkeypatch.setattr(queue, "enqueue_external", reject)
    with pytest.raises(ValueError, match="rejected"): service.start(["one"], {})
    assert service.current()["one"] == original["id"]
    assert service.result(original["id"], "one")["status"] == "COMPLETED"
    assert len(list(service.root.glob("lab_*"))) == 1


@pytest.mark.parametrize("failure_point", ["after_registration", "during_prune"])
def test_restart_recovers_index_and_pruning(lab, monkeypatch, failure_point):
    service, queue, _ = lab
    original = service.start(["one"], {})
    complete(queue, original["id"])
    atomic_write_json(service.root / original["id"] / "one/result.json", {"status": "COMPLETED"})
    if failure_point == "after_registration":
        register = queue.enqueue_external
        def fail(**kwargs):
            register(**kwargs)
            raise OSError("simulated interruption")
        monkeypatch.setattr(queue, "enqueue_external", fail)
    else:
        def fail(*_): raise OSError("simulated interruption")
        monkeypatch.setattr(service, "prune", fail)
    with pytest.raises(OSError, match="interruption"): service.start(["one"], {})
    recovered = AnnotationLabService(service.runs, queue.project_root, queue)
    current = recovered.list()
    assert len(current) == 1 and current[0]["id"] != original["id"]
    assert current[0]["status"] == "QUEUED"
    assert not (service.root / original["id"]).exists()
    assert original["id"] not in queue.repository.queue_ids()
