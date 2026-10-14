import asyncio
import json
import os
from types import SimpleNamespace

import pytest

from backend.app.api import offline_jobs
from backend.app.api.models import DeleteJobRequest
from backend.app.core.exceptions import ConflictError
from backend.app.storage.database import connect
from backend.app.storage.repositories import RunLabelRepository, RunRepository
from test_evaluation_api import make_service


def training_job(tmp_path, *, status="COMPLETED", kind="training"):
    jobs = make_service(tmp_path)
    registry = tmp_path / "registry"
    registry.mkdir()
    jobs.manager.policy_catalog = SimpleNamespace(registry=registry, base_models={"base": {}}, refresh=lambda: None)
    identifier = "owned"
    directory = jobs.jobs_root / identifier
    directory.mkdir()
    output = jobs.training_root / identifier
    output.mkdir()
    job = {"schema_version": 1, "id": identifier, "kind": kind, "status": status,
           "dataset_id": "dataset", "parameters": {}, "pid": None,
           "output_path": str(output), "created_at": "2026-10-14T00:00:00+00:00"}
    if kind != "training":
        job["executor"] = "local"
    jobs._save_work(job)
    return jobs, job, output, registry


def exported(output, registry, *, complete=True):
    run = output / "run"
    run.mkdir()
    child = registry / "test-child"
    child.mkdir()
    (child / "actor.pt").write_bytes(b"model")
    (run / "export.json").write_text(json.dumps({"policy_id": child.name, "registry": str(registry),
                                                 "previous_latest": "previous-child"}))
    if complete:
        (run / "summary.json").write_text(json.dumps({"policy_overlay": str(child / "policy.yaml")}))
    old = registry / "previous-child"
    old.mkdir()
    (registry / "latest").symlink_to(child.name)
    return child, old


@pytest.mark.parametrize("complete", [True, False])
def test_cleanup_removes_training_files_exports_and_indexes_and_restores_latest(tmp_path, complete):
    jobs, job, output, registry = training_job(tmp_path)
    child, old = exported(output, registry, complete=complete)
    unrelated = jobs.jobs_root / "unrelated"
    unrelated.mkdir()
    (unrelated / "notes.txt").write_text("keep")
    result = jobs.delete_job(job["id"], job["id"], delete_exported_models=True)
    assert result["models_deleted"] == [child.name]
    assert not child.exists() and not output.exists() and not (jobs.jobs_root / job["id"]).exists()
    assert (registry / "latest").resolve() == old.resolve() and unrelated.is_dir()
    with connect(jobs.repository.database_path) as database:
        for table in ("offline_jobs", "training_runs", "annotation_runs"):
            assert database.execute(f"SELECT COUNT(*) FROM {table} WHERE id=?", (job["id"],)).fetchone()[0] == 0


def test_normal_job_deletion_preserves_exported_policy(tmp_path):
    jobs, job, output, registry = training_job(tmp_path)
    child, _ = exported(output, registry)
    jobs.delete_job(job["id"], job["id"])
    assert child.is_dir() and not output.exists()


@pytest.mark.parametrize("status", ["QUEUED", "STARTING", "RUNNING", "STOPPING"])
def test_active_jobs_are_never_deleted(tmp_path, status):
    jobs, job, output, _ = training_job(tmp_path, status=status)
    with pytest.raises(ConflictError, match="finish"):
        jobs.delete_job(job["id"], job["id"])
    assert output.is_dir() and (jobs.jobs_root / job["id"]).is_dir()


@pytest.mark.parametrize("local", [False, True])
def test_terminal_status_waits_for_worker_release(tmp_path, local):
    jobs, job, output, _ = training_job(tmp_path)
    if local:
        jobs._local_tasks[job["id"]] = object()
    else:
        job["pid"] = os.getpid()
        jobs._save_work(job)
    with pytest.raises(ConflictError):
        jobs.delete_job(job["id"], job["id"])
    assert output.is_dir()


def test_path_and_base_identity_guards_precede_deletion(tmp_path):
    jobs, job, output, registry = training_job(tmp_path)
    child, _ = exported(output, registry)
    with pytest.raises(ValueError, match="exactly"):
        jobs.delete_job(job["id"], "other")
    marker = output / "run" / "export.json"
    marker.write_text(json.dumps({"policy_id": "base", "registry": str(registry)}))
    with pytest.raises(ValueError, match="base model"):
        jobs.delete_job(job["id"], job["id"], delete_exported_models=True)
    assert child.exists() and output.exists()
    marker.write_text(json.dumps({"policy_id": child.name, "registry": str(tmp_path / "foreign")}))
    with pytest.raises(ValueError, match="registry"):
        jobs.delete_job(job["id"], job["id"], delete_exported_models=True)
    job["output_path"] = str(tmp_path)
    jobs._save_work(job)
    with pytest.raises(ValueError, match="managed root"):
        jobs.delete_job(job["id"], job["id"])
    assert output.exists()


def test_cleanup_does_not_replace_a_newer_latest_pointer(tmp_path):
    jobs, job, output, registry = training_job(tmp_path)
    child, old = exported(output, registry)
    latest = registry / "latest"
    latest.unlink()
    latest.symlink_to(old.name)
    jobs.delete_job(job["id"], job["id"], delete_exported_models=True)
    assert latest.resolve() == old.resolve() and not child.exists()


def test_cleanup_restores_latest_through_multiple_exports(tmp_path):
    jobs, job, output, registry = training_job(tmp_path)
    first, old = exported(output, registry)
    second_run = output / "second-run"
    second_run.mkdir()
    second = registry / "second-child"
    second.mkdir()
    (second_run / "export.json").write_text(json.dumps({"policy_id": second.name,
        "registry": str(registry), "previous_latest": first.name}))
    latest = registry / "latest"
    latest.unlink()
    latest.symlink_to(second.name)
    jobs.delete_job(job["id"], job["id"], delete_exported_models=True)
    assert latest.resolve() == old.resolve()
    assert not first.exists() and not second.exists()


def test_checkpoint_referenced_by_pending_resume_cannot_be_deleted(tmp_path):
    jobs, job, output, _ = training_job(tmp_path)
    other = {**job, "id": "resume", "status": "QUEUED",
             "parameters": {"resume_checkpoint": str(output / "run/checkpoints/step_2")}}
    (jobs.jobs_root / other["id"]).mkdir()
    jobs._save_work(other)
    jobs._local_tasks[other["id"]] = object()
    with pytest.raises(ConflictError, match="Checkpoint"):
        jobs.delete_job(job["id"], job["id"])
    assert output.exists()


def test_local_task_cleanup_preserves_created_dataset(tmp_path):
    jobs, job, output, _ = training_job(tmp_path, kind="dataset")
    jobs.delete_job(job["id"], job["id"])
    assert output.is_dir()
    assert not (jobs.jobs_root / job["id"]).exists()


def test_delete_endpoint_and_run_label_cleanup(tmp_path):
    jobs, job, output, _ = training_job(tmp_path)
    request = SimpleNamespace(app=SimpleNamespace(state=SimpleNamespace(offline_job_service=jobs)))
    result = asyncio.run(offline_jobs.delete_job(job["id"], DeleteJobRequest(confirm_job_id=job["id"]), request))
    assert result["deleted"] == job["id"] and not output.exists()
    runs = RunRepository(jobs.repository.database_path, "test")
    labels = RunLabelRepository(jobs.repository.database_path, "test")
    labels.set_test("test-run", True)
    runs.delete("test-run")
    assert not labels.test_ids()
