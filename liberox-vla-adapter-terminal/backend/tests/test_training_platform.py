from __future__ import annotations

import json
import hashlib
import threading
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest

from backend.app.services.offline_job_service import OfflineJobService, _stable_hash
from backend.app.services.training_dataset_service import TrainingDatasetService, _split
from backend.app.workers import offline_job_runner


class FakeRuns:
    def __init__(self, runs):
        self.runs = runs

    def list_runs(self):
        return list(self.runs)


def make_run(
    root: Path,
    run_id: str,
    *,
    task_id: str = "LEVEL1::pick",
    kind: str = "original",
    source: str = "policy",
    resume_step: int = 0,
    success: bool = False,
    created_at: str = "2026-08-24T10:00:00+00:00",
):
    run_dir = root / "runs" / task_id.replace("::", "_") / "2026-08-24" / run_id
    episode = run_dir / "episodes" / "episode_000"
    episode.mkdir(parents=True)
    action_count = 17
    np.savez_compressed(episode / "trajectory.npz", action=np.zeros((action_count, 7)))
    np.savez_compressed(
        episode / "trajectory_observations.npz",
        agentview_image=np.zeros((action_count + 1, 2, 2, 3), dtype=np.uint8),
        wrist_image=np.zeros((action_count + 1, 2, 2, 3), dtype=np.uint8),
    )
    manifest = {
        "id": run_id,
        "kind": kind,
        "task_id": task_id,
        "status": "COMPLETED",
        "success": success,
    }
    (run_dir / "run.json").write_text(json.dumps(manifest), encoding="utf-8")
    return {
        **manifest,
        "error": None,
        "task": "pick the bowl",
        "task_name": "pick",
        "created_at": created_at,
        "action_count": action_count,
        "output_dir": str(run_dir),
        "trajectory": str(episode / "trajectory.npz"),
        "control_mode": "manual" if source == "human" else "policy",
        "resume_step": resume_step if kind == "branch" else None,
        "parent_session_id": "root" if kind == "branch" else None,
        "root_session_id": "root" if kind == "branch" else run_id,
    }


def service(tmp_path: Path, runs):
    config = SimpleNamespace(
        project_root=tmp_path / "projects" / "test",
        catalog_path=tmp_path / "catalog.sqlite3",
        project_id="test",
    )
    return TrainingDatasetService(FakeRuns(runs), config)


def test_selection_is_reproducible_and_classifies_branch_suffixes(tmp_path: Path):
    runs = [
        make_run(tmp_path, "root", created_at="2026-08-24T10:00:00+00:00"),
        make_run(
            tmp_path, "human", kind="branch", source="human", resume_step=5,
            success=True, created_at="2026-08-24T10:01:00+00:00",
        ),
        make_run(
            tmp_path, "requery", kind="branch", source="policy_requery",
            resume_step=9, created_at="2026-08-24T10:02:00+00:00",
        ),
    ]
    current = service(tmp_path, runs)
    selection = {
        "mode": "random", "size": 2, "seed": 19,
        "source_types": ["inference", "manual", "policy_requery"],
        "outcomes": ["success", "failure"],
    }
    assert current.preview("LEVEL1::pick", selection)["run_ids"] == current.preview(
        "LEVEL1::pick", selection
    )["run_ids"]
    listed = {item["id"]: item for item in current.list_runs()}
    assert listed["human"]["source_type"] == "manual"
    assert listed["human"]["training_start_step"] == 0
    assert listed["human"]["training_action_count"] == 17
    assert listed["human"]["resume_step"] == 5
    assert listed["requery"]["source_type"] == "policy_requery"
    page = current.list_runs_page(page=1, page_size=2)
    assert page["total"] == 3
    assert len(page["items"]) == 2
    assert page["pages"] == 2


def test_frozen_dataset_hashes_sources_and_marks_changed_source_broken(tmp_path: Path):
    root_id = next(
        f"root-{index}" for index in range(1000)
        if _split(f"root-{index}", 7, 0.9) == "validation"
    )
    run = make_run(tmp_path, root_id)
    current = service(tmp_path, [run])
    dataset = current.create(
        name="one task", task_id="LEVEL1::pick",
        selection={"mode": "manual", "run_ids": [root_id]},
        validation_fraction=0.9,
    )
    assert dataset["status"] == "FROZEN"
    assert dataset["members"][0]["split"] == "train"
    assert current.references_for_run(root_id)[0]["id"] == dataset["id"]

    with Path(run["trajectory"]).open("ab") as stream:
        stream.write(b"changed")
    broken = current.verify(dataset["id"])
    assert broken["integrity_status"] == "BROKEN"
    with pytest.raises(Exception, match="integrity"):
        current.require_ready_for_annotation(dataset["id"])


def test_failed_reannotation_preserves_previous_ready_version(tmp_path: Path):
    run = make_run(tmp_path, "root")
    current = service(tmp_path, [run])
    dataset = current.create(
        name="stable annotation", task_id="LEVEL1::pick",
        selection={"mode": "manual", "run_ids": ["root"]},
    )
    current.update_annotation(dataset["id"], "RUNNING", annotation_id="ann-good")
    current.update_annotation(dataset["id"], "READY", annotation_id="ann-good")
    current.update_annotation(dataset["id"], "RUNNING", annotation_id="ann-bad")
    result = current.update_annotation(dataset["id"], "ERROR", annotation_id="ann-bad")
    assert result["annotation_status"] == "READY"
    assert result["annotation_id"] == "ann-good"
    assert result["last_annotation_status"] == "ERROR"
    assert len(result["annotation_history"]) == 2


def test_immutable_membership_tampering_is_detected(tmp_path: Path):
    run = make_run(tmp_path, "root")
    current = service(tmp_path, [run])
    dataset = current.create(
        name="immutable", task_id="LEVEL1::pick",
        selection={"mode": "manual", "run_ids": ["root"]},
    )
    manifest = current.root / dataset["id"] / "dataset.json"
    payload = json.loads(manifest.read_text(encoding="utf-8"))
    payload["members"][0]["resume_step"] = 3
    manifest.write_text(json.dumps(payload), encoding="utf-8")
    assert current.get(dataset["id"])["integrity_status"] == "BROKEN"


def test_cancel_frozen_dataset_only_removes_unused_manifest(tmp_path: Path):
    run = make_run(tmp_path, "root")
    current = service(tmp_path, [run])
    dataset = current.create(
        name="unused", task_id="LEVEL1::pick",
        selection={"mode": "manual", "run_ids": ["root"]},
    )
    directory = current.root / dataset["id"]
    result = current.delete_unannotated(dataset["id"], dataset["id"])
    assert result == {
        "deleted": dataset["id"], "annotation_status": "NOT_STARTED",
        "source_runs_deleted": False, "shared_cache_deleted": False,
    }
    assert not directory.exists()
    assert Path(run["trajectory"]).is_file()
    assert current.references_for_run("root") == []


def test_cancel_frozen_dataset_rejects_used_or_parent_versions(tmp_path: Path):
    run = make_run(tmp_path, "root")
    current = service(tmp_path, [run])
    parent = current.create(
        name="parent", task_id="LEVEL1::pick",
        selection={"mode": "manual", "run_ids": ["root"]},
    )
    current.create(
        name="child", task_id="LEVEL1::pick",
        selection={"mode": "manual", "run_ids": ["root"]},
        parent_dataset_id=parent["id"],
    )
    with pytest.raises(Exception, match="derived versions"):
        current.delete_unannotated(parent["id"], parent["id"])
    current.update_annotation(parent["id"], "RUNNING", annotation_id="ann")
    with pytest.raises(Exception, match="active"):
        current.delete_unannotated(parent["id"], parent["id"])
    with pytest.raises(ValueError, match="exactly match"):
        current.delete_unannotated(parent["id"], "wrong")


def test_delete_annotated_dataset_removes_version_but_preserves_source(tmp_path: Path):
    run = make_run(tmp_path, "root")
    current = service(tmp_path, [run])
    dataset = current.create(
        name="annotated", task_id="LEVEL1::pick",
        selection={"mode": "manual", "run_ids": ["root"]},
    )
    current.update_annotation(dataset["id"], "RUNNING", annotation_id="ann")
    current.update_annotation(dataset["id"], "READY", annotation_id="ann")
    annotation = current.root / dataset["id"] / "annotations" / "ann"
    annotation.mkdir(parents=True)
    result = current.delete_dataset(dataset["id"], dataset["id"])
    assert result["annotation_status"] == "READY"
    assert result["source_runs_deleted"] is False
    assert not annotation.parent.parent.exists()
    assert Path(run["trajectory"]).is_file()


def test_job_service_blocks_dataset_deletion_with_training_history():
    jobs = object.__new__(OfflineJobService)
    jobs.lock = threading.RLock()
    jobs.list = lambda: [{
        "id": "train", "kind": "training", "status": "COMPLETED",
        "dataset_id": "ds",
    }]
    jobs.repository = SimpleNamespace(references_for_dataset=lambda _: [])
    jobs.datasets = SimpleNamespace(
        delete_dataset=lambda *_: pytest.fail("dataset deletion must be blocked")
    )
    with pytest.raises(Exception, match="training history"):
        jobs.delete_dataset("ds", "ds")


def test_forced_dataset_deletion_retains_and_marks_training_history(tmp_path: Path):
    jobs = object.__new__(OfflineJobService)
    jobs.lock = threading.RLock()
    jobs.jobs_root = tmp_path / "jobs"
    job_dir = jobs.jobs_root / "train"
    job_dir.mkdir(parents=True)
    job_path = job_dir / "job.json"
    payload = {
        "schema_version": 1, "id": "train", "kind": "training",
        "status": "COMPLETED", "dataset_id": "ds",
        "created_at": "2026-08-24T10:00:00+00:00",
    }
    job_path.write_text(json.dumps(payload), encoding="utf-8")
    jobs.list = lambda: [payload]
    upserts = []
    jobs.repository = SimpleNamespace(
        references_for_dataset=lambda _: [],
        upsert=lambda job, path: upserts.append((job, path)),
    )
    jobs.datasets = SimpleNamespace(delete_dataset=lambda *_: {
        "deleted": "ds", "annotation_status": "READY",
        "source_runs_deleted": False, "shared_cache_deleted": False,
    })

    result = jobs.delete_dataset("ds", "ds", force=True)

    retained = json.loads(job_path.read_text(encoding="utf-8"))
    assert result["retained_training_jobs"] == ["train"]
    assert retained["source_dataset_deleted"] is True
    assert retained["source_dataset_deleted_at"]
    assert upserts[0][1] == job_path


def test_training_parameters_reject_unknown_and_unsafe_values():
    with pytest.raises(ValueError, match="Unknown"):
        OfflineJobService._validate_training_parameters({"command": "rm"})
    with pytest.raises(ValueError, match="expectile"):
        OfflineJobService._validate_training_parameters({"expectile": 1.1})
    with pytest.raises(ValueError, match="train_steps"):
        OfflineJobService._validate_training_parameters({"train_steps": 0})
    OfflineJobService._validate_training_parameters({
        "train_steps": 20, "critic_warmup_steps": 5,
        "policy_peak_lr": 1e-4, "resume_checkpoint": None,
    })


def test_gpu_launch_reservation_blocks_simulation_race():
    jobs = object.__new__(OfflineJobService)
    jobs.launch_reserved = True
    jobs.has_active_job = lambda: False
    jobs._external_gpu_lock = lambda: False
    with pytest.raises(Exception, match="using the GPU"):
        jobs.assert_simulation_allowed()


def test_detached_runner_persists_fake_conda_stage(tmp_path: Path, monkeypatch):
    job_dir = tmp_path / "job"
    job_dir.mkdir()
    (job_dir / "job.json").write_text(json.dumps({
        "schema_version": 1, "id": "job", "kind": "annotation",
        "status": "STARTING", "gpu_lock_path": str(tmp_path / "gpu.lock"),
        "stages": [{
            "id": "prepare", "label": "prepare", "environment": "fake-env",
            "argv": ["python", "fake.py"], "cwd": str(tmp_path),
        }],
    }), encoding="utf-8")

    class Child:
        def wait(self): return 0
        def poll(self): return 0
        def terminate(self): pass

    commands = []
    monkeypatch.setattr(offline_job_runner.signal, "signal", lambda *_: None)
    monkeypatch.setattr(
        offline_job_runner.subprocess, "Popen",
        lambda command, **kwargs: commands.append((command, kwargs)) or Child(),
    )
    assert offline_job_runner.Runner(job_dir).run() == 0
    result = json.loads((job_dir / "job.json").read_text(encoding="utf-8"))
    assert result["status"] == "COMPLETED"
    assert commands[0][0][:5] == [
        "conda", "run", "--no-capture-output", "-n", "fake-env",
    ]
    assert "任务完成" in (job_dir / "job.log").read_text(encoding="utf-8")


def test_resume_checkpoint_catalog_only_returns_compatible_directories(tmp_path: Path):
    jobs = object.__new__(OfflineJobService)
    jobs.training_root = tmp_path / "training"
    datasets_root = tmp_path / "datasets"
    work = datasets_root / "ds" / "annotations" / "ann" / "work"
    (work / "rewards").mkdir(parents=True)
    prepared = {"dataset_sha256": "dataset-hash"}
    reward = {"schema_version": 1, "dataset_sha256": "dataset-hash", "episodes": []}
    (work / "dataset_manifest.json").write_text(json.dumps(prepared), encoding="utf-8")
    (work / "rewards" / "reward_manifest.json").write_text(
        json.dumps(reward), encoding="utf-8"
    )
    jobs.datasets = SimpleNamespace(
        root=datasets_root,
        get=lambda dataset_id: {
            "id": dataset_id, "annotation_id": "ann", "annotation_status": "READY",
        },
    )
    compatible = jobs.training_root / "job" / "run" / "checkpoints" / "step_00000020"
    incompatible = jobs.training_root / "other" / "run" / "checkpoints" / "step_00000020"
    compatible.mkdir(parents=True)
    incompatible.mkdir(parents=True)
    (compatible / "checkpoint.json").write_text(json.dumps({
        "dataset_sha256": "dataset-hash", "reward_sha256": _stable_hash(reward),
    }), encoding="utf-8")
    (incompatible / "checkpoint.json").write_text(json.dumps({
        "dataset_sha256": "other", "reward_sha256": "other",
    }), encoding="utf-8")

    listed = jobs.available_checkpoints("ds")
    assert [item["path"] for item in listed] == [str(compatible.resolve())]
    assert jobs._resolve_resume(str(compatible)) == str(compatible.resolve())
    with pytest.raises(FileNotFoundError):
        jobs._resolve_resume(str(compatible / "trainer.pt"))


def test_training_revalidates_completed_annotation_hashes(tmp_path: Path):
    jobs = object.__new__(OfflineJobService)
    jobs.cache_root = tmp_path / "cache"
    jobs.cache_root.mkdir()
    datasets_root = tmp_path / "datasets"
    work = datasets_root / "ds" / "annotations" / "ann" / "work"
    (work / "rewards").mkdir(parents=True)
    annotation = jobs.cache_root / "value.npz"
    annotation.write_bytes(b"reward")
    digest = hashlib.sha256(b"reward").hexdigest()
    prepared = {
        "source_dataset_id": "ds", "source_dataset_sha256": "frozen",
        "dataset_sha256": "prepared", "episodes": [{"run_id": "run"}],
    }
    reward = {
        "complete": True, "dataset_sha256": "prepared",
        "episodes": [{
            "run_id": "run", "annotation_path": str(annotation),
            "annotation_sha256": digest,
        }],
    }
    (work / "dataset_manifest.json").write_text(json.dumps(prepared), encoding="utf-8")
    (work / "rewards" / "reward_manifest.json").write_text(
        json.dumps(reward), encoding="utf-8"
    )
    jobs.datasets = SimpleNamespace(root=datasets_root)
    dataset = {"id": "ds", "annotation_id": "ann", "dataset_sha256": "frozen"}
    assert jobs._validated_annotation_work(dataset)[0] == work
    annotation.write_bytes(b"changed")
    with pytest.raises(Exception, match="hash changed"):
        jobs._validated_annotation_work(dataset)
