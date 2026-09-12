from __future__ import annotations

import copy
import json
import threading
from pathlib import Path
from types import SimpleNamespace

import pytest
import yaml

from backend.app.services.offline_job_service import OfflineJobService
from backend.app.storage.database import connect
from backend.app.storage.repositories import OfflineJobRepository


WORKSPACE = Path(__file__).resolve().parents[3]


def make_jobs(tmp_path: Path):
    jobs = object.__new__(OfflineJobService)
    jobs.lock = threading.RLock()
    jobs.jobs_root = tmp_path / "jobs"
    jobs.training_root = tmp_path / "training"
    jobs.cache_root = tmp_path / "cache"
    jobs.launch_reserved = False
    jobs.ui_config = SimpleNamespace(
        offline_rl_root=WORKSPACE / "vla-adapter-rynn-iql",
        train_environment="vla-liberox", reward_environment="rynnvalue-reward",
    )
    base = yaml.safe_load(jobs.base_config_path.read_text(encoding="utf-8"))
    jobs._load_base_config = lambda: copy.deepcopy(base)
    jobs.available_checkpoints = lambda _: []
    frozen = tmp_path / "datasets" / "ds" / "dataset.json"
    frozen.parent.mkdir(parents=True)
    members = [{
        "run_id": "run", "split": "validation",
        "artifacts": {"trajectory": {"path": "/source/trajectory.npz", "sha256": "frozen"}},
    }]
    frozen.write_text(json.dumps({"members": members}), encoding="utf-8")
    dataset = {
        "id": "ds", "task_id": "LEVEL1::pick", "member_count": 1,
        "action_count": 16, "chunk_count": 2,
        "annotation_id": None, "annotation_status": "PENDING",
        "dataset_sha256": "frozen-dataset",
        "integrity_status": "HEALTHY", "validation_fraction": .2,
        "split_seed": 7, "success_consecutive_steps": 5,
    }
    calls = []

    def require_ready():
        calls.append("requires_rynn")
        if dataset["annotation_status"] != "READY":
            raise ValueError("Dataset annotation is not ready")
        return copy.deepcopy(dataset)

    jobs.datasets = SimpleNamespace(
        root=frozen.parent.parent,
        require_ready_for_training=lambda _: require_ready(),
        require_ready_for_annotation=lambda _: calls.append("requires_healthy") or copy.deepcopy(dataset),
        get=lambda _: copy.deepcopy(dataset),
    )
    snapshots = {"run": {"keyframes": [], "trajectory_sha256": "frozen"}}

    def validate_members(current, threshold):
        assert current == members
        assert threshold == 5
        calls.append("stage_validation")
        return copy.deepcopy(snapshots)

    jobs.stage_annotations = SimpleNamespace(validate_members=validate_members)
    jobs._prepare_launch = lambda: calls.append("gpu_reservation")
    jobs._validated_annotation_work = lambda _: (
        tmp_path / "old-rynn-work", {}, {},
    )
    jobs._new_job = lambda **kwargs: calls.append("spawn") or kwargs
    return jobs, base, dataset, calls, snapshots


@pytest.mark.parametrize("source", ["sparse", "stage"])
def test_direct_sources_train_without_any_rynn_annotation(tmp_path, source):
    jobs, _, _, calls, _ = make_jobs(tmp_path)
    jobs._validated_annotation_work = lambda _: pytest.fail("must not read RynnValue")
    result = jobs.start_training("ds", {
        "reward_source": source, "reward_stage_exponent": 3.0,
        "reward_gamma": .92,
    })
    assert "requires_rynn" not in calls
    assert [stage["id"] for stage in result["stages"]] == ["prepare", "rewards", "train"]
    assert {stage["environment"] for stage in result["stages"]} == {"vla-liberox"}
    assert all("annotate_rewards.py" not in " ".join(stage["argv"]) for stage in result["stages"])
    raw = yaml.safe_load(result["config_path"].read_text(encoding="utf-8"))
    assert raw["reward"]["source"] == source
    assert raw["reward"]["rynnvalue"] is False
    assert raw["reward"]["stage_exponent"] == 3.0
    assert raw["reward"]["gamma"] == .92
    assert Path(raw["paths"]["work_dir"]).parent == result["config_path"].parent
    assert result["parameters"]["annotation_id"] is None
    assert result["parameters"]["reward"]["source"] == source
    assert jobs.launch_reserved is False


@pytest.mark.parametrize("source", ["sparse", "stage", "rynnvalue"])
def test_training_launch_and_recovery_use_real_catalog(tmp_path, monkeypatch, source):
    jobs, _, dataset, _, _ = make_jobs(tmp_path)
    database_path = tmp_path / "catalog.sqlite3"
    jobs.repository = OfflineJobRepository(database_path, "test")
    jobs.gpu_lock_path = tmp_path / ".gpu-task.lock"
    jobs._new_job = lambda **kwargs: OfflineJobService._new_job(jobs, **kwargs)
    jobs._prepare_launch = lambda: setattr(jobs, "launch_reserved", True)
    dataset["annotation_id"] = "ann"  # Direct rewards must not inherit an older model evaluation.
    if source == "rynnvalue":
        dataset.update(annotation_status="READY", annotation_id="ann")
    spawned = []

    def launch(*args, **kwargs):
        spawned.append(args)
        return SimpleNamespace(pid=123456)

    monkeypatch.setattr("backend.app.services.offline_job_service.subprocess.Popen", launch)
    result = jobs.start_training("ds", {"reward_source": source})
    assert len(spawned) == 1
    assert result["status"] == "STARTING"
    assert jobs.launch_reserved is False
    annotation_id = "ann" if source == "rynnvalue" else None
    path = jobs.jobs_root / result["id"] / "job.json"
    assert json.loads(path.read_text())["parameters"]["annotation_id"] == annotation_id

    with connect(database_path) as database:
        column = next(row for row in database.execute("PRAGMA table_info(training_runs)")
                      if row["name"] == "annotation_id")
        assert column["notnull"] == 1  # Existing catalogs require no schema migration.
        row = database.execute("SELECT * FROM training_runs WHERE id = ?", (result["id"],)).fetchone()
        assert row["annotation_id"] == (annotation_id or "")
        assert row["status"] == "STARTING"
        assert len(jobs.repository.references_for_dataset("ds")) == 1

    # The pre-fix failure left a STARTING manifest before any runner was spawned.
    # A restarted service must index it as failed instead of hiding it on every poll.
    orphan = json.loads(path.read_text())
    orphan.update(created_at="2000-01-01T00:00:00+00:00", launcher_pid=None)
    path.write_text(json.dumps(orphan))
    jobs.repository = OfflineJobRepository(database_path, "test")
    recovered = jobs.get(result["id"])
    assert recovered["status"] == "FAILED"
    assert len(spawned) == 1  # Recovery never restarts training automatically.
    assert recovered["parameters"]["annotation_id"] == annotation_id
    with connect(database_path) as database:
        for table in ("offline_jobs", "training_runs"):
            row = database.execute(f"SELECT status FROM {table} WHERE id = ?", (result["id"],)).fetchone()
            assert row["status"] == "FAILED"


def test_stage_freezes_all_members_before_gpu_and_retains_original_snapshot(tmp_path):
    jobs, _, _, calls, snapshots = make_jobs(tmp_path)
    result = jobs.start_training("ds", {"reward_source": "stage"})
    assert calls == ["requires_healthy", "stage_validation", "gpu_reservation", "spawn"]
    raw = yaml.safe_load(result["config_path"].read_text(encoding="utf-8"))
    snapshot_path = Path(raw["data"]["stage_annotations_manifest"])
    assert snapshot_path.parent == result["config_path"].parent
    assert json.loads(snapshot_path.read_text()) == {
        "schema_version": 1, "annotations": snapshots,
    }
    snapshots["run"]["keyframes"].append({"step": 5, "kind": "positive"})
    assert json.loads(snapshot_path.read_text())["annotations"]["run"]["keyframes"] == []


def test_missing_stage_member_stops_before_job_or_gpu_creation(tmp_path):
    jobs, _, _, calls, _ = make_jobs(tmp_path)

    def reject(*_):
        raise ValueError("Missing Stage annotations: run")

    jobs.stage_annotations.validate_members = reject
    with pytest.raises(ValueError, match="Missing Stage annotations: run"):
        jobs.start_training("ds", {"reward_source": "stage"})
    assert calls == ["requires_healthy"]
    assert not jobs.jobs_root.exists()
    assert not jobs.training_root.exists()


def test_rynnvalue_preserves_existing_ready_gate_and_reward_only_stage(tmp_path):
    jobs, _, dataset, calls, _ = make_jobs(tmp_path)
    with pytest.raises(ValueError, match="annotation is not ready"):
        jobs.start_training("ds", {"reward_source": "rynnvalue"})
    assert calls == ["requires_rynn"]
    dataset.update(annotation_status="READY", annotation_id="ann")
    calls.clear()
    result = jobs.start_training("ds", {"reward_source": "rynnvalue"})
    assert calls == ["requires_rynn", "gpu_reservation", "spawn"]
    assert [stage["id"] for stage in result["stages"]] == ["rewards", "train"]
    raw = yaml.safe_load(result["config_path"].read_text())
    assert raw["reward"]["rynnvalue"] is True
    assert "stage_annotations_manifest" not in raw["data"]
    assert result["parameters"]["annotation_id"] == "ann"


def test_source_defaults_and_legacy_checkbox_remain_supported(tmp_path):
    jobs, base, _, _, _ = make_jobs(tmp_path)
    base["reward"].update(source="stage", rynnvalue=False, stage_exponent=4.0)
    defaults = jobs.defaults()["advanced"]
    assert defaults["reward_source"] == "stage"
    assert defaults["reward_stage_exponent"] == 4.0
    assert jobs._reward_source(base, {"reward_rynnvalue": True}) == "rynnvalue"
    result = jobs.start_training("ds", {"reward_rynnvalue": False})
    assert result["parameters"]["reward"]["source"] == "sparse"


@pytest.mark.parametrize("annotation_ready", [False, True])
def test_checkpoint_catalog_keeps_direct_sources_and_excludes_other_datasets(
    tmp_path, annotation_ready,
):
    jobs, _, dataset, _, _ = make_jobs(tmp_path)
    del jobs.available_checkpoints
    if annotation_ready:
        dataset.update(annotation_status="READY", annotation_id="ann")
        work = jobs.datasets.root / "ds" / "annotations" / "ann" / "work"
        work.mkdir(parents=True)
        (work / "dataset_manifest.json").write_text(json.dumps({
            "dataset_sha256": "rynn-prepared",
        }))
    expected = []
    for name, selected_dataset, source, frozen_hash in (
        ("stage", "ds", "stage", "frozen-dataset"),
        ("sparse", "ds", "sparse", "frozen-dataset"),
        ("other", "different", "stage", "frozen-dataset"),
        ("changed", "ds", "stage", "different-frozen-hash"),
    ):
        checkpoint = jobs.training_root / name / "run" / "checkpoints" / "step_00000032"
        checkpoint.mkdir(parents=True)
        (checkpoint / "checkpoint.json").write_text(json.dumps({
            "dataset_sha256": "direct-prepared", "reward_sha256": source,
        }))
        job_dir = jobs.jobs_root / name
        job_dir.mkdir(parents=True)
        (job_dir / "job.json").write_text(json.dumps({
            "id": name, "kind": "training", "dataset_id": selected_dataset,
            "parameters": {
                "source_dataset_sha256": frozen_hash, "reward": {"source": source},
            },
        }))
        if name in {"stage", "sparse"}:
            expected.append(str(checkpoint.resolve()))
    listed = jobs.available_checkpoints("ds")
    assert sorted(item["path"] for item in listed) == sorted(expected)
    assert {item["label"].split(" · ")[-1] for item in listed} == {"stage", "sparse"}


@pytest.mark.parametrize("parameter", [
    {"reward_source": "robometer"}, {"reward_source": None},
    {"reward_source": []}, {"reward_stage_exponent": 0},
    {"reward_stage_exponent": True}, {"reward_stage_exponent": float("nan")},
    {"reward_stage_exponent": float("inf")}, {"reward_stage_exponent": "2"},
])
def test_reward_controls_reject_invalid_values(parameter):
    with pytest.raises(ValueError, match="reward_"):
        OfflineJobService._validate_training_parameters(parameter)
