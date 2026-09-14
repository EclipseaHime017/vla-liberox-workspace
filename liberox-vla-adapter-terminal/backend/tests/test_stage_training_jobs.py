from __future__ import annotations

import copy
import json
import threading
from pathlib import Path
from types import SimpleNamespace

import pytest
import yaml

from backend.app.services.offline_job_service import OfflineJobService, _training_replay_counts
from backend.app.storage.database import connect
from backend.app.storage.repositories import OfflineJobRepository


WORKSPACE = Path(__file__).resolve().parents[3]


def _full_recording_episode(run_id, chunks, *, terminal_step=None):
    values = [
        {"start": start, "end": end, "length": end - start, "action_source": "policy",
         "copied_prefix": copied}
        for start, end, copied in chunks
    ]
    return {
        "run_id": run_id, "root_run_id": "run", "kind": "original" if run_id == "run" else "branch",
        "split": "train", "recorded_action_count": values[-1]["end"],
        "action_count": values[-1]["end"] if terminal_step is None else terminal_step + 1,
        "terminal_step": terminal_step,
        "chunks": values if terminal_step is None else [
            chunk for chunk in values if chunk["end"] <= terminal_step + 1
        ],
        "evaluation_chunks": values,
    }


@pytest.mark.parametrize("legacy", [False, True])
def test_training_replay_counts_include_tail_and_deduplicate_copied_prefix(legacy):
    episodes = [
        _full_recording_episode("run", [(0, 8, False), (8, 10, False), (10, 17, False)],
                                terminal_step=9 if legacy else None),
        _full_recording_episode("branch", [(0, 8, True), (8, 10, False), (10, 18, False), (18, 25, False)],
                                terminal_step=9 if legacy else None),
    ]
    before = copy.deepcopy(episodes)
    assert _training_replay_counts({"episodes": episodes}) == {"action_count": 34, "chunk_count": 6}
    assert episodes == before


def test_training_replay_counts_reject_missing_recorded_tail():
    episode = _full_recording_episode("run", [(0, 5, False), (5, 17, False)], terminal_step=4)
    del episode["evaluation_chunks"]
    with pytest.raises(ValueError, match="full-recording replay"):
        _training_replay_counts({"episodes": [episode]})


def test_training_replay_counts_exclude_validation_recordings():
    train = _full_recording_episode("run", [(0, 5, False), (5, 17, False)], terminal_step=4)
    validation = _full_recording_episode("held-out", [(0, 8, False)])
    validation.update(root_run_id="held-out", split="validation")
    assert _training_replay_counts({"episodes": [validation, train]}) == {
        "action_count": 17, "chunk_count": 2,
    }


def test_new_training_job_counts_full_pinned_recording_without_changing_history(tmp_path):
    from test_dataset_reward_versions import setup_jobs, finish

    jobs, dataset = setup_jobs(tmp_path)
    # Old replay stopped after action 5; success confirmation and a controller
    # break at 7 make full replay contain more chunks than the frozen estimate.
    episode = _full_recording_episode(
        "run", [(0, 5, False), (5, 7, False), (7, 15, False), (15, 17, False)],
        terminal_step=4,
    )
    for chunk in episode["evaluation_chunks"]:
        if chunk["start"] >= 7:
            chunk["action_source"] = "human"
    version = finish(jobs, dataset, jobs.start_annotation(dataset["id"], source="sparse"),
                     prepared_episodes=[episode])
    frozen_path = jobs.datasets.root / dataset["id"] / "dataset.json"
    prepared_path = Path(version["prepared_manifest_path"])
    reward_path = Path(version["reward_manifest_path"])
    historical = jobs.jobs_root / "historical" / "job.json"
    historical.parent.mkdir(parents=True)
    historical.write_text(json.dumps({"parameters": {"action_count": 5, "chunk_count": 1}}))
    frozen_before = json.loads(frozen_path.read_text())
    paths = [prepared_path, reward_path, historical]
    originals = {path: path.read_bytes() for path in paths}

    result = jobs.start_training(dataset["id"], {})

    assert dataset["chunk_count"] == 3
    assert result["parameters"]["action_count"] == 17
    assert result["parameters"]["chunk_count"] == 4
    assert result["parameters"]["replay_policy"] == "full_recording_v1"
    assert result["parameters"]["reward_version_id"] == version["id"]
    assert {path: path.read_bytes() for path in paths} == originals
    frozen_after = json.loads(frozen_path.read_text())
    # Existing source-integrity checks update verification timestamps only.
    for key in frozen_before:
        if key not in {"updated_at", "last_verified_at"}:
            assert frozen_after[key] == frozen_before[key]


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


@pytest.mark.parametrize("source", ["sparse", "stage", "rynnvalue"])
def test_training_launch_and_recovery_use_real_catalog(tmp_path, monkeypatch, source):
    from test_dataset_reward_versions import setup_jobs, finish

    jobs, dataset = setup_jobs(tmp_path)
    version = finish(jobs, dataset, jobs.start_annotation(dataset["id"], source=source))
    database_path = tmp_path / "catalog.sqlite3"
    jobs.repository = OfflineJobRepository(database_path, "test")
    jobs.gpu_lock_path = tmp_path / ".gpu-task.lock"
    jobs._new_job = lambda **kwargs: OfflineJobService._new_job(jobs, **kwargs)
    jobs._prepare_launch = lambda: setattr(jobs, "launch_reserved", True)
    spawned = []

    def launch(*args, **kwargs):
        spawned.append(args)
        return SimpleNamespace(pid=123456)

    monkeypatch.setattr("backend.app.services.offline_job_service.subprocess.Popen", launch)
    result = jobs.start_training(dataset["id"], {"reward_source": source})
    assert len(spawned) == 1
    assert result["status"] == "STARTING"
    assert jobs.launch_reserved is False
    annotation_id = version["id"]
    path = jobs.jobs_root / result["id"] / "job.json"
    assert json.loads(path.read_text())["parameters"]["annotation_id"] == annotation_id
    assert [stage["id"] for stage in json.loads(path.read_text())["stages"]] == ["train"]

    with connect(database_path) as database:
        column = next(row for row in database.execute("PRAGMA table_info(training_runs)")
                      if row["name"] == "annotation_id")
        assert column["notnull"] == 1
        row = database.execute("SELECT * FROM training_runs WHERE id = ?", (result["id"],)).fetchone()
        assert row["annotation_id"] == annotation_id
        assert row["status"] == "STARTING"

    orphan = json.loads(path.read_text())
    orphan.update(created_at="2000-01-01T00:00:00+00:00", launcher_pid=None)
    path.write_text(json.dumps(orphan))
    jobs.repository = OfflineJobRepository(database_path, "test")
    recovered = jobs.get(result["id"])
    assert recovered["status"] == "FAILED"
    assert len(spawned) == 1
    assert recovered["parameters"]["annotation_id"] == annotation_id
    with connect(database_path) as database:
        for table in ("offline_jobs", "training_runs"):
            row = database.execute(f"SELECT status FROM {table} WHERE id = ?", (result["id"],)).fetchone()
            assert row["status"] == "FAILED"


def test_old_training_record_with_null_annotation_still_indexes(tmp_path):
    repository = OfflineJobRepository(tmp_path / "catalog.sqlite3", "test")
    payload = {"id": "old", "kind": "training", "status": "FAILED", "dataset_id": "ds",
               "created_at": "2026-01-01", "parameters": {"annotation_id": None}}
    path = tmp_path / "job.json"
    path.write_text(json.dumps(payload))
    repository.upsert(payload, path)
    with connect(tmp_path / "catalog.sqlite3") as database:
        row = database.execute("SELECT annotation_id FROM training_runs WHERE id='old'").fetchone()
        assert row["annotation_id"] == ""


def test_source_defaults_and_legacy_checkbox_remain_supported(tmp_path):
    jobs, base, _, _, _ = make_jobs(tmp_path)
    base["reward"].update(source="stage", rynnvalue=False, stage_exponent=4.0)
    defaults = jobs.defaults()["advanced"]
    assert defaults["reward_source"] == "stage"
    assert defaults["reward_stage_exponent"] == 4.0
    assert jobs._reward_source(base, {"reward_rynnvalue": True}) == "rynnvalue"
    assert jobs._reward_source(base, {"reward_rynnvalue": False}) == "sparse"


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
