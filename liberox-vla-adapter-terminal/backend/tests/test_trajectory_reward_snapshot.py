import json
import os
import threading
from pathlib import Path
from types import SimpleNamespace

import pytest
import numpy as np

from backend.app.services import trajectory_reward_snapshot as snapshots
from backend.app.services.offline_job_service import OfflineJobService
from test_dataset_evaluation_detail import digest, global_fixture
from test_dataset_reward_versions import setup_jobs


def test_existing_first_snapshot_skips_observation_hashing(tmp_path, monkeypatch):
    result, service, _, versions = global_fixture(tmp_path)
    first = snapshots.ensure_first_reward_snapshot(result["run"], service)
    later = versions[("b", "v1")]
    original_hash = snapshots._hash

    def checked_hash(path):
        assert Path(path).name != "trajectory_observations.npz"
        return original_hash(path)

    monkeypatch.setattr(snapshots, "_hash", checked_hash)
    bound = snapshots.bind_reward_snapshot(Path(later["prepared_manifest_path"]),
                                          Path(later["reward_manifest_path"]))
    assert bound == {"bound": [], "skipped": ["r"], "count": 0}
    assert snapshots.read_reward_snapshot(result["run"])["metadata"] == first["metadata"]


def test_existing_global_rynn_has_precedence_over_dataset_stage(tmp_path):
    from backend.app.services.trajectory_evaluation_service import OFFICIAL_ARRAY_KEYS, PBRS_ARRAY_KEYS

    result, service, _, _ = global_fixture(tmp_path)
    trajectory = Path(result["run"]["trajectory"])
    values = trajectory.with_name("rynnvalue_evaluation.npz")
    np.savez(trajectory, env_action=np.zeros((20, 7)))
    np.savez(values, boundary_steps=[0, 20], **{key: [0.] for key in OFFICIAL_ARRAY_KEYS | PBRS_ARRAY_KEYS})
    trajectory.with_name("rynnvalue_evaluation.json").write_text(json.dumps({
        "schema_version": 6, "run_id": "r",
        "trajectory_sha256": digest(trajectory), "values_sha256": digest(values),
        "observations_sha256": digest(trajectory.with_name("trajectory_observations.npz")),
    }))
    assert snapshots.ensure_first_reward_snapshot(result["run"], service) is None
    assert not trajectory.with_name(snapshots.SIDECAR).exists()


def test_invalid_replacement_preserves_first_snapshot(tmp_path):
    result, service, _, versions = global_fixture(tmp_path)
    first = snapshots.ensure_first_reward_snapshot(result["run"], service)
    trajectory = Path(result["run"]["trajectory"])
    sidecar = trajectory.with_name(snapshots.SIDECAR)
    original = sidecar.read_bytes()
    later = versions[("b", "v1")]
    manifest = json.loads(Path(later["reward_manifest_path"]).read_text())
    Path(manifest["episodes"][0]["reward_path"]).write_bytes(b"broken")
    with pytest.raises(ValueError, match="hash mismatch"):
        snapshots.bind_reward_snapshot(Path(later["prepared_manifest_path"]),
                                       Path(later["reward_manifest_path"]), overwrite=True)
    assert sidecar.read_bytes() == original
    assert snapshots.read_reward_snapshot(result["run"])["metadata"] == first["metadata"]


def test_identical_observation_copy_revalidates_without_dataset(tmp_path):
    result, service, _, _ = global_fixture(tmp_path)
    original = snapshots.ensure_first_reward_snapshot(result["run"], service)
    trajectory = Path(result["run"]["trajectory"])
    observations = trajectory.with_name("trajectory_observations.npz")
    temporary = observations.with_name("copied-observations.npz")
    temporary.write_bytes(observations.read_bytes())
    os.replace(temporary, observations)
    service.list = lambda *_: pytest.fail("owned snapshot must survive dataset deletion")
    assert snapshots.read_reward_snapshot(result["run"]) is None
    assert snapshots.needs_snapshot_validation(result["run"])
    restored = snapshots.ensure_first_reward_snapshot(result["run"], service)
    assert restored["metadata"]["values_sha256"] == original["metadata"]["values_sha256"]
    assert restored["metadata"]["evaluated_at"] == original["metadata"]["evaluated_at"]
    assert not snapshots.needs_snapshot_validation(result["run"])


def test_changed_observations_do_not_silently_replace_first_result(tmp_path):
    result, service, _, _ = global_fixture(tmp_path)
    snapshots.ensure_first_reward_snapshot(result["run"], service)
    trajectory = Path(result["run"]["trajectory"])
    sidecar = trajectory.with_name(snapshots.SIDECAR)
    original = sidecar.read_bytes()
    trajectory.with_name("trajectory_observations.npz").write_bytes(b"changed observations")
    service.list = lambda *_: pytest.fail("invalid first result cannot be substituted")
    assert snapshots.needs_snapshot_validation(result["run"])
    assert snapshots.ensure_first_reward_snapshot(result["run"], service) is None
    assert sidecar.read_bytes() == original
    assert not snapshots.needs_snapshot_validation(result["run"])


def test_stage_global_overwrite_does_not_force_model_or_change_recipe(tmp_path):
    jobs, dataset = setup_jobs(tmp_path)
    first = jobs.start_annotation(dataset["id"], source="stage", overwrite_global=True,
                                  stage_exponent=4)
    import yaml
    raw = yaml.safe_load(first["config_path"].read_text())
    assert first["parameters"]["overwrite_global"] is True
    assert first["parameters"]["force_model"] is False
    assert "overwrite_global" not in raw["reward"]
    assert raw["reward"]["stage_exponent"] == 4
    with pytest.raises(ValueError, match="boolean"):
        jobs.start_annotation(dataset["id"], source="stage", overwrite_global="yes")


def test_dataset_completion_preserves_first_until_explicit_stage_overwrite(tmp_path):
    result, datasets, _, versions = global_fixture(tmp_path)
    datasets.run_service = SimpleNamespace(get_run=lambda _: result["run"])
    for version in versions.values():
        version["run_ids"] = ["r"]
    jobs = object.__new__(OfflineJobService)
    jobs.datasets = datasets
    jobs.trajectory_evaluations = jobs.robometer_evaluations = None
    record = {"kind": "annotation", "dataset_id": "b", "id": "v1", "parameters": {}}
    jobs._bind_completed_result(record)
    assert snapshots.read_reward_snapshot(result["run"])["metadata"]["reward_config"]["stage_exponent"] == 2
    record["parameters"]["overwrite_global"] = True
    jobs._bind_completed_result(record)
    updated = snapshots.read_reward_snapshot(result["run"])["metadata"]
    assert updated["reward_config"]["stage_exponent"] == 4
    assert updated["origin"] == "manual"


def test_robometer_global_overwrite_never_changes_training_reward(tmp_path):
    result, datasets, _, _ = global_fixture(tmp_path)
    first = snapshots.ensure_first_reward_snapshot(result["run"], datasets)
    datasets.run_service = SimpleNamespace(get_run=lambda _: result["run"])
    datasets.get_version = lambda *_: {"evaluator": "robometer", "run_ids": ["r"],
                                      "robometer_manifest_path": str(tmp_path / "robometer.json")}
    jobs = object.__new__(OfflineJobService)
    jobs.datasets = datasets
    calls = []
    jobs.robometer_evaluations = SimpleNamespace(bind=lambda *args, **kwargs: calls.append((args, kwargs)))
    jobs._bind_completed_result({"kind": "annotation", "dataset_id": "b", "id": "robo",
                                "parameters": {"overwrite_global": True}})
    assert calls[0][1] == {"overwrite": True, "run_ids": ["r"]}
    assert snapshots.read_reward_snapshot(result["run"])["metadata"] == first["metadata"]


def test_completed_results_publish_in_single_background_queue(tmp_path):
    jobs = object.__new__(OfflineJobService)
    jobs.jobs_root = tmp_path
    jobs.lock = threading.RLock()
    jobs.repository = SimpleNamespace(upsert=lambda *_: None)
    entered, release = threading.Event(), threading.Event()
    calls = []

    def bind(job):
        calls.append(job["id"])
        if job["id"] == "first":
            entered.set()
            assert release.wait(3)
        return {"bound": [job["id"]]}

    jobs._bind_completed_result = bind
    records = []
    for identity in ("first", "second"):
        path = tmp_path / identity / "job.json"
        path.parent.mkdir()
        record = {"schema_version": 1, "id": identity, "kind": "annotation", "status": "COMPLETED"}
        path.write_text(json.dumps(record))
        records.append(record)
    try:
        jobs._schedule_result_binding(records[0])
        assert entered.wait(3)
        jobs._schedule_result_binding(records[0])
        jobs._schedule_result_binding(records[1])
        assert calls == ["first"]  # the second hash/bind work is queued
        assert all(record["trajectory_binding_status"] == "RUNNING" for record in records)
    finally:
        release.set()
        jobs.close()
    # Shutdown can cancel a queued item. Its durable RUNNING marker is retried
    # after restart, whereas the completed first result must not be duplicated.
    _, completed = jobs._load_job("first")
    jobs._schedule_result_binding(completed)
    assert calls.count("first") == 1
    assert completed["trajectory_binding_status"] == "READY"


def test_binding_error_keeps_diagnostic_job_status(tmp_path):
    jobs = object.__new__(OfflineJobService)
    jobs.jobs_root = tmp_path
    jobs.lock = threading.RLock()
    jobs.repository = SimpleNamespace(upsert=lambda *_: None)
    path = tmp_path / "job" / "job.json"
    path.parent.mkdir()
    record = {"schema_version": 1, "id": "job", "kind": "annotation", "status": "COMPLETED"}
    path.write_text(json.dumps(record))

    def fail(_):
        raise ValueError("invalid values")

    jobs._bind_completed_result = fail
    jobs._schedule_result_binding(record)
    jobs._binding_workers["job"].result(timeout=3)
    jobs.close()
    _, final = jobs._load_job("job")
    assert final["trajectory_binding_error"] == "ValueError: invalid values"
    public = jobs._public_job(final)
    assert public["status"] == "COMPLETED"
    assert public["warning"] == "ValueError: invalid values"
