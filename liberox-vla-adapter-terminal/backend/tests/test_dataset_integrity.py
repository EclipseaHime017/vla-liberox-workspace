import asyncio
import json
from pathlib import Path

import numpy as np
import pytest

from backend.app.api import training_datasets as api
from backend.app.services.dataset_integrity import verify_dataset
from backend.app.services.stage_annotation_service import StageAnnotationService
from backend.app.workers.finalize_reward_version import seal
from test_global_reward_inheritance import setup_recording, native_rynn
from vla_rynn_iql.config import load_train_config
from vla_rynn_iql.data import prepare_dataset
from vla_rynn_iql.rewards import materialize_reward_manifest


def setup(tmp_path, monkeypatch):
    jobs, dataset, run = setup_recording(tmp_path, monkeypatch)
    jobs.stage_annotations = StageAnnotationService(jobs.datasets.run_service, jobs.ui_config.offline_rl_root)
    jobs.stage_annotations.bind_datasets(jobs.datasets)
    return jobs, dataset, run


def rows(report):
    return {row["source"]: row for row in report["sources"]}


def test_verification_reports_global_local_and_effective_without_model_or_observation_decode(tmp_path, monkeypatch):
    jobs, dataset, run = setup(tmp_path, monkeypatch)
    native_rynn(jobs, dataset, run)
    original = np.load
    def no_observations(path, *args, **kwargs):
        assert "trajectory_observations" not in str(path)
        return original(path, *args, **kwargs)
    monkeypatch.setattr(np, "load", no_observations)
    jobs.start_annotation = lambda *_args, **_kw: pytest.fail("verification must not evaluate")
    protected = {p: p.read_bytes() for p in Path(run["trajectory"]).parent.iterdir() if p.is_file()}
    report = verify_dataset(jobs, dataset["id"])
    assert report["integrity_status"] == "HEALTHY"
    by_source = rows(report)
    for source in ("sparse", "stage", "rynnvalue"):
        assert by_source[source]["global"]["ready"] == 1
        assert by_source[source]["dataset"]["ready"] == 0
        assert by_source[source]["effective"]["ready"] == by_source[source]["inherited"] == 1
    for source in ("final", "robometer"):
        assert by_source[source]["effective"]["missing"] == 1
        assert any("全局评价不可用" in item["error"] for item in by_source[source]["issues"])
    assert not jobs.jobs_root.exists()
    assert all(p.read_bytes() == value for p, value in protected.items())


def test_local_result_is_not_hidden_by_global_and_broken_arrays_remain_errors(tmp_path, monkeypatch):
    jobs, dataset, run = setup(tmp_path, monkeypatch)
    job = jobs.start_annotation(dataset["id"], source="stage", stage_exponent=4)
    config = load_train_config(job["config_path"])
    prepare_dataset(config)
    materialize_reward_manifest(config)
    version = seal(job["output_path"] / "version.json")
    jobs.datasets.update_version(dataset["id"], version)
    row = rows(verify_dataset(jobs, dataset["id"]))["stage"]
    assert row["dataset"]["ready"] == row["effective"]["ready"] == 1
    assert row["inherited"] == 0
    index = json.loads(Path(version["reward_manifest_path"]).read_text())
    Path(index["episodes"][0]["reward_path"]).write_bytes(b"broken")
    row = rows(verify_dataset(jobs, dataset["id"]))["stage"]
    assert row["global"]["ready"] == 1
    assert row["dataset"]["invalid"] == row["effective"]["invalid"] == 1
    assert row["inherited"] == 0


def test_global_and_owned_labels_are_checked_independently(tmp_path, monkeypatch):
    jobs, dataset, run = setup(tmp_path, monkeypatch)
    Path(run["trajectory"]).with_name("stage_annotation.json").write_text("{}")
    report = rows(verify_dataset(jobs, dataset["id"]))
    assert report["keyframes"]["global"]["invalid"] == 1
    assert report["keyframes"]["effective"]["ready"] == 1
    assert report["stage"]["global"]["invalid"] == 1
    assert report["stage"]["effective"]["ready"] == 1


def test_robometer_integrity_uses_saved_outputs_even_without_model_environment(tmp_path, monkeypatch):
    from test_dataset_transfer import robometer
    jobs, dataset, run = setup(tmp_path, monkeypatch)
    robometer(run)
    row = rows(verify_dataset(jobs, dataset["id"]))["robometer"]
    assert row["global"]["ready"] == row["effective"]["ready"] == row["inherited"] == 1
    values = Path(run["trajectory"]).with_name("robometer_evaluation.npz")
    values.write_bytes(b"corrupt")
    row = rows(verify_dataset(jobs, dataset["id"]))["robometer"]
    assert row["global"]["invalid"] == row["effective"]["invalid"] == 1


def test_late_global_labels_are_inherited_by_training_and_reward_generation_without_editor(tmp_path, monkeypatch):
    jobs, _, run = setup(tmp_path, monkeypatch)
    path = Path(run["trajectory"]).with_name("stage_annotation.json")
    labels = path.read_bytes()
    path.unlink()
    dataset = jobs.datasets.create(name="before-labels", task_id="LEVEL1::pick",
        selection={"mode": "manual", "run_ids": ["run"]})
    missing = jobs.defaults(dataset["id"], "stage")["reward_availability"]
    assert not missing["ready"] and missing["missing_run_ids"] == ["run"]
    assert "数据集与全局均缺少" in missing["errors"][0]["error"]
    path.write_bytes(labels)
    assert jobs.defaults(dataset["id"], "stage")["reward_availability"]["ready"]
    job = jobs.start_annotation(dataset["id"], source="stage", stage_exponent=4)
    snapshot = json.loads((job["output_path"] / "stage_annotations.json").read_text())
    assert snapshot["annotations"]["run"]["keyframes"] == json.loads(labels)["keyframes"]


def test_required_missing_global_reward_blocks_training_instead_of_changing_source(tmp_path, monkeypatch):
    from backend.app.core.exceptions import ConflictError
    jobs, dataset, _ = setup(tmp_path, monkeypatch)
    for source in ("rynnvalue", "final"):
        available = jobs.defaults(dataset["id"], source)["reward_availability"]
        assert not available["ready"] and available["missing_run_ids"] == ["run"]
        with pytest.raises(ConflictError, match="缺少可用"):
            jobs.start_training(dataset["id"], {"reward_source": source})
    assert not jobs.jobs_root.exists()


def test_verify_endpoint_queues_full_report(tmp_path, monkeypatch):
    jobs, dataset, _ = setup(tmp_path, monkeypatch)
    monkeypatch.setattr(api, "offline_job_service", lambda _: jobs)
    monkeypatch.setattr(api, "training_dataset_service", lambda _: jobs.datasets)
    queued = asyncio.run(api.verify(dataset["id"], None))
    assert queued["status"] == "QUEUED" and queued["kind"] == "verification"
    _, future, _ = jobs._local_tasks[queued["id"]]
    jobs._dispatch_training_queue()
    result = future.result(10)
    saved = jobs.get(queued["id"])
    assert saved["status"] == "COMPLETED"
    assert saved["result"]["verification"] == result
    assert "完整性检查完成" == saved["stage_label"]
