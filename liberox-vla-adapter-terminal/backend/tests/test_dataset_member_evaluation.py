import asyncio
import json
from pathlib import Path

import numpy as np
import pytest
from fastapi import FastAPI, Request

from backend.app.api.training_datasets import members as members_endpoint
from backend.app.services.robometer_evaluation_service import RobometerEvaluationService
from backend.app.services.trajectory_evaluation_service import TrajectoryEvaluationService
from test_training_platform import make_run, service


def setup_dataset(tmp_path, count=1):
    runs = [make_run(tmp_path, f"run-{index}") for index in range(count)]
    datasets = service(tmp_path, runs)
    datasets.evaluations = TrajectoryEvaluationService(datasets.run_service, tmp_path)
    datasets.robometer_evaluations = RobometerEvaluationService(datasets.run_service, tmp_path)
    dataset = datasets.create(name="test", task_id="LEVEL1::pick",
                              selection={"mode": "manual", "run_ids": [run["id"] for run in runs]})
    return datasets, dataset, runs


def native(run, source):
    path = Path(run["trajectory"]).with_name(f"{source}_evaluation.json")
    path.write_text(json.dumps({"schema_version": 6 if source == "rynnvalue" else 1,
        "run_id": run["id"], "boundary_count": 3, "sample_count": 3,
        "annotator": {"model": f"global-{source}"}}))
    # Deliberately not an NPZ: a catalog request must never decode these arrays.
    path.with_suffix(".npz").write_bytes(b"values")


def local(datasets, dataset, source, identifier):
    directory = datasets.root / dataset["id"] / "annotations" / identifier
    directory.mkdir(parents=True)
    values = directory / "values.npz"
    values.write_bytes(b"local values")
    manifest = directory / "manifest.json"
    manifest.write_text(json.dumps({"complete": True, "episodes": [
        {"run_id": member["run_id"], "annotation_path": str(values), "reward_path": str(values)}
        for member in dataset["members"]]}))
    version = {"id": identifier, "dataset_id": dataset["id"], "dataset_sha256": dataset["dataset_sha256"],
        "evaluator": source, "status": "READY", "complete": True, "parameters": {},
        "completed_at": "2026-09-15T00:00:00Z",
        "reward_manifest_path": str(manifest), "robometer_manifest_path": str(manifest)}
    (directory / "version.json").write_text(json.dumps(version))
    datasets.update_version(dataset["id"], version)
    return values


def test_members_endpoint_inherits_both_global_statuses_without_array_io(tmp_path, monkeypatch):
    datasets, dataset, runs = setup_dataset(tmp_path, 2)
    for source in ("rynnvalue", "robometer"):
        native(runs[0], source)
    monkeypatch.setattr(np, "load", lambda *_args, **_kwargs: pytest.fail("list must not decode arrays"))
    monkeypatch.setattr(Path, "read_bytes", lambda *_: pytest.fail("list must not hash artifacts"))
    async def invoke(function, *args, **kwargs):
        return function(*args, **kwargs)
    monkeypatch.setattr("backend.app.api.training_datasets.run_in_threadpool", invoke)
    app = FastAPI()
    app.state.training_dataset_service = datasets
    page = asyncio.run(members_endpoint(dataset["id"], Request({"type": "http", "app": app}),
                                        page=1, page_size=5))
    by_id = {item["id"]: item for item in page["items"]}
    for field in ("rynn_evaluation", "robometer_evaluation"):
        assert by_id[runs[0]["id"]][field]["status"] == "READY"
        assert by_id[runs[0]["id"]][field]["origin"] == "global"
        assert by_id[runs[1]["id"]][field]["status"] == "NOT_EVALUATED"
    assert not datasets.get(dataset["id"])["evaluation_versions"]


def test_each_local_evaluator_overrides_only_its_own_global_status(tmp_path):
    datasets, dataset, (run,) = setup_dataset(tmp_path)
    for source in ("rynnvalue", "robometer"):
        native(run, source)
    local(datasets, dataset, "rynnvalue", "rynn")
    local(datasets, dataset, "stage", "stage")  # Active training reward is not a model output.
    item = datasets.members_page(dataset["id"])["items"][0]
    assert item["rynn_evaluation"]["origin"] == "dataset"
    assert item["rynn_evaluation"]["version_id"] == "rynn"
    assert item["robometer_evaluation"]["origin"] == "global"
    assert item["robometer_evaluation"]["status"] == "READY"
    local(datasets, dataset, "robometer", "robo")
    item = datasets.members_page(dataset["id"])["items"][0]
    assert item["robometer_evaluation"]["version_id"] == "robo"
    assert item["robometer_evaluation"]["origin"] == "dataset"
    assert item["rynn_evaluation"]["version_id"] == "rynn"


@pytest.mark.parametrize("source,field", [("rynnvalue", "rynn_evaluation"), ("robometer", "robometer_evaluation")])
def test_missing_local_artifact_is_not_hidden_by_global_evaluation(tmp_path, source, field):
    datasets, dataset, (run,) = setup_dataset(tmp_path)
    native(run, source)
    values = local(datasets, dataset, source, "local")
    values.unlink()
    status = datasets.members_page(dataset["id"])["items"][0][field]
    assert status["origin"] == "dataset"
    assert status["status"] == "ERROR"
    assert "unavailable" in status["error"]


def test_global_reward_snapshot_and_legacy_dataset_outputs_are_visible(tmp_path):
    datasets, dataset, (run,) = setup_dataset(tmp_path)
    sidecar = Path(run["trajectory"]).with_name("trajectory_reward.rynnvalue.json")
    values = sidecar.with_suffix(".npz")
    values.write_bytes(b"reward snapshot")
    sidecar.write_text(json.dumps({"schema_version": 1, "run_id": run["id"],
                                  "source": "rynnvalue", "values_file": values.name}))
    old_dataset = datasets.create(name="legacy", task_id=dataset["task_id"],
                                  selection={"mode": "manual", "run_ids": [run["id"]]})
    local(datasets, old_dataset, "robometer", "legacy-robo")
    item = datasets.members_page(dataset["id"])["items"][0]
    assert item["rynn_evaluation"]["status"] == "READY"
    assert item["rynn_evaluation"]["origin"] == "global"
    assert item["robometer_evaluation"]["status"] == "READY"
    assert item["robometer_evaluation"]["origin"] == "global"


def test_only_requested_page_reads_global_evaluations_and_refreshes(tmp_path, monkeypatch):
    datasets, dataset, runs = setup_dataset(tmp_path, 6)
    calls = []
    status = datasets.evaluations.status
    monkeypatch.setattr(datasets.evaluations, "status", lambda run: (calls.append(run["id"]), status(run))[1])
    page = datasets.members_page(dataset["id"], page=2, page_size=5)
    assert len(page["items"]) == 1 and page["total"] == 6 and page["pages"] == 2
    run = next(run for run in runs if run["id"] == page["items"][0]["id"])
    assert calls == [run["id"]]
    assert page["items"][0]["rynn_evaluation"]["status"] == "NOT_EVALUATED"
    native(run, "rynnvalue")
    assert datasets.members_page(dataset["id"], page=2, page_size=5)["items"][0]["rynn_evaluation"]["status"] == "READY"
