import hashlib
import asyncio
import json
from pathlib import Path

import numpy as np
import pytest
from fastapi import FastAPI, Request

from backend.app.api.datasets import get_stage_annotation, save_stage_annotation
from backend.app.api.models import StageAnnotationRequest
from backend.app.core.exceptions import ConflictError
from backend.app.services.dataset_stage_annotations import _digest
from backend.app.services.stage_annotation_service import StageAnnotationService
from test_training_platform import make_run, service


def setup(tmp_path):
    runs = [make_run(tmp_path, name) for name in ("one", "two")]
    for run in runs:
        np.savez_compressed(run["trajectory"], env_action=np.zeros((17, 7)),
            done=np.zeros(17, dtype=bool), time_seconds=np.arange(18) / 20)
    datasets = service(tmp_path, runs)
    stages = StageAnnotationService(datasets.run_service,
        Path(__file__).resolve().parents[3] / "vla-adapter-rynn-iql")
    stages._defaults = lambda: (5, 2.)
    stages.bind_datasets(datasets)
    return datasets, stages, runs


def create(datasets, name="A", run_ids=None, **kwargs):
    return datasets.create(name=name, task_id="LEVEL1::pick",
        selection={"mode": "manual", "run_ids": run_ids or ["one"]}, **kwargs)


def save(stages, step, dataset_id=None, run_id="one"):
    view = stages.detail(run_id, dataset_id)
    frames = [] if step is None else [{"step": step, "kind": "positive"}]
    return stages.save(run_id, frames, 2, view["revision"], dataset_id)


def test_copy_once_and_independent_edits_in_global_parent_and_siblings(tmp_path):
    datasets, stages, runs = setup(tmp_path)
    save(stages, 3)
    first, sibling = create(datasets), create(datasets, "B")
    selection = (datasets.root / first["id"] / "dataset.json").read_bytes()
    assert stages.detail("one", first["id"])["origin"] == "global_copy"
    global_before = Path(runs[0]["trajectory"]).with_name("stage_annotation.json").read_bytes()
    save(stages, 6, first["id"])
    child = datasets.derive(first["id"], name="child",
        selection={"mode": "manual", "run_ids": ["one", "two"]})
    save(stages, 9, first["id"])
    assert Path(runs[0]["trajectory"]).with_name("stage_annotation.json").read_bytes() == global_before
    save(stages, 12)
    assert stages.detail("one", first["id"])["keyframes"][0]["step"] == 9
    assert stages.detail("one", sibling["id"])["keyframes"][0]["step"] == 3
    child_view = stages.detail("one", child["id"])
    assert child_view["origin"] == "parent_copy" and child_view["keyframes"][0]["step"] == 6
    assert stages.detail("two", child["id"])["status"] == "missing"
    assert (datasets.root / first["id"] / "dataset.json").read_bytes() == selection


def test_missing_is_persistent_and_inheritance_is_explicit_with_scope_revision(tmp_path):
    datasets, stages, _ = setup(tmp_path)
    first, other = create(datasets), create(datasets, "B")
    before = stages.detail("one", first["id"])
    save(stages, 4)
    assert stages.detail("one", first["id"])["status"] == "missing"
    with pytest.raises(ConflictError):
        stages.save("one", [], 2, before["revision"], other["id"])
    inherited = stages.save("one", [], 2, before["revision"], first["id"], inherit_global=True)
    assert inherited["origin"] == "global_copy" and inherited["keyframes"][0]["step"] == 4
    save(stages, 7)
    assert stages.detail("one", first["id"])["keyframes"][0]["step"] == 4
    with pytest.raises(ConflictError):
        stages.save("one", [], 2, before["revision"], first["id"], inherit_global=True)
    with pytest.raises(ValueError, match="不属于"):
        stages.detail("two", first["id"])
    with pytest.raises(ValueError, match="不属于"):
        stages.save("two", [], 2, stages.detail("two")["revision"], first["id"])


def test_member_validation_uses_owned_marks_and_frozen_trajectory(tmp_path):
    datasets, stages, runs = setup(tmp_path)
    save(stages, 3)
    dataset = create(datasets)
    save(stages, 8, dataset["id"])
    save(stages, 10)
    _, frozen = datasets._load(dataset["id"])
    labels = stages.validate_members(frozen["members"], 5, dataset_id=dataset["id"])
    assert labels["one"]["keyframes"][0]["step"] == 8
    np.savez_compressed(runs[0]["trajectory"], env_action=np.ones((17, 7)),
        done=np.zeros(17, dtype=bool), time_seconds=np.arange(18) / 20)
    with pytest.raises(ValueError, match="冻结数据集不匹配"):
        stages.validate_members(frozen["members"], 5, dataset_id=dataset["id"])


def legacy_snapshot(datasets, dataset, *, corrupt=False):
    labels = datasets.stage_labels.load(datasets._load(dataset["id"])[1])["annotations"]
    snapshot = {"schema_version": 1, "annotations": {"one": labels["one"]["annotation"]}}
    root = datasets.root / dataset["id"] / "annotations" / "old"
    root.mkdir(parents=True)
    path = root / "stage_annotations.json"
    path.write_text(json.dumps(snapshot))
    manifest = root / "reward_manifest.json"
    manifest.write_text(json.dumps({"stage_annotations_path": str(path),
        "stage_annotations_sha256": _digest(snapshot), "episodes": [{"run_id": "one",
            "stage_annotation_sha256": snapshot["annotations"]["one"]["annotation_sha256"]}]}))
    version = {"id": "old", "dataset_id": dataset["id"], "dataset_sha256": dataset["dataset_sha256"],
        "evaluator": "stage", "status": "READY", "reward_manifest_path": str(manifest),
        "reward_manifest_sha256": hashlib.sha256(manifest.read_bytes()).hexdigest()}
    (root / "version.json").write_text(json.dumps(version))
    datasets._update_metadata(dataset["id"], {"evaluation_versions": [version],
        "evaluation_version_ids": {"stage": "old"}})
    if corrupt:
        manifest.write_text(manifest.read_text() + " ")  # Valid JSON, broken sealed hash.
    (datasets.root / dataset["id"] / "stage_annotations.json").unlink()
    return path, manifest


@pytest.mark.parametrize("corrupt", [False, True])
def test_legacy_prefers_its_reward_snapshot_and_never_hides_corruption(tmp_path, corrupt):
    datasets, stages, _ = setup(tmp_path)
    save(stages, 3)
    dataset = create(datasets)
    paths = legacy_snapshot(datasets, dataset, corrupt=corrupt)
    protected = {p: p.read_bytes() for p in paths}
    save(stages, 12)
    restored = stages.detail("one", dataset["id"])
    assert restored["origin"] == "reward_snapshot"
    if corrupt:
        assert restored["status"] == "stale" and "哈希" in restored["error"]
        assert restored["keyframes"] == []
    else:
        assert restored["status"] == "ready" and restored["keyframes"][0]["step"] == 3
    save(stages, 6, dataset["id"])
    assert all(p.read_bytes() == value for p, value in protected.items())


def test_multiple_member_saves_do_not_clobber_other_members(tmp_path):
    datasets, stages, _ = setup(tmp_path)
    dataset = create(datasets, run_ids=["one", "two"])
    first = stages.detail("one", dataset["id"])
    second = stages.detail("two", dataset["id"])
    stages.save("one", [{"step": 3, "kind": "positive"}], 2, first["revision"], dataset["id"])
    stages.save("two", [{"step": 6, "kind": "positive"}], 2, second["revision"], dataset["id"])
    assert stages.detail("one", dataset["id"])["keyframes"][0]["step"] == 3
    assert stages.detail("two", dataset["id"])["keyframes"][0]["step"] == 6


def test_dataset_api_scope_and_explicit_inherit(tmp_path, monkeypatch):
    datasets, stages, _ = setup(tmp_path)
    dataset = create(datasets)
    save(stages, 4)
    app = FastAPI()
    app.state.stage_annotation_service = stages
    request = Request({"type": "http", "app": app})
    async def dispatch(function, *args, **kwargs):
        return function(*args, **kwargs)
    monkeypatch.setattr("backend.app.api.datasets.run_in_threadpool", dispatch)
    async def exercise():
        current = await get_stage_annotation("one", request, dataset["id"])
        assert current["status"] == "missing"
        saved = await save_stage_annotation("one", StageAnnotationRequest(
            keyframes=[], revision=current["revision"], dataset_id=dataset["id"], inherit_global=True), request)
        assert saved["dataset_id"] == dataset["id"] and saved["keyframes"][0]["step"] == 4
        assert (await get_stage_annotation("one", request))["dataset_id"] is None
    asyncio.run(exercise())
