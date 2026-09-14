from __future__ import annotations

import hashlib
import json
import threading
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest
import yaml

from backend.app.core.exceptions import ConflictError
from backend.app.services.offline_job_service import OfflineJobService
from backend.app.workers.finalize_reward_version import seal
from test_training_platform import make_run, service


WORKSPACE = Path(__file__).resolve().parents[3]


def setup_jobs(tmp_path):
    datasets = service(tmp_path, [make_run(tmp_path, "run")])
    dataset = datasets.create(name="one", task_id="LEVEL1::pick",
                              selection={"mode": "manual", "run_ids": ["run"]})
    jobs = object.__new__(OfflineJobService)
    jobs.datasets = datasets
    jobs.lock = threading.RLock()
    jobs.jobs_root = tmp_path / "jobs"
    jobs.training_root = tmp_path / "training"
    jobs.cache_root = tmp_path / "cache"
    jobs.launch_reserved = False
    jobs.trajectory_evaluations = None
    jobs.stage_annotations = SimpleNamespace(validate_members=lambda members, threshold: {
        "run": {"schema_version": 2, "keyframes": [{"step": 5, "kind": "positive"}],
                "annotation_sha256": "label"}})
    jobs.ui_config = SimpleNamespace(offline_rl_root=WORKSPACE / "vla-adapter-rynn-iql",
        robometer_root=WORKSPACE / "vla-adapter-robometer", train_environment="vla-liberox",
        reward_environment="rynnvalue-reward", robometer_environment="robometer-reward")
    jobs._prepare_launch = lambda: None
    jobs._new_job = lambda **kwargs: kwargs
    jobs.available_checkpoints = lambda _: []
    return jobs, dataset


def finish(jobs, dataset, job):
    work = job["output_path"] / "work"
    prepared = {"dataset_sha256": "prepared", "source_dataset_sha256": dataset["dataset_sha256"],
                "source_dataset_id": dataset["id"], "episodes": [{"run_id": "run"}]}
    (work / "dataset_manifest.json").write_text(json.dumps(prepared))
    rewards = work / "rewards"
    rewards.mkdir()
    values = rewards / "values.npz"
    np.savez(values, final_reward=np.array([-1., 0.]))
    digest = hashlib.sha256(values.read_bytes()).hexdigest()
    manifest = {"complete": True, "dataset_sha256": "prepared", "episodes": [{
        "run_id": "run", "reward_path": str(values), "reward_sha256": digest,
        "annotation_path": str(values), "annotation_sha256": digest}]}
    if job["parameters"]["source"] == "rynnvalue":
        raw = yaml.safe_load(job["config_path"].read_text())
        annotations = work / "annotations"
        annotations.mkdir(exist_ok=True)
        raw_values = annotations / "official.npz"
        np.savez(raw_values, value=np.array([1., 0.]))
        annotations_index = {"complete": True, "kind": "rynnvalue_annotation", "dataset_sha256": "prepared",
            "annotation_config": {key: raw["reward"][key] for key in (
                "model", "revision", "dtype", "max_frames", "robot_description", "camera_description")},
            "episodes": [{"run_id": "run", "annotation_path": str(raw_values),
                          "annotation_sha256": hashlib.sha256(raw_values.read_bytes()).hexdigest()}]}
        (annotations / "annotation_manifest.json").write_text(json.dumps(annotations_index))
    (rewards / "reward_manifest.json").write_text(json.dumps(manifest))
    version = seal(job["output_path"] / "version.json")
    jobs.datasets.update_version(dataset["id"], version)
    return version


def test_direct_generation_is_cpu_only_and_snapshots_latest_labels(tmp_path):
    jobs, dataset = setup_jobs(tmp_path)
    jobs._prepare_launch = lambda: pytest.fail("must not reserve or unload GPU")
    first = jobs.start_annotation(dataset["id"], source="stage", stage_exponent=2.0)
    assert [s["id"] for s in first["stages"]] == ["prepare", "rewards", "seal"]
    assert first["parameters"]["requires_gpu"] is False
    version = finish(jobs, dataset, first)
    original_bytes = Path(version["reward_manifest_path"]).read_bytes()
    jobs.stage_annotations.validate_members = lambda *_: {"run": {"annotation_sha256": "new-label", "keyframes": []}}
    second = jobs.start_annotation(dataset["id"], source="stage", stage_exponent=4.0)
    snapshot = json.loads((second["output_path"] / "stage_annotations.json").read_text())
    assert snapshot["annotations"]["run"]["annotation_sha256"] == "new-label"
    assert Path(version["reward_manifest_path"]).read_bytes() == original_bytes
    assert jobs.datasets.get(dataset["id"])["reward_version_id"] == version["id"]


def test_failed_regeneration_and_robometer_do_not_replace_reward(tmp_path):
    jobs, dataset = setup_jobs(tmp_path)
    version = finish(jobs, dataset, jobs.start_annotation(dataset["id"], source="sparse"))
    jobs.datasets.update_version(dataset["id"], {**version, "id": "failed", "status": "ERROR"})
    jobs.datasets.update_version(dataset["id"], {**version, "id": "robo", "evaluator": "robometer"})
    current = jobs.datasets.get(dataset["id"])
    assert current["reward_version_id"] == version["id"]
    assert current["robometer_version_id"] == "robo"


def test_training_pins_sealed_version_and_never_recomputes(tmp_path):
    jobs, dataset = setup_jobs(tmp_path)
    version = finish(jobs, dataset, jobs.start_annotation(dataset["id"], source="stage", stage_exponent=4.0, gamma=.92))
    jobs.stage_annotations.validate_members = lambda *_: pytest.fail("training must consume snapshot")
    train = jobs.start_training(dataset["id"], {"train_steps": 100})
    assert [s["id"] for s in train["stages"]] == ["train"]
    raw = yaml.safe_load(train["config_path"].read_text())
    assert raw["reward"]["gamma"] == .92
    assert raw["reward"]["stage_exponent"] == 4
    assert raw["reward"]["manifest_path"] == version["reward_manifest_path"]
    assert raw["reward"]["version_id"] == version["id"]
    defaults = jobs.defaults(dataset["id"])
    assert defaults["reward_parameters_locked"] is False
    assert "reward_gamma" in defaults["reward_editable_parameters"]
    assert "reward_stage_exponent" in defaults["reward_locked_parameters"]
    with pytest.raises(ConflictError, match="locked"):
        jobs.start_training(dataset["id"], {"reward_stage_exponent": 2.0})


def test_unprepared_dataset_cannot_train_and_missing_labels_stop_generation(tmp_path):
    jobs, dataset = setup_jobs(tmp_path)
    with pytest.raises(ConflictError):
        jobs.start_training(dataset["id"], {})
    def missing(*_):
        raise ValueError("run: missing keyframes")
    jobs.stage_annotations.validate_members = missing
    with pytest.raises(ValueError, match="missing keyframes"):
        jobs.start_annotation(dataset["id"], source="stage")
    assert not jobs.jobs_root.exists()


@pytest.mark.parametrize("options", [{"source": "stage", "max_frames": 4},
    {"source": "stage", "stage_exponent": .5}, {"source": "sparse", "gamma": float("nan")},
    {"source": "sparse", "force_model": True}, {"source": "robometer", "sampling_hz": 21}])
def test_invalid_version_parameters_fail_before_job(tmp_path, options):
    jobs, dataset = setup_jobs(tmp_path)
    with pytest.raises(ValueError):
        jobs.start_annotation(dataset["id"], **options)
    assert not jobs.jobs_root.exists()


def test_activation_is_explicit_and_preserves_other_dataset(tmp_path):
    jobs, dataset = setup_jobs(tmp_path)
    first = finish(jobs, dataset, jobs.start_annotation(dataset["id"], source="stage", stage_exponent=2.0))
    second = finish(jobs, dataset, jobs.start_annotation(dataset["id"], source="stage", stage_exponent=4.0))
    jobs.datasets.get_version(dataset["id"], first["id"])
    assert jobs.datasets.get(dataset["id"])["reward_version_id"] == second["id"]
    assert jobs.datasets.activate_version(dataset["id"], first["id"])["reward_version_id"] == first["id"]
    other = jobs.datasets.derive(dataset["id"], name="two", selection={"mode": "manual", "run_ids": ["run"]})
    assert other["reward_version_id"] is None
    assert jobs.datasets.members_page(dataset["id"])["items"][0]["id"] == "run"


def test_seal_is_immutable_and_detects_corruption(tmp_path):
    jobs, dataset = setup_jobs(tmp_path)
    job = jobs.start_annotation(dataset["id"], source="sparse")
    version = finish(jobs, dataset, job)
    with pytest.raises(ValueError, match="cannot be overwritten"):
        seal(job["output_path"] / "version.json")
    Path(version["reward_manifest_path"]).write_text("corrupt")
    with pytest.raises(ValueError, match="integrity"):
        jobs.start_training(dataset["id"], {})


def test_rynn_output_reuse_across_datasets_and_force_bypass(tmp_path):
    jobs, dataset = setup_jobs(tmp_path)
    old = finish(jobs, dataset, jobs.start_annotation(dataset["id"], source="rynnvalue"))
    other = jobs.datasets.derive(dataset["id"], name="other", selection={"mode": "manual", "run_ids": ["run"]})
    reused = jobs.start_annotation(other["id"], source="rynnvalue", gamma=.92)
    seeded = json.loads((reused["output_path"] / "work" / "annotations" / "annotation_manifest.json").read_text())
    assert seeded["episodes"][0]["annotation_path"].startswith(old["work_dir"])
    completed = finish(jobs, other, reused)
    forced = jobs.start_annotation(other["id"], source="rynnvalue", force_model=True)
    assert not (forced["output_path"] / "work" / "annotations" / "annotation_manifest.json").exists()
    assert "--overwrite" in next(stage for stage in forced["stages"] if stage["id"] == "annotate")["argv"]
    assert Path(completed["annotation_manifest_path"]).is_file()


def test_freeze_and_derive_api_do_not_start_evaluation(tmp_path, monkeypatch):
    import asyncio
    from fastapi import FastAPI, Request
    from backend.app.api import training_datasets as api
    from backend.app.api.models import CreateTrainingDatasetRequest, DeriveTrainingDatasetRequest

    async def invoke(function, *args, **kwargs):
        return function(*args, **kwargs)
    monkeypatch.setattr(api, "run_in_threadpool", invoke)

    jobs, dataset = setup_jobs(tmp_path)
    jobs.start_annotation = lambda *_args, **_kwargs: pytest.fail("freeze must not evaluate")
    app = FastAPI()
    app.state.training_dataset_service = jobs.datasets
    app.state.offline_job_service = jobs
    request = Request({"type": "http", "app": app})
    body = {"name": "new", "task_id": "LEVEL1::pick", "selection": {"mode": "manual", "run_ids": ["run"]}}
    created = asyncio.run(api.create(CreateTrainingDatasetRequest(**body), request))
    assert created["reward_version_id"] is None and created["evaluation_versions"] == []
    derived = asyncio.run(api.derive(dataset["id"], DeriveTrainingDatasetRequest(**{
        "name": "derived", "selection": body["selection"]}), request))
    assert derived["reward_version_id"] is None


def test_hidden_recipe_parameters_are_pinned_not_taken_from_new_base(tmp_path):
    jobs, dataset = setup_jobs(tmp_path)
    finish(jobs, dataset, jobs.start_annotation(dataset["id"], source="stage"))
    old_load = jobs._load_base_config
    def changed_base():
        raw = old_load()
        raw["reward"]["shaping_weight"] = 99
        return raw
    jobs._load_base_config = changed_base
    assert jobs.defaults(dataset["id"])["advanced"]["reward_shaping_weight"] == .1
    with pytest.raises(ConflictError, match="locked"):
        jobs.start_training(dataset["id"], {"reward_shaping_weight": 99})


@pytest.mark.parametrize("source", ["sparse", "stage", "rynnvalue"])
@pytest.mark.parametrize("overrides", [
    {"reward_gamma": .92}, {"reward_accumulate_primitive_steps": True},
    {"reward_gamma": .95, "reward_accumulate_primitive_steps": True},
])
def test_training_discount_overrides_do_not_mutate_evaluation(tmp_path, source, overrides):
    jobs, dataset = setup_jobs(tmp_path)
    version = finish(jobs, dataset, jobs.start_annotation(dataset["id"], source=source))
    root = Path(version["work_dir"]).parent
    original = {path: path.read_bytes() for path in root.rglob("*") if path.is_file()}
    train = jobs.start_training(dataset["id"], overrides)
    raw = yaml.safe_load(train["config_path"].read_text())
    assert [stage["id"] for stage in train["stages"]] == ["train"]
    assert raw["reward"]["version_id"] == version["id"]
    assert raw["reward"]["manifest_path"] == version["reward_manifest_path"]
    assert raw["reward"]["gamma"] == overrides.get("reward_gamma", .99)
    assert raw["reward"]["accumulate_primitive_steps"] is overrides.get("reward_accumulate_primitive_steps", False)
    assert train["parameters"]["reward"]["gamma"] == raw["reward"]["gamma"]
    assert original == {path: path.read_bytes() for path in root.rglob("*") if path.is_file()}
    assert jobs.datasets.get(dataset["id"])["reward_version_id"] == version["id"]


@pytest.mark.parametrize("override", [
    {"reward_source": "sparse"}, {"reward_stage_exponent": 3},
    {"reward_shaping_weight": .2}, {"reward_rynnvalue": True},
])
def test_other_reward_recipe_fields_remain_locked(tmp_path, override):
    jobs, dataset = setup_jobs(tmp_path)
    finish(jobs, dataset, jobs.start_annotation(dataset["id"], source="stage"))
    with pytest.raises(ConflictError, match="locked"):
        jobs.start_training(dataset["id"], override)


@pytest.mark.parametrize("override", [
    {"reward_gamma": float("nan")}, {"reward_gamma": None}, {"reward_gamma": 1.1},
    {"reward_accumulate_primitive_steps": None}, {"reward_accumulate_primitive_steps": 1},
])
def test_training_discount_overrides_are_strict(override):
    with pytest.raises(ValueError):
        OfflineJobService._validate_training_parameters(override)
