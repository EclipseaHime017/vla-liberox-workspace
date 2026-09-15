import copy
import json
from pathlib import Path

import numpy as np
import pytest
import yaml

from backend.app.core.exceptions import ConflictError
from backend.app.services.trajectory_reward_snapshot import bind_reward_snapshot, read_reward_snapshot, snapshot_path
from backend.app.workers.finalize_reward_version import seal
from test_dataset_reward_versions import setup_jobs, finish
from test_dataset_evaluation_detail import global_fixture, digest, attach_dataset_context


def full_reward(jobs, dataset, source="stage", exponent=2):
    job = jobs.start_annotation(dataset["id"], source=source,
                                **({"stage_exponent": exponent} if source == "stage" else {}))
    raw = yaml.safe_load(job["config_path"].read_text())
    _, frozen = jobs.datasets._load(dataset["id"])
    artifact = frozen["members"][0]["artifacts"]
    work = job["output_path"] / "work"
    episode = {"run_id": "run", "root_run_id": "run", "parent_run_id": None,
        "kind": "original", "split": "train", "success": False, "terminal_step": None,
        "action_count": 17, "recorded_action_count": 17, "resume_step": None,
        "prompt": "pick bowl", "observation_orientation": "libero_raw",
        "trajectory_path": artifact["trajectory"]["path"], "trajectory_sha256": artifact["trajectory"]["sha256"],
        "observations_path": artifact["observations"]["path"], "observations_sha256": artifact["observations"]["sha256"],
        "source_manifest": artifact["manifest"]["path"], "source_manifest_sha256": artifact["manifest"]["sha256"],
        "chunks": [{"start": a, "end": b, "length": b-a, "action_source": "policy",
                    "transition_type": "policy", "interrupted": False, "copied_prefix": False}
                   for a, b in [(0, 8), (8, 16), (16, 17)]], "reward_boundaries": [0, 8, 16, 17]}
    prepared = {"schema_version": 4, "dataset_sha256": "prepared", "source_dataset_id": dataset["id"],
        "source_dataset_sha256": dataset["dataset_sha256"], "episodes": [episode],
        **{key: raw["data"][key] for key in ("action_horizon", "action_dim", "proprio_dim", "control_hz")}}
    (work / "dataset_manifest.json").write_text(json.dumps(prepared))
    rewards = work / "rewards"
    rewards.mkdir()
    arrays = rewards / "values.npz"
    scores = -1 + (np.arange(18)/17)**exponent
    np.savez(arrays, final_reward=scores[[8, 16, 17]], boundary_steps=[0, 8, 16, 17],
             stage_score=scores, time_seconds=np.arange(18)/20)
    recipe = {"source": source, "rynnvalue": False, "gamma": raw["reward"]["gamma"],
              "accumulate_primitive_steps": False}
    if source == "stage":
        recipe["stage_exponent"] = exponent
    index = {"schema_version": 1, "kind": "derived_iql_reward", "complete": True,
        "dataset_sha256": "prepared", "reward_config": recipe, "episodes": [{
            "run_id": "run", "reward_path": str(arrays), "annotation_path": str(arrays),
            "reward_sha256": digest(arrays), "annotation_sha256": digest(arrays)}]}
    (rewards / "reward_manifest.json").write_text(json.dumps(index))
    version = seal(job["output_path"] / "version.json")
    jobs.datasets.update_version(dataset["id"], version)
    return version


def test_each_dataset_source_remains_selectable_after_other_evaluations(tmp_path):
    jobs, dataset = setup_jobs(tmp_path)
    saved = {source: finish(jobs, dataset, jobs.start_annotation(dataset["id"], source=source))
             for source in ("rynnvalue", "sparse", "stage")}
    for source, version in saved.items():
        defaults = jobs.defaults(dataset["id"], source)
        assert defaults["reward_version"]["id"] == version["id"]
        assert defaults["reward_availability"]["ready"]
        train = jobs.start_training(dataset["id"], {"reward_source": source})
        assert train["parameters"]["reward_version_id"] == version["id"]
    current = jobs.datasets.get(dataset["id"])
    assert current["evaluation_version_ids"] == {key: val["id"] for key, val in saved.items()}
    newer = finish(jobs, dataset, jobs.start_annotation(dataset["id"], source="stage"))
    assert jobs.datasets.get(dataset["id"])["evaluation_version_ids"] == {
        **current["evaluation_version_ids"], "stage": newer["id"]}


def test_global_pin_is_independent_of_dataset_evaluation_and_copies_exact_arrays(tmp_path):
    from vla_rynn_iql.config import load_train_config as load_config
    from vla_rynn_iql.rewards import load_pinned_reward_index, reward_manifest_digest

    jobs, owner = setup_jobs(tmp_path)
    version = full_reward(jobs, owner)
    bind_reward_snapshot(Path(version["prepared_manifest_path"]), Path(version["reward_manifest_path"]))
    target = jobs.datasets.create(name="global fallback", task_id="LEVEL1::pick",
        selection={"mode": "manual", "run_ids": ["run"]})
    defaults = jobs.defaults(target["id"], "stage")
    assert defaults["reward_version"] is None
    assert defaults["reward_availability"] == {"ready": True, "origin": "global", "missing_run_ids": [], "errors": [], "pending": False}
    original = json.loads(Path(version["reward_manifest_path"]).read_text())
    train = jobs.start_training(target["id"], {"reward_source": "stage"})
    index = load_pinned_reward_index(load_config(train["config_path"]))
    assert index["binding_kind"] == "global_trajectory_snapshots"
    assert Path(index["episodes"][0]["reward_path"]).read_bytes() == Path(original["episodes"][0]["reward_path"]).read_bytes()
    again = jobs.start_training(target["id"], {"reward_source": "stage"})
    assert reward_manifest_digest(load_pinned_reward_index(load_config(again["config_path"]))) == reward_manifest_digest(index)
    assert not jobs.datasets.get(target["id"])["evaluation_version_ids"]
    assert train["parameters"]["reward_version_id"] != again["parameters"]["reward_version_id"]
    local = full_reward(jobs, target, exponent=4)
    assert jobs.defaults(target["id"], "stage")["reward_version"]["id"] == local["id"]
    assert read_reward_snapshot(jobs.datasets.run_service.get_run("run"), "stage")["metadata"]["reward_config"]["stage_exponent"] == 2
    assert load_pinned_reward_index(load_config(train["config_path"]))["episodes"] == index["episodes"]


def test_source_selection_never_falls_back_to_another_type(tmp_path):
    jobs, dataset = setup_jobs(tmp_path)
    full_reward(jobs, dataset)
    assert not jobs.defaults(dataset["id"], "sparse")["reward_availability"]["ready"]
    with pytest.raises(ConflictError, match="sparse"):
        jobs.start_training(dataset["id"], {"reward_source": "sparse"})


def test_legacy_evaluation_without_version_config_does_not_break_defaults(tmp_path):
    jobs, dataset = setup_jobs(tmp_path)
    jobs.datasets.update_annotation(dataset["id"], "READY", annotation_id="legacy", annotation_config={"max_frames": 4})
    before = (jobs.datasets.root / dataset["id"] / "dataset.json").read_bytes()
    defaults = jobs.defaults(dataset["id"], "rynnvalue")
    assert defaults["reward_version"]["legacy"]
    assert not defaults["reward_availability"]["ready"]
    assert "旧评价" in defaults["reward_availability"]["message"]
    assert defaults["basic"]["train_steps"] > 0
    assert (jobs.datasets.root / dataset["id"] / "dataset.json").read_bytes() == before


def test_missing_run_stays_a_member_error_instead_of_raising_from_defaults(tmp_path):
    jobs, dataset = setup_jobs(tmp_path)
    jobs.datasets.run_service.runs.clear()
    jobs.schedule_first_reward_snapshot = lambda *_: pytest.fail("missing run cannot be validated")
    defaults = jobs.defaults(dataset["id"], "stage")
    availability = defaults["reward_availability"]
    assert not availability["ready"] and not availability["pending"]
    assert availability["missing_run_ids"] == ["run"]
    assert "轨迹记录" in availability["errors"][0]["error"]


def test_missing_saved_config_reports_unavailable_without_using_global(tmp_path):
    jobs, dataset = setup_jobs(tmp_path)
    version = full_reward(jobs, dataset)
    # Keep the selected version but simulate an incomplete deployment copy.
    Path(version["config_path"]).rename(Path(version["config_path"]).with_suffix(".missing"))
    defaults = jobs.defaults(dataset["id"], "stage")
    assert defaults["reward_availability"]["origin"] == "dataset"
    assert not defaults["reward_availability"]["ready"]
    assert "配置" in defaults["reward_availability"]["message"]
    assert defaults["reward_version"]["id"] == version["id"]


def test_training_defaults_api_keeps_legacy_data_inspectable(tmp_path, monkeypatch):
    import asyncio
    from fastapi import FastAPI, Request
    from backend.app.api import offline_jobs as api

    async def invoke(function, *args):
        return function(*args)
    monkeypatch.setattr(api, "run_in_threadpool", invoke)

    jobs, dataset = setup_jobs(tmp_path)
    jobs.datasets.update_annotation(dataset["id"], "READY", annotation_id="legacy")
    app = FastAPI()
    app.state.offline_job_service = jobs
    request = Request({"type": "http", "app": app})
    response = asyncio.run(api.defaults(request, dataset["id"], "rynnvalue"))
    assert not response["reward_availability"]["ready"]
    assert "旧评价" in response["reward_availability"]["message"]


@pytest.mark.parametrize("endpoint", ["defaults", "train"])
def test_training_api_does_not_disguise_config_key_errors_as_missing_runs(tmp_path, caplog, endpoint, monkeypatch):
    import asyncio
    from fastapi import FastAPI, HTTPException, Request
    from backend.app.api import offline_jobs as api
    from backend.app.api.models import TrainingRunRequest

    async def invoke(function, *args):
        return function(*args)
    monkeypatch.setattr(api, "run_in_threadpool", invoke)

    jobs, _ = setup_jobs(tmp_path)
    def broken(*_):
        raise KeyError("critic_optimizer")
    jobs.defaults = jobs.start_training = broken
    app = FastAPI()
    app.state.offline_job_service = jobs
    request = Request({"type": "http", "app": app})
    with pytest.raises(HTTPException) as caught:
        asyncio.run(api.defaults(request, None, None) if endpoint == "defaults" else
                    api.train(TrainingRunRequest(dataset_id="dataset", parameters={}), request))
    assert caught.value.status_code == 500
    assert "critic_optimizer" in caught.value.detail
    assert "Run not found" not in caught.value.detail
    assert "KeyError" in caplog.text


def test_legacy_global_stage_and_new_sparse_coexist_and_overwrite_independently(tmp_path):
    result, datasets, _, versions = global_fixture(tmp_path)
    stage = versions[("a", "v1")]
    bind_reward_snapshot(Path(stage["prepared_manifest_path"]), Path(stage["reward_manifest_path"]))
    path = snapshot_path(result["run"], "stage")
    legacy = path.with_name("trajectory_reward.json")
    path.rename(legacy)
    before = legacy.read_bytes()
    sparse_index = json.loads(Path(stage["reward_manifest_path"]).read_text())
    sparse_index["reward_config"] = {"source": "sparse", "gamma": .99, "accumulate_primitive_steps": False}
    sparse = tmp_path / "sparse.json"
    sparse.write_text(json.dumps(sparse_index))
    bind_reward_snapshot(Path(stage["prepared_manifest_path"]), sparse)
    assert read_reward_snapshot(result["run"], "sparse")["metadata"]["source"] == "sparse"
    assert read_reward_snapshot(result["run"], "stage")["metadata"]["source"] == "stage"
    detail = attach_dataset_context(copy.deepcopy(result), datasets, "b", None)
    assert set(detail["reward_evaluations"]) == {"stage", "sparse"}
    assert detail["evaluation_sources"]["stage"]["origin"] == "dataset"
    assert detail["evaluation_sources"]["sparse"]["origin"] == "global"
    bind_reward_snapshot(Path(stage["prepared_manifest_path"]), sparse, overwrite=True)
    assert legacy.read_bytes() == before
