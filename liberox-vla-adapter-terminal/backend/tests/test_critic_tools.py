from __future__ import annotations

import json
import os
import shutil
from pathlib import Path
from types import SimpleNamespace

import pytest
import yaml
from fastapi import FastAPI
from fastapi.testclient import TestClient

from backend.app.api import tools
from backend.app.core.exceptions import ConflictError
from backend.app.services.critic_diagnostic_service import CriticDiagnosticService
from backend.app.services.offline_job_service import OfflineJobService, _sha256, _stable_hash
from backend.app.storage.database import connect
from backend.app.storage.files import atomic_write_json
from test_dataset_reward_versions import finish, setup_jobs
from test_evaluation_api import FakeManager


@pytest.fixture
def diagnostic(tmp_path, monkeypatch):
    jobs, dataset = setup_jobs(tmp_path)
    version = finish(jobs, dataset, jobs.start_annotation(dataset["id"], source="stage", gamma=.92))
    jobs._new_job = OfflineJobService._new_job.__get__(jobs)
    jobs._prepare_launch = OfflineJobService._prepare_launch.__get__(jobs)
    jobs.manager = FakeManager(tmp_path)
    jobs.manager.provider.unload = lambda: pytest.fail("queueing must not load/unload the model")

    def freeze(catalog, task_id, destination):
        destination.mkdir()
        bddl, init = catalog.paths(task_id)
        shutil.copyfile(bddl, destination / "task.bddl")
        shutil.copyfile(init, destination / "initial_states.init")
        return SimpleNamespace(paths=lambda _: (destination / "task.bddl", destination / "initial_states.init"),
                               manifest={"task_revision_sha256": "f" * 64})

    monkeypatch.setattr("backend.app.services.task_registry.freeze_task", freeze)
    service = CriticDiagnosticService(jobs)
    request = {"dataset_id": dataset["id"], "reward_source": "stage", "task_id": dataset["task_id"],
               "policy_id": "base", "max_steps": 80, "open_loop_steps": 4,
               "seed": 7, "init_state_index": 1, "train_steps": 9, "batch_size": 2}
    return service, jobs, dataset, version, request


def test_defaults_only_expose_complete_saved_rewards(diagnostic):
    tool, jobs, dataset, version, _ = diagnostic
    assert tool.defaults() == {"train_steps": 3000, "batch_size": 32, "reward_sources": [],
                               "default_reward_source": None}
    assert tool.defaults(dataset["id"])["reward_sources"] == ["stage"]
    assert tool.defaults(dataset["id"])["default_reward_source"] == "stage"
    jobs.datasets.update_version(dataset["id"], {**version, "id": "bad", "status": "ERROR"})
    assert tool.defaults(dataset["id"])["reward_sources"] == ["stage"]


def test_start_freezes_data_rewards_model_and_task_without_training_registration(diagnostic):
    tool, jobs, dataset, version, request = diagnostic
    paths = [Path(version[key]) for key in ("config_path", "prepared_manifest_path", "reward_manifest_path")]
    before = {path: path.read_bytes() for path in paths}
    job = tool.start(request)
    assert job["kind"] == "critic_diagnostic" and job["status"] == "QUEUED"
    assert job["dataset_id"] == dataset["id"]
    directory = jobs.jobs_root / job["id"]
    raw = yaml.safe_load((directory / "effective_config.yaml").read_text())
    train = yaml.safe_load(Path(raw["training_config"]).read_text())
    sealed = yaml.safe_load(Path(version["config_path"]).read_text())
    assert train["reward"]["gamma"] == .92
    assert train["reward"]["accumulate_primitive_steps"] == sealed["reward"]["accumulate_primitive_steps"]
    assert Path(train["paths"]["annotation_cache"]).is_relative_to(directory)
    assert raw["rollout"]["success_consecutive_steps"] == sealed["data"]["success_consecutive_steps"]
    assert train["reward"]["version_id"] == version["id"]
    assert Path(train["reward"]["manifest_path"]).is_relative_to(directory)
    assert _sha256(Path(train["reward"]["manifest_path"])) == train["reward"]["manifest_sha256"]
    index = json.loads(Path(train["reward"]["manifest_path"]).read_text())
    original_index = json.loads(Path(version["reward_manifest_path"]).read_text())
    entry = index["episodes"][0]
    assert entry["reward_path"] == entry["annotation_path"]
    assert Path(entry["reward_path"]).is_relative_to(directory / "train" / "rewards")
    assert _sha256(Path(entry["reward_path"])) == original_index["episodes"][0]["reward_sha256"]
    for key, value in original_index["episodes"][0].items():
        if not key.endswith("_path"):
            assert entry[key] == value
    assert (directory / "train" / "work" / "dataset_manifest.json").read_bytes() == before[Path(version["prepared_manifest_path"])]
    assert raw["policy_snapshot"]["policy_id"] == request["policy_id"]
    assert len(raw["policy_snapshot"]["content_sha256"]) == 64
    assert train["model"]["contract"] == raw["policy_snapshot"]["model_contract"]
    assert raw["rollout"]["open_loop_steps"] == 4
    assert raw["task_snapshot"]["task_revision_sha256"] == "f" * 64
    assert Path(raw["task_snapshot"]["bddl_path"]).is_relative_to(directory)
    _, persisted = jobs._load_job(job["id"])
    assert [stage["id"] for stage in persisted["stages"]] == ["rollout", "fit"]
    assert all(stage["environment"] == jobs.ui_config.train_environment for stage in persisted["stages"])
    assert all("run_critic_diagnostic.py" in stage["argv"][1] for stage in persisted["stages"])
    with connect(jobs.repository.database_path) as database:
        assert database.execute("SELECT COUNT(*) FROM training_runs").fetchone()[0] == 0
        assert database.execute("SELECT COUNT(*) FROM runs").fetchone()[0] == 0
    assert {path: path.read_bytes() for path in paths} == before
    assert not jobs.training_root.exists()


def test_trained_policy_is_selected_and_frozen_without_parent_fallback(diagnostic, tmp_path):
    from backend.app.policies.catalog import PolicyCatalog
    from test_policy_overlay import BASE_MODELS, STATS, _overlay

    tool, jobs, _, _, request = diagnostic
    manifest = _overlay(tmp_path / "registry")
    jobs.manager.policy_catalog = PolicyCatalog(manifest.parent.parent, base_models=BASE_MODELS,
                                                base_revisions={"base": "a" * 40})
    job = tool.start({**request, "policy_id": "trained"})
    config = yaml.safe_load(Path(job["config_path"]).read_text())
    snapshot = config["policy_snapshot"]
    assert snapshot["policy_id"] == "trained" and snapshot["manifest"] == str(manifest)
    assert snapshot["stats_key"] == STATS
    assert Path(snapshot["action_head"]).is_file()
    raw = yaml.safe_load(Path(config["training_config"]).read_text())
    assert raw["model"]["stats_key"] == STATS
    (manifest.parent / "action_head.pt").write_bytes(b"tampered")
    with pytest.raises(ValueError, match="hash mismatch"):
        tool.start({**request, "policy_id": "trained"})


def test_unverified_model_snapshot_is_rejected(diagnostic, monkeypatch):
    tool, jobs, _, _, request = diagnostic
    original = jobs._evaluation_snapshots
    def invalid(*args):
        task, policy = original(*args)
        return task, {**policy, "content_sha256": None}
    monkeypatch.setattr(jobs, "_evaluation_snapshots", invalid)
    with pytest.raises(ValueError, match="Unverified policy snapshot"):
        tool.start(request)
    assert not list(jobs.jobs_root.glob("critic_*"))


def test_sealed_success_rule_and_data_selection_do_not_follow_live_settings(diagnostic, monkeypatch):
    tool, jobs, dataset, version, request = diagnostic
    preview = jobs.preview_evaluation
    def current_rule(*args):
        value = preview(*args)
        value["config"]["success_streak"] = 99
        return value
    monkeypatch.setattr(jobs, "preview_evaluation", current_rule)
    jobs.datasets.update_training_options(dataset["id"], include_post_success=False)
    job = tool.start(request)
    effective = yaml.safe_load(Path(job["config_path"]).read_text())
    raw = yaml.safe_load(Path(effective["training_config"]).read_text())
    sealed = yaml.safe_load(Path(version["config_path"]).read_text())
    assert effective["rollout"]["success_consecutive_steps"] == sealed["data"]["success_consecutive_steps"]
    assert effective["rollout"]["success_consecutive_steps"] != 99
    assert raw["data"]["include_post_success"] == sealed["data"]["include_post_success"]
    before = Path(raw["data"]["selection_manifest"]).read_bytes()
    jobs.datasets.update_training_options(dataset["id"], include_post_success=True)
    assert Path(raw["data"]["selection_manifest"]).read_bytes() == before


def test_saved_configuration_change_after_validation_is_rejected(diagnostic, monkeypatch):
    tool, jobs, _, version, request = diagnostic
    original = jobs.datasets.validate_version
    def changed(*args):
        value = original(*args)
        Path(value["config_path"]).write_text("data: {}\nreward: {}\n")
        return value
    monkeypatch.setattr(jobs.datasets, "validate_version", changed)
    with pytest.raises(ValueError, match="changed while copying"):
        tool.start(request)
    assert not list(jobs.jobs_root.glob("critic_*"))


@pytest.mark.parametrize("changes,match", [
    ({"task_id": "other"}, "must match"), ({"policy_id": "missing"}, "unknown policy"),
    ({"open_loop_steps": 1000}, "open_loop_steps"),
    ({"init_state_index": 100}, "init_state_indices"),
    ({"reward_source": "sparse"}, "no saved reward"),
])
def test_invalid_selection_does_not_create_job(diagnostic, changes, match):
    tool, jobs, _, _, request = diagnostic
    with pytest.raises(ValueError, match=match):
        tool.start({**request, **changes})
    assert not list(jobs.jobs_root.glob("critic_*"))


def test_source_reward_corruption_stops_before_job(diagnostic):
    tool, jobs, _, version, request = diagnostic
    index = json.loads(Path(version["reward_manifest_path"]).read_text())
    Path(index["episodes"][0]["reward_path"]).write_bytes(b"corrupt")
    with pytest.raises(ValueError, match="integrity"):
        tool.start(request)
    assert not list(jobs.jobs_root.glob("critic_*"))


def test_preflight_failure_cleans_owned_files_only(diagnostic, monkeypatch):
    tool, jobs, _, _, request = diagnostic
    other = jobs.jobs_root / "user-job"
    other.mkdir(parents=True)
    (other / "keep.txt").write_text("keep")
    monkeypatch.setattr(tool, "_copy_rewards", lambda *_: (_ for _ in ()).throw(ValueError("copy failed")))
    with pytest.raises(ValueError, match="copy failed"):
        tool.start(request)
    assert not list(jobs.jobs_root.glob("critic_*"))
    assert (other / "keep.txt").read_text() == "keep"


def test_stage_snapshot_and_optional_provenance_preserved(diagnostic):
    tool, jobs, _, version, request = diagnostic
    manifest_path = Path(version["reward_manifest_path"])
    manifest = json.loads(manifest_path.read_text())
    snapshot = manifest_path.with_name("stage_annotations.json")
    annotations = {"run": {"keyframes": [{"step": 5, "kind": "positive"}]}}
    atomic_write_json(snapshot, annotations)
    manifest.update(stage_annotations_path=str(snapshot), stage_annotations_sha256=_stable_hash(annotations))
    manifest["episodes"][0].update(official_annotation_path="/missing/old-official.npz",
                                     official_annotation_sha256="a" * 64,
                                     saved_reward_config={"source": "stage", "stage_exponent": 2.})
    atomic_write_json(manifest_path, manifest)
    version["reward_manifest_sha256"] = _sha256(manifest_path)
    atomic_write_json(Path(version["work_dir"]).parent / "version.json", version)
    job = tool.start(request)
    index = json.loads((jobs.jobs_root / job["id"] / "train/rewards/reward_manifest.json").read_text())
    assert json.loads(Path(index["stage_annotations_path"]).read_text()) == annotations
    assert index["stage_annotations_sha256"] == _stable_hash(annotations)
    assert index["episodes"][0]["official_annotation_path"] == "/missing/old-official.npz"
    assert index["episodes"][0]["saved_reward_config"] == manifest["episodes"][0]["saved_reward_config"]


def test_diagnostic_uses_shared_fifo_and_resource_lease(diagnostic, monkeypatch):
    tool, jobs, _, _, request = diagnostic
    jobs.manager.provider = None
    first, first_done = jobs.submit_local("simulation", lambda: "done")
    diagnostic_job = tool.start(request)
    launches = []
    def spawn(payload, *, lease=None):
        assert lease is not None
        launches.append(payload["id"])
        _, current = jobs._load_job(payload["id"])
        current.update(status="RUNNING", pid=os.getpid())
        jobs._save_work(current)
    monkeypatch.setattr(jobs, "_spawn_job", spawn)
    jobs._dispatch_training_queue()
    assert first_done.result(3) == "done"
    assert jobs.get(first["id"])["status"] == "COMPLETED"
    jobs._dispatch_training_queue()
    assert launches == [diagnostic_job["id"]]
    jobs._dispatch_training_queue()
    assert launches == [diagnostic_job["id"]]


def test_detail_only_reads_small_metadata_and_whitelisted_artifacts(diagnostic):
    tool, jobs, _, _, request = diagnostic
    job = tool.start(request)
    artifacts = jobs.jobs_root / job["id"] / "artifacts"
    artifacts.mkdir()
    (artifacts / "qv.csv").write_bytes(b"not valid CSV, should never parse")
    (artifacts / "qv.npz").write_bytes(b"not valid NPZ, should never parse")
    atomic_write_json(artifacts / "summary.json", {"count": 3})
    atomic_write_json(artifacts / "progress.json", {"step": 2})
    detail = tool.get(job["id"])
    assert detail["diagnostic_summary"] == {"count": 3}
    assert detail["critic_progress"] == {"step": 2}
    assert "qv.npz" in detail["artifacts"]
    assert "progress.json" not in detail["artifacts"]
    with pytest.raises(ValueError, match="Unknown"):
        tool.artifact(job["id"], "../job.json")
    (artifacts / "qv.png").symlink_to(artifacts / "qv.csv")
    with pytest.raises(FileNotFoundError):
        tool.artifact(job["id"], "qv.png")
    shutil.rmtree(artifacts)
    outside = jobs.project_root / "external-artifacts"
    outside.mkdir()
    (outside / "qv.csv").write_text("private outside file")
    atomic_write_json(outside / "summary.json", {"private": True})
    artifacts.symlink_to(outside, target_is_directory=True)
    with pytest.raises(FileNotFoundError):
        tool.artifact(job["id"], "qv.csv")
    assert "diagnostic_summary" not in tool.get(job["id"])


def test_stop_and_delete_own_job_without_removing_dataset(diagnostic):
    tool, jobs, dataset, _, request = diagnostic
    job = tool.start(request)
    assert tool.stop(job["id"])["status"] == "CANCELED"
    with pytest.raises(ValueError, match="exactly match"):
        tool.delete(job["id"], "wrong")
    assert tool.delete(job["id"], job["id"])["deleted"] == job["id"]
    assert not (jobs.jobs_root / job["id"]).exists()
    assert jobs.datasets.get(dataset["id"])["id"] == dataset["id"]


def test_other_job_kinds_and_active_jobs_are_not_deletable(diagnostic):
    tool, jobs, _, _, request = diagnostic
    job = tool.start(request)
    with pytest.raises(ConflictError, match="finish"):
        tool.delete(job["id"], job["id"])
    other, _ = jobs.submit_local("simulation", lambda: None)
    with pytest.raises(ValueError, match="not an owned"):
        tool.get(other["id"])
    assert len(tool.list()["jobs"]) == 1


def test_api_configuration_routes_and_validation(diagnostic):
    tool, jobs, _, _, request = diagnostic
    app = FastAPI()
    app.state.critic_diagnostic_service = tool
    app.include_router(tools.router)
    with TestClient(app) as client:
        assert client.get("/api/tools/critic/defaults").status_code == 200
        bad = client.post("/api/tools/critic", json={**request, "reward_gamma": .5})
        assert bad.status_code == 422
        assert client.post("/api/tools/critic", json={**request, "train_steps": True}).status_code == 422
        result = client.post("/api/tools/critic", json=request)
        assert result.status_code == 201, result.text
        identifier = result.json()["id"]
        assert client.get("/api/tools/critic").json()["jobs"][0]["id"] == identifier
        detail = client.get(f"/api/tools/critic/{identifier}")
        assert detail.status_code == 200
        config = client.get(f"/api/tools/critic/{identifier}/artifacts/effective_config.yaml")
        assert config.status_code == 200
        assert yaml.safe_load(config.text)["policy_snapshot"]["policy_id"] == "base"
        assert client.get(f"/api/tools/critic/{identifier}/artifacts/job.json").status_code == 422
        assert client.post(f"/api/tools/critic/{identifier}/stop").status_code == 200
        assert client.request("DELETE", f"/api/tools/critic/{identifier}", json={"confirm_job_id": identifier}).status_code == 200
