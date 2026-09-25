from __future__ import annotations

import asyncio
import json
from pathlib import Path

import pytest
import yaml
from pydantic import ValidationError

from backend.app.api.models import DatasetTrainingOptionsRequest
from backend.app.services.offline_job_service import _training_replay_counts
from test_dataset_reward_versions import setup_jobs, finish
from test_stage_training_jobs import _full_recording_episode


def test_setting_updates_same_dataset_without_rewriting_identity_or_evaluation(tmp_path):
    jobs, dataset = setup_jobs(tmp_path)
    version = finish(jobs, dataset, jobs.start_annotation(dataset["id"], source="sparse"))
    before = jobs.datasets.get(dataset["id"])
    paths = [Path(version[key]) for key in ("config_path", "prepared_manifest_path", "reward_manifest_path")]
    values = {p: p.read_bytes() for p in paths}
    after = jobs.datasets.update_training_options(dataset["id"], include_post_success=False)
    assert after["include_post_success"] is False
    for key in ("id", "members", "dataset_sha256", "reward_version_id", "evaluation_versions"):
        assert after[key] == before[key]
    assert jobs.datasets.verify(dataset["id"])["integrity_status"] == "HEALTHY"
    assert {p: p.read_bytes() for p in paths} == values
    jobs.datasets.update_training_options(dataset["id"], include_post_success=True)
    assert jobs.datasets.get(dataset["id"])["include_post_success"] is True


def test_old_datasets_default_to_full_recording_and_invalid_values_are_rejected(tmp_path):
    jobs, dataset = setup_jobs(tmp_path)
    path = jobs.datasets.root / dataset["id"] / "dataset.json"
    value = json.loads(path.read_text())
    value.pop("include_post_success")
    path.write_text(json.dumps(value))
    assert jobs.datasets.get(dataset["id"])["include_post_success"] is True
    for invalid in (0, 1, "false", None):
        with pytest.raises((TypeError, ValidationError)):
            DatasetTrainingOptionsRequest(include_post_success=invalid)
        with pytest.raises(TypeError):
            jobs.datasets.update_training_options(dataset["id"], include_post_success=invalid)


@pytest.mark.parametrize("algorithm", ["iql", "bc"])
@pytest.mark.parametrize("queued", [False, True])
def test_future_jobs_snapshot_option_without_changing_queued_or_running_jobs(tmp_path, algorithm, queued):
    jobs, dataset = setup_jobs(tmp_path)
    version = None
    if algorithm == "iql":
        episode = _full_recording_episode("run", [(0, 5, False), (5, 13, False), (13, 17, False)], terminal_step=4)
        version = finish(jobs, dataset, jobs.start_annotation(dataset["id"], source="sparse"),
                         prepared_episodes=[episode])
    jobs.datasets.update_training_options(dataset["id"], include_post_success=False)
    jobs._wake_training_queue = lambda: None
    launch = jobs.enqueue_training if queued else jobs.start_training
    result = launch(dataset["id"], {"algorithm": algorithm})
    path = Path(result["config_path"])
    before = path.read_bytes()
    config = yaml.safe_load(before)
    assert config["data"]["include_post_success"] is False
    assert result["parameters"]["include_post_success"] is False
    assert result["parameters"]["replay_policy"] == "confirmed_success_v1"
    if version:
        assert result["parameters"]["action_count"] == 5
        assert result["parameters"]["chunk_count"] == 1
        assert config["reward"]["version_id"] == version["id"]
    jobs.datasets.update_training_options(dataset["id"], include_post_success=True)
    assert path.read_bytes() == before
    next_job = launch(dataset["id"], {"algorithm": algorithm})
    assert yaml.safe_load(Path(next_job["config_path"]).read_text())["data"]["include_post_success"] is True


def test_truncated_counts_keep_confirmation_and_deduplicate_prefixes():
    episodes = [
        _full_recording_episode("run", [(0, 8, False), (8, 10, False), (10, 17, False)], terminal_step=9),
        _full_recording_episode("branch", [(0, 8, True), (8, 10, False), (10, 18, False)], terminal_step=9),
    ]
    assert _training_replay_counts({"episodes": episodes}, False) == {"action_count": 12, "chunk_count": 3}


def test_verify_does_not_overwrite_concurrent_training_setting(tmp_path, monkeypatch):
    from backend.app.services import training_dataset_service as module
    jobs, dataset = setup_jobs(tmp_path)
    original = module._sha256
    changed = False

    def hash_and_update(path):
        nonlocal changed
        if not changed:
            changed = True
            jobs.datasets.update_training_options(dataset["id"], include_post_success=False)
        return original(path)

    monkeypatch.setattr(module, "_sha256", hash_and_update)
    assert jobs.datasets.verify(dataset["id"])["include_post_success"] is False


def test_training_options_api_delegates_only_metadata_update(tmp_path, monkeypatch):
    from backend.app.api import training_datasets as api
    jobs, dataset = setup_jobs(tmp_path)
    monkeypatch.setattr(api, "training_dataset_service", lambda _: jobs.datasets)

    async def direct(function, *args, **kwargs):
        return function(*args, **kwargs)

    monkeypatch.setattr(api, "run_in_threadpool", direct)
    result = asyncio.run(api.training_options(dataset["id"],
        DatasetTrainingOptionsRequest(include_post_success=False), None))
    assert result["include_post_success"] is False
