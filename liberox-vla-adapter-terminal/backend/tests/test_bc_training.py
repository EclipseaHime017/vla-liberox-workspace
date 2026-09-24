from pathlib import Path
import json

import pytest
import yaml

from test_dataset_reward_versions import setup_jobs
from backend.app.services.offline_job_service import OfflineJobService


@pytest.mark.parametrize("queued", [False, True])
def test_bc_can_train_unannotated_dataset_without_reward_binding(tmp_path, queued):
    jobs, dataset = setup_jobs(tmp_path)
    jobs.pinned_reward = lambda *_: pytest.fail("BC must not bind rewards")
    jobs._wake_training_queue = lambda: None
    result = (jobs.enqueue_training if queued else jobs.start_training)(
        dataset["id"], {"algorithm": "bc", "train_steps": 32, "actor_lr_warmup_steps": 4})
    assert [stage["id"] for stage in result["stages"]] == ["prepare", "train"]
    assert result["parameters"]["reward"] is None
    assert result["parameters"]["annotation_id"] is None
    assert result["parameters"]["algorithm"] == "bc"
    raw = yaml.safe_load(result["config_path"].read_text())
    assert raw["training"]["method"] == "bc"
    assert raw["training"]["actor_lr_warmup_steps"] == 4
    assert raw["iql"]["train_steps"] == 32
    assert raw["reward"]["manifest_path"] is None
    assert Path(raw["data"]["selection_manifest"]).is_file()
    assert raw["data"]["selection_manifest"] != str(jobs.datasets.root / dataset["id"] / "dataset.json")


def test_bc_defaults_never_inspect_global_evaluations(tmp_path, monkeypatch):
    jobs, dataset = setup_jobs(tmp_path)
    monkeypatch.setattr("backend.app.services.global_reward_binding.global_members", lambda *_: pytest.fail("reward lookup"))
    defaults = jobs.defaults(dataset["id"], algorithm="bc")
    assert defaults["algorithm"] == "bc"
    assert defaults["reward_version"] is None
    assert defaults["reward_availability"] is None
    assert "actor_lr_warmup_steps" in defaults["basic"]
    assert "critic_warmup_steps" not in defaults["basic"]
    assert not any(key.startswith("reward_") for key in defaults["advanced"])
    with pytest.raises(ValueError, match="does not accept reward"):
        jobs.start_training(dataset["id"], {"algorithm": "bc", "reward_gamma": .9})
    with pytest.raises(ValueError, match="algorithm"):
        jobs.defaults(dataset["id"], algorithm="unknown")


def test_defaults_after_bc_checkpoint_accepts_null_reward(tmp_path):
    jobs, dataset = setup_jobs(tmp_path)
    jobs.available_checkpoints = OfflineJobService.available_checkpoints.__get__(jobs)
    checkpoint = jobs.training_root / "train_bc" / "run" / "checkpoints" / "step_00000002"
    checkpoint.mkdir(parents=True)
    (checkpoint / "checkpoint.json").write_text(json.dumps({"algorithm": "bc", "step": 2}))
    job_dir = jobs.jobs_root / "train_bc"
    job_dir.mkdir(parents=True)
    (job_dir / "job.json").write_text(json.dumps({"id": "train_bc", "kind": "training",
        "dataset_id": dataset["id"], "parameters": {"algorithm": "bc", "reward": None}}))
    assert len(jobs.available_checkpoints()) == 1
    assert jobs.defaults()["algorithm"] == "iql"
    assert jobs.defaults(dataset["id"], algorithm="bc")["algorithm"] == "bc"
