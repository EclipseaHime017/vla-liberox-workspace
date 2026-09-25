from pathlib import Path
import json

import pytest
import yaml

from test_dataset_reward_versions import setup_jobs
from test_dataset_reward_versions import finish
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
    assert raw["training"]["train_steps"] == 32
    assert raw["reward"] == {} and raw["iql"] == {}
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
    with pytest.raises(ValueError, match="IQL parameters"):
        jobs.start_training(dataset["id"], {"algorithm": "bc", "critic_lr": .01})


def test_defaults_and_launch_use_independent_training_entries(tmp_path, monkeypatch):
    from backend.app.services.inherited_reward_inputs import offline_module
    jobs, dataset = setup_jobs(tmp_path)
    sources = offline_module(jobs.ui_config.offline_rl_root, "config_sources")
    original = sources._source_layers
    reads = []

    def entries(path, *args, **kwargs):
        layers = original(path, *args, **kwargs)
        # Traversal selects the sibling entry before reading its dependencies.
        selected = kwargs.get("method") or path.stem
        reads.append(selected)
        result = layers[-1]
        if path.parent.name == "training" and selected == "bc":
            result["training"].update(actor_lr_warmup_steps=11, policy_peak_lr=.0002)
        elif path.parent.name == "training" and selected == "iql":
            result["training"].update(actor_lr_warmup_steps=22, policy_peak_lr=.0004)
            result["iql"].update(critic_warmup_steps=33, critic_lr=.0008)
        return layers

    monkeypatch.setattr(sources, "_source_layers", entries)
    bc = jobs.defaults(algorithm="bc")
    assert bc["basic"]["actor_lr_warmup_steps"] == 11
    assert bc["advanced"]["policy_peak_lr"] == .0002
    assert "iql" not in reads
    iql = jobs.defaults(algorithm="iql")
    assert iql["basic"]["actor_lr_warmup_steps"] == 22
    assert iql["basic"]["critic_warmup_steps"] == 33
    assert iql["advanced"]["policy_peak_lr"] == .0004
    assert iql["advanced"]["critic_lr"] == .0008
    job = jobs.start_training(dataset["id"], {"algorithm": "bc"})
    raw = yaml.safe_load(job["config_path"].read_text())
    assert raw["training"]["policy_peak_lr"] == .0002
    assert raw["training"]["actor_lr_warmup_steps"] == 11
    assert not raw["iql"] and not raw["reward"]


def test_sealed_reward_does_not_override_selected_model(tmp_path, monkeypatch):
    from backend.app.services.inherited_reward_inputs import offline_module
    jobs, dataset = setup_jobs(tmp_path)
    version = finish(jobs, dataset, jobs.start_annotation(dataset["id"], source="sparse"))
    sealed = Path(version["config_path"]).read_bytes()
    sources = offline_module(jobs.ui_config.offline_rl_root, "config_sources")
    original = sources.read_source

    def presets(path):
        value = original(path)
        if path.name == "vla_adapter.yaml":
            value["model"]["base_checkpoint"] = "/new/base/checkpoint"
        return value

    monkeypatch.setattr(sources, "read_source", presets)
    job = jobs.start_training(dataset["id"], {"model_family": "vla_adapter"})
    raw = yaml.safe_load(job["config_path"].read_text())
    assert raw["model"]["base_checkpoint"] == "/new/base/checkpoint"
    assert raw["reward"]["version_id"] == version["id"]
    assert Path(version["config_path"]).read_bytes() == sealed


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


@pytest.mark.parametrize("algorithm", ["iql", "bc"])
def test_model_selection_is_pinned_independently_of_algorithm_and_rewards(tmp_path, algorithm):
    jobs, dataset = setup_jobs(tmp_path)
    if algorithm == "iql":
        version = finish(jobs, dataset, jobs.start_annotation(dataset["id"], source="sparse"))
        before = Path(version["config_path"]).read_bytes()
    defaults = jobs.defaults(algorithm=algorithm)
    assert defaults["models"][0]["id"] == "vla_adapter"
    assert defaults["model"]["model_backbone"] == "frozen"
    assert "freeze_backbone" not in defaults["fixed"]
    result = jobs.start_training(dataset["id"], {
        "algorithm": algorithm, "model_family": "vla_adapter", "model_backbone": "lora",
        "model_action_head": "frozen", "model_proprio_projector": "frozen",
        "model_lora_rank": 8, "model_lora_alpha": 16, "model_lora_dropout": .1,
    })
    raw = yaml.safe_load(result["config_path"].read_text())
    assert raw["model"] == result["parameters"]["model"] == {
        "base_checkpoint": "VLA-Adapter/LIBERO-Object-Pro", "stats_key": "libero_object", "use_pro_version": True,
        "family": "vla_adapter", "backbone": "lora", "action_head": "frozen",
        "proprio_projector": "frozen", "lora": {"rank": 8, "alpha": 16, "dropout": .1}}
    assert "vla" not in raw
    if algorithm == "iql":
        assert Path(version["config_path"]).read_bytes() == before


@pytest.mark.parametrize("parameters", [
    {"model_family": "fake"}, {"model_backbone": "invalid"}, {"model_lora_rank": 0},
    {"model_action_head": "frozen", "model_proprio_projector": "frozen"},
])
def test_invalid_model_config_rejected_before_creating_jobs(tmp_path, parameters):
    jobs, dataset = setup_jobs(tmp_path)
    with pytest.raises(ValueError):
        jobs.start_training(dataset["id"], {"algorithm": "bc", **parameters})
    assert not jobs.jobs_root.exists()
