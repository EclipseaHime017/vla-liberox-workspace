import copy
import json
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest
import yaml

from backend.app.core.exceptions import ConflictError
from backend.app.services.dataset_evaluation_detail import _reward_arrays_detail
from test_dataset_reward_versions import setup_jobs, finish


def test_final_only_does_not_load_models_and_locks_semantics(tmp_path):
    jobs, dataset = setup_jobs(tmp_path)
    jobs._prepare_launch = lambda: pytest.fail("GPU reservation")
    job = jobs.start_annotation(dataset["id"], source="final", alpha=.5, shaping_weight=0., stage_exponent=4.)
    assert [step["id"] for step in job["stages"]] == ["prepare", "rewards", "seal"]
    raw = yaml.safe_load(job["config_path"].read_text())
    assert raw["reward"]["source"] == "final"
    assert raw["reward"]["final_normalization"] == "initial_chunk_v1"
    assert raw["reward"]["alpha"] == .5
    assert raw["reward"]["rynnvalue"] is False
    version = finish(jobs, dataset, job)
    settings = jobs.defaults(dataset["id"], "final")
    assert settings["reward_availability"]["ready"]
    assert settings["advanced"]["reward_alpha"] == .5
    _, selected = jobs.pinned_reward(jobs.datasets.get(dataset["id"]), {
        "reward_source": "final", "reward_gamma": .92, "reward_accumulate_primitive_steps": True})
    assert selected["reward_gamma"] == .92
    assert selected["reward_version_id"] == version["id"]
    for key, value in {"reward_alpha": .2, "reward_stage_exponent": 2., "reward_shaping_weight": .1,
                       "reward_fusion_mode": "multiplicative"}.items():
        with pytest.raises(ConflictError, match="locked"):
            jobs.pinned_reward(jobs.datasets.get(dataset["id"]), {"reward_source": "final", key: value})


def test_sparse_final_needs_no_labels_and_force_model_rejected(tmp_path):
    jobs, dataset = setup_jobs(tmp_path)
    jobs.stage_annotations.validate_members = lambda *_: pytest.fail("Stage lookup")
    job = jobs.start_annotation(dataset["id"], source="final", alpha=0., shaping_weight=0.)
    assert not (job["output_path"] / "stage_annotations.json").exists()
    with pytest.raises(ValueError, match="does not run a model"):
        jobs.start_annotation(dataset["id"], source="final", force_model=True)


@pytest.mark.parametrize("legacy", [False, True])
def test_training_preserves_saved_normalization_not_base_defaults(tmp_path, legacy):
    jobs, dataset = setup_jobs(tmp_path)
    job = jobs.start_annotation(dataset["id"], source="final", alpha=0., shaping_weight=0.)
    if legacy:
        saved = yaml.safe_load(job["config_path"].read_text())
        saved["reward"].pop("final_normalization")
        job["config_path"].write_text(yaml.safe_dump(saved))
    finish(jobs, dataset, job)
    train = jobs.start_training(dataset["id"], {"reward_gamma": .92})
    effective = yaml.safe_load(train["config_path"].read_text())
    assert effective["reward"]["final_normalization"] == ("none" if legacy else "initial_chunk_v1")
    assert effective["reward"]["gamma"] == .92


def test_new_final_normalizes_even_when_base_yaml_has_no_rule(tmp_path):
    jobs, dataset = setup_jobs(tmp_path)
    effective_config = jobs._effective_config

    def legacy_base(value):
        raw = effective_config(value)
        raw["reward"].pop("final_normalization", None)
        return raw

    jobs._effective_config = legacy_base
    job = jobs.start_annotation(dataset["id"], source="final", alpha=0., shaping_weight=0.)
    assert yaml.safe_load(job["config_path"].read_text())["reward"]["final_normalization"] == "initial_chunk_v1"


def test_multiplication_normalizes_evaluation_and_training_to_macro(tmp_path):
    jobs, dataset = setup_jobs(tmp_path)
    job = jobs.start_annotation(dataset["id"], source="final", fusion_mode="multiplicative",
                                shaping_weight=0., accumulate_primitive_steps=True)
    assert job["parameters"]["accumulate_primitive_steps"] is False
    assert yaml.safe_load(job["config_path"].read_text())["reward"]["accumulate_primitive_steps"] is False
    finish(jobs, dataset, job)
    defaults = jobs.defaults(dataset["id"], "final")
    assert defaults["advanced"]["reward_accumulate_primitive_steps"] is False
    assert "reward_accumulate_primitive_steps" not in defaults["reward_editable_parameters"]
    _, parameters = jobs.pinned_reward(jobs.datasets.get(dataset["id"]),
        {"reward_source": "final", "reward_gamma": .92, "reward_accumulate_primitive_steps": True})
    assert parameters["reward_accumulate_primitive_steps"] is False
    assert parameters["reward_gamma"] == .92


def test_mixed_global_defaults_are_macro_even_when_first_member_is_additive(tmp_path, monkeypatch):
    jobs, dataset = setup_jobs(tmp_path)
    records = [(None, {"reward_config": {"source": "final", "fusion_mode": mode,
                 "accumulate_primitive_steps": True}}, None, None, None)
               for mode in ("additive", "multiplicative")]
    monkeypatch.setattr("backend.app.services.global_reward_binding.global_members", lambda *_: (records, []))
    defaults = jobs.defaults(dataset["id"], "final")
    assert defaults["reward_availability"]["ready"]
    assert defaults["advanced"]["reward_accumulate_primitive_steps"] is False
    assert defaults["reward_editable_parameters"] == ["reward_gamma"]


def test_all_runs_serial_models_then_final_and_publishes_atomically(tmp_path):
    jobs, dataset = setup_jobs(tmp_path)
    old = finish(jobs, dataset, jobs.start_annotation(dataset["id"], source="sparse"))
    # A tiny official-checkout placeholder: the job is planned, never executed.
    root = tmp_path / "robometer-project"
    (root / "configs").mkdir(parents=True)
    checkout = tmp_path / "official" / "robometer"
    checkout.mkdir(parents=True)
    (checkout / "__init__.py").touch()
    settings = yaml.safe_load((jobs.ui_config.robometer_root / "configs" / "robometer_evaluation.yaml").read_text())
    settings["paths"]["robometer_root"] = str(checkout.parent)
    (root / "configs" / "robometer_evaluation.yaml").write_text(yaml.safe_dump(settings))
    jobs.ui_config.robometer_root = root
    job = jobs.start_annotation(dataset["id"], source="all", alpha=.5, batch_size=2, robometer_batch_size=3)
    assert yaml.safe_load(job["config_path"].read_text())["reward"]["final_normalization"] == "initial_chunk_v1"
    ids = job["parameters"]["related_version_ids"]
    assert len(ids) == 3 and ids[-1] == job["config_path"].parent.name
    stages = [s["id"] for s in job["stages"]]
    assert stages.index("rynnvalue_annotate") < stages.index("robometer_robometer") < stages.index("final_rewards")
    for step in job["stages"]:
        if step["id"] in {"rynnvalue_annotate", "robometer_robometer"}:
            assert "--overwrite" in step["argv"]
        if step["id"] == "final_rewards":
            assert "--annotation-manifest" in step["argv"]
    versions = [jobs.datasets.get_version(dataset["id"], identifier) for identifier in ids]
    first_ready = {**versions[0], "status": "READY", "complete": True}
    with pytest.raises(ValueError, match="All evaluation"):
        jobs.datasets.update_versions(dataset["id"], [first_ready, *versions[1:]])
    assert jobs.datasets.get(dataset["id"])["reward_version_id"] == old["id"]
    # The first child may be sealed on disk while the second model fails.
    # Reconciliation (including a backend restart) must not publish that child.
    sealed_path = jobs.datasets.root / dataset["id"] / "annotations" / ids[0] / "version.json"
    sealed_path.write_text(json.dumps(first_ready))
    failed = {"id": ids[-1], "kind": "annotation", "dataset_id": dataset["id"],
              "created_at": "2026-09-16T00:00:00+00:00", "completed_at": "2026-09-16T00:01:00+00:00",
              "status": "FAILED", "error": "second model failed", "parameters": job["parameters"]}
    jobs._load_job = lambda _: (tmp_path / "job.json", failed)
    jobs.repository = SimpleNamespace(upsert=lambda *_: None)
    jobs._public_job = lambda value: value
    jobs._schedule_result_binding = lambda _: None
    for _ in range(2):
        assert jobs._reconcile(ids[-1])["status"] == "FAILED"
        assert all(jobs.datasets.get_version(dataset["id"], identifier)["status"] == "ERROR" for identifier in ids)
    assert jobs.datasets.get(dataset["id"])["reward_version_id"] == old["id"]
    published = jobs.datasets.update_versions(dataset["id"], [{**v, "status": "READY", "complete": True} for v in versions])
    assert published["evaluation_version_ids"]["rynnvalue"] == ids[0]
    assert published["robometer_version_id"] == ids[1]
    assert published["reward_version_id"] == ids[2]


def test_final_prefers_dataset_rynn_and_rejects_corruption_instead_of_global_fallback(tmp_path, monkeypatch):
    jobs, dataset = setup_jobs(tmp_path)
    job = jobs.start_annotation(dataset["id"], source="rynnvalue", max_frames=32)
    version = finish(jobs, dataset, job)
    current = jobs.datasets.get(dataset["id"])
    monkeypatch.setattr("backend.app.services.global_reward_binding.global_members",
                        lambda *_args, **_kwargs: pytest.fail("must not fall back to global outputs"))
    references = jobs._final_model_inputs(current)
    index = json.loads(Path(version["reward_manifest_path"]).read_text())
    assert references["episodes"][0]["annotation_path"] == index["episodes"][0]["reward_path"]
    Path(index["episodes"][0]["reward_path"]).write_bytes(b"broken")
    with pytest.raises(ValueError, match="integrity failed"):
        jobs._final_model_inputs(current)


def test_final_detail_exposes_independent_original_and_fused_curves():
    arrays = dict(boundary_steps=np.array([0, 2]), final_reward=np.array([-1.]),
                  raw_final_reward=np.array([-.4]), final_reward_reference=np.array(-.4),
                  final_reward_scale=np.array(2.5),
                  original_final_reward=np.array([-.8]), sparse_reward=np.array([-1.]),
                  pbrs_shaping_reward=np.array([2.]), dense_reward=np.array([.2]),
                  stage_score=np.array([-1., -.7, -.5]), time_seconds=np.array([0., .05, .1]))
    saved = copy.deepcopy(arrays)
    result, model = _reward_arrays_detail({"id": "final", "evaluator": "final"},
        {"reward_config": {"source": "final", "alpha": .5}}, {}, arrays, [0., .05, .1])
    assert result["final_reward"] == [-1.]
    assert result["raw_final_reward"] == [-.4]
    assert result["final_reward_reference"] == -.4
    assert result["final_reward_scale"] == 2.5
    assert result["original_final_reward"] == [-.8]
    assert result["stage_scores"] == [-1., -.7, -.5]
    assert model is None  # A fusion is not a new RynnValue model evaluation.
    for key in arrays:
        np.testing.assert_array_equal(arrays[key], saved[key])
