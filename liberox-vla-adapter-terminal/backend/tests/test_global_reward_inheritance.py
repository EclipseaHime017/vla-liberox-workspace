import copy
import hashlib
import json
from pathlib import Path

import numpy as np
import pytest

from backend.app.core.exceptions import ConflictError
from backend.app.services.dataset_evaluation_detail import attach_dataset_context, attach_inherited_rewards
from backend.app.services.inherited_reward_inputs import reconstruct_episode
from vla_rynn_iql.config import load_train_config
from vla_rynn_iql.data import prepare_dataset
from vla_rynn_iql.rewards import load_pinned_reward_index, materialize_reward_manifest, reward_derivation_config
from vla_rynn_iql.stage_rewards import build_stage_annotation
import test_dataset_reward_versions as fixture
from test_training_platform import make_run


def digest(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def setup_recording(tmp_path, monkeypatch):
    run = make_run(tmp_path, "run", kind="branch", source="human", resume_step=3, success=True)
    path = Path(run["trajectory"])
    count = 17
    done = np.arange(count) >= 9
    raw = np.zeros((count, 7), np.float32)
    raw[:, 6] = .5
    np.savez_compressed(path, time_seconds=np.arange(count+1)/20,
        eef_position=np.zeros((count+1, 3)), eef_axis_angle=np.zeros((count+1, 3)),
        gripper_qpos=np.zeros((count+1, 2)), raw_action=raw, env_action=np.zeros((count, 7)),
        done=done, action_source=np.array(["policy"]*3 + ["human"]*(count-3)))
    manifest = Path(run["output_dir"]) / "run.json"
    manifest.write_text(json.dumps(run))
    monkeypatch.setattr(fixture, "make_run", lambda *_: run)
    jobs, dataset = fixture.setup_jobs(tmp_path)
    load_base = jobs._load_base_config
    def test_config():
        raw = load_base()
        raw["data"]["project_id"] = "test"
        return raw
    jobs._load_base_config = test_config
    label = build_stage_annotation(run_id="run", trajectory_sha256=digest(path), done=done,
                                   keyframes=[{"step": 6, "kind": "positive"}])
    path.with_name("stage_annotation.json").write_text(json.dumps(label))
    return jobs, dataset, run


def native_rynn(jobs, dataset, run):
    _, frozen = jobs.datasets._load(dataset["id"])
    episode, _ = reconstruct_episode(jobs, dataset, frozen["members"][0])
    boundaries = episode["reward_boundaries"]
    path = Path(run["trajectory"]).with_name("rynnvalue_evaluation.npz")
    n = len(boundaries)
    np.savez_compressed(path, boundary_steps=np.asarray(boundaries),
        absolute_temporal_distance_seconds=np.arange(n, dtype=float)[::-1, None],
        absolute_value_entropy_nats=np.zeros((n, 1)), absolute_value_logits=np.zeros((n, 1, 256)),
        relative_temporal_distance_seconds=np.zeros(n), relative_value_logits=np.zeros((n, 256)),
        pbrs_shaping_reward=np.ones(n-1), pbrs_chunk_reward=np.arange(n-1, dtype=np.float32))
    raw = jobs._load_base_config()["reward"]
    raw["rynnvalue"] = True
    payload = {"schema_version": 6, "run_id": "run", "values_sha256": digest(path),
        "trajectory_sha256": digest(Path(run["trajectory"])),
        "observations_sha256": frozen["members"][0]["artifacts"]["observations"]["sha256"],
        "reward_config": reward_derivation_config(raw),
        "annotation_config": {key: raw[key] for key in ("model", "revision", "dtype", "max_frames",
                                                       "robot_description", "camera_description")},
        "annotator": {"model": raw["model"], "resolved_revision": raw["revision"]},
        "official_outputs": {"analysis": {"generated_text": "Success: Yes"}}}
    path.with_suffix(".json").write_text(json.dumps(payload))
    return path


@pytest.mark.parametrize("source", ["rynnvalue", "stage", "sparse"])
def test_new_dataset_trains_from_global_inputs_without_any_evaluation_job(tmp_path, monkeypatch, source):
    jobs, dataset, run = setup_recording(tmp_path, monkeypatch)
    values = native_rynn(jobs, dataset, run) if source == "rynnvalue" else None
    protected = {path: path.read_bytes() for path in Path(run["trajectory"]).parent.iterdir() if path.is_file()}
    jobs.start_annotation = jobs.start_trajectory_evaluation = lambda *_args, **_kw: pytest.fail("must not evaluate")
    defaults = jobs.defaults(dataset["id"], source)
    assert defaults["reward_availability"]["ready"], defaults["reward_availability"]
    assert defaults["reward_availability"]["origin"] == "global"
    assert defaults["reward_version"] is None
    assert not jobs.jobs_root.exists()  # Opening training does not materialize a job.
    train = jobs.start_training(dataset["id"], {"reward_source": source})
    config = load_train_config(train["config_path"])
    reward = load_pinned_reward_index(config)
    entry = reward["episodes"][0]
    assert entry["source_origin"] == "global"
    if values:
        assert Path(entry["reward_path"]).read_bytes() == values.read_bytes()
    assert jobs.datasets.get(dataset["id"])["evaluation_origins"] == dict.fromkeys(
        ("sparse", "stage", "rynnvalue", "robometer", "final"), "global")
    assert not jobs.datasets.get(dataset["id"])["evaluation_versions"]
    assert all(path.read_bytes() == data for path, data in protected.items())

    # An independently prepared episode has exactly the inherited indices,
    # including the interrupted prefix and the successful recording's tail.
    reference_raw = copy.deepcopy(config.raw)
    reference_raw["paths"]["work_dir"] = str(tmp_path / "reference")
    reference_raw["reward"].update(manifest_path=None, manifest_sha256=None, version_id=None)
    from vla_rynn_iql.config import LoadedConfig
    reference_config = LoadedConfig(config.path, reference_raw)
    reference_path = prepare_dataset(reference_config).manifest
    reference = json.loads(reference_path.read_text())["episodes"][0]
    inherited = json.loads((Path(config.raw["paths"]["work_dir"])/"dataset_manifest.json").read_text())["episodes"][0]
    for key in ("chunks", "evaluation_chunks", "reward_boundaries", "terminal_step", "recorded_action_count"):
        assert inherited[key] == reference[key]
    assert inherited["chunks"][0]["length"] == 3
    assert inherited["chunks"][-1]["end"] == 17
    if source != "rynnvalue":
        reference_index = json.loads(materialize_reward_manifest(reference_config).read_text())
        with np.load(reference_index["episodes"][0]["reward_path"]) as a, np.load(entry["reward_path"]) as b:
            assert set(a.files) == set(b.files)
            for key in a.files:
                np.testing.assert_array_equal(a[key], b[key])


def test_inheritance_never_decodes_observations_on_training_page(tmp_path, monkeypatch):
    jobs, dataset, run = setup_recording(tmp_path, monkeypatch)
    native_rynn(jobs, dataset, run)
    original = np.load
    def no_observations(file, *args, **kwargs):
        assert "trajectory_observations.npz" not in str(file)
        return original(file, *args, **kwargs)
    monkeypatch.setattr(np, "load", no_observations)
    assert jobs.defaults(dataset["id"], "rynnvalue")["reward_availability"]["ready"]
    assert jobs.defaults(dataset["id"], "stage")["reward_availability"]["ready"]


def test_detail_inherits_stage_labels_and_sparse_without_writing_a_dataset_result(tmp_path, monkeypatch):
    jobs, dataset, run = setup_recording(tmp_path, monkeypatch)
    result = {"run": run, "series": {"time_seconds": (np.arange(18)/20).tolist()}}
    result = attach_dataset_context(result, jobs.datasets, dataset["id"], None)
    result = attach_inherited_rewards(result, jobs, dataset["id"])
    assert set(result["reward_evaluations"]) == {"sparse", "stage"}
    assert result["evaluation_sources"]["stage"]["origin"] == "global"
    assert len(result["reward_evaluations"]["stage"]["stage_scores"]) == 18
    assert not jobs.datasets.get(dataset["id"])["evaluation_versions"]


def test_global_overwrite_affects_future_inheritance_not_existing_training_pin(tmp_path, monkeypatch):
    jobs, dataset, run = setup_recording(tmp_path, monkeypatch)
    values = native_rynn(jobs, dataset, run)
    first = jobs.start_training(dataset["id"], {"reward_source": "rynnvalue"})
    pinned = load_pinned_reward_index(load_train_config(first["config_path"]))
    old_path = Path(pinned["episodes"][0]["reward_path"])
    old_bytes = old_path.read_bytes()
    with np.load(values) as archive:
        arrays = {key: archive[key] for key in archive.files}
    arrays["pbrs_chunk_reward"] += 1
    np.savez_compressed(values, **arrays)
    sidecar = values.with_suffix(".json")
    payload = json.loads(sidecar.read_text())
    payload["values_sha256"] = digest(values)
    sidecar.write_text(json.dumps(payload))
    second = jobs.start_training(dataset["id"], {"reward_source": "rynnvalue"})
    new = load_pinned_reward_index(load_train_config(second["config_path"]))
    assert new["episodes"][0]["reward_sha256"] != pinned["episodes"][0]["reward_sha256"]
    assert old_path.read_bytes() == old_bytes


def test_missing_or_corrupt_global_labels_block_only_the_requested_source(tmp_path, monkeypatch):
    jobs, dataset, run = setup_recording(tmp_path, monkeypatch)
    label = Path(run["trajectory"]).with_name("stage_annotation.json")
    label.write_text("{}")
    assert not jobs.defaults(dataset["id"], "stage")["reward_availability"]["ready"]
    assert jobs.defaults(dataset["id"], "sparse")["reward_availability"]["ready"]
    with pytest.raises(ConflictError):
        jobs.start_training(dataset["id"], {"reward_source": "stage"})


def test_dataset_reevaluation_overrides_only_stage_while_sibling_inherits_globals(tmp_path, monkeypatch):
    from backend.app.services.stage_annotation_service import StageAnnotationService
    from backend.app.workers.finalize_reward_version import seal

    jobs, dataset, run = setup_recording(tmp_path, monkeypatch)
    native = native_rynn(jobs, dataset, run)
    before = native.read_bytes()
    sibling = jobs.datasets.derive(dataset["id"], name="sibling",
                                   selection={"mode": "manual", "run_ids": ["run"]})
    jobs.stage_annotations = StageAnnotationService(jobs.datasets.run_service, jobs.ui_config.offline_rl_root)
    job = jobs.start_annotation(dataset["id"], source="stage", stage_exponent=4)
    config = load_train_config(job["config_path"])
    prepare_dataset(config)
    materialize_reward_manifest(config)
    version = seal(job["output_path"] / "version.json")
    jobs.datasets.update_version(dataset["id"], version)
    jobs._bind_completed_result({"kind": "annotation", "dataset_id": dataset["id"],
                                "id": version["id"], "parameters": {}})
    local = jobs.defaults(dataset["id"], "stage")
    assert local["reward_availability"]["origin"] == "dataset"
    assert local["advanced"]["reward_stage_exponent"] == 4
    assert jobs.defaults(dataset["id"], "rynnvalue")["reward_availability"]["origin"] == "global"
    inherited = jobs.defaults(sibling["id"], "stage")
    assert inherited["reward_availability"]["origin"] == "global"
    assert inherited["advanced"]["reward_stage_exponent"] == 2
    assert native.read_bytes() == before


def test_corrupt_global_rynn_does_not_evaluate_or_switch_reward_source(tmp_path, monkeypatch):
    jobs, dataset, run = setup_recording(tmp_path, monkeypatch)
    values = native_rynn(jobs, dataset, run)
    values.write_bytes(b"corrupt")
    defaults = jobs.defaults(dataset["id"], "rynnvalue")
    assert not defaults["reward_availability"]["ready"]
    assert defaults["reward_availability"]["missing_run_ids"] == ["run"]
    assert defaults["advanced"]["reward_source"] == "rynnvalue"
    with pytest.raises(ConflictError):
        jobs.start_training(dataset["id"], {"reward_source": "rynnvalue"})
    assert not jobs.jobs_root.exists()
