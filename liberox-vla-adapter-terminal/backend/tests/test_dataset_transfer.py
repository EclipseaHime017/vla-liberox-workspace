import json
from pathlib import Path

import numpy as np
import pytest
import yaml

from backend.app.services.dataset_transfer import export_dataset
from backend.app.services.training_dataset_service import TrainingDatasetService
from backend.app.services.stage_annotation_service import StageAnnotationService
from backend.app.workers.finalize_reward_version import seal
from test_global_reward_inheritance import setup_recording, native_rynn, digest
from test_training_platform import make_run
from vla_rynn_iql.config import load_train_config
from vla_rynn_iql.data import prepare_dataset
from vla_rynn_iql.rewards import materialize_reward_manifest


def export(jobs, dataset, destination):
    config = jobs.datasets.root.parent / "export-test.yaml"
    config.write_text(yaml.safe_dump(jobs._load_base_config()))
    return export_dataset(jobs.datasets.root.parent, dataset["id"], destination,
                          config, jobs.ui_config.offline_rl_root)


def local_stage(jobs, dataset, exponent=4):
    jobs.stage_annotations = StageAnnotationService(jobs.datasets.run_service, jobs.ui_config.offline_rl_root)
    job = jobs.start_annotation(dataset["id"], source="stage", stage_exponent=exponent)
    config = load_train_config(job["config_path"])
    prepare_dataset(config)
    materialize_reward_manifest(config)
    version = seal(job["output_path"] / "version.json")
    jobs.datasets.update_version(dataset["id"], version)
    return version


def robometer(run):
    episode = Path(run["trajectory"]).parent
    values = episode / "robometer_evaluation.npz"
    np.savez_compressed(values, observation_steps=np.array([0, 17]),
        time_seconds=np.array([0., .85]), progress_pred=np.array([.1, .9]), success_probs=np.array([.2, .8]))
    from backend.app.services.robometer_evaluation_service import SCHEMA_VERSION
    metadata = {"schema_version": SCHEMA_VERSION, "run_id": run["id"], "values_file": values.name,
        "values_sha256": digest(values), "sample_count": 2,
        "trajectory_sha256": digest(Path(run["trajectory"])),
        "observations_sha256": digest(episode / "trajectory_observations.npz"),
        "manifest_sha256": digest(Path(run["output_dir"]) / "run.json"),
        "annotator": {"model": "locked-model", "revision": "locked-revision"}}
    values.with_suffix(".json").write_text(json.dumps(metadata))


def test_full_members_and_all_evaluations_are_portable_readonly(tmp_path, tmp_path_factory, monkeypatch):
    jobs, dataset, run = setup_recording(tmp_path, monkeypatch)
    native = native_rynn(jobs, dataset, run)
    robometer(run)
    version = local_stage(jobs, dataset)
    episode = Path(run["trajectory"]).parent
    (episode / "agentview.mp4").write_bytes(b"video")
    (episode / "factr_samples.csv").write_text("step,joint\n0,1\n")
    make_run(tmp_path, "unselected")
    protected = {p: digest(p) for p in Path(run["output_dir"]).rglob("*") if p.is_file()}
    destination = tmp_path_factory.mktemp("exports") / "dataset"
    export(jobs, dataset, destination)
    index = json.loads((destination / "export.json").read_text())
    assert index["complete"]
    assert [r["run_id"] for r in index["members"]] == ["run"]
    assert len(list((destination / "runs").rglob("run.json"))) == 1
    copied = next((destination / "runs").rglob("trajectory.npz")).parent
    assert (copied / "rynnvalue_evaluation.npz").read_bytes() == native.read_bytes()
    assert (copied / "agentview.mp4").read_bytes() == b"video"
    assert (copied / "factr_samples.csv").read_text() == "step,joint\n0,1\n"
    assert (copied / "stage_annotation.json").read_bytes() == (episode / "stage_annotation.json").read_bytes()
    stage = json.loads((copied / "trajectory_reward.stage.json").read_text())
    assert stage["reward_config"]["stage_exponent"] == 4
    assert stage["origin"] == "dataset" and stage["evaluation_id"] == version["id"]
    assert stage["stage_annotations_snapshot"]["annotations"]["run"]["keyframes"]
    assert index["members"][0]["evaluations"]["final"]["status"] == "NOT_EVALUATED"
    assert protected == {p: digest(p) for p in Path(run["output_dir"]).rglob("*") if p.is_file()}
    for relative, item in index["files"].items():
        assert digest(destination / relative) == item["sha256"]
    assert TrainingDatasetService.classify({**run, "trajectory": str(copied / "trajectory.npz"),
        "output_dir": str(copied.parent.parent)})[0] == "manual"
    Path(run["output_dir"]).rename(tmp_path / "offline")
    for source in ("rynnvalue", "stage"):
        metadata = json.loads((copied / f"trajectory_reward.{source}.json").read_text())
        assert digest(copied / metadata["values_file"]) == metadata["values_sha256"]
        assert digest(copied / "trajectory.npz") == metadata["trajectory_sha256"]


def test_legacy_global_in_dataset_directory_is_materialized(tmp_path, tmp_path_factory, monkeypatch):
    jobs, dataset, run = setup_recording(tmp_path, monkeypatch)
    original = local_stage(jobs, dataset)
    sibling = jobs.datasets.derive(dataset["id"], name="sibling",
        selection={"mode": "manual", "run_ids": ["run"]})
    # With a current label, implicit global Stage takes precedence over history.
    labels = Path(run["trajectory"]).with_name("stage_annotation.json")
    labeled_export = tmp_path_factory.mktemp("exports") / "labels"
    export(jobs, sibling, labeled_export)
    assert not list(labeled_export.rglob("trajectory_reward.stage.json"))
    labels.unlink()
    destination = tmp_path_factory.mktemp("exports") / "sibling"
    export(jobs, sibling, destination)
    metadata = json.loads(next(destination.rglob("trajectory_reward.stage.json")).read_text())
    assert metadata["origin"] == "global" and metadata["evaluation_id"] == original["id"]
    assert metadata["reward_config"]["stage_exponent"] == 4


class FakeAnnotator:
    metadata = {"provider": "fake", "revision": "fixed"}

    def predict(self, prompt, frames):
        count = len(frames)
        return {"absolute_temporal_distance_seconds": np.ones((count, 1)),
            "absolute_value_entropy_nats": np.zeros((count, 1)),
            "absolute_value_logits": np.zeros((count, 1, 256)),
            "relative_temporal_distance_seconds": np.zeros(count),
            "relative_value_logits": np.zeros((count, 256))}

    def analyze(self, prompt, frames):
        return {"generated_text": "- Match: Yes\n- Success: No", "generated_token_ids": [1, 2]}


def test_sealed_rynn_and_final_keep_independent_inputs(tmp_path, tmp_path_factory, monkeypatch):
    from vla_rynn_iql.rewards import annotate_manifest
    jobs, dataset, run = setup_recording(tmp_path, monkeypatch)
    local_stage(jobs, dataset, exponent=4)
    for source, options in (("rynnvalue", {}), ("final", {"alpha": .5, "stage_exponent": 2, "shaping_weight": 0})):
        job = jobs.start_annotation(dataset["id"], source=source, **options)
        config = load_train_config(job["config_path"])
        prepare_dataset(config)
        if source == "rynnvalue":
            annotate_manifest(config, FakeAnnotator())
        materialize_reward_manifest(config)
        version = seal(job["output_path"] / "version.json")
        jobs.datasets.update_version(dataset["id"], version)
    destination = tmp_path_factory.mktemp("exports") / "all"
    export(jobs, dataset, destination)
    episode = next(destination.rglob("trajectory.npz")).parent
    rynn = json.loads((episode / "trajectory_reward.rynnvalue.json").read_text())
    final = json.loads((episode / "trajectory_reward.final.json").read_text())
    stage = json.loads((episode / "trajectory_reward.stage.json").read_text())
    assert "Success: No" in str(rynn["official_outputs"])
    assert final["reward_config"]["source"] == "final"
    assert final["reward_config"]["stage_exponent"] == 2
    assert stage["reward_config"]["stage_exponent"] == 4
    assert (episode / rynn["entry"]["official_annotation_path"]).is_file()


def test_global_snapshot_outlives_original_annotation_directory(tmp_path, tmp_path_factory, monkeypatch):
    from backend.app.services.trajectory_reward_snapshot import bind_reward_snapshot
    from vla_rynn_iql.rewards import annotate_manifest
    jobs, dataset, run = setup_recording(tmp_path, monkeypatch)
    job = jobs.start_annotation(dataset["id"], source="rynnvalue")
    config = load_train_config(job["config_path"])
    prepare_dataset(config)
    annotate_manifest(config, FakeAnnotator())
    materialize_reward_manifest(config)
    version = seal(job["output_path"] / "version.json")
    bind_reward_snapshot(Path(version["prepared_manifest_path"]), Path(version["reward_manifest_path"]))
    # Keep global snapshot but remove the job directories it no longer depends on.
    job["output_path"].rename(tmp_path / "removed-annotation")
    destination = tmp_path_factory.mktemp("exports") / "global"
    export(jobs, dataset, destination)
    metadata = json.loads(next(destination.rglob("trajectory_reward.rynnvalue.json")).read_text())
    assert metadata["origin"] == "global"
    assert "official_annotation_path" not in metadata["entry"]


def test_dataset_robometer_overrides_only_its_own_exported_sidecar(tmp_path, tmp_path_factory, monkeypatch):
    jobs, dataset, run = setup_recording(tmp_path, monkeypatch)
    native_rynn(jobs, dataset, run)
    robometer(run)
    episode = Path(run["trajectory"]).parent
    old_bytes = (episode / "robometer_evaluation.npz").read_bytes()
    metadata = json.loads((episode / "robometer_evaluation.json").read_text())
    directory = jobs.datasets.root / dataset["id"] / "annotations/robo"
    work = directory / "work/robometer"
    work.mkdir(parents=True)
    values = work / "outputs.npz"
    with np.load(episode / "robometer_evaluation.npz") as archive:
        arrays = {k: archive[k] for k in archive.files}
    arrays["progress_pred"] = np.array([.2, .6])
    np.savez_compressed(values, **arrays)
    manifest = {"complete": True, "schema_version": metadata["schema_version"],
        "annotator": metadata["annotator"], "episodes": [{**metadata,
            "annotation_path": str(values), "values_sha256": digest(values)}]}
    (work / "robometer_manifest.json").write_text(json.dumps(manifest))
    config = directory / "input.yaml"
    config.write_text("sampling_hz: 3\n")
    version = {"id": "robo", "dataset_id": dataset["id"], "dataset_sha256": dataset["dataset_sha256"],
        "evaluator": "robometer", "work_dir": str(work.parent), "config_path": str(config),
        "run_ids": ["run"], "parameters": {}, "created_at": "2026-01-01"}
    (directory / "version.json").write_text(json.dumps(version))
    jobs.datasets.update_version(dataset["id"], seal(directory / "version.json"))
    destination = tmp_path_factory.mktemp("exports") / "robo"
    export(jobs, dataset, destination)
    copied = next(destination.rglob("robometer_evaluation.npz"))
    assert copied.read_bytes() == values.read_bytes()
    assert (episode / "robometer_evaluation.npz").read_bytes() == old_bytes
    assert next(destination.rglob("rynnvalue_evaluation.npz")).read_bytes() == (episode / "rynnvalue_evaluation.npz").read_bytes()


def test_ready_legacy_reward_without_version_file_can_be_exported(tmp_path, tmp_path_factory, monkeypatch):
    jobs, dataset, _ = setup_recording(tmp_path, monkeypatch)
    version = local_stage(jobs, dataset)
    (jobs.datasets.root / dataset["id"] / "annotations" / version["id"] / "version.json").unlink()
    destination = tmp_path_factory.mktemp("exports") / "legacy"
    export(jobs, dataset, destination)
    metadata = json.loads(next(destination.rglob("trajectory_reward.stage.json")).read_text())
    assert metadata["reward_config"]["stage_exponent"] == 4


def test_sources_changed_during_copy_fail_without_publishing(tmp_path, tmp_path_factory, monkeypatch):
    from backend.app.services import dataset_transfer
    jobs, dataset, run = setup_recording(tmp_path, monkeypatch)
    copy = dataset_transfer._copy_run
    def change(source, target, progress):
        signatures = copy(source, target, progress)
        Path(run["trajectory"]).with_name("new-annotation.json").write_text("{}")
        return signatures
    monkeypatch.setattr(dataset_transfer, "_copy_run", change)
    destination = tmp_path_factory.mktemp("exports") / "changing"
    with pytest.raises(ValueError, match="Source changed"):
        export(jobs, dataset, destination)
    assert not destination.exists()


@pytest.mark.parametrize("damage", ["native", "explicit", "observation", "symlink"])
def test_corruption_fails_atomically_without_source_writes(tmp_path, tmp_path_factory, monkeypatch, damage):
    jobs, dataset, run = setup_recording(tmp_path, monkeypatch)
    values = native_rynn(jobs, dataset, run)
    if damage == "native":
        values.write_bytes(b"broken")
    elif damage == "explicit":
        version = local_stage(jobs, dataset)
        Path(version["reward_manifest_path"]).write_text("{}")
    elif damage == "observation":
        Path(run["trajectory"]).with_name("trajectory_observations.npz").write_bytes(b"broken")
    else:
        Path(run["trajectory"]).with_name("external").symlink_to(tmp_path)
    root = tmp_path_factory.mktemp("exports")
    with pytest.raises(ValueError):
        export(jobs, dataset, root / "failed")
    assert not (root / "failed").exists()
    assert not list(root.glob(".dataset-export-*"))
    assert Path(run["trajectory"]).exists()


def test_export_does_not_require_rewards_or_generate_them(tmp_path, tmp_path_factory, monkeypatch):
    jobs, dataset, run = setup_recording(tmp_path, monkeypatch)
    destination = tmp_path_factory.mktemp("exports") / "unrated"
    export(jobs, dataset, destination)
    assert next(destination.rglob("stage_annotation.json")).is_file()
    assert not list(destination.rglob("trajectory_reward.*.json"))
    with pytest.raises(ValueError, match="already exists"):
        export(jobs, dataset, destination)
    with pytest.raises(ValueError, match="outside dataset-root"):
        export(jobs, dataset, tmp_path / "exports")
