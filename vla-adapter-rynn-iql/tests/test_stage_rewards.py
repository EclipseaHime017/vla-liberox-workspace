from __future__ import annotations

import copy
import importlib.util
import json
import sys
import zipfile
from pathlib import Path

import numpy as np
import pytest
import yaml

from vla_rynn_iql.config import load_train_config
from vla_rynn_iql.data import prepare_dataset
from vla_rynn_iql.io import atomic_json
from vla_rynn_iql.rewards import load_reward_index, load_stage_annotations, reward_manifest_digest
from vla_rynn_iql.stage_rewards import (
    build_stage_annotation, stage_anchors, stage_chunk_reward, stage_scores,
    stage_annotation_context, validate_stage_annotation,
)


def annotation(frames, *, done=None, exponent=2):
    done = [False] * 20 if done is None else done
    labels = build_stage_annotation(run_id="run", trajectory_sha256="a" * 64,
                                    done=done, keyframes=frames, exponent=exponent)
    return stage_annotation_context(labels, done=done, success_consecutive_steps=5,
                                    exponent=exponent)


def test_success_positive_negative_normalization_and_post_success_tail():
    payload = annotation([
        {"step": 2, "kind": "positive"}, {"step": 4, "kind": "negative"},
        {"step": 6, "kind": "positive"}, {"step": 8, "kind": "positive"},
    ], done=[False] * 10 + [True] * 10)
    assert payload["success_step"] == 15
    anchors = stage_anchors(payload)
    np.testing.assert_allclose([item["score"] for item in anchors], [-1, -2/3, -1, -2/3, -1/3, 0])
    scores = stage_scores(payload)
    assert len(scores) == 21
    assert np.all(scores[15:] == 0)
    assert scores[1] == pytest.approx(-1 + (1/3) * .5 ** 2)


def test_failure_empty_and_negative_first_are_not_clipped():
    np.testing.assert_array_equal(stage_scores(annotation([])), np.full(21, -1))
    payload = annotation([{"step": 3, "kind": "negative"}])
    assert stage_scores(payload)[3] == -2
    assert stage_scores(payload)[-1] == -2
    payload = annotation([{"step": 2, "kind": "positive"}, {"step": 4, "kind": "negative"}])
    assert stage_scores(payload)[2] == -.5
    assert stage_scores(payload)[4] == -1


def test_intermediate_positive_scores_preserve_exact_formula():
    payload = annotation([
        {"step": 1, "kind": "positive"}, {"step": 2, "kind": "positive"},
        {"step": 3, "kind": "negative"}, {"step": 4, "kind": "negative"},
    ], done=[False] * 10 + [True] * 10)
    assert stage_scores(payload)[2] == 1
    assert stage_scores(payload)[15] == 0


@pytest.mark.parametrize("frames", [
    [{"step": 0, "kind": "positive"}], [{"step": 21, "kind": "negative"}],
    [{"step": 1, "kind": "success"}], [{"step": True, "kind": "positive"}],
    [{"step": 1, "kind": "positive"}, {"step": 1, "kind": "negative"}],
])
def test_keyframe_errors(frames):
    with pytest.raises((ValueError, TypeError)):
        annotation(frames)


def test_invalid_denominator_and_duplicate_success_rejected():
    with pytest.raises(ValueError, match="denominator"):
        stage_scores(annotation([{"step": 2, "kind": "negative"}], done=[False] * 10 + [True] * 10))
    with pytest.raises(ValueError, match="precede"):
        stage_scores(annotation([{"step": 15, "kind": "positive"}], done=[False] * 10 + [True] * 10))
    with pytest.raises(ValueError, match="finite"):
        annotation([], exponent=float("nan"))


def test_hash_and_foreign_trajectory_rejected_but_threshold_is_recipe_local():
    payload = build_stage_annotation(run_id="run", trajectory_sha256="a" * 64,
                                     done=[False] * 20, keyframes=[])
    validate_stage_annotation(payload, run_id="run", trajectory_sha256="a" * 64,
                              done=[False] * 20, success_consecutive_steps=5)
    assert validate_stage_annotation(payload, run_id="run", trajectory_sha256="a" * 64,
                                     done=[False] * 20, success_consecutive_steps=6) == payload
    for overrides in [{"run_id": "other"}, {"trajectory_sha256": "b" * 64}]:
        args = dict(run_id="run", trajectory_sha256="a" * 64,
                    done=[False] * 20, success_consecutive_steps=5)
        args.update(overrides)
        with pytest.raises(ValueError, match="stale"):
            validate_stage_annotation(payload, **args)


def test_chunk_endpoint_and_cumulative_actual_length():
    scores = np.asarray([-1, -.9, -.8, -.7, -.6])
    assert stage_chunk_reward(scores, 1, 3, .9, False) == -.6
    assert stage_chunk_reward(scores, 1, 3, .9, True) == pytest.approx(-.8 - .9 * .7 - .9**2 * .6)


def _select(config, source):
    config.raw["reward"].update(source=source, rynnvalue=source == "rynnvalue")


def _save_annotations(config, manifest):
    for episode in manifest["episodes"]:
        path = Path(episode["trajectory_path"])
        with np.load(path) as data:
            done = data["done"]
        payload = build_stage_annotation(
            run_id=episode["run_id"], trajectory_sha256=episode["trajectory_sha256"],
            done=done, keyframes=[{"step": 4, "kind": "positive"}],
            success_consecutive_steps=config.raw["data"]["success_consecutive_steps"],
        )
        atomic_json(path.parent / "stage_annotation.json", payload)


def test_sparse_does_not_require_any_annotation(configured, monkeypatch):
    prepare_dataset(configured)
    _select(configured, "sparse")
    monkeypatch.setattr("vla_rynn_iql.rewards.load_annotation_index", lambda _: pytest.fail("RynnValue lookup"))
    index = load_reward_index(configured)
    assert index["reward_config"]["source"] == "sparse"
    with np.load(index["episodes"][0]["reward_path"]) as data:
        assert np.array_equal(data["final_reward"], data["sparse_reward"])
    assert not (Path(configured.raw["paths"]["work_dir"]) / "annotations").exists()


def test_stage_all_members_required_and_frozen_snapshot_reusable(configured, monkeypatch):
    manifest_path = prepare_dataset(configured).manifest
    manifest = json.loads(manifest_path.read_text())
    _select(configured, "stage")
    monkeypatch.setattr("vla_rynn_iql.rewards.load_annotation_index", lambda _: pytest.fail("RynnValue lookup"))
    with pytest.raises(ValueError, match="root") as error:
        load_reward_index(configured)
    assert "branch" in str(error.value)
    _save_annotations(configured, manifest)
    index = load_reward_index(configured)
    saved = Path(index["stage_annotations_path"])
    assert saved.is_file()
    modified_ns = [Path(entry["reward_path"]).stat().st_mtime_ns for entry in index["episodes"]]
    assert load_reward_index(configured) == index
    assert modified_ns == [Path(entry["reward_path"]).stat().st_mtime_ns for entry in index["episodes"]]
    configured.raw["data"]["stage_annotations_manifest"] = str(saved)
    for episode in manifest["episodes"]:
        (Path(episode["trajectory_path"]).parent / "stage_annotation.json").unlink()
    assert load_reward_index(configured) == index
    configured.raw["reward"]["stage_exponent"] = 3
    updated = load_reward_index(configured)
    assert updated["stage_annotations_sha256"] == index["stage_annotations_sha256"]
    assert updated["episodes"][0]["reward_sha256"] != index["episodes"][0]["reward_sha256"]
    with np.load(updated["episodes"][0]["reward_path"]) as arrays:
        assert "absolute_temporal_distance_seconds" not in arrays
        assert "pbrs_shaping_reward" not in arrays


def test_stage_annotation_success_threshold_is_dataset_local(configured):
    manifest = json.loads(prepare_dataset(configured).manifest.read_text())
    _save_annotations(configured, manifest)
    _select(configured, "stage")
    configured.raw["data"]["success_consecutive_steps"] = 6
    assert len(load_stage_annotations(configured, manifest)["annotations"]) == 2


def test_reward_config_source_compatibility_and_validation(configured, tmp_path):
    path = tmp_path / "settings.yaml"
    raw = copy.deepcopy(configured.raw)
    del raw["reward"]["source"]
    raw["reward"]["rynnvalue"] = False
    path.write_text(yaml.safe_dump(raw))
    assert load_train_config(path).raw["reward"]["source"] == "sparse"
    raw["reward"]["source"] = "stage"
    raw["reward"]["stage_exponent"] = 1
    path.write_text(yaml.safe_dump(raw))
    assert load_train_config(path).raw["reward"]["source"] == "stage"
    raw["reward"]["rynnvalue"] = True
    path.write_text(yaml.safe_dump(raw))
    with pytest.raises(ValueError, match="conflicts"):
        load_train_config(path)


def test_direct_reward_checkpoint_identity_independent_of_job_paths(configured, tmp_path):
    manifest = json.loads(prepare_dataset(configured).manifest.read_text())
    _save_annotations(configured, manifest)
    _select(configured, "stage")
    first = load_reward_index(configured)
    first_digest = reward_manifest_digest(first)
    configured.raw["paths"]["work_dir"] = str(tmp_path / "different-job" / "work")
    prepare_dataset(configured)
    second = load_reward_index(configured)
    assert second["stage_annotations_path"] != first["stage_annotations_path"]
    assert second["episodes"][0]["reward_path"] != first["episodes"][0]["reward_path"]
    assert reward_manifest_digest(second) == first_digest
    configured.raw["reward"]["stage_exponent"] = 3
    assert reward_manifest_digest(load_reward_index(configured)) != first_digest
    configured.raw["reward"]["stage_exponent"] = 2
    episode = manifest["episodes"][0]
    path = Path(episode["trajectory_path"])
    with np.load(path) as arrays:
        changed = build_stage_annotation(
            run_id=episode["run_id"], trajectory_sha256=episode["trajectory_sha256"],
            done=arrays["done"], keyframes=[{"step": 2, "kind": "positive"}],
        )
    atomic_json(path.parent / "stage_annotation.json", changed)
    assert reward_manifest_digest(load_reward_index(configured)) != first_digest
    _select(configured, "sparse")
    assert reward_manifest_digest(load_reward_index(configured)) != first_digest


def test_rynnvalue_digest_keeps_legacy_full_manifest_hash():
    from vla_rynn_iql.io import stable_hash
    payload = {"reward_config": {"rynnvalue": True}, "episodes": [], "some_path": "/legacy"}
    assert reward_manifest_digest(payload) == stable_hash(payload)


def test_stage_sidecars_survive_full_raw_npz_zip_import(configured, tmp_path):
    manifest = json.loads(prepare_dataset(configured).manifest.read_text())
    _save_annotations(configured, manifest)
    dataset = Path(configured.raw["paths"]["dataset_sources"][0])
    archive = tmp_path / "export.zip"
    with zipfile.ZipFile(archive, "w") as bundle:
        for item in dataset.rglob("*"):
            if item.is_file():
                bundle.write(item, item.relative_to(dataset))
    configured.raw["paths"]["dataset_sources"] = [str(archive)]
    configured.raw["paths"]["work_dir"] = str(tmp_path / "zip-work")
    _select(configured, "stage")
    prepare_dataset(configured)
    assert len(load_reward_index(configured)["episodes"]) == 2


def test_stage_lightweight_csv_video_import_never_rebinds_annotation_hash(configured):
    manifest = json.loads(prepare_dataset(configured).manifest.read_text())
    manifest["episodes"][0]["observation_orientation"] = "vla_policy"
    with pytest.raises(ValueError, match="original trajectory.npz"):
        load_stage_annotations(configured, manifest)


def test_missing_stage_labels_stop_before_vla_components_load(configured, monkeypatch):
    from vla_rynn_iql import training
    prepare_dataset(configured)
    _select(configured, "stage")
    monkeypatch.setattr(training, "_device", lambda _: training.torch.device("cpu"))
    monkeypatch.setattr(training.torch.cuda, "is_available", lambda: False)
    monkeypatch.setattr(training, "load_components", lambda _: pytest.fail("VLA loaded before validation"))
    with pytest.raises(ValueError, match="Stage annotations missing"):
        training.train(configured)
    assert not Path(configured.raw["paths"]["output_dir"]).exists()


def test_stage_replay_uses_actual_chunk_endpoint_without_reward_model(configured):
    from test_replay import _stats
    from vla_rynn_iql.replay import ReplayDataset
    manifest = json.loads(prepare_dataset(configured).manifest.read_text())
    _save_annotations(configured, manifest)
    _select(configured, "stage")
    snapshot = load_stage_annotations(configured, manifest)
    replay = ReplayDataset(configured, _stats(7), _stats(8))
    post_success = []
    for index, (episode, chunk_index, _) in enumerate(replay.items):
        item = replay[index]
        chunk = episode["chunks"][chunk_index]
        with np.load(episode["trajectory_path"], allow_pickle=False) as source:
            context = stage_annotation_context(snapshot["annotations"][episode["run_id"]],
                done=source["done"], success_consecutive_steps=5)
        score = stage_scores(context)
        assert item["reward"].item() == pytest.approx(score[chunk["end"]])
        assert item["action_mask"].sum().item() == chunk["length"]
        if episode["terminal_step"] is not None and chunk["start"] > episode["terminal_step"]:
            post_success.append(item)
            assert item["reward"].item() == 0.0
    assert [item["start"] for item in post_success] == [18]
    assert post_success[0]["chunk_length"].item() == 4
    assert post_success[0]["bootstrap_mask"].item() == 0.0


@pytest.mark.parametrize("source", ["sparse", "stage"])
def test_terminal_direct_pipeline_skips_model_and_binding(configured, tmp_path, monkeypatch, source):
    from test_terminal_pipeline import _terminal_config
    manifest = json.loads(prepare_dataset(configured).manifest.read_text())
    if source == "stage":
        _save_annotations(configured, manifest)
    path = _terminal_config(tmp_path, configured.path)
    raw = yaml.safe_load(path.read_text())
    raw["overrides"]["reward"].update(source=source, stage_exponent=2)
    path.write_text(yaml.safe_dump(raw))
    script = Path(__file__).parents[1] / "scripts" / "train_terminal.py"
    spec = importlib.util.spec_from_file_location("test_stage_terminal", script)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    monkeypatch.setattr(sys, "argv", [str(script), "--config", str(path), "--yes"])
    environments = []
    monkeypatch.setattr(module, "_verify_conda_environments", environments.append)
    monkeypatch.setattr(module, "bind_reward_manifest", lambda *_: pytest.fail("RynnValue binding"))
    monkeypatch.setattr(module.StageRunner, "install_signal_handlers", lambda _: None)
    stages = []

    def run_stage(self, stage_id, _label, environment, script, config_path, extra=None):
        stages.append(stage_id)
        effective = load_train_config(config_path)
        if stage_id == "prepare":
            prepare_dataset(effective)
        elif stage_id == "rewards":
            load_reward_index(effective)
        elif stage_id == "train":
            if source == "stage":
                assert Path(effective.raw["data"]["stage_annotations_manifest"]).is_file()
            atomic_json(Path(extra[1]), {"policy_overlay": "test-policy.yaml"})
        else:
            pytest.fail(f"Unexpected stage {stage_id}")

    monkeypatch.setattr(module.StageRunner, "stage", run_stage)
    assert module.main() == 0
    assert stages == ["prepare", "rewards", "train"]
    assert environments == [{"vla-liberox"}]
    pipeline = next((tmp_path / "pipelines" / "runs").glob("*/pipeline.json"))
    state = json.loads(pipeline.read_text())
    assert state["stages"]["annotate"]["status"] == "SKIPPED"
    assert state["stages"]["bind"]["status"] == "SKIPPED"
