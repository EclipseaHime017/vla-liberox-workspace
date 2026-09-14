"""Human labels and immutable, dataset-local reward versions are separate inputs."""
from __future__ import annotations

import copy
import importlib.util
import json
import shutil
from pathlib import Path

import numpy as np
import pytest
import yaml

from vla_rynn_iql.config import load_train_config
from vla_rynn_iql.data import action_source_segments, prepare_dataset, replay_chunks
from vla_rynn_iql.io import atomic_json, sha256_file, stable_hash
from vla_rynn_iql.replay import ReplayDataset
from vla_rynn_iql.rewards import (
    annotate_manifest, chunk_reward_components, load_reward_index,
    materialize_reward_manifest, reward_manifest_digest, sparse_macro_reward,
    sparse_primitive_return,
)
from vla_rynn_iql.stage_rewards import (
    build_stage_annotation, stage_annotation_context, stage_scores, validate_stage_annotation,
)
from test_rewards import CountingAnnotator
from test_replay import _stats
from test_stage_rewards import _save_annotations, _select


def _stage(configured):
    manifest = json.loads(prepare_dataset(configured).manifest.read_text())
    _save_annotations(configured, manifest)
    _select(configured, "stage")
    return manifest


def _pin(configured, index):
    path = Path(index["episodes"][0]["reward_path"]).parent / "reward_manifest.json"
    configured.raw["reward"].update(
        manifest_path=str(path), manifest_sha256=sha256_file(path), version_id="test-version",
    )
    return path


def _legacy_prepared_snapshot(path):
    """Reproduce a saved schema-4 manifest with the success-truncated replay."""
    prepared = json.loads(path.read_text())
    prepared.pop("replay_policy", None)
    for episode in prepared["episodes"]:
        endpoint = (episode["terminal_step"] + 1 if episode["terminal_step"] is not None
                    else episode["recorded_action_count"])
        episode["action_count"] = endpoint
        episode["chunks"] = [copy.deepcopy(chunk) for chunk in episode["evaluation_chunks"]
                             if chunk["end"] <= endpoint]
        with np.load(episode["trajectory_path"], allow_pickle=False) as arrays:
            episode["action_source_segments"] = action_source_segments(
                arrays["action_source"].tolist(), endpoint)
    # This is the dataset identity formula used by schema 4 before complete
    # recorded trajectories became the replay policy.
    prepared["dataset_sha256"] = stable_hash([{key: episode[key] for key in (
        "run_id", "source_manifest_sha256", "trajectory_sha256", "observations_sha256",
        "task_id", "prompt", "resume_step", "action_count", "recorded_action_count",
        "terminal_step", "trailing_action_count", "post_terminal_false_count",
        "recorded_success", "raw_done_true_count", "success_consecutive_steps",
        "success_streak_start", "action_source_segments", "recorded_action_source_segments",
        "chunks", "evaluation_chunks", "split",
    )} for episode in prepared["episodes"]])
    prepared["trajectory_chunk_count"] = sum(len(episode["chunks"]) for episode in prepared["episodes"])
    # This fixture has no identical copied-prefix chunk in its parent trajectory.
    prepared["chunk_count"] = prepared["trajectory_chunk_count"]
    atomic_json(path, prepared)
    return prepared


def _assert_replay_uses_saved_post_success_rewards(configured, index):
    replay = ReplayDataset(configured, _stats(7), _stats(8), reward_index=index)
    success_chunks, tail_chunks = [], []
    for item_index, (episode, chunk_index, reward_path) in enumerate(replay.items):
        item = replay[item_index]
        chunk = replay_chunks(episode)[chunk_index]
        with np.load(reward_path, allow_pickle=False) as arrays:
            assert item["reward"].item() == float(arrays["final_reward"][chunk_index])
        assert item["action_mask"].sum().item() == chunk["length"]
        if episode["terminal_step"] is None:
            continue
        if chunk["end"] == episode["terminal_step"] + 1:
            success_chunks.append(item)
        if chunk["start"] > episode["terminal_step"]:
            tail_chunks.append(item)
    assert [item["start"] for item in success_chunks] == [13]
    assert success_chunks[0]["bootstrap_mask"].item() == 1.0
    assert [item["start"] for item in tail_chunks] == [18]
    assert tail_chunks[0]["chunk_length"].item() == 4
    assert tail_chunks[0]["action_mask"].tolist() == [True] * 4 + [False] * 4
    assert tail_chunks[0]["bootstrap_mask"].item() == 0.0
    if index["reward_config"].get("source") in {"stage", "sparse"}:
        assert tail_chunks[0]["reward"].item() == 0.0


def test_label_identity_excludes_recipe_and_old_labels_need_no_resave():
    done = [False] * 10 + [True] * 10
    options = dict(run_id="run", trajectory_sha256="a" * 64, done=done,
                   keyframes=[{"step": 5, "kind": "positive"}])
    labels = build_stage_annotation(**options, exponent=2)
    assert build_stage_annotation(**options, exponent=4, success_consecutive_steps=6) == labels
    assert set(labels) == {"schema_version", "run_id", "trajectory_sha256",
                           "action_count", "keyframes", "annotation_sha256"}
    legacy = {**labels, "schema_version": 1, "exponent": 2.0,
              "success_step": 15, "success_consecutive_steps": 5}
    legacy.pop("annotation_sha256")
    legacy["annotation_sha256"] = stable_hash(legacy)
    before = copy.deepcopy(legacy)
    validated = validate_stage_annotation(legacy, run_id="run", trajectory_sha256="a" * 64,
        done=done, success_consecutive_steps=6)
    assert validated == legacy == before
    first = stage_annotation_context(validated, done=done, success_consecutive_steps=6, exponent=2)
    second = stage_annotation_context(validated, done=done, success_consecutive_steps=6, exponent=4)
    assert first["success_step"] == second["success_step"] == 16
    assert not np.array_equal(stage_scores(first), stage_scores(second))


def test_formula_error_is_not_corrupt_or_missing_labels():
    done = [False] * 10 + [True] * 10
    labels = build_stage_annotation(run_id="run", trajectory_sha256="a" * 64, done=done,
                                    keyframes=[{"step": 2, "kind": "negative"}])
    assert validate_stage_annotation(labels, run_id="run", trajectory_sha256="a" * 64,
        done=done, success_consecutive_steps=5) == labels
    with pytest.raises(ValueError, match="denominator"):
        stage_scores(stage_annotation_context(labels, done=done, success_consecutive_steps=5))


def test_code_change_and_forced_rebuild_do_not_overwrite_any_previous_array(configured, monkeypatch):
    _stage(configured)
    old = load_reward_index(configured)
    old_paths = [Path(item["reward_path"]) for item in old["episodes"]]
    old_bytes = [path.read_bytes() for path in old_paths]
    monkeypatch.setattr("vla_rynn_iql.rewards.reward_implementation_fingerprint", lambda _: "new-formula")
    new = load_reward_index(configured)
    assert new["derivation_implementation_sha256"] == "new-formula"
    assert new["episodes"][0]["reward_path"] != str(old_paths[0])
    forced = json.loads(materialize_reward_manifest(configured, force=True).read_text())
    assert forced["episodes"][0]["reward_path"] != new["episodes"][0]["reward_path"]
    assert [path.read_bytes() for path in old_paths] == old_bytes
    assert json.loads((old_paths[0].parent / "reward_manifest.json").read_text()) == old


def test_pinned_training_reads_snapshot_despite_new_code_or_deleted_live_labels(configured, monkeypatch):
    prepared = _stage(configured)
    pinned = load_reward_index(configured)
    _pin(configured, pinned)
    for episode in prepared["episodes"]:
        (Path(episode["trajectory_path"]).parent / "stage_annotation.json").unlink()
    monkeypatch.setattr("vla_rynn_iql.rewards.materialize_reward_manifest",
                        lambda *_: pytest.fail("Pinned training must not materialize"))
    monkeypatch.setattr("vla_rynn_iql.rewards.reward_implementation_fingerprint",
                        lambda _: pytest.fail("Historical version must not use current formula"))
    assert load_reward_index(configured) == pinned
    configured.raw["reward"]["stage_exponent"] = 4
    with pytest.raises(ValueError, match="parameters conflict"):
        load_reward_index(configured)


def test_pinned_version_rejects_manifest_and_arrays_corruption(configured):
    _stage(configured)
    index = load_reward_index(configured)
    path = _pin(configured, index)
    original = path.read_bytes()
    path.write_bytes(original + b" ")
    with pytest.raises(ValueError, match="manifest.*hash"):
        load_reward_index(configured)
    path.write_bytes(original)
    array_path = Path(index["episodes"][0]["reward_path"])
    array_path.write_bytes(array_path.read_bytes() + b"tampered")
    with pytest.raises(ValueError, match="arrays.*corrupted"):
        load_reward_index(configured)


def test_pinned_version_checks_chunk_alignment_not_only_array_length(configured):
    _stage(configured)
    index = load_reward_index(configured)
    path = _pin(configured, index)
    index["episodes"][0]["evaluation_chunks"][0]["start"] += 1
    atomic_json(path, index)
    configured.raw["reward"]["manifest_sha256"] = sha256_file(path)
    with pytest.raises(ValueError, match="evaluation_chunks mismatch"):
        load_reward_index(configured)


def test_direct_versions_include_complete_recorded_timeline(configured):
    prepared = _stage(configured)
    index = load_reward_index(configured)
    for episode, entry in zip(prepared["episodes"], index["episodes"]):
        with np.load(entry["reward_path"]) as arrays:
            count = episode["recorded_action_count"]
            np.testing.assert_array_equal(arrays["observation_steps"], np.arange(count + 1))
            np.testing.assert_allclose(arrays["time_seconds"], np.arange(count + 1) / 20)
            assert len(arrays["stage_score"]) == count + 1
            assert len(arrays["environment_done"]) == count
            assert len(arrays["final_reward"]) == len(episode["evaluation_chunks"])


def test_forced_model_evaluation_keeps_previous_official_outputs(configured):
    prepare_dataset(configured)
    annotator = CountingAnnotator()
    original = json.loads(annotate_manifest(configured, annotator).read_text())
    first_calls = annotator.calls
    old_paths = [Path(item["annotation_path"]) for item in original["episodes"]]
    old_bytes = [path.read_bytes() for path in old_paths]
    new = json.loads(annotate_manifest(configured, annotator, overwrite=True).read_text())
    assert annotator.calls > first_calls
    assert new["episodes"][0]["annotation_path"] != original["episodes"][0]["annotation_path"]
    assert [path.read_bytes() for path in old_paths] == old_bytes


def test_pinned_config_requires_complete_reference_and_resolves_relative_path(configured, tmp_path):
    raw = copy.deepcopy(configured.raw)
    path = tmp_path / "pinned.yaml"
    raw["reward"]["manifest_path"] = "versions/reward_manifest.json"
    path.write_text(yaml.safe_dump(raw))
    with pytest.raises(ValueError, match="together"):
        load_train_config(path)
    raw["reward"].update(manifest_sha256="a" * 64, version_id="version-a")
    path.write_text(yaml.safe_dump(raw))
    parsed = load_train_config(path)
    assert parsed.raw["reward"]["manifest_path"] == str(tmp_path / "versions/reward_manifest.json")


def test_reward_config_change_reuses_rynn_outputs_and_preserves_previous_reward(configured, monkeypatch):
    prepare_dataset(configured)
    annotate_manifest(configured, CountingAnnotator())
    old = load_reward_index(configured)
    old_path = Path(old["episodes"][0]["reward_path"])
    old_bytes = old_path.read_bytes()
    monkeypatch.setattr("vla_rynn_iql.rewards.RynnValueAnnotator",
                        lambda _: pytest.fail("Reward recipe changes never load the model"))
    configured.raw["reward"].update(gamma=.8, shaping_weight=.7, accumulate_primitive_steps=True)
    new = load_reward_index(configured)
    assert new["episodes"][0]["reward_path"] != str(old_path)
    assert old_path.read_bytes() == old_bytes


@pytest.fixture
def backend_seal():
    """Use the real pure-data worker, without importing FastAPI or backend services."""
    path = (Path(__file__).resolve().parents[2] / "liberox-vla-adapter-terminal"
            / "backend/app/workers/finalize_reward_version.py")
    if not path.is_file():
        pytest.skip("Backend worker is absent in this standalone training checkout")
    spec = importlib.util.spec_from_file_location("reward_version_seal_integration", path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module.seal


@pytest.mark.parametrize("source", ["stage", "sparse", "rynnvalue"])
def test_real_backend_seal_to_pinned_training_interface(configured, tmp_path, monkeypatch, backend_seal, source):
    prepared_path = prepare_dataset(configured).manifest
    prepared = json.loads(prepared_path.read_text())
    _select(configured, source)
    if source == "stage":
        _save_annotations(configured, prepared)
    if source == "rynnvalue":
        annotate_manifest(configured, CountingAnnotator())
    generated = json.loads(materialize_reward_manifest(configured).read_text())
    original_arrays = {item["reward_path"]: Path(item["reward_path"]).read_bytes()
                       for item in generated["episodes"]}
    sealed_root = tmp_path / "sealed-version"
    sealed_root.mkdir()
    config_path = tmp_path / "generation_config.yaml"
    config_path.write_text(yaml.safe_dump(configured.raw))
    version_path = sealed_root / "version.json"
    atomic_json(version_path, {
        "id": "version-one", "evaluator": source, "complete": False,
        "dataset_sha256": prepared["source_dataset_sha256"],
        "run_ids": [episode["run_id"] for episode in prepared["episodes"]],
        "work_dir": configured.raw["paths"]["work_dir"], "config_path": str(config_path),
    })
    version = backend_seal(version_path)
    assert version["complete"] is True
    configured.raw["reward"].update(manifest_path=version["reward_manifest_path"],
        manifest_sha256=version["reward_manifest_sha256"], version_id=version["id"])
    pinned = load_reward_index(configured)
    assert all(Path(path).read_bytes() == content for path, content in original_arrays.items())
    with monkeypatch.context() as no_derivation:
        no_derivation.setattr("vla_rynn_iql.rewards.materialize_reward_manifest",
                              lambda *_: pytest.fail("Training regenerated a sealed reward"))
        assert load_reward_index(configured) == pinned
    if source != "rynnvalue":
        return

    old_work = Path(configured.raw["paths"]["work_dir"])
    annotation_manifest = old_work / "annotations/annotation_manifest.json"
    sealed_annotations = json.loads(annotation_manifest.read_text())
    assert pinned["annotation_manifest_sha256"] == stable_hash(sealed_annotations)
    local_official = {item["annotation_path"]: Path(item["annotation_path"]).read_bytes()
                      for item in sealed_annotations["episodes"]}
    assert all(Path(path).parent == old_work / "official_outputs" for path in local_official)
    # Delete only the test fixture's disposable global cache. A new version must
    # recover from copied, dataset-bound official outputs and never load a model.
    cache = Path(configured.raw["paths"]["annotation_cache"])
    assert cache.is_relative_to(tmp_path)
    shutil.rmtree(cache)
    config_copy = copy.deepcopy(configured.raw)
    configured.raw["reward"].update(manifest_path=None, manifest_sha256=None, version_id=None, gamma=.8)
    configured.raw["paths"]["work_dir"] = str(tmp_path / "new-generation-work")
    prepare_dataset(configured)
    new_work = Path(configured.raw["paths"]["work_dir"])
    (new_work / "annotations").mkdir()
    shutil.copyfile(annotation_manifest, new_work / "annotations/annotation_manifest.json")
    monkeypatch.setattr("vla_rynn_iql.rewards.RynnValueAnnotator",
                        lambda _: pytest.fail("Sealed official outputs should avoid model forward"))
    annotate_manifest(configured)
    updated = load_reward_index(configured)
    assert updated["reward_config"]["gamma"] == .8
    assert all(Path(path).read_bytes() == content for path, content in original_arrays.items())
    assert all(Path(path).read_bytes() == content for path, content in local_official.items())
    # Previously saved effective config and reward pin still reference old numbers.
    configured.raw.clear()
    configured.raw.update(config_copy)
    assert load_reward_index(configured) == pinned


@pytest.mark.parametrize("source", ["stage", "sparse", "rynnvalue"])
@pytest.mark.parametrize("cumulative", [False, True])
@pytest.mark.parametrize("legacy_prepared", [False, True], ids=["full-recording", "legacy-schema-4"])
def test_training_local_gamma_reduction_keeps_dataset_and_semantics_pinned(
    configured, monkeypatch, source, cumulative, legacy_prepared,
):
    prepared_path = prepare_dataset(configured).manifest
    prepared = (_legacy_prepared_snapshot(prepared_path) if legacy_prepared
                else json.loads(prepared_path.read_text()))
    branch = next(episode for episode in prepared["episodes"] if episode["run_id"] == "branch")
    assert branch["action_count"] == (18 if legacy_prepared else 22)
    assert branch["reward_boundaries"] == [0, 5, 13, 18, 22]
    _select(configured, source)
    if source == "stage":
        _save_annotations(configured, prepared)
    if source == "rynnvalue":
        annotate_manifest(configured, CountingAnnotator())
    original = load_reward_index(configured)
    manifest_path = _pin(configured, original)
    frozen_root = Path(configured.raw["paths"]["work_dir"])
    original_files = {str(path): path.read_bytes() for path in frozen_root.rglob("*") if path.is_file()}
    official_files = {item["official_annotation_path"]: Path(item["official_annotation_path"]).read_bytes()
                      for item in original["episodes"] if "official_annotation_path" in item}
    monkeypatch.setattr("vla_rynn_iql.rewards.materialize_reward_manifest",
                        lambda *_: pytest.fail("Pinned training must not regenerate evaluation rewards"))
    monkeypatch.setattr("vla_rynn_iql.rewards.annotate_manifest",
                        lambda *_: pytest.fail("Training must not repeat model evaluation"))
    monkeypatch.setattr("vla_rynn_iql.rewards.RynnValueAnnotator",
                        lambda _: pytest.fail("Training must not evaluate a model"))
    monkeypatch.setattr("vla_rynn_iql.stage_rewards.stage_scores",
                        lambda *_: pytest.fail("Training must not recompute the Stage curve"))
    monkeypatch.setattr("vla_rynn_iql.rewards.load_stage_annotations",
                        lambda *_: pytest.fail("Training must not read live keyframes"))
    with monkeypatch.context() as same_recipe:
        same_recipe.setattr("vla_rynn_iql.rewards._adapt_pinned_training_rewards",
                            lambda *_: pytest.fail("Equal recipe must read the original immutable numbers"))
        assert load_reward_index(configured) == original
        _assert_replay_uses_saved_post_success_rewards(configured, original)
        assert not Path(configured.raw["paths"]["output_dir"]).exists()
    gamma = .87
    configured.raw["reward"].update(gamma=gamma, accumulate_primitive_steps=cumulative)
    adapted = load_reward_index(configured)
    assert adapted["source_reward_manifest_sha256"] == sha256_file(manifest_path)
    assert adapted["source_reward_version_id"] == "test-version"
    assert adapted["reward_config"]["gamma"] == gamma
    assert adapted["reward_config"]["accumulate_primitive_steps"] is cumulative
    assert reward_manifest_digest(adapted) != reward_manifest_digest(original)
    _assert_replay_uses_saved_post_success_rewards(configured, adapted)
    for episode, old, new in zip(prepared["episodes"], original["episodes"], adapted["episodes"]):
        assert Path(new["reward_path"]).is_relative_to(
            Path(configured.raw["paths"]["output_dir"]) / "reward_adaptations")
        done = np.zeros(episode["recorded_action_count"], dtype=bool)
        if episode["terminal_step"] is not None:
            done[episode["terminal_step"]:] = True
        with np.load(old["reward_path"]) as original_values, np.load(new["reward_path"]) as values:
            expected = []
            for chunk in episode["evaluation_chunks"]:
                start, length, end = chunk["start"], chunk["length"], chunk["end"]
                if source == "stage":
                    scores = original_values["stage_score"]
                    expected.append(float(np.dot(gamma ** np.arange(length), scores[start + 1:end + 1]))
                                    if cumulative else scores[end])
                    np.testing.assert_array_equal(values["stage_score"], scores)
                elif source == "rynnvalue":
                    lookup = dict(zip(original_values["boundary_steps"],
                                      original_values["absolute_temporal_distance_seconds"][:, 0]))
                    expected.append(chunk_reward_components(done, start, length, lookup[start], lookup[end],
                        gamma, configured.raw["reward"]["shaping_weight"], cumulative)[2])
                    np.testing.assert_array_equal(values["absolute_temporal_distance_seconds"],
                                                  original_values["absolute_temporal_distance_seconds"])
                else:
                    expected.append(sparse_primitive_return(done, start, length, gamma) if cumulative
                                    else sparse_macro_reward(done, start, length))
            np.testing.assert_allclose(values["final_reward"], expected, rtol=1e-6, atol=1e-6)
    assert {str(path): path.read_bytes() for path in frozen_root.rglob("*") if path.is_file()} == original_files
    assert all(Path(path).read_bytes() == content for path, content in official_files.items())
    again = load_reward_index(configured)
    assert again["episodes"][0]["reward_path"] != adapted["episodes"][0]["reward_path"]
    assert reward_manifest_digest(again) == reward_manifest_digest(adapted)


@pytest.mark.parametrize("field,value", [("stage_exponent", 4), ("shaping_weight", .9)])
def test_training_reduction_does_not_unlock_stage_recipe(configured, field, value):
    _stage(configured)
    pinned = load_reward_index(configured)
    _pin(configured, pinned)
    configured.raw["reward"].update(gamma=.8, accumulate_primitive_steps=True)
    configured.raw["reward"][field] = value
    with pytest.raises(ValueError, match="parameters conflict"):
        load_reward_index(configured)
    assert not Path(configured.raw["paths"]["output_dir"]).exists()


def test_training_local_reduction_rejects_writes_inside_sealed_work(configured):
    _stage(configured)
    index = load_reward_index(configured)
    _pin(configured, index)
    configured.raw["reward"]["gamma"] = .8
    configured.raw["paths"]["output_dir"] = configured.raw["paths"]["work_dir"]
    with pytest.raises(ValueError, match="outside the sealed"):
        load_reward_index(configured)
    assert not (Path(configured.raw["paths"]["work_dir"]) / "reward_adaptations").exists()


def test_materialize_with_pinned_override_returns_training_local_manifest(configured, monkeypatch):
    _stage(configured)
    pinned = load_reward_index(configured)
    source_path = _pin(configured, pinned)
    original_bytes = source_path.read_bytes()
    assert materialize_reward_manifest(configured) == source_path
    configured.raw["reward"].update(gamma=.8, accumulate_primitive_steps=True)
    monkeypatch.setattr("vla_rynn_iql.rewards.load_stage_annotations",
                        lambda *_: pytest.fail("Pinned materialization must not reload keyframes"))
    path = materialize_reward_manifest(configured)
    assert path != source_path
    adapted = json.loads(path.read_text())
    assert adapted["reward_config"]["gamma"] == .8
    assert adapted["reward_config"]["accumulate_primitive_steps"] is True
    assert adapted["training_adaptation_manifest_path"] == str(path)
    assert source_path.read_bytes() == original_bytes


@pytest.mark.parametrize("field", ["trajectory_sha256", "observations_sha256", "prompt"])
def test_shared_annotation_manifest_rejects_different_source_with_equal_run_and_boundaries(configured, field):
    from vla_rynn_iql.rewards import _reusable_manifest_entry
    prepared = json.loads(prepare_dataset(configured).manifest.read_text())
    manifest = json.loads(annotate_manifest(configured, CountingAnnotator()).read_text())
    episode = copy.deepcopy(prepared["episodes"][0])
    assert _reusable_manifest_entry(episode, configured.raw["reward"], manifest) is not None
    episode[field] = "different-task" if field == "prompt" else "b" * 64
    assert _reusable_manifest_entry(episode, configured.raw["reward"], manifest) is None
