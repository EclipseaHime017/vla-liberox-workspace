from __future__ import annotations

import copy
import json
from pathlib import Path

import numpy as np
import pytest

from vla_rynn_iql.data import prepare_dataset
from vla_rynn_iql.fusion_rewards import fused_reward_arrays
from vla_rynn_iql.fusion_rewards import import_saved_model_inputs
from vla_rynn_iql.io import sha256_file
from vla_rynn_iql.rewards import load_reward_index
from test_stage_rewards import _save_annotations
from test_rewards import FakeAnnotator


def example(length=3, *, terminal=None):
    return {"run_id": "sample", "recorded_action_count": length, "terminal_step": terminal,
            "reward_boundaries": [0, length], "chunks": [{"start": 0, "end": length, "length": length}]}


def recipe(**overrides):
    return {"source": "final", "fusion_mode": "additive", "alpha": 0., "shaping_weight": 0.,
            "gamma": .9, "accumulate_primitive_steps": False, **overrides}


@pytest.mark.parametrize("length", [3, 8])
@pytest.mark.parametrize("cumulative", [False, True])
def test_additive_three_existing_rewards(length, cumulative):
    episode = example(length)
    scores = np.linspace(-1., -.2, length + 1)
    signals = {"stage_score": scores, "absolute_temporal_distance_seconds": np.array([[3.], [1.]])}
    settings = recipe(accumulate_primitive_steps=cumulative)
    b = -sum(.9 ** h for h in range(length)) if cumulative else -1.
    f = 3 - .9 ** (length if cumulative else 1)
    s = np.dot(.9 ** np.arange(length), scores[1:]) if cumulative else scores[-1]
    assert fused_reward_arrays(episode, settings, {})["final_reward"][0] == pytest.approx(b)
    rynn = fused_reward_arrays(episode, {**settings, "shaping_weight": .1}, signals)
    assert rynn["final_reward"][0] == pytest.approx(b + .1 * f)
    stage = fused_reward_arrays(episode, {**settings, "alpha": 1.}, signals)
    assert stage["final_reward"][0] == pytest.approx(s)
    merged = fused_reward_arrays(episode, {**settings, "alpha": .5, "shaping_weight": .1}, signals)
    assert merged["final_reward"][0] == pytest.approx(.5 * b + .5 * s + .1 * f)
    assert merged["original_final_reward"][0] == pytest.approx(b + .1 * f)


def test_multiplication_has_no_floor_and_ignores_cumulative():
    settings = recipe(fusion_mode="multiplicative")
    scores = np.array([-1., -.8, -.4, 0.])
    assert fused_reward_arrays(example(), settings, {"stage_score": scores})["final_reward"][0] == 0
    settings["accumulate_primitive_steps"] = True
    result = fused_reward_arrays(example(), settings, {"stage_score": scores})
    assert result["final_reward"][0] == 0
    assert result["sparse_reward"][0] == -1


def test_negative_scores_are_not_clipped_and_positive_scores_rejected():
    settings = recipe(alpha=1.)
    assert fused_reward_arrays(example(), settings, {"stage_score": np.full(4, -2.)})["final_reward"][0] == -2
    with pytest.raises(ValueError, match="exceeds zero"):
        fused_reward_arrays(example(), settings, {"stage_score": np.array([-1., .1, 0., 0.])})
    with pytest.raises(ValueError, match="Stage"):
        fused_reward_arrays(example(), settings, {})
    with pytest.raises(ValueError, match="RynnValue"):
        fused_reward_arrays(example(), recipe(shaping_weight=.1), {})


@pytest.mark.parametrize("length", [3, 8])
def test_multiplication_uses_chunk_potential_without_interpolation(length):
    scores = np.linspace(-1., -.1, length + 1)
    signals = {"stage_score": scores,
               "absolute_temporal_distance_seconds": np.array([[3.], [1.]])}
    settings = recipe(fusion_mode="multiplicative", shaping_weight=.2, accumulate_primitive_steps=True)
    for gamma in [.9, .92]:
        settings["gamma"] = gamma
        result = fused_reward_arrays(example(length), settings, signals)
        expected = .1 * (-1 + .2 * (3 - gamma))
        assert result["final_reward"][0] == pytest.approx(expected)
        assert result["pbrs_shaping_reward"][0] == pytest.approx(3 - gamma)
        assert "primitive_potential" not in result


def test_mixed_global_results_cannot_bootstrap_multiplication_with_primitive_discount(configured):
    manifest = json.loads(prepare_dataset(configured).manifest.read_text())
    _save_annotations(configured, manifest)
    configured.raw["reward"].update(recipe(alpha=.5), stage_exponent=2.)
    saved = load_reward_index(configured)
    saved["binding_kind"] = "global_trajectory_snapshots"
    for index, entry in enumerate(saved["episodes"]):
        entry["saved_reward_config"] = {**saved["reward_config"],
            "fusion_mode": "additive" if index == 0 else "multiplicative"}
    path = Path(configured.raw["paths"]["work_dir"]) / "composed.json"
    path.write_text(json.dumps(saved))
    configured.raw["reward"].update(manifest_path=str(path), manifest_sha256=sha256_file(path),
                                   version_id="mixed", accumulate_primitive_steps=True)
    with pytest.raises(ValueError, match="entire run"):
        load_reward_index(configured)


def test_preserves_post_success_zero_cost():
    settings = recipe(alpha=.5)
    data = fused_reward_arrays(example(8, terminal=0), settings, {"stage_score": np.zeros(9)})
    assert data["final_reward"].tolist() == [0.]


@pytest.mark.parametrize("mode", ["additive", "multiplicative"])
@pytest.mark.parametrize("cumulative", [False, True])
def test_rescale_anchors_full_recording_and_preserves_original_components(mode, cumulative):
    chunks = [{"start": 0, "end": 3, "length": 3}, {"start": 3, "end": 6, "length": 3}]
    episode = {**example(6, terminal=5), "reward_boundaries": [0, 3, 6],
               "evaluation_chunks": chunks, "chunks": chunks[1:], "resume_step": 3}
    signals = {"stage_score": np.array([-1., -.9, -.8, -.7, -.6, -.2, 0.]),
               "absolute_temporal_distance_seconds": np.array([[6.], [4.], [0.]])}
    settings = recipe(fusion_mode=mode, alpha=.5, shaping_weight=.1, accumulate_primitive_steps=cumulative)
    original = fused_reward_arrays(episode, settings, signals)
    result = fused_reward_arrays(episode, {**settings, "final_normalization": "initial_chunk_v1"}, signals)
    assert result["final_reward"][0] == -1.
    np.testing.assert_allclose(result["final_reward"], original["final_reward"] / -original["final_reward"][0])
    np.testing.assert_allclose(result["raw_final_reward"], original["final_reward"])
    assert result["final_reward_reference"] == pytest.approx(original["final_reward"][0])
    assert result["final_reward_scale"] == pytest.approx(1 / -original["final_reward"][0])
    for key in ("sparse_reward", "pbrs_shaping_reward", "dense_reward", "original_final_reward", "stage_score"):
        np.testing.assert_array_equal(result[key], original[key])
    np.testing.assert_array_equal(result["pbrs_chunk_reward"], result["final_reward"])


@pytest.mark.parametrize("distance", [1., 2.])
def test_nonnegative_initial_final_is_rejected(distance):
    with pytest.raises(ValueError, match="Initial Final Reward must be negative.*sample"):
        fused_reward_arrays(example(), recipe(shaping_weight=1., final_normalization="initial_chunk_v1"),
            {"absolute_temporal_distance_seconds": np.array([[distance], [0.]])})


def test_negative_near_zero_is_not_clamped_and_missing_prefix_is_rejected():
    settings = recipe(alpha=1., final_normalization="initial_chunk_v1")
    result = fused_reward_arrays(example(), settings, {"stage_score": np.array([-1., -.8, -.4, -1e-12])})
    assert result["final_reward"][0] == -1
    assert result["final_reward_reference"] == -1e-12
    episode = {**example(), "chunks": [{"start": 1, "end": 3, "length": 2}]}
    with pytest.raises(ValueError, match="full trajectory from step 0"):
        fused_reward_arrays(episode, settings, {"stage_score": np.full(4, -.5)})


def test_final_cpu_materialization_and_pinned_training_reduction(configured, monkeypatch):
    manifest = json.loads(prepare_dataset(configured).manifest.read_text())
    _save_annotations(configured, manifest)
    configured.raw["reward"].update(recipe(alpha=.6), stage_exponent=2.)
    monkeypatch.setattr("vla_rynn_iql.rewards.RynnValueAnnotator", lambda _: pytest.fail("model load"))
    index = load_reward_index(configured)
    path = Path(configured.raw["paths"]["work_dir"]) / "rewards" / "reward_manifest.json"
    saved = path.read_bytes()
    values = {entry["reward_path"]: Path(entry["reward_path"]).read_bytes() for entry in index["episodes"]}
    assert load_reward_index(configured) == index
    training = copy.deepcopy(configured)
    training.raw["reward"].update(manifest_path=str(path), manifest_sha256=sha256_file(path), version_id="fixed",
                                   gamma=.92, accumulate_primitive_steps=True)
    adapted = load_reward_index(training)
    assert adapted["reward_config"]["gamma"] == .92
    assert path.read_bytes() == saved
    assert all(Path(name).read_bytes() == content for name, content in values.items())
    configured.raw["reward"]["stage_exponent"] = 4.
    changed = load_reward_index(configured)
    assert changed["stage_annotations_sha256"] == index["stage_annotations_sha256"]
    assert changed["episodes"][0]["reward_sha256"] != index["episodes"][0]["reward_sha256"]


@pytest.mark.parametrize("normalization", ["none", "initial_chunk_v1"])
def test_pinned_gamma_change_recomputes_reference_without_changing_saved_semantics(configured, normalization):
    from vla_rynn_iql.rewards import annotate_manifest
    manifest = json.loads(prepare_dataset(configured).manifest.read_text())
    _save_annotations(configured, manifest)
    annotate_manifest(configured, FakeAnnotator())
    configured.raw["reward"].update(recipe(alpha=.5, shaping_weight=.1), final_normalization=normalization)
    index = load_reward_index(configured)
    path = Path(configured.raw["paths"]["work_dir"]) / "rewards" / "reward_manifest.json"
    before = {e["run_id"]: Path(e["reward_path"]).read_bytes() for e in index["episodes"]}
    configured.raw["reward"].update(manifest_path=str(path), manifest_sha256=sha256_file(path), version_id="pinned",
                                   gamma=.92, accumulate_primitive_steps=True)
    adapted = load_reward_index(configured)
    for episode, entry, old in zip(manifest["episodes"], adapted["episodes"], index["episodes"]):
        assert Path(old["reward_path"]).read_bytes() == before[old["run_id"]]
        with np.load(entry["reward_path"]) as values, np.load(old["reward_path"]) as saved:
            expected = fused_reward_arrays(episode, adapted["reward_config"], {k: saved[k] for k in saved.files})
            np.testing.assert_array_equal(values["final_reward"], expected["final_reward"])
            if normalization == "initial_chunk_v1":
                assert values["final_reward"][0] == -1
                assert values["final_reward_reference"] != saved["final_reward_reference"]
                assert values["final_reward_reference"] == values["raw_final_reward"][0]
            else:
                assert "final_normalization" not in adapted["reward_config"]
                assert "final_reward_scale" not in values


def test_missing_rynn_is_error_not_a_model_forward(configured, monkeypatch):
    prepare_dataset(configured)
    configured.raw["reward"].update(recipe(shaping_weight=.1))
    monkeypatch.setattr("vla_rynn_iql.rewards.RynnValueAnnotator", lambda _: pytest.fail("model load"))
    with pytest.raises(ValueError, match="Required RynnValue"):
        load_reward_index(configured)


def test_regenerate_unscaled_final_reuses_inputs_and_keeps_old_arrays(configured, monkeypatch):
    from vla_rynn_iql.rewards import annotate_manifest, OFFICIAL_OUTPUT_KEYS
    manifest = json.loads(prepare_dataset(configured).manifest.read_text())
    _save_annotations(configured, manifest)
    annotation_path = annotate_manifest(configured, FakeAnnotator())
    annotations_before = annotation_path.read_bytes()
    monkeypatch.setattr("vla_rynn_iql.rewards.RynnValueAnnotator", lambda _: pytest.fail("model load"))
    configured.raw["reward"].update(recipe(alpha=.5, shaping_weight=.1), final_normalization="none")
    old = load_reward_index(configured)
    before = {entry["reward_path"]: Path(entry["reward_path"]).read_bytes() for entry in old["episodes"]}
    configured.raw["reward"]["final_normalization"] = "initial_chunk_v1"
    new = load_reward_index(configured)
    assert annotation_path.read_bytes() == annotations_before
    assert old["stage_annotations_sha256"] == new["stage_annotations_sha256"]
    assert new["reward_config"]["final_normalization"] == "initial_chunk_v1"
    for old_entry, new_entry in zip(old["episodes"], new["episodes"]):
        assert old_entry["reward_path"] != new_entry["reward_path"]
        assert Path(old_entry["reward_path"]).read_bytes() == before[old_entry["reward_path"]]
        with np.load(old_entry["reward_path"]) as old_values, np.load(new_entry["reward_path"]) as values:
            assert values["final_reward"][0] == -1
            for key in (*OFFICIAL_OUTPUT_KEYS, "stage_score", "original_final_reward"):
                np.testing.assert_array_equal(values[key], old_values[key])
    assert load_reward_index(configured) == new


@pytest.mark.parametrize("damage", ["snapshot", "duplicate_member", "chunk_metadata", "symlink"])
def test_bad_cache_is_rebuilt_without_overwriting_old_arrays(configured, damage):
    manifest = json.loads(prepare_dataset(configured).manifest.read_text())
    _save_annotations(configured, manifest)
    configured.raw["reward"].update(recipe(alpha=.6), stage_exponent=2.)
    index = load_reward_index(configured)
    cached = copy.deepcopy(index)
    path = Path(configured.raw["paths"]["work_dir"]) / "rewards" / "reward_manifest.json"
    entry = cached["episodes"][0]
    old_values = Path(entry["reward_path"])
    original_hash = sha256_file(old_values)
    if damage == "snapshot":
        Path(cached["stage_annotations_path"]).write_text("{}")
    elif damage == "duplicate_member":
        cached["episodes"][-1] = entry
    elif damage == "chunk_metadata":
        entry["evaluation_chunks"][0]["end"] += 1
    else:
        link = old_values.with_name("linked.npz")
        link.symlink_to(old_values)
        entry.update(reward_path=str(link), annotation_path=str(link))
    path.write_text(json.dumps(cached))
    repaired = load_reward_index(configured)
    assert repaired["episodes"][0]["reward_path"] != index["episodes"][0]["reward_path"]
    assert sha256_file(old_values) == original_hash
    assert load_reward_index(configured) == repaired


def test_reuses_official_outputs_without_mutation_or_forward(configured, monkeypatch):
    from vla_rynn_iql.rewards import annotate_manifest, OFFICIAL_OUTPUT_KEYS
    from vla_rynn_iql.evaluation_store import bind_reward_manifest
    manifest = json.loads(prepare_dataset(configured).manifest.read_text())
    _save_annotations(configured, manifest)
    annotation_path = annotate_manifest(configured, FakeAnnotator())
    annotations = json.loads(annotation_path.read_text())
    original = {e["run_id"]: (e["annotation_path"], sha256_file(Path(e["annotation_path"]))) for e in annotations["episodes"]}
    monkeypatch.setattr("vla_rynn_iql.rewards.RynnValueAnnotator", lambda _: pytest.fail("model load"))
    configured.raw["reward"].update(recipe(alpha=.5, shaping_weight=.1))
    index = load_reward_index(configured)
    for entry in index["episodes"]:
        source, digest = original[entry["run_id"]]
        assert sha256_file(Path(source)) == digest
        with np.load(source) as raw, np.load(entry["reward_path"]) as fused:
            for key in OFFICIAL_OUTPUT_KEYS:
                np.testing.assert_array_equal(raw[key], fused[key])
            np.testing.assert_allclose(fused["original_final_reward"], fused["sparse_reward"] + fused["dense_reward"])
    work = Path(configured.raw["paths"]["work_dir"])
    with pytest.raises(ValueError, match="Only original"):
        bind_reward_manifest(work / "dataset_manifest.json", work / "rewards" / "reward_manifest.json")


def test_dataset_final_uses_saved_inference_recipes_not_current_defaults(configured, monkeypatch, tmp_path):
    from vla_rynn_iql.rewards import annotate_manifest
    manifest = json.loads(prepare_dataset(configured).manifest.read_text())
    annotations = json.loads(annotate_manifest(configured, FakeAnnotator()).read_text())
    refs = {"source_dataset_sha256": manifest.get("source_dataset_sha256"), "episodes": []}
    for episode, entry in zip(manifest["episodes"], annotations["episodes"]):
        refs["episodes"].append({**episode, **entry,
            "annotation_config": {**annotations["annotation_config"], "max_frames": 32}})
    path = tmp_path / "model_inputs.json"
    path.write_text(json.dumps(refs))
    configured.raw["reward"].update(recipe(shaping_weight=.1))
    monkeypatch.setattr("vla_rynn_iql.rewards.RynnValueAnnotator", lambda _: pytest.fail("model load"))
    import_saved_model_inputs(configured, path)
    result = load_reward_index(configured)
    assert all(entry["annotation_config"]["max_frames"] == 32 for entry in result["episodes"])
    refs["episodes"][0]["trajectory_sha256"] = "changed"
    path.write_text(json.dumps(refs))
    with pytest.raises(ValueError, match="trajectory_sha256"):
        import_saved_model_inputs(configured, path)
