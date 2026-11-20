from __future__ import annotations

import copy
import json
from pathlib import Path

import numpy as np
import pytest

from core.data import prepare_dataset
from rewards.fusion import fused_reward_arrays
from rewards.fusion import import_saved_model_inputs
from core.io import sha256_file
from rewards.manifest import load_reward_index
from test_stage_rewards import _save_annotations
from test_rewards import FakeAnnotator


def example(length=3, *, terminal=None):
    return {"run_id": "sample", "recorded_action_count": length, "terminal_step": terminal,
            "chunks": [{"start": 0, "end": length, "length": length}]}


def recipe(**overrides):
    return {"source": "final", "fusion_mode": "additive", "alpha": 0., "shaping_weight": 0.,
            "gamma": .9, **overrides}


def signals(length):
    return {"stage_score": np.linspace(-1., -.2, length + 1),
            "absolute_temporal_distance_seconds": np.linspace(3., 1., length + 1)[:, None]}


@pytest.mark.parametrize("length", [3, 8])
def test_additive_three_existing_rewards_per_action(length):
    episode, saved = example(length), signals(length)
    phi = -saved["absolute_temporal_distance_seconds"][:, 0]
    b = np.full(length, -1.)
    f = .9 * phi[1:] - phi[:-1]
    s = saved["stage_score"][1:]
    np.testing.assert_allclose(fused_reward_arrays(episode, recipe(), {})["final_reward"], b)
    rynn = fused_reward_arrays(episode, recipe(shaping_weight=.1), saved)
    np.testing.assert_allclose(rynn["final_reward"], b + .1 * f, rtol=1e-6)
    stage = fused_reward_arrays(episode, recipe(alpha=1.), saved)
    np.testing.assert_allclose(stage["final_reward"], s, rtol=1e-6)
    merged = fused_reward_arrays(episode, recipe(alpha=.5, shaping_weight=.1), saved)
    np.testing.assert_allclose(merged["final_reward"], .5 * b + .5 * s + .1 * f, rtol=1e-6)
    np.testing.assert_allclose(merged["original_final_reward"], b + .1 * f, rtol=1e-6)
    # Shaping summed over the recording telescopes to its two ends.
    total = np.dot(.9 ** np.arange(length), merged["pbrs_shaping_reward"])
    assert total == pytest.approx(.9 ** length * phi[-1] - phi[0], rel=1e-5)


def test_multiplication_scales_each_action_by_the_next_stage_score_without_floor():
    saved = {"stage_score": np.array([-1., -.8, -.4, 0.]),
             "absolute_temporal_distance_seconds": np.array([[3.], [2.], [1.], [0.]])}
    result = fused_reward_arrays(example(), recipe(fusion_mode="multiplicative", shaping_weight=.2), saved)
    original = result["sparse_reward"] + .2 * result["pbrs_shaping_reward"]
    np.testing.assert_allclose(result["final_reward"], -saved["stage_score"][1:] * original, rtol=1e-6)
    assert result["final_reward"][-1] == 0


def test_negative_scores_are_not_clipped_and_positive_scores_rejected():
    settings = recipe(alpha=1.)
    assert fused_reward_arrays(example(), settings, {"stage_score": np.full(4, -2.)})["final_reward"].tolist() == [-2.] * 3
    with pytest.raises(ValueError, match="exceeds zero"):
        fused_reward_arrays(example(), settings, {"stage_score": np.array([-1., .1, 0., 0.])})
    with pytest.raises(ValueError, match="Stage"):
        fused_reward_arrays(example(), settings, {})
    with pytest.raises(ValueError, match="RynnValue"):
        fused_reward_arrays(example(), recipe(shaping_weight=.1), {})
    with pytest.raises(ValueError, match="RynnValue"):
        fused_reward_arrays(example(), recipe(shaping_weight=.1),
                            {"absolute_temporal_distance_seconds": np.ones((2, 1))})


def test_mixed_global_results_bind_additive_and_multiplicative_rewards(configured):
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
                                   version_id="mixed")
    assert len(load_reward_index(configured)["episodes"]) == len(manifest["episodes"])


def test_preserves_post_success_zero_cost():
    data = fused_reward_arrays(example(8, terminal=0), recipe(alpha=.5), {"stage_score": np.zeros(9)})
    assert data["final_reward"].tolist() == [0.] * 8


def test_final_cpu_materialization_and_pinned_training_reduction(configured, monkeypatch):
    manifest = json.loads(prepare_dataset(configured).manifest.read_text())
    _save_annotations(configured, manifest)
    configured.raw["reward"].update(recipe(alpha=.6), stage_exponent=2.)
    monkeypatch.setattr("rewards.rynnvalue.annotate.RynnValueAnnotator", lambda _: pytest.fail("model load"))
    index = load_reward_index(configured)
    path = Path(configured.raw["paths"]["work_dir"]) / "rewards" / "reward_manifest.json"
    saved = path.read_bytes()
    values = {entry["reward_path"]: Path(entry["reward_path"]).read_bytes() for entry in index["episodes"]}
    assert load_reward_index(configured) == index
    training = copy.deepcopy(configured)
    training.raw["reward"].update(manifest_path=str(path), manifest_sha256=sha256_file(path), version_id="fixed",
                                   gamma=.92)
    adapted = load_reward_index(training)
    assert adapted["reward_config"]["gamma"] == .92
    assert path.read_bytes() == saved
    assert all(Path(name).read_bytes() == content for name, content in values.items())
    configured.raw["reward"]["stage_exponent"] = 4.
    changed = load_reward_index(configured)
    assert changed["stage_annotations_sha256"] == index["stage_annotations_sha256"]
    assert changed["episodes"][0]["reward_sha256"] != index["episodes"][0]["reward_sha256"]


def test_pinned_gamma_change_recomputes_rewards_without_changing_saved_semantics(configured):
    from rewards.rynnvalue.annotate import annotate_manifest
    manifest = json.loads(prepare_dataset(configured).manifest.read_text())
    _save_annotations(configured, manifest)
    annotate_manifest(configured, FakeAnnotator())
    configured.raw["reward"].update(recipe(alpha=.5, shaping_weight=.1))
    index = load_reward_index(configured)
    path = Path(configured.raw["paths"]["work_dir"]) / "rewards" / "reward_manifest.json"
    before = {e["run_id"]: Path(e["reward_path"]).read_bytes() for e in index["episodes"]}
    configured.raw["reward"].update(manifest_path=str(path), manifest_sha256=sha256_file(path), version_id="pinned",
                                   gamma=.92)
    adapted = load_reward_index(configured)
    for episode, entry, old in zip(manifest["episodes"], adapted["episodes"], index["episodes"]):
        assert Path(old["reward_path"]).read_bytes() == before[old["run_id"]]
        with np.load(entry["reward_path"]) as values, np.load(old["reward_path"]) as saved:
            expected = fused_reward_arrays(episode, adapted["reward_config"], {k: saved[k] for k in saved.files})
            np.testing.assert_array_equal(values["final_reward"], expected["final_reward"])
            assert not np.array_equal(values["pbrs_shaping_reward"], saved["pbrs_shaping_reward"])


def test_missing_rynn_is_error_not_a_model_forward(configured, monkeypatch):
    prepare_dataset(configured)
    configured.raw["reward"].update(recipe(shaping_weight=.1))
    monkeypatch.setattr("rewards.rynnvalue.annotate.RynnValueAnnotator", lambda _: pytest.fail("model load"))
    with pytest.raises(ValueError, match="Required RynnValue"):
        load_reward_index(configured)


@pytest.mark.parametrize("damage", ["snapshot", "duplicate_member", "terminal_step", "symlink"])
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
    elif damage == "terminal_step":
        entry["terminal_step"] = 0
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
    from rewards.rynnvalue.annotate import annotate_manifest
    from rewards.rynnvalue.outputs import OFFICIAL_OUTPUT_KEYS
    from rewards.evaluation_store import bind_reward_manifest
    manifest = json.loads(prepare_dataset(configured).manifest.read_text())
    _save_annotations(configured, manifest)
    annotation_path = annotate_manifest(configured, FakeAnnotator())
    annotations = json.loads(annotation_path.read_text())
    original = {e["run_id"]: (e["annotation_path"], sha256_file(Path(e["annotation_path"]))) for e in annotations["episodes"]}
    monkeypatch.setattr("rewards.rynnvalue.annotate.RynnValueAnnotator", lambda _: pytest.fail("model load"))
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
    from rewards.rynnvalue.annotate import annotate_manifest
    manifest = json.loads(prepare_dataset(configured).manifest.read_text())
    annotations = json.loads(annotate_manifest(configured, FakeAnnotator()).read_text())
    refs = {"source_dataset_sha256": manifest.get("source_dataset_sha256"), "episodes": []}
    for episode, entry in zip(manifest["episodes"], annotations["episodes"]):
        refs["episodes"].append({**episode, **entry,
            "annotation_config": {**annotations["annotation_config"], "max_frames": 32}})
    path = tmp_path / "model_inputs.json"
    path.write_text(json.dumps(refs))
    configured.raw["reward"].update(recipe(shaping_weight=.1))
    monkeypatch.setattr("rewards.rynnvalue.annotate.RynnValueAnnotator", lambda _: pytest.fail("model load"))
    import_saved_model_inputs(configured, path)
    result = load_reward_index(configured)
    assert all(entry["annotation_config"]["max_frames"] == 32 for entry in result["episodes"])
    refs["episodes"][0]["trajectory_sha256"] = "changed"
    path.write_text(json.dumps(refs))
    with pytest.raises(ValueError, match="trajectory_sha256"):
        import_saved_model_inputs(configured, path)
