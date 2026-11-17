import copy
import json
import os
from pathlib import Path
import subprocess
import sys

import numpy as np
import pytest

from vla_rynn_iql.config import validate_train_config
from vla_rynn_iql.data import prepare_dataset
from vla_rynn_iql.fusion_rewards import fused_reward_arrays
from vla_rynn_iql.io import sha256_file
from vla_rynn_iql.rewards import load_reward_index, reward_derivation_config
from test_fusion_rewards import example, recipe
from test_stage_rewards import _save_annotations


@pytest.mark.parametrize("cumulative", [False, True])
@pytest.mark.parametrize("alpha", [0., .5, 1.])
def test_base_coefficient_is_independent_of_stage(cumulative, alpha):
    signals = {"stage_score": np.array([-1., -.8, -.4, -.2]),
               "absolute_temporal_distance_seconds": np.array([[3.], [1.]])}
    settings = recipe(alpha=alpha, shaping_weight=.1, accumulate_primitive_steps=cumulative)
    legacy = fused_reward_arrays(example(), settings, signals)
    explicit_legacy = fused_reward_arrays(example(), {**settings, "base_coefficient": 1 - alpha}, signals)
    for key in legacy:
        np.testing.assert_array_equal(legacy[key], explicit_legacy[key])
    independent = fused_reward_arrays(example(), {**settings, "base_coefficient": 1., "final_normalization": "none"}, signals)
    np.testing.assert_allclose(independent["final_reward"], legacy["final_reward"] + alpha * legacy["sparse_reward"], rtol=1e-6)
    assert "final_reward_scale" not in independent
    np.testing.assert_array_equal(independent["dense_reward"], legacy["dense_reward"])


def test_absent_coefficient_preserves_config_and_recipe_identity(configured):
    baseline = validate_train_config(copy.deepcopy(configured.raw), configured.path)
    with_null = copy.deepcopy(configured.raw)
    with_null["reward"]["base_coefficient"] = None
    normalized = validate_train_config(with_null, configured.path)
    assert normalized.digest == baseline.digest
    assert reward_derivation_config(normalized.section("reward")) == reward_derivation_config(baseline.section("reward"))
    assert "base_coefficient" not in normalized.section("reward")


@pytest.mark.parametrize("value", [-.1, True, "1", float("inf"), float("nan")])
def test_invalid_coefficient_rejected(configured, value):
    raw = copy.deepcopy(configured.raw)
    raw["reward"].update(source="final", base_coefficient=value)
    with pytest.raises((ValueError, TypeError)):
        validate_train_config(raw, configured.path)


@pytest.mark.parametrize("mode,source", [("multiplicative", "final"), ("additive", "rynnvalue")])
def test_coefficient_cannot_silently_affect_other_reward_modes(configured, mode, source):
    raw = copy.deepcopy(configured.raw)
    raw["reward"].update(source=source, fusion_mode=mode, base_coefficient=1.)
    with pytest.raises(ValueError, match="additive Final Reward"):
        validate_train_config(raw, configured.path)


def test_materialized_and_pinned_gamma_reduction_keep_coefficient(configured):
    manifest = json.loads(prepare_dataset(configured).manifest.read_text())
    _save_annotations(configured, manifest)
    configured.raw["reward"].update(recipe(alpha=.5, base_coefficient=1., final_normalization="none"), stage_exponent=2.)
    index = load_reward_index(configured)
    assert index["reward_config"]["base_coefficient"] == 1.
    path = Path(configured.raw["paths"]["work_dir"]) / "rewards/reward_manifest.json"
    protected = {p: p.read_bytes() for p in [path, *(Path(e["reward_path"]) for e in index["episodes"])]}
    training = copy.deepcopy(configured)
    training.raw["reward"].update(manifest_path=str(path), manifest_sha256=sha256_file(path), version_id="fixed", gamma=.92)
    adapted = load_reward_index(training)
    assert adapted["reward_config"]["base_coefficient"] == 1.
    assert adapted["reward_config"].get("final_normalization", "none") == "none"
    for episode, entry in zip(manifest["episodes"], adapted["episodes"]):
        with np.load(entry["reward_path"]) as arrays:
            expected = fused_reward_arrays(episode, adapted["reward_config"], {k: arrays[k] for k in arrays.files})
            np.testing.assert_array_equal(arrays["final_reward"], expected["final_reward"])
            assert "final_reward_scale" not in arrays
    assert all(p.read_bytes() == data for p, data in protected.items())
    training.raw["reward"]["base_coefficient"] = .5
    with pytest.raises(ValueError, match="conflict"):
        load_reward_index(training)


@pytest.mark.skipif(not os.environ.get("W1_SCREEN_SCRIPT"), reason="W-1 checkout not provided")
def test_w1_reads_generated_independent_base_reward(configured):
    from test_rewards import FakeAnnotator
    from vla_rynn_iql.rewards import annotate_manifest
    prepared = prepare_dataset(configured).manifest
    _save_annotations(configured, json.loads(prepared.read_text()))
    configured.raw["reward"].update(recipe(alpha=.5, base_coefficient=1., shaping_weight=.1,
                                          final_normalization="none"), stage_exponent=2.)
    annotate_manifest(configured, FakeAnnotator())
    load_reward_index(configured)
    work = Path(configured.raw["paths"]["work_dir"])
    rewards = work / "rewards/reward_manifest.json"
    output = work / "screen.json"
    result = subprocess.run([sys.executable, os.environ["W1_SCREEN_SCRIPT"],
        "--dataset-manifest", str(prepared), "--reward-manifest", str(rewards),
        "--gamma", ".9", "--split", "all", "--output", str(output)], capture_output=True, text=True)
    assert result.returncode == 0, result.stdout + result.stderr
    report = json.loads(output.read_text())
    assert report["handoff_summary"]["count"] == 1
    assert report["normalization_scales"]["mean"] == 1.
    assert report["reward_manifest_sha256"] == sha256_file(rewards)


# Server-branch coverage merged into main.

def test_server_does_not_offer_semantic_coefficient_override(tmp_path):
    import yaml
    from vla_rynn_iql.server_config import load_server_config
    root = Path(__file__).resolve().parents[1]
    raw = yaml.safe_load((root / "configs/server_pipeline.yaml").read_text())
    raw["overrides"]["reward"]["base_coefficient"] = 1.
    path = tmp_path / "server.yaml"
    path.write_text(yaml.safe_dump(raw))
    with pytest.raises(ValueError, match="base_coefficient|reward overrides"):
        load_server_config(path)
