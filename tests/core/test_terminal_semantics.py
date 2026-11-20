"""W-5 opt-in endpoint semantics, without models or remote access."""
import json
from pathlib import Path

import numpy as np
import pytest

from core.config import load_train_config
from core.data import prepare_dataset
from train.replay import ReplayDataset
from rewards.rynnvalue.annotate import annotate_manifest
from rewards.pbrs import shaping_rewards
from rewards.manifest import load_reward_index, reward_derivation_config
from core.io import sha256_file
from test_replay import FakeAnnotator, _stats


@pytest.mark.parametrize("include_post_success", [True, False])
def test_failed_end_bootstraps_only_when_enabled(configured, include_post_success):
    configured.raw["data"]["include_post_success"] = include_post_success
    prepare_dataset(configured)
    annotate_manifest(configured, FakeAnnotator())
    legacy = ReplayDataset(configured, _stats(7), _stats(8))
    index = load_reward_index(configured)
    records = []
    for i, (episode, _, start) in enumerate(legacy.items):
        end = legacy.transition_end(episode, start)
        endpoint = (episode["terminal_step"] + 1 if not include_post_success and episode["terminal_step"] is not None
                    else episode["recorded_action_count"])
        if end == endpoint:
            records.append((i, episode["terminal_step"] is not None))
    assert len(records) == 2
    assert all(legacy[i]["bootstrap_mask"].item() == 0 for i, _ in records)
    configured.raw["data"]["bootstrap_failed_end"] = True
    opted = ReplayDataset(configured, _stats(7), _stats(8), reward_index=index)
    for i, successful in records:
        sample = opted[i]
        assert sample["bootstrap_mask"].item() == (0 if successful else 1)
        assert sample["next_proprio"].shape == legacy[i]["next_proprio"].shape
        np.testing.assert_array_equal(sample["next_pixels"], legacy[i]["next_pixels"])


def test_success_potential_zero_after_confirming_action():
    done = np.array([False, False, True, True])
    potential = -np.array([8., 6., 4., 2., 0.])
    zeroed = shaping_rewards(potential, done, .9, zero_success_potential=True)
    plain = shaping_rewards(potential, done, .9)
    np.testing.assert_array_equal(zeroed[:2], plain[:2])
    assert zeroed[2] == 4  # the observation after the confirming action has phi = 0
    assert zeroed[3] == 0
    failure = np.zeros(4, bool)
    np.testing.assert_array_equal(shaping_rewards(potential, failure, .9, zero_success_potential=True),
                                  shaping_rewards(potential, failure, .9))


def test_false_preserves_recipe_and_config_accepts_boolean(configured):
    recipe = reward_derivation_config(configured.raw["reward"])
    explicit = dict(configured.raw["reward"], zero_success_potential=False)
    assert reward_derivation_config(explicit) == recipe
    assert reward_derivation_config(dict(explicit, zero_success_potential=True))["zero_success_potential"] is True
    for section, key in (("data", "bootstrap_failed_end"), ("reward", "zero_success_potential")):
        assert load_train_config(configured.path, overrides={section: {key: True}}).section(section)[key] is True
        with pytest.raises(TypeError):
            load_train_config(configured.path, overrides={section: {key: "true"}})


def test_pinned_reward_adapts_success_potential_without_touching_source(configured):
    manifest = json.loads(prepare_dataset(configured).manifest.read_text())
    annotate_manifest(configured, FakeAnnotator())
    original = load_reward_index(configured)
    originals = {item["reward_path"]: Path(item["reward_path"]).read_bytes() for item in original["episodes"]}
    pinned = Path(original["episodes"][0]["reward_path"]).parent / "reward_manifest.json"
    configured.raw["reward"].update(manifest_path=str(pinned), manifest_sha256=sha256_file(pinned),
                                   version_id="test", zero_success_potential=True)
    adapted = load_reward_index(configured)
    assert adapted["reward_config"]["zero_success_potential"] is True
    by_id = {entry["run_id"]: entry for entry in adapted["episodes"]}
    for episode in manifest["episodes"]:
        rid = episode["run_id"]
        old_entry = next(e for e in original["episodes"] if e["run_id"] == rid)
        with np.load(old_entry["reward_path"]) as old, np.load(by_id[rid]["reward_path"]) as new:
            if episode["terminal_step"] is None:
                np.testing.assert_array_equal(new["final_reward"], old["final_reward"])
            else:
                tail = new["pbrs_shaping_reward"][episode["terminal_step"] + 1:]
                assert len(tail)
                np.testing.assert_array_equal(tail, 0)
    assert all(Path(path).read_bytes() == content for path, content in originals.items())


def test_fusion_success_potential_matches_rynnvalue_zero_alpha(configured):
    from rewards.fusion import fused_reward_arrays
    manifest = json.loads(prepare_dataset(configured).manifest.read_text())
    episode = next(ep for ep in manifest["episodes"] if ep["terminal_step"] is not None)
    observations = episode["recorded_action_count"] + 1
    signals = {"absolute_temporal_distance_seconds": np.ones((observations, 1), dtype=np.float32) * 4}
    recipe = dict(reward_derivation_config(configured.raw["reward"]), source="final",
                  alpha=0., fusion_mode="additive", zero_success_potential=True)
    arrays = fused_reward_arrays(episode, recipe, signals)
    np.testing.assert_array_equal(arrays["pbrs_shaping_reward"][episode["terminal_step"] + 1:], 0)


def test_resume_rejects_changed_failed_bootstrap(configured, tmp_path, monkeypatch):
    from types import SimpleNamespace
    import torch
    from train.trainer import _restore_checkpoint
    from core.registry import model_contract
    from rewards.manifest import reward_manifest_digest
    configured.raw["data"]["bootstrap_failed_end"] = True
    metadata = {"algorithm": "iql", "model_contract": model_contract(configured.raw["model"]),
                "base_checkpoint": configured.raw["model"]["base_checkpoint"], "stats_key": "test",
                "dataset_sha256": "dataset", "reward_sha256": reward_manifest_digest({})}
    (tmp_path / "checkpoint.json").write_text(json.dumps(metadata))
    def refuse(*args, **kwargs):
        pytest.fail("must reject before loading checkpoint weights")
    monkeypatch.setattr(torch, "load", refuse)
    with pytest.raises(ValueError, match="failed-end bootstrap"):
        _restore_checkpoint(tmp_path, SimpleNamespace(stats_key="test"), None, None, None,
                            torch.device("cpu"), configured, {"episodes": []}, {})
