import json
from pathlib import Path

import pytest
import torch
import numpy as np

from vla_rynn_iql.base_models import model_contract
from vla_rynn_iql.data import add_episode_chunks, iter_unique_replay_chunks, load_manifest, prepare_dataset
from vla_rynn_iql.replay import ActionDataset, ReplayDataset
from vla_rynn_iql.rewards import annotate_manifest
from vla_rynn_iql.training import _sample_batch
from test_replay import FakeAnnotator, _stats


@pytest.mark.parametrize("flip_horizontal", [False, True])
def test_instruction_segments_train_actor_but_keep_full_task_reward(configured, flip_horizontal):
    source = Path(configured.section("paths")["dataset_sources"][0])
    contract = model_contract(configured.section("model"))
    contract["observation"] = dict(image_flip_vertical=True, image_flip_horizontal=flip_horizontal)
    configured.raw["model"]["contract"] = contract
    image = np.arange(16 * 16 * 3, dtype=np.uint8).reshape(16, 16, 3)
    for observations in source.rglob("trajectory_observations.npz"):
        with np.load(observations, allow_pickle=False) as arrays:
            shape = arrays["agentview_image"].shape
        np.savez_compressed(observations, agentview_image=np.broadcast_to(image, shape),
                            wrist_image=np.broadcast_to(image + 1, shape))
    path = next(source.rglob("branch/run.json"))
    run = json.loads(path.read_text())
    run["trajectory_segments"] = [
        dict(start=0, end=5, origin_run_id="root", policy_instruction="first step"),
        dict(start=5, end=10, origin_run_id="earlier-human", policy_instruction="second step"),
        dict(start=10, end=22, origin_run_id="branch", policy_instruction="finish task")]
    run["resume_step"], run["handoff_steps"] = 10, [5, 10]
    path.write_text(json.dumps(run))
    prepare_dataset(configured)
    manifest = load_manifest(configured)
    episode = next(ep for ep in manifest["episodes"] if ep["run_id"] == "branch")
    assert episode["prompt"] == "do the task"
    assert any(chunk["start"] == 10 for chunk in episode["chunks"])
    assert all(not chunk["start"] < 10 < chunk["end"] for chunk in episode["chunks"])
    class GoalAnnotator(FakeAnnotator):
        def predict(self, prompt, frames):
            assert prompt == "do the task"
            return super().predict(prompt, frames)
        def analyze(self, prompt, frames):
            assert prompt == "do the task"
            return super().analyze(prompt, frames)
    annotate_manifest(configured, GoalAnnotator())
    for kind in (ActionDataset, ReplayDataset):
        dataset = kind(configured, _stats(7), _stats(8), split="train")
        actual = [dataset[index]["prompt"] for index in range(len(dataset))]
        assert "second step" in actual and "finish task" in actual
        expected = image[::-1, ::-1] if flip_horizontal else image[::-1]
        np.testing.assert_array_equal(dataset[0]["agent_image"].numpy(), expected)
        np.testing.assert_array_equal(dataset[0]["wrist_image"].numpy(), expected + 1)
        generator = torch.Generator().manual_seed(7)
        for _ in range(10):
            batch = _sample_batch(dataset, 4, generator)
            assert len(set(batch["prompt"])) == 1


def test_exact_origin_dedup_keeps_distinct_human_attempts_and_short_interruptions():
    def episode(run_id, origin, copied, end=8):
        return dict(run_id=run_id, root_run_id="root", kind="branch", split="train",
            recorded_action_count=end, terminal_step=None,
            chunks=[dict(start=0, end=end, length=end, action_source="human",
                         origin_run_id=origin, copied_prefix=copied)])
    source = episode("z-producer", "z-producer", False)
    copied = episode("a-copy", "z-producer", True)
    distinct = episode("other", "other", False)
    short = episode("short-copy", "z-producer", True, 5)
    chosen = list(iter_unique_replay_chunks([copied, source, distinct, short]))
    assert {ep["run_id"] for ep, _ in chosen} == {"z-producer", "other", "short-copy"}


@pytest.mark.parametrize("boundary", [4.5, True, -1, 11])
def test_invalid_handoff_boundaries_are_rejected(boundary):
    episode = dict(run_id="run", kind="original", resume_step=None, terminal_step=None,
        recorded_action_count=10, handoff_steps=[boundary],
        action_source_segments=[dict(start=0, end=10, action_source="policy")])
    with pytest.raises(ValueError, match="integer recording boundaries"):
        add_episode_chunks(episode, 8)
