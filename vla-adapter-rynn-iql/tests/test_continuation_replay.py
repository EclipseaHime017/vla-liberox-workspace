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
def test_producer_segments_keep_full_task_prompt_for_actor_and_reward(configured, flip_horizontal):
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
        dict(start=0, end=5, origin_run_id="root"),
        dict(start=5, end=10, origin_run_id="earlier-human"),
        dict(start=10, end=22, origin_run_id="branch")]
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
        assert set(actual) == {"do the task"}
        expected = image[::-1, ::-1] if flip_horizontal else image[::-1]
        np.testing.assert_array_equal(dataset[0]["agent_image"].numpy(), expected)
        np.testing.assert_array_equal(dataset[0]["wrist_image"].numpy(), expected + 1)
        generator = torch.Generator().manual_seed(7)
        for _ in range(10):
            batch = _sample_batch(dataset, 4, generator)
            assert set(batch["prompt"]) == {"do the task"}


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


@pytest.mark.parametrize("source_kind", ["run", "trajectory", "segment"])
@pytest.mark.parametrize("matches_task", [False, True])
def test_old_recordings_never_silently_relabel_custom_actions(configured, source_kind, matches_task):
    source = Path(configured.section("paths")["dataset_sources"][0])
    path = next(source.rglob("branch/run.json"))
    run = json.loads(path.read_text())
    instruction = "do the task" if matches_task else "a different task"
    metadata = dict(task="do the task")
    if source_kind == "run":
        run["policy_instruction"] = instruction
        path.write_text(json.dumps(run))
    elif source_kind == "trajectory":
        metadata["policy_instruction"] = instruction
    else:
        metadata["trajectory_segments"] = [
            dict(start=0, end=22, origin_run_id="branch", open_loop_steps=8,
                 policy_instruction=instruction)]
    trajectory = path.parent / "episodes/episode_000/trajectory.npz"
    with np.load(trajectory, allow_pickle=False) as archive:
        arrays = dict(archive)
    arrays["metadata_json"] = np.asarray(json.dumps(metadata))
    np.savez_compressed(trajectory, **arrays)
    before = path.read_bytes(), trajectory.read_bytes()
    if matches_task:
        prepare_dataset(configured)
        dataset = ActionDataset(configured, _stats(7), _stats(8))
        assert {dataset[index]["prompt"] for index in range(len(dataset))} == {"do the task"}
        assert all("policy_instruction" not in chunk
                   for episode in dataset.manifest["episodes"] for chunk in episode["chunks"])
    else:
        with pytest.raises(ValueError, match="custom instruction differs from the full task prompt"):
            prepare_dataset(configured)
    assert (path.read_bytes(), trajectory.read_bytes()) == before


@pytest.mark.parametrize("section", ["chunks", "evaluation_chunks", "trajectory_segments"])
def test_prepared_data_rejects_old_custom_language(configured, section):
    prepared = prepare_dataset(configured)
    manifest = load_manifest(configured)
    episode = manifest["episodes"][0]
    if section == "trajectory_segments":
        episode[section] = [dict(policy_instruction="different task")]
    else:
        episode[section][0]["policy_instruction"] = "different task"
    prepared.manifest.write_text(json.dumps(manifest))
    before = prepared.manifest.read_bytes()
    with pytest.raises(ValueError, match="custom instruction differs from the full task prompt"):
        load_manifest(configured)
    assert prepared.manifest.read_bytes() == before
