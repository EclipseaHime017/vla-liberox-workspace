from __future__ import annotations

from pathlib import Path
import json
import shutil

import numpy as np

from vla_rynn_iql.data import prepare_dataset
from vla_rynn_iql.replay import ReplayDataset
from vla_rynn_iql.rewards import annotate_manifest


class FakeAnnotator:
    metadata = {"provider": "fake", "revision": "test"}

    def predict(self, prompt, frames):
        values = np.arange(len(frames), 0, -1, dtype=np.float32)
        return {
            "absolute_temporal_distance_seconds": values[:, None],
            "absolute_value_entropy_nats": np.zeros((len(frames), 1), np.float32),
            "absolute_value_logits": np.zeros((len(frames), 1, 256), np.float32),
            "relative_temporal_distance_seconds": np.zeros(len(frames), np.float32),
            "relative_value_logits": np.zeros((len(frames), 256), np.float32),
        }

    def analyze(self, prompt, frames):
        return {
            "generated_text": "- Match: Yes\n- Success: No",
            "generated_token_ids": [1, 2],
            "parsed_for_display": {"match": "Yes", "success": "No"},
        }


def _stats(dim: int) -> dict[str, list[float]]:
    return {"q01": [0.0] * dim, "q99": [1.0] * dim}


def test_transient_raw_done_does_not_terminate_an_earlier_chunk(configured):
    source = Path(configured.section("paths")["dataset_sources"][0])
    trajectory = next(source.rglob("branch/episodes/episode_000/trajectory.npz"))
    with np.load(trajectory, allow_pickle=False) as archive:
        arrays = {key: archive[key] for key in archive.files}
    arrays["done"] = arrays["done"].copy()
    arrays["done"][8] = True
    np.savez_compressed(trajectory, **arrays)

    prepare_dataset(configured)
    annotate_manifest(configured, FakeAnnotator())
    replay = ReplayDataset(configured, _stats(7), _stats(8), split="train")
    def replay_index(start: int) -> int:
        return next(
            index for index, (episode, chunk_index, _) in enumerate(replay.items)
            if episode["run_id"] == "branch"
            and episode["chunks"][chunk_index]["start"] == start
        )

    assert replay[replay_index(5)]["bootstrap_mask"].item() == 1.0
    assert replay[replay_index(13)]["bootstrap_mask"].item() == 0.0


def test_replay_exposes_variable_duration_transition_metadata(configured):
    prepare_dataset(configured)
    annotate_manifest(configured, FakeAnnotator())
    replay = ReplayDataset(configured, _stats(7), _stats(8), split="train")
    item = next(
        replay[index] for index, (episode, chunk_index, _) in enumerate(replay.items)
        if episode["run_id"] == "branch"
        and episode["chunks"][chunk_index]["interrupted"]
    )
    assert item["chunk_length"].item() == 5
    assert item["action_mask"].tolist() == [True] * 5 + [False] * 3
    assert item["action_source"] == "policy"
    assert item["transition_type"] == "policy_interrupted"
    assert item["bootstrap_mask"].item() == 1.0


def test_replay_deduplicates_sibling_interrupted_prefixes(configured):
    source = Path(configured.section("paths")["dataset_sources"][0])
    branch_run = next(source.rglob("branch/run.json")).parent
    sibling_run = branch_run.parent / "branch-sibling"
    shutil.copytree(branch_run, sibling_run)
    run = json.loads((sibling_run / "run.json").read_text(encoding="utf-8"))
    run["id"] = "branch-sibling"
    (sibling_run / "run.json").write_text(json.dumps(run), encoding="utf-8")

    prepare_dataset(configured)
    annotate_manifest(configured, FakeAnnotator())
    replay = ReplayDataset(configured, _stats(7), _stats(8), split="train")
    interrupted = [
        (episode["root_run_id"], episode["chunks"][chunk_index]["start"],
         episode["chunks"][chunk_index]["end"])
        for episode, chunk_index, _ in replay.items
        if episode["chunks"][chunk_index]["interrupted"]
    ]
    assert interrupted == [("root", 0, 5)]


def test_replay_deduplicates_full_copied_prefix_against_parent(configured):
    source = Path(configured.section("paths")["dataset_sources"][0])
    run_json = next(source.rglob("branch/run.json"))
    run = json.loads(run_json.read_text(encoding="utf-8"))
    run["resume_step"] = 13
    run_json.write_text(json.dumps(run), encoding="utf-8")
    trajectory = run_json.parent / "episodes" / "episode_000" / "trajectory.npz"
    with np.load(trajectory, allow_pickle=False) as archive:
        arrays = {key: archive[key] for key in archive.files}
    sources = np.asarray(["policy"] * len(arrays["env_action"]), dtype="<U32")
    sources[13:] = "human"
    arrays["action_source"] = sources
    np.savez_compressed(trajectory, **arrays)

    prepare_dataset(configured)
    annotate_manifest(configured, FakeAnnotator())
    replay = ReplayDataset(configured, _stats(7), _stats(8), split="train")
    copied_zero_to_eight = [
        (episode["run_id"], episode["chunks"][chunk_index]["start"],
         episode["chunks"][chunk_index]["end"])
        for episode, chunk_index, _ in replay.items
        if episode["root_run_id"] == "root"
        and episode["chunks"][chunk_index]["start"] == 0
        and episode["chunks"][chunk_index]["end"] == 8
    ]
    assert copied_zero_to_eight == [("root", 0, 8)]
    assert any(
        episode["run_id"] == "branch"
        and episode["chunks"][chunk_index]["start"] == 8
        and episode["chunks"][chunk_index]["end"] == 13
        for episode, chunk_index, _ in replay.items
    )
