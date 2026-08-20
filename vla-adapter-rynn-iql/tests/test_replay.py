from __future__ import annotations

from pathlib import Path

import numpy as np

from vla_rynn_iql.data import prepare_dataset
from vla_rynn_iql.replay import ReplayDataset
from vla_rynn_iql.rewards import annotate_manifest


class FakeAnnotator:
    metadata = {"provider": "fake", "revision": "test"}

    def predict(self, prompt, frames):
        values = np.arange(len(frames), 0, -1, dtype=np.float32)
        return values, np.zeros_like(values)


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
