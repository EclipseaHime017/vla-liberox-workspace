from __future__ import annotations

import numpy as np
import torch

from vla_rynn_iql.data import prepare_dataset
from vla_rynn_iql.rewards import (
    RynnValueAnnotator, annotate_manifest, annotation_windows, shaped_chunk_reward,
)
from vla_rynn_iql.runtime import run_cuda_stage


class FakeAnnotator:
    metadata = {"provider": "fake", "revision": "test"}

    def predict(self, prompt, frames):
        values = np.arange(len(frames), 0, -1, dtype=np.float32)
        return values, np.zeros_like(values)


class CountingAnnotator(FakeAnnotator):
    def __init__(self):
        self.calls = 0

    def predict(self, prompt, frames):
        self.calls += 1
        return super().predict(prompt, frames)


def test_overlapping_windows_cover_tail():
    windows = annotation_windows(130, 64, 8)
    assert windows[0] == (0, 64)
    assert windows[-1] == (66, 130)


def test_chunk_reward_uses_gamma_to_actual_length():
    reward = shaped_chunk_reward(
        np.asarray([False, False, True]), 0, 3, value_start=3, value_end=0,
        gamma=0.9, shaping_weight=0.1,
    )
    assert np.isclose(reward, -1.0 - 0.9 + 0.3)


def test_fake_annotation_pipeline(configured):
    prepare_dataset(configured)
    result = annotate_manifest(configured, FakeAnnotator())
    assert result.is_file()


def test_reward_cache_resumes_without_reannotation(configured):
    prepare_dataset(configured)
    annotator = CountingAnnotator()
    annotate_manifest(configured, annotator)
    first_calls = annotator.calls
    annotate_manifest(configured, annotator)
    assert first_calls > 0
    assert annotator.calls == first_calls


def test_official_prefix_shape_reduction_uses_last_slot():
    values = torch.arange(8, dtype=torch.float32).reshape(2, 4)
    reduced = RynnValueAnnotator._last_slot(values, sample_count=2)
    torch.testing.assert_close(reduced, torch.tensor([3.0, 7.0]))


def test_cuda_oom_reports_stage_without_cpu_fallback():
    def fail():
        raise RuntimeError("CUDA out of memory while allocating tensor")

    with np.testing.assert_raises_regex(RuntimeError, "reward annotation failed"):
        run_cuda_stage("reward annotation", fail)
