from __future__ import annotations

import json
from types import SimpleNamespace

import numpy as np
import torch

from vla_rynn_iql.data import prepare_dataset
from vla_rynn_iql.rewards import (
    RynnValueAnnotator, annotate_manifest, annotation_windows, shaped_chunk_reward,
    validate_rynnvalue_config_contract, validate_rynnvalue_runtime_dtype,
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


def test_sparse_reward_uses_only_debounced_terminal(configured):
    configured.raw["reward"]["shaping_weight"] = 0.0
    prepare_dataset(configured)
    index_path = annotate_manifest(configured, FakeAnnotator())
    index = json.loads(index_path.read_text(encoding="utf-8"))
    branch = next(item for item in index["episodes"] if item["run_id"] == "branch")
    with np.load(branch["annotation_path"], allow_pickle=False) as annotation:
        rewards = annotation["chunk_reward"]
    gamma = float(configured.section("reward")["gamma"])
    # The final five-step chunk has four -1 costs followed by the confirmed
    # terminal's zero cost. Raw done=True began four steps earlier.
    expected = -sum(gamma ** offset for offset in range(4))
    assert np.isclose(rewards[-1], expected)


def test_official_prefix_shape_reduction_uses_last_slot():
    values = torch.arange(8, dtype=torch.float32).reshape(2, 4)
    reduced = RynnValueAnnotator._last_slot(values, sample_count=2)
    torch.testing.assert_close(reduced, torch.tensor([3.0, 7.0]))


def _fake_rynn_config():
    return SimpleNamespace(
        model_type="rynn_value_lang",
        text_config=SimpleNamespace(hidden_size=2560),
        value_token_repeat=8,
        value_tokenizer_config=SimpleNamespace(bins=256),
        value_head_config=SimpleNamespace(head_type="bro"),
        num_value_heads=1,
    )


def test_rynnvalue_contract_uses_repeated_qwen_hidden_states():
    contract = validate_rynnvalue_config_contract(
        _fake_rynn_config(), SimpleNamespace(value_token_repeat=8)
    )
    assert contract["qwen_hidden_size"] == 2560
    assert contract["value_head_input_size"] == 20480


def test_rynnvalue_contract_rejects_processor_repeat_mismatch():
    with np.testing.assert_raises_regex(RuntimeError, "repeat mismatch"):
        validate_rynnvalue_config_contract(
            _fake_rynn_config(), SimpleNamespace(value_token_repeat=1)
        )


class _TinyRynnValue(torch.nn.Module):
    def __init__(self, dtype):
        super().__init__()
        projection = torch.nn.Module()
        projection.input_layer = torch.nn.Linear(16, 4, dtype=dtype)
        head = torch.nn.Module()
        head.proj = projection
        self.value_heads = torch.nn.ModuleList([head])
        self.backbone = torch.nn.Linear(4, 4, dtype=dtype)


def test_rynnvalue_runtime_rejects_float32_head_for_bfloat16_model():
    model = _TinyRynnValue(torch.float32)
    with np.testing.assert_raises_regex(RuntimeError, "not converted"):
        validate_rynnvalue_runtime_dtype(model, torch.bfloat16, 16)
    model.to(dtype=torch.bfloat16)
    result = validate_rynnvalue_runtime_dtype(model, torch.bfloat16, 16)
    assert result["value_head_dtype"] == "torch.bfloat16"


def test_cuda_oom_reports_stage_without_cpu_fallback():
    def fail():
        raise RuntimeError("CUDA out of memory while allocating tensor")

    with np.testing.assert_raises_regex(RuntimeError, "reward annotation failed"):
        run_cuda_stage("reward annotation", fail)
