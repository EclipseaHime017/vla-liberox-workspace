from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest
import torch
import yaml

from vla_rynn_iql.config import load_train_config
from vla_rynn_iql.data import prepare_dataset
from vla_rynn_iql.replay import ReplayDataset
from vla_rynn_iql.rewards import annotate_manifest
from vla_rynn_iql.server_config import (
    ReplayCacheConfig,
    load_server_config,
    validate_global_batch,
)
from vla_rynn_iql.server_replay import (
    CachedReplayDataset,
    DeterministicDistributedBatchSampler,
    build_replay_cache,
    validate_replay_cache,
)


class FakeAnnotator:
    metadata = {"provider": "fake", "revision": "server-test"}

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
            "generated_token_ids": [1],
            "parsed_for_display": {"match": "Yes", "success": "No"},
        }


def _stats(dim: int) -> dict[str, list[float]]:
    return {"q01": [0.0] * dim, "q99": [1.0] * dim}


def _server_yaml(tmp_path: Path, base: Path) -> Path:
    path = tmp_path / "server.yaml"
    path.write_text(yaml.safe_dump({
        "schema_version": 1,
        "base_config": str(base),
        "pipeline_root": str(tmp_path / "pipelines"),
        "environments": {
            "prepare": "vla-liberox",
            "annotate": "rynnvalue-reward",
            "train": "vla-liberox",
        },
        "selection": {
            "task_id": "task",
            "mode": "all",
            "seed": 7,
            "source_types": ["inference", "manual", "policy_requery"],
            "outcomes": ["success", "failure"],
            "size": None,
            "quotas": [],
        },
        "overrides": {
            "paths": {}, "data": {}, "reward": {}, "vla": {}, "iql": {}, "logging": {},
        },
        "distributed": {
            "gpu_ids": [0, 1],
            "backend": "nccl",
            "zero_stage": 1,
            "timeout_seconds": 60,
            "data_workers_per_rank": 0,
            "prefetch_factor": 2,
            "pin_memory": False,
            "persistent_workers": False,
        },
        "replay_cache": {
            "enabled": True,
            "root": str(tmp_path / "cache"),
            "rebuild": False,
        },
    }, sort_keys=False), encoding="utf-8")
    return path


def test_server_config_rejects_gpu_and_global_batch_errors(configured, tmp_path):
    path = _server_yaml(tmp_path, configured.path)
    server = load_server_config(path)
    raw = configured.raw.copy()
    raw["iql"] = dict(configured.raw["iql"])
    raw["iql"]["micro_batch_size"] = 8
    assert validate_global_batch(raw, server.distributed) == (8, 4)
    raw["iql"]["micro_batch_size"] = 3
    with pytest.raises(ValueError, match="divisible"):
        validate_global_batch(raw, server.distributed)

    invalid = yaml.safe_load(path.read_text(encoding="utf-8"))
    invalid["distributed"]["gpu_ids"] = [0, 0]
    path.write_text(yaml.safe_dump(invalid), encoding="utf-8")
    with pytest.raises(ValueError, match="unique"):
        load_server_config(path)

    for key, value, message in (
        ("backend", "gloo", "backend=nccl"),
        ("zero_stage", 2, "zero_stage=1"),
    ):
        path = _server_yaml(tmp_path, configured.path)
        invalid = yaml.safe_load(path.read_text(encoding="utf-8"))
        invalid["distributed"][key] = value
        path.write_text(yaml.safe_dump(invalid), encoding="utf-8")
        with pytest.raises(ValueError, match=message):
            load_server_config(path)

    path = _server_yaml(tmp_path, configured.path)
    invalid = yaml.safe_load(path.read_text(encoding="utf-8"))
    invalid["distributed"]["persistent_workers"] = True
    path.write_text(yaml.safe_dump(invalid), encoding="utf-8")
    with pytest.raises(ValueError, match="data_workers_per_rank"):
        load_server_config(path)


def test_server_yaml_rejects_duplicate_and_unknown_keys(configured, tmp_path):
    path = _server_yaml(tmp_path, configured.path)
    path.write_text("schema_version: 1\nschema_version: 1\n", encoding="utf-8")
    with pytest.raises(Exception, match="duplicate key"):
        load_server_config(path)

    path = _server_yaml(tmp_path, configured.path)
    raw = yaml.safe_load(path.read_text(encoding="utf-8"))
    raw["distributed"]["unknown"] = 1
    path.write_text(yaml.safe_dump(raw), encoding="utf-8")
    with pytest.raises(ValueError, match="Unknown distributed"):
        load_server_config(path)


def test_distributed_sampler_preserves_global_sequence_across_world_sizes():
    one = DeterministicDistributedBatchSampler(
        dataset_size=97, global_batch_size=8, rank=0, world_size=1,
        seed=11, start_step=0, end_step=5,
    )
    rank_zero = DeterministicDistributedBatchSampler(
        dataset_size=97, global_batch_size=8, rank=0, world_size=2,
        seed=11, start_step=0, end_step=5,
    )
    rank_one = DeterministicDistributedBatchSampler(
        dataset_size=97, global_batch_size=8, rank=1, world_size=2,
        seed=11, start_step=0, end_step=5,
    )
    assert list(one) == [left + right for left, right in zip(rank_zero, rank_one)]

    resumed = DeterministicDistributedBatchSampler(
        dataset_size=97, global_batch_size=8, rank=0, world_size=1,
        seed=11, start_step=3, end_step=5,
    )
    assert list(resumed) == list(one)[3:]


def test_mmap_cache_matches_existing_replay_dataset(configured, tmp_path):
    prepare_dataset(configured)
    annotate_manifest(configured, FakeAnnotator())
    cache_config = ReplayCacheConfig(True, tmp_path / "server-cache", False)
    cache_path, reused = build_replay_cache(configured, cache_config)
    assert reused is False
    assert validate_replay_cache(cache_path, configured)["item_count"] > 0
    second_path, reused = build_replay_cache(configured, cache_config)
    assert second_path == cache_path
    assert reused is True

    ordinary = ReplayDataset(configured, _stats(7), _stats(8), split="train")
    cached = CachedReplayDataset(configured, cache_path, _stats(7), _stats(8))
    assert len(cached) == len(ordinary)
    tensor_keys = {
        "pixels", "next_pixels", "proprio", "next_proprio", "actions",
        "action_mask", "reward", "bootstrap_mask", "chunk_length",
        "agent_image", "wrist_image",
    }
    for index in range(len(ordinary)):
        before, after = ordinary[index], cached[index]
        for key in tensor_keys:
            assert torch.equal(before[key], after[key]), (index, key)
        for key in (
            "prompt", "run_id", "start", "action_source", "transition_type", "interrupted",
        ):
            assert before[key] == after[key], (index, key)


def test_ui_training_entry_remains_the_single_gpu_script():
    workspace = Path(__file__).resolve().parents[2]
    service = (
        workspace
        / "liberox-vla-adapter-terminal/backend/app/services/offline_job_service.py"
    ).read_text(encoding="utf-8")
    assert '"scripts" / "train_iql.py"' in service
    assert "train_iql_distributed.py" not in service
    train_script = (
        workspace / "vla-adapter-rynn-iql/scripts/train_iql.py"
    ).read_text(encoding="utf-8")
    single_gpu_training = (
        workspace / "vla-adapter-rynn-iql/src/vla_rynn_iql/training.py"
    ).read_text(encoding="utf-8")
    assert "distributed_training" not in train_script
    assert "train_iql_distributed" not in single_gpu_training
