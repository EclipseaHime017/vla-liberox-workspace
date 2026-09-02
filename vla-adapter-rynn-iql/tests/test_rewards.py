from __future__ import annotations

import json
import shutil
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import torch
import yaml

import vla_rynn_iql.rewards as rewards_module
from vla_rynn_iql.data import load_manifest, prepare_dataset
from vla_rynn_iql.config import load_train_config
from vla_rynn_iql.io import sha256_file, stable_hash
from vla_rynn_iql.rewards import (
    ANNOTATION_SCHEMA_VERSION,
    RynnValueAnnotator, annotate_manifest, chunk_reward_components,
    load_reward_index, shaped_chunk_reward,
    validate_rynnvalue_config_contract, validate_rynnvalue_runtime_dtype,
)
from vla_rynn_iql.runtime import run_cuda_stage


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
            "parsed_for_display": {"description": None, "match": "Yes", "success": "No"},
        }


class CountingAnnotator(FakeAnnotator):
    def __init__(self):
        self.calls = 0

    def predict(self, prompt, frames):
        self.calls += 1
        return super().predict(prompt, frames)


def test_chunk_reward_treats_the_action_chunk_as_one_macro_transition():
    sparse, shaping, reward = chunk_reward_components(
        np.asarray([False, False, False]), 0, 3, value_start=3, value_end=1,
        gamma=0.9, shaping_weight=0.1,
    )
    assert np.isclose(sparse, -1.0)
    assert np.isclose(shaping, 2.1)
    assert np.isclose(reward, -0.79)

    terminal_sparse, terminal_shaping, terminal_reward = chunk_reward_components(
        np.asarray([False, False, True]), 0, 3, value_start=3, value_end=0,
        gamma=0.9, shaping_weight=0.1,
    )
    assert np.isclose(terminal_sparse, 0.0)
    assert np.isclose(terminal_shaping, 3.0)
    assert np.isclose(terminal_reward, 0.3)
    assert np.isclose(
        shaped_chunk_reward(
            np.asarray([False, False, False]), 0, 3, value_start=3, value_end=1,
            gamma=0.9, shaping_weight=0.1,
        ),
        reward,
    )


def test_chunk_reward_can_accumulate_primitive_step_costs():
    sparse, shaping, reward = chunk_reward_components(
        np.asarray([False, False, False]), 0, 3, value_start=3, value_end=1,
        gamma=0.9, shaping_weight=0.1, accumulate_primitive_steps=True,
    )
    assert np.isclose(sparse, -(1.0 + 0.9 + 0.9**2))
    assert np.isclose(shaping, 3.0 - 0.9**3)
    assert np.isclose(reward, sparse + 0.1 * shaping)


def test_fake_annotation_pipeline(configured):
    prepare_dataset(configured)
    result = annotate_manifest(configured, FakeAnnotator())
    assert result.is_file()


def test_annotation_preserves_every_official_output_and_separates_pbrs(configured):
    prepare_dataset(configured)
    result = annotate_manifest(configured, FakeAnnotator())
    manifest = json.loads(result.read_text(encoding="utf-8"))
    assert manifest["schema_version"] == ANNOTATION_SCHEMA_VERSION
    episode = manifest["episodes"][0]
    assert episode["official_outputs"]["inference_method"] == "prefix_uniform_last_slot"
    assert episode["official_outputs"]["analysis"]["generated_token_ids"] == [1, 2]
    assert episode["pbrs_reward"]["array_keys"] == [
        "pbrs_shaping_reward", "pbrs_chunk_reward",
    ]
    with np.load(episode["annotation_path"], allow_pickle=False) as annotation:
        assert set(annotation.files) == {
            "boundary_steps",
            "absolute_temporal_distance_seconds",
            "absolute_value_entropy_nats",
            "absolute_value_logits",
            "relative_temporal_distance_seconds",
            "relative_value_logits",
            "pbrs_shaping_reward",
            "pbrs_chunk_reward",
        }
        assert annotation["absolute_temporal_distance_seconds"].ndim == 2
        assert annotation["absolute_value_logits"].shape[-1] == 256
        assert annotation["relative_value_logits"].shape[-1] == 256


def test_successful_branch_annotation_keeps_recorded_post_terminal_tail(configured):
    prepare_dataset(configured)
    result = annotate_manifest(configured, FakeAnnotator())
    manifest = json.loads(result.read_text(encoding="utf-8"))
    branch = next(item for item in manifest["episodes"] if item["run_id"] == "branch")
    with np.load(branch["annotation_path"], allow_pickle=False) as annotation:
        assert int(annotation["boundary_steps"][0]) == 0
        assert int(annotation["boundary_steps"][-1]) == 22
        assert len(annotation["pbrs_shaping_reward"]) == len(annotation["boundary_steps"]) - 1
        assert len(annotation["pbrs_chunk_reward"]) == len(annotation["boundary_steps"]) - 1


def test_reward_cache_resumes_without_reannotation(configured):
    prepare_dataset(configured)
    annotator = CountingAnnotator()
    annotate_manifest(configured, annotator)
    first_calls = annotator.calls
    annotate_manifest(configured, annotator)
    assert first_calls > 0
    assert annotator.calls == first_calls


def test_reward_mode_change_reuses_heads_and_rebuilds_manifest(
    configured, monkeypatch,
):
    prepare_dataset(configured)
    annotate_manifest(configured, FakeAnnotator())
    configured.raw["reward"]["accumulate_primitive_steps"] = True

    class UnexpectedModelLoad:
        def __init__(self, _config):
            raise AssertionError("reward-only relabel must not load RynnValue")

    monkeypatch.setattr(rewards_module, "RynnValueAnnotator", UnexpectedModelLoad)
    rebuilt = load_reward_index(configured)
    assert rebuilt["accumulate_primitive_steps"] is True
    assert rebuilt["reward_config"]["accumulate_primitive_steps"] is True
    with np.load(rebuilt["episodes"][0]["annotation_path"], allow_pickle=False) as arrays:
        assert arrays["pbrs_chunk_reward"][0] < -1.0


def test_manifest_without_explicit_reward_mode_is_rebuilt(configured, monkeypatch):
    prepare_dataset(configured)
    manifest_path = annotate_manifest(configured, FakeAnnotator())
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    del manifest["reward_config"]["accumulate_primitive_steps"]
    manifest_path.write_text(json.dumps(manifest), encoding="utf-8")

    class UnexpectedModelLoad:
        def __init__(self, _config):
            raise AssertionError("mode migration must reuse stored RynnValue heads")

    monkeypatch.setattr(rewards_module, "RynnValueAnnotator", UnexpectedModelLoad)
    rebuilt = load_reward_index(configured)
    assert rebuilt["reward_config"]["accumulate_primitive_steps"] is False


def test_v4_sidecars_reuse_official_outputs_and_recompute_macro_rewards(
    configured, monkeypatch,
):
    prepare_dataset(configured)
    first_path = annotate_manifest(configured, FakeAnnotator())
    first_index = json.loads(first_path.read_text(encoding="utf-8"))
    prepared = load_manifest(configured)
    prepared_by_id = {item["run_id"]: item for item in prepared["episodes"]}
    reward_cfg = configured.section("reward")

    for reward in first_index["episodes"]:
        episode = prepared_by_id[reward["run_id"]]
        episode_dir = Path(episode["trajectory_path"]).parent
        values_path = episode_dir / "rynnvalue_evaluation.npz"
        with np.load(reward["annotation_path"], allow_pickle=False) as source:
            arrays = {name: source[name] for name in source.files}
        # Simulate an obsolete v4 reduction. The migration must keep only the
        # official model heads and deterministically recompute these rewards.
        arrays["pbrs_shaping_reward"] = np.full_like(
            arrays["pbrs_shaping_reward"], 999.0,
        )
        arrays["pbrs_chunk_reward"] = np.full_like(
            arrays["pbrs_chunk_reward"], 999.0,
        )
        np.savez_compressed(values_path, **arrays)
        (episode_dir / "rynnvalue_evaluation.json").write_text(
            json.dumps({
                "schema_version": 4,
                "run_id": episode["run_id"],
                "trajectory_sha256": episode["trajectory_sha256"],
                "observations_sha256": episode["observations_sha256"],
                "values_sha256": sha256_file(values_path),
                "annotator": {
                    "model": reward_cfg["model"],
                    "requested_revision": reward_cfg["revision"],
                    "resolved_revision": reward_cfg["revision"],
                },
                "reward_config": reward_cfg,
                "official_outputs": reward["official_outputs"],
            }),
            encoding="utf-8",
        )

    shutil.rmtree(Path(configured.section("paths")["work_dir"]) / "rewards")
    shutil.rmtree(Path(configured.section("paths")["annotation_cache"]))

    class UnexpectedModelLoad:
        def __init__(self, _config):
            raise AssertionError("v4 official outputs should avoid loading RynnValue")

    monkeypatch.setattr(rewards_module, "RynnValueAnnotator", UnexpectedModelLoad)
    migrated_path = annotate_manifest(configured)
    migrated = json.loads(migrated_path.read_text(encoding="utf-8"))
    assert migrated["schema_version"] == ANNOTATION_SCHEMA_VERSION
    for reward in migrated["episodes"]:
        with np.load(reward["annotation_path"], allow_pickle=False) as arrays:
            assert not np.any(arrays["pbrs_shaping_reward"] == 999.0)
            assert not np.any(arrays["pbrs_chunk_reward"] == 999.0)


def test_tampered_reward_cache_is_recomputed(configured):
    prepare_dataset(configured)
    annotate_manifest(configured, CountingAnnotator())
    index = json.loads(
        (Path(configured.section("paths")["work_dir"]) / "rewards" / "reward_manifest.json")
        .read_text(encoding="utf-8")
    )
    with Path(index["episodes"][0]["annotation_path"]).open("ab") as stream:
        stream.write(b"tampered")
    second = CountingAnnotator()
    annotate_manifest(configured, second)
    assert second.calls > 0


def test_reward_cache_is_reused_across_dataset_versions(configured, tmp_path):
    prepare_dataset(configured)
    first = CountingAnnotator()
    annotate_manifest(configured, first)
    assert first.calls > 0

    source = Path(configured.section("paths")["dataset_sources"][0])
    run_json = next(source.rglob("root/run.json"))
    episode = run_json.parent / "episodes" / "episode_000"
    members = [{
            "run_id": "root", "split": "train", "resume_step": 0,
            "end_step": 17,
            "artifacts": {
                name: {"path": str(path), "sha256": sha256_file(path), "size": path.stat().st_size}
                for name, path in {
                    "manifest": run_json, "trajectory": episode / "trajectory.npz",
                    "observations": episode / "trajectory_observations.npz",
                }.items()
            },
        }]
    immutable = {
        "task_id": "LEVEL1::task", "selection": {"mode": "manual"},
        "validation_fraction": 0.2, "split_seed": 7,
        "success_consecutive_steps": 5, "members": members,
    }
    selection = {
        "schema_version": 1, "id": "ds_second", "project_id": "libero_x_vla",
        **immutable, "dataset_sha256": stable_hash(immutable),
    }
    selection_path = tmp_path / "second-dataset.json"
    selection_path.write_text(json.dumps(selection), encoding="utf-8")
    raw = yaml.safe_load(configured.path.read_text(encoding="utf-8"))
    raw["data"]["selection_manifest"] = str(selection_path)
    raw["data"]["task_ids"] = ["LEVEL1::task"]
    raw["paths"]["work_dir"] = str(tmp_path / "second-work")
    path = tmp_path / "second-config.yaml"
    path.write_text(yaml.safe_dump(raw), encoding="utf-8")
    second_config = load_train_config(path)
    prepare_dataset(second_config)
    second = CountingAnnotator()
    annotate_manifest(second_config, second)
    assert second.calls == 0


def test_sparse_reward_uses_only_debounced_terminal(configured):
    configured.raw["reward"]["shaping_weight"] = 0.0
    prepare_dataset(configured)
    prepared_branch = next(
        item for item in load_manifest(configured)["episodes"]
        if item["run_id"] == "branch"
    )
    index_path = annotate_manifest(configured, FakeAnnotator())
    index = json.loads(index_path.read_text(encoding="utf-8"))
    branch = next(item for item in index["episodes"] if item["run_id"] == "branch")
    with np.load(branch["annotation_path"], allow_pickle=False) as annotation:
        rewards = annotation["pbrs_chunk_reward"]
    # The final chunk is one completing macro action, irrespective of its five
    # executed low-level actions or earlier transient done=True samples.
    expected = 0.0
    # Evaluation continues through the recorded post-terminal tail. The final
    # replay chunk is therefore not necessarily the final diagnostic chunk.
    assert np.isclose(rewards[len(prepared_branch["chunks"]) - 1], expected)
    assert np.isclose(rewards[-1], 0.0)


def test_official_prefix_shape_reduction_uses_last_slot():
    values = torch.arange(8, dtype=torch.float32).reshape(1, 8)
    reduced = RynnValueAnnotator._absolute_last_slots(
        values, sample_count=2, head_count=1
    )
    torch.testing.assert_close(reduced, torch.tensor([[3.0], [7.0]]))


def _fake_rynn_config():
    return SimpleNamespace(
        model_type="rynn_value_lang",
        text_config=SimpleNamespace(hidden_size=2560),
        value_token_repeat=8,
        value_tokenizer_config=SimpleNamespace(bins=256),
        value_head_config=SimpleNamespace(head_type="bro"),
        num_value_heads=1,
        relative_value_token_repeat=8,
        relative_value_tokenizer_config=SimpleNamespace(bins=256),
        relative_value_head_config=SimpleNamespace(head_type="bro"),
    )


def test_rynnvalue_contract_uses_repeated_qwen_hidden_states():
    contract = validate_rynnvalue_config_contract(
        _fake_rynn_config(),
        SimpleNamespace(value_token_repeat=8, relative_value_token_repeat=8),
    )
    assert contract["qwen_hidden_size"] == 2560
    assert contract["value_head_input_size"] == 20480


def test_rynnvalue_contract_rejects_processor_repeat_mismatch():
    with np.testing.assert_raises_regex(RuntimeError, "repeat mismatch"):
        validate_rynnvalue_config_contract(
            _fake_rynn_config(),
            SimpleNamespace(value_token_repeat=1, relative_value_token_repeat=8),
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


def test_complete_runtime_contract_requires_relative_head():
    model = _TinyRynnValue(torch.bfloat16)
    with np.testing.assert_raises_regex(RuntimeError, "no dedicated relative value head"):
        validate_rynnvalue_runtime_dtype(model, torch.bfloat16, 16, 16)


def test_cuda_oom_reports_stage_without_cpu_fallback():
    def fail():
        raise RuntimeError("CUDA out of memory while allocating tensor")

    with np.testing.assert_raises_regex(RuntimeError, "reward annotation failed"):
        run_cuda_stage("reward annotation", fail)
