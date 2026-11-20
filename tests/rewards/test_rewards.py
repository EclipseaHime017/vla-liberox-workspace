from __future__ import annotations

import json
import shutil
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest
import torch
import yaml

import rewards.rynnvalue.annotate as rynnvalue_module
from core.data import load_manifest, prepare_dataset
from core.config import load_train_config
from rewards.evaluation_store import bind_reward_manifest
from core.io import sha256_file, stable_hash
from rewards.rynnvalue.outputs import ANNOTATION_SCHEMA_VERSION
from rewards.rynnvalue.model import RynnValueAnnotator, validate_rynnvalue_config_contract, validate_rynnvalue_runtime_dtype
from rewards.rynnvalue.annotate import annotate_manifest
from rewards.pbrs import discounted_sum, shaping_rewards, sparse_rewards
from rewards.manifest import load_reward_index, materialize_reward_manifest
from core.runtime import run_cuda_stage


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


class RecordingAnnotator(FakeAnnotator):
    def __init__(self):
        self.frame_counts = []

    def predict(self, prompt, frames):
        self.frame_counts.append(len(frames))
        return super().predict(prompt, frames)


class CountingAnnotator(FakeAnnotator):
    def __init__(self):
        self.calls = 0

    def predict(self, prompt, frames):
        self.calls += 1
        return super().predict(prompt, frames)


def test_per_step_rewards_and_shaping_telescope_over_an_interval():
    done = np.asarray([False, False, True])
    assert sparse_rewards(done).tolist() == [-1.0, -1.0, 0.0]
    potential = -np.asarray([3.0, 2.0, 1.5, 0.0])
    shape = shaping_rewards(potential, done, 0.9)
    np.testing.assert_allclose(shape, 0.9 * potential[1:] - potential[:-1])
    # Per-step shaping sums to gamma^L * phi(end) - phi(start) over any interval.
    assert discounted_sum(shape, 0, 3, 0.9) == pytest.approx(0.9 ** 3 * potential[3] - potential[0])
    assert discounted_sum(sparse_rewards(done), 0, 3, 0.9) == pytest.approx(-(1 + 0.9))
    # Zero success potential: the observation after the confirming action has phi = 0.
    zeroed = shaping_rewards(potential, done, 0.9, zero_success_potential=True)
    assert zeroed[2] == pytest.approx(0.9 * 0.0 - potential[2])
    with pytest.raises(ValueError, match="observations"):
        shaping_rewards(potential[:-1], done, 0.9)


def test_fake_annotation_pipeline(configured):
    prepare_dataset(configured)
    result = annotate_manifest(configured, FakeAnnotator())
    assert result.is_file()


def test_annotation_preserves_only_official_outputs_and_rewards_are_separate(configured):
    prepare_dataset(configured)
    result = annotate_manifest(configured, FakeAnnotator())
    manifest = json.loads(result.read_text(encoding="utf-8"))
    assert manifest["schema_version"] == ANNOTATION_SCHEMA_VERSION
    episode = manifest["episodes"][0]
    assert episode["official_outputs"]["inference_method"] == "prefix_uniform_last_slot"
    assert episode["official_outputs"]["evaluated_steps"] == "every_observation"
    assert episode["official_outputs"]["analysis"]["generated_token_ids"] == [1, 2]
    with np.load(episode["annotation_path"], allow_pickle=False) as annotation:
        assert set(annotation.files) == {
            "absolute_temporal_distance_seconds",
            "absolute_value_entropy_nats",
            "absolute_value_logits",
            "relative_temporal_distance_seconds",
            "relative_value_logits",
        }
        assert annotation["absolute_temporal_distance_seconds"].ndim == 2
        assert annotation["absolute_value_logits"].shape[-1] == 256
        assert annotation["relative_value_logits"].shape[-1] == 256
    reward_manifest = json.loads(
        materialize_reward_manifest(configured).read_text(encoding="utf-8")
    )
    reward = reward_manifest["episodes"][0]
    with np.load(reward["reward_path"], allow_pickle=False) as arrays:
        assert {"sparse_reward", "pbrs_shaping_reward", "dense_reward", "final_reward"}.issubset(arrays.files)
        assert not {"boundary_steps", "pbrs_chunk_reward"} & set(arrays.files)


def test_annotation_evaluates_every_observation_with_time_uniform_prefixes(configured):
    prepare_dataset(configured)
    annotator = RecordingAnnotator()
    result = annotate_manifest(configured, annotator)
    manifest = json.loads(result.read_text(encoding="utf-8"))
    prepared = {item["run_id"]: item for item in load_manifest(configured)["episodes"]}
    branch = next(item for item in manifest["episodes"] if item["run_id"] == "branch")
    assert prepared["branch"]["recorded_action_count"] == 22
    with np.load(branch["annotation_path"], allow_pickle=False) as annotation:
        assert annotation["absolute_temporal_distance_seconds"].shape == (23, 1)
    assert set(annotator.frame_counts) == {item["recorded_action_count"] + 1 for item in prepared.values()}
    rewards = json.loads(materialize_reward_manifest(configured).read_text(encoding="utf-8"))
    branch_reward = next(item for item in rewards["episodes"] if item["run_id"] == "branch")
    with np.load(branch_reward["reward_path"], allow_pickle=False) as arrays:
        assert arrays["pbrs_shaping_reward"].shape == (22,)
        assert arrays["final_reward"].shape == (22,)


def test_reward_cache_resumes_without_reannotation(configured):
    prepare_dataset(configured)
    annotator = CountingAnnotator()
    annotate_manifest(configured, annotator)
    first_calls = annotator.calls
    annotate_manifest(configured, annotator)
    assert first_calls > 0
    assert annotator.calls == first_calls


def test_gamma_change_reuses_heads_and_rebuilds_manifest(
    configured, monkeypatch,
):
    prepare_dataset(configured)
    annotation_path = annotate_manifest(configured, FakeAnnotator())
    annotation_before = json.loads(annotation_path.read_text(encoding="utf-8"))
    configured.raw["reward"]["gamma"] = 0.95

    class UnexpectedModelLoad:
        def __init__(self, _config):
            raise AssertionError("reward-only relabel must not load RynnValue")

    monkeypatch.setattr(rynnvalue_module, "RynnValueAnnotator", UnexpectedModelLoad)
    rebuilt = load_reward_index(configured)
    assert rebuilt["reward_config"]["gamma"] == 0.95
    annotation_after = json.loads(annotation_path.read_text(encoding="utf-8"))
    assert annotation_after == annotation_before
    with np.load(rebuilt["episodes"][0]["reward_path"], allow_pickle=False) as arrays:
        assert arrays["final_reward"].shape == arrays["sparse_reward"].shape
        np.testing.assert_allclose(arrays["final_reward"],
                                   arrays["sparse_reward"] + 0.1 * arrays["pbrs_shaping_reward"], rtol=1e-6)


def test_disabling_rynnvalue_reuses_evaluation_and_materializes_sparse_only(
    configured, monkeypatch,
):
    prepare_dataset(configured)
    annotation_path = annotate_manifest(configured, FakeAnnotator())
    annotation_before = json.loads(annotation_path.read_text(encoding="utf-8"))
    configured.raw["reward"].update(source="sparse", rynnvalue=False)

    class UnexpectedModelLoad:
        def __init__(self, _config):
            raise AssertionError("reward ablation must not load RynnValue")

    monkeypatch.setattr(rynnvalue_module, "RynnValueAnnotator", UnexpectedModelLoad)
    rebuilt = load_reward_index(configured)
    assert rebuilt["reward_config"]["rynnvalue"] is False
    assert json.loads(annotation_path.read_text(encoding="utf-8")) == annotation_before
    with np.load(rebuilt["episodes"][0]["reward_path"], allow_pickle=False) as arrays:
        assert "pbrs_shaping_reward" not in arrays
        assert "dense_reward" not in arrays
        assert np.array_equal(arrays["final_reward"], arrays["sparse_reward"])


def test_sidecars_from_older_evaluations_are_not_reused(configured):
    prepare_dataset(configured)
    annotate_manifest(configured, FakeAnnotator())
    work_dir = Path(configured.section("paths")["work_dir"])
    bind_reward_manifest(work_dir / "dataset_manifest.json", materialize_reward_manifest(configured))
    for episode in load_manifest(configured)["episodes"]:
        sidecar = Path(episode["trajectory_path"]).parent / "rynnvalue_evaluation.json"
        payload = json.loads(sidecar.read_text(encoding="utf-8"))
        payload["schema_version"] = ANNOTATION_SCHEMA_VERSION - 1
        sidecar.write_text(json.dumps(payload), encoding="utf-8")
    shutil.rmtree(work_dir / "annotations")
    shutil.rmtree(work_dir / "rewards")
    shutil.rmtree(Path(configured.section("paths")["annotation_cache"]))
    annotator = CountingAnnotator()
    annotate_manifest(configured, annotator)
    assert annotator.calls == len(load_manifest(configured)["episodes"])


def test_existing_sidecar_survives_reward_parameter_changes_without_model_forward(
    configured, monkeypatch,
):
    prepare_dataset(configured)
    annotate_manifest(configured, FakeAnnotator())
    first_rewards = materialize_reward_manifest(configured)
    work_dir = Path(configured.section("paths")["work_dir"])
    bind_reward_manifest(work_dir / "dataset_manifest.json", first_rewards)

    prepared = load_manifest(configured)
    for episode in prepared["episodes"]:
        sidecar = Path(episode["trajectory_path"]).parent / "rynnvalue_evaluation.json"
        payload = json.loads(sidecar.read_text(encoding="utf-8"))
        payload["annotator"] = {
            "model": configured.raw["reward"]["model"],
            "requested_revision": configured.raw["reward"]["revision"],
            "resolved_revision": configured.raw["reward"]["revision"],
        }
        sidecar.write_text(json.dumps(payload), encoding="utf-8")

    shutil.rmtree(work_dir / "annotations")
    shutil.rmtree(work_dir / "rewards")
    shutil.rmtree(Path(configured.section("paths")["annotation_cache"]))
    configured.raw["reward"]["gamma"] = 0.8
    configured.raw["reward"]["shaping_weight"] = 0.25

    class UnexpectedModelLoad:
        def __init__(self, _config):
            raise AssertionError("valid saved evaluations must not run RynnValue again")

    monkeypatch.setattr(rynnvalue_module, "RynnValueAnnotator", UnexpectedModelLoad)
    migrated = annotate_manifest(configured)
    assert migrated.is_file()
    rewards = load_reward_index(configured)
    assert rewards["reward_config"]["gamma"] == 0.8
    assert rewards["reward_config"]["shaping_weight"] == 0.25


def test_tampered_reward_cache_is_recomputed(configured):
    prepare_dataset(configured)
    annotate_manifest(configured, CountingAnnotator())
    index = json.loads(
        (Path(configured.section("paths")["work_dir"]) / "annotations" / "annotation_manifest.json")
        .read_text(encoding="utf-8")
    )
    with Path(index["episodes"][0]["annotation_path"]).open("ab") as stream:
        stream.write(b"tampered")
    second = CountingAnnotator()
    annotate_manifest(configured, second)
    assert second.calls > 0


def test_reward_cache_is_reused_across_dataset_versions(configured, tmp_path, monkeypatch):
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
    # A different package can seed official per-trajectory outputs without
    # loading a model even when its own cache is empty.
    import shutil
    from rewards.rynnvalue import annotate as rynnvalue
    parent_index = Path(configured.section("paths")["work_dir"]) / "annotations/annotation_manifest.json"
    child_index = Path(second_config.section("paths")["work_dir"]) / "annotations/annotation_manifest.json"
    shutil.copyfile(parent_index, child_index)
    second_config.raw["paths"]["annotation_cache"] = str(tmp_path / "empty-cache")
    def unexpected(_):
        raise AssertionError("Compatible output reuse must not construct RynnValue")
    monkeypatch.setattr(rynnvalue, "RynnValueAnnotator", unexpected)
    annotate_manifest(second_config, reuse_only=True)


def test_sparse_reward_uses_only_debounced_terminal(configured):
    configured.raw["reward"]["shaping_weight"] = 0.0
    prepare_dataset(configured)
    prepared_branch = next(
        item for item in load_manifest(configured)["episodes"]
        if item["run_id"] == "branch"
    )
    index_path = annotate_manifest(configured, FakeAnnotator())
    index = load_reward_index(configured)
    branch = next(item for item in index["episodes"] if item["run_id"] == "branch")
    with np.load(branch["reward_path"], allow_pickle=False) as annotation:
        rewards = annotation["final_reward"]
    # Only actions from the one completing the confirmation streak are free of the
    # step cost, irrespective of earlier transient done=True samples.
    terminal = prepared_branch["terminal_step"]
    assert rewards.shape == (prepared_branch["recorded_action_count"],)
    assert np.all(rewards[:terminal] == -1.0)
    # Training also retains the recorded tail, whose saved sparse reward is zero.
    assert terminal < len(rewards) - 1
    assert np.all(rewards[terminal:] == 0.0)


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
