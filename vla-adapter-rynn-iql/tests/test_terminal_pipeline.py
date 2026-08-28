from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pytest
import yaml

from vla_rynn_iql.config import load_train_config
from vla_rynn_iql.data import prepare_dataset
from vla_rynn_iql.evaluation_store import (
    bind_reward_manifest,
    valid_bound_evaluation,
)
from vla_rynn_iql.rewards import annotate_manifest
from vla_rynn_iql.terminal_pipeline import (
    build_selection_manifest,
    dataset_roots,
    discover_candidates,
    load_terminal_config,
    mark_prepare_cache,
    merged_training_config,
    prepare_cache_valid,
    prepare_fingerprint,
    resolve_task_id,
    reward_cache_valid,
    select_candidates,
)


class FakeAnnotator:
    metadata = {"provider": "fake", "revision": "terminal-test"}

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
            "parsed_for_display": {"description": None, "match": "Yes", "success": "No"},
        }


def _terminal_config(tmp_path: Path, base: Path, **selection_changes) -> Path:
    selection = {
        "task_id": "task",
        "mode": "quota",
        "seed": 17,
        "source_types": ["inference", "manual", "policy_requery"],
        "outcomes": ["success", "failure"],
        "size": None,
        "quotas": [
            {"source_type": "inference", "outcome": "failure", "count": 1, "order": "random"},
            {"source_type": "manual", "outcome": "success", "count": 1, "order": "random"},
        ],
    }
    selection.update(selection_changes)
    path = tmp_path / "terminal.yaml"
    path.write_text(yaml.safe_dump({
        "schema_version": 1,
        "base_config": str(base),
        "pipeline_root": str(tmp_path / "pipelines"),
        "environments": {
            "prepare": "vla-liberox", "annotate": "rynnvalue-reward", "train": "vla-liberox",
        },
        "selection": selection,
        "overrides": {
            "paths": {}, "data": {}, "reward": {}, "vla": {}, "iql": {}, "logging": {},
        },
    }, sort_keys=False), encoding="utf-8")
    return path


def _selection(configured, tmp_path: Path):
    terminal = load_terminal_config(_terminal_config(tmp_path, configured.path))
    raw = merged_training_config(terminal)
    roots = dataset_roots(raw, terminal.pipeline_root / "imports")
    candidates, rejected = discover_candidates(roots, raw["data"]["project_id"])
    assert not rejected
    task_id = resolve_task_id(terminal.selection["task_id"], candidates)
    selected = select_candidates(candidates, terminal.selection, task_id)
    manifest = build_selection_manifest(selected, raw, terminal.selection, task_id)
    return terminal, raw, selected, manifest


def test_terminal_yaml_rejects_duplicate_keys(configured, tmp_path: Path):
    path = tmp_path / "duplicate.yaml"
    path.write_text("schema_version: 1\nschema_version: 1\n", encoding="utf-8")
    with pytest.raises(Exception, match="duplicate key"):
        load_terminal_config(path)


def test_terminal_yaml_rejects_invalid_environment_and_unknown_override(
    configured, tmp_path: Path,
):
    path = _terminal_config(tmp_path, configured.path)
    raw = yaml.safe_load(path.read_text(encoding="utf-8"))
    raw["environments"]["train"] = "invalid environment"
    path.write_text(yaml.safe_dump(raw), encoding="utf-8")
    with pytest.raises(ValueError, match="Conda environment"):
        load_terminal_config(path)

    raw["environments"]["train"] = "vla-liberox"
    raw["overrides"]["iql"]["unknown"] = 1
    path.write_text(yaml.safe_dump(raw), encoding="utf-8")
    with pytest.raises(ValueError, match="Unknown overrides.iql"):
        load_terminal_config(path)


def test_quota_selection_is_deterministic_and_root_grouped(configured, tmp_path: Path):
    _, raw, selected, manifest = _selection(configured, tmp_path)
    assert [(item.source_type, item.outcome) for item in selected] == [
        ("inference", "failure"), ("manual", "success"),
    ]
    assert {member["root_run_id"] for member in manifest["members"]} == {"root"}
    assert len({member["split"] for member in manifest["members"]}) == 1
    assert manifest["task_id"] == "LEVEL1::task"
    assert manifest["project_id"] == raw["data"]["project_id"]


def test_quota_selection_fails_instead_of_shrinking(configured, tmp_path: Path):
    terminal = load_terminal_config(_terminal_config(tmp_path, configured.path))
    raw = merged_training_config(terminal)
    candidates, _ = discover_candidates(
        dataset_roots(raw, terminal.pipeline_root / "imports"), raw["data"]["project_id"],
    )
    selection = dict(terminal.selection)
    selection["quotas"] = [
        {"source_type": "manual", "outcome": "success", "count": 2, "order": "random"},
    ]
    with pytest.raises(ValueError, match="only 1"):
        select_candidates(candidates, selection, "LEVEL1::task")


@pytest.mark.parametrize("mode,size", [("all", None), ("random", 1)])
def test_all_and_random_selection_modes(configured, tmp_path: Path, mode: str, size: int | None):
    path = _terminal_config(
        tmp_path, configured.path, mode=mode, size=size, quotas=[],
    )
    terminal = load_terminal_config(path)
    raw = merged_training_config(terminal)
    candidates, _ = discover_candidates(
        dataset_roots(raw, terminal.pipeline_root / "imports"), raw["data"]["project_id"],
    )
    first = select_candidates(candidates, terminal.selection, "LEVEL1::task")
    second = select_candidates(candidates, terminal.selection, "LEVEL1::task")
    assert [item.run_id for item in first] == [item.run_id for item in second]
    assert len(first) == (2 if mode == "all" else 1)


def test_prepare_fingerprint_cache_and_bound_evaluation(configured, tmp_path: Path):
    terminal, raw, _, selection = _selection(configured, tmp_path)
    selection_path = tmp_path / "frozen" / "dataset.json"
    selection_path.parent.mkdir(parents=True)
    selection_path.write_text(json.dumps(selection), encoding="utf-8")
    fingerprint = prepare_fingerprint(selection, raw)
    work = terminal.pipeline_root / "cache" / fingerprint / "work"
    raw["paths"]["work_dir"] = str(work)
    raw["data"]["task_ids"] = [selection["task_id"]]
    raw["data"]["selection_manifest"] = str(selection_path)
    effective_path = tmp_path / "effective.yaml"
    effective_path.write_text(yaml.safe_dump(raw, sort_keys=False), encoding="utf-8")
    effective = load_train_config(effective_path)

    prepare_dataset(effective)
    assert not prepare_cache_valid(work, fingerprint)
    mark_prepare_cache(work, fingerprint, selection["dataset_sha256"])
    assert prepare_cache_valid(work, fingerprint)

    reward_path = annotate_manifest(effective, FakeAnnotator())
    assert reward_cache_valid(work, effective.section("reward"))
    binding = bind_reward_manifest(work / "dataset_manifest.json", reward_path)
    assert binding["bound_count"] == 2
    assert binding["skipped_count"] == 0
    prepared = json.loads((work / "dataset_manifest.json").read_text(encoding="utf-8"))
    assert all(valid_bound_evaluation(episode) is not None for episode in prepared["episodes"])

    second = bind_reward_manifest(work / "dataset_manifest.json", reward_path)
    assert second["bound_count"] == 0
    assert second["skipped_count"] == 2

    changed_reward = dict(effective.section("reward"))
    changed_reward["shaping_weight"] = 0.2
    assert not reward_cache_valid(work, changed_reward)

    episode = prepared["episodes"][0]
    values = Path(episode["trajectory_path"]).parent / "rynnvalue_evaluation.npz"
    values.write_bytes(values.read_bytes() + b"corrupt")
    assert valid_bound_evaluation(episode) is None


def test_training_only_override_does_not_change_prepare_fingerprint(configured, tmp_path: Path):
    _, raw, _, selection = _selection(configured, tmp_path)
    before = prepare_fingerprint(selection, raw)
    raw["iql"]["train_steps"] += 1000
    assert prepare_fingerprint(selection, raw) == before
    raw["data"]["success_consecutive_steps"] += 1
    assert prepare_fingerprint(selection, raw) != before
