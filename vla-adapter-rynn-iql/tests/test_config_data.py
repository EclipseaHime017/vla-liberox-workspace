from __future__ import annotations

from pathlib import Path
import zipfile

import numpy as np
import yaml

import pytest

from vla_rynn_iql.config import load_train_config
from vla_rynn_iql.data import load_manifest, prepare_dataset


def test_duplicate_yaml_key_is_rejected(tmp_path: Path):
    path = tmp_path / "bad.yaml"
    path.write_text("schema_version: 1\nschema_version: 1\n", encoding="utf-8")
    with pytest.raises(Exception, match="duplicate key"):
        load_train_config(path)


def test_unknown_config_key_is_rejected(configured, tmp_path: Path):
    raw = yaml.safe_load(configured.path.read_text(encoding="utf-8"))
    raw["iql"]["mystery"] = 1
    path = tmp_path / "unknown.yaml"
    path.write_text(yaml.safe_dump(raw), encoding="utf-8")
    with pytest.raises(ValueError, match="Unknown config.iql keys"):
        load_train_config(path)


def test_rynnvalue_checkout_path_is_resolved(configured):
    expected = configured.path.parent / "RynnValue"
    assert Path(configured.section("paths")["rynnvalue_root"]) == expected.resolve()


def test_reward_dtype_must_match_pinned_bfloat16_checkpoint(configured, tmp_path: Path):
    raw = yaml.safe_load(configured.path.read_text(encoding="utf-8"))
    raw["reward"]["dtype"] = "float32"
    path = tmp_path / "float32-reward.yaml"
    path.write_text(yaml.safe_dump(raw), encoding="utf-8")
    with pytest.raises(ValueError, match="reward.dtype=bfloat16"):
        load_train_config(path)


def test_success_confirmation_threshold_is_validated(configured, tmp_path: Path):
    raw = yaml.safe_load(configured.path.read_text(encoding="utf-8"))
    raw["data"]["success_consecutive_steps"] = 0
    path = tmp_path / "invalid-success-threshold.yaml"
    path.write_text(yaml.safe_dump(raw), encoding="utf-8")
    with pytest.raises(ValueError, match="success_consecutive_steps"):
        load_train_config(path)


def test_branch_prefix_is_not_added_as_new_replay(configured):
    prepare_dataset(configured)
    manifest = load_manifest(configured)
    episodes = {episode["run_id"]: episode for episode in manifest["episodes"]}
    assert [chunk["start"] for chunk in episodes["root"]["chunks"]] == [0, 8, 16]
    assert [chunk["start"] for chunk in episodes["branch"]["chunks"]] == [5, 13]
    assert episodes["branch"]["reward_boundaries"] == [5, 13, 18]
    assert episodes["root"]["split"] == episodes["branch"]["split"]


def test_latched_done_tail_is_excluded_from_replay_without_changing_source(configured):
    source = Path(configured.section("paths")["dataset_sources"][0])
    trajectory = next(source.rglob("branch/episodes/episode_000/trajectory.npz"))
    original_bytes = trajectory.read_bytes()

    prepare_dataset(configured)
    branch = next(
        episode for episode in load_manifest(configured)["episodes"]
        if episode["run_id"] == "branch"
    )

    assert branch["recorded_action_count"] == 22
    assert branch["terminal_step"] == 17
    assert branch["success_streak_start"] == 13
    assert branch["action_count"] == 18
    assert branch["trailing_action_count"] == 4
    assert branch["post_terminal_false_count"] == 0
    assert branch["chunks"][-1] == {"start": 13, "length": 5, "end": 18}
    assert trajectory.read_bytes() == original_bytes


def test_transient_success_requires_a_new_complete_streak(configured):
    source = Path(configured.section("paths")["dataset_sources"][0])
    trajectory = next(source.rglob("branch/episodes/episode_000/trajectory.npz"))
    with np.load(trajectory, allow_pickle=False) as archive:
        arrays = {key: archive[key] for key in archive.files}
    arrays["done"] = arrays["done"].copy()
    arrays["done"][15] = False
    np.savez_compressed(trajectory, **arrays)

    prepare_dataset(configured)
    branch = next(
        episode for episode in load_manifest(configured)["episodes"]
        if episode["run_id"] == "branch"
    )
    assert branch["terminal_step"] == 20
    assert branch["success_streak_start"] == 16
    assert branch["action_count"] == 21
    assert branch["trailing_action_count"] == 1
    assert branch["post_terminal_false_count"] == 0


def test_unconfirmed_success_pulses_are_treated_as_failure(configured):
    source = Path(configured.section("paths")["dataset_sources"][0])
    trajectory = next(source.rglob("branch/episodes/episode_000/trajectory.npz"))
    with np.load(trajectory, allow_pickle=False) as archive:
        arrays = {key: archive[key] for key in archive.files}
    arrays["done"] = np.zeros_like(arrays["done"], dtype=bool)
    arrays["done"][[13, 15, 17, 19, 21]] = True
    np.savez_compressed(trajectory, **arrays)

    prepare_dataset(configured)
    branch = next(
        episode for episode in load_manifest(configured)["episodes"]
        if episode["run_id"] == "branch"
    )
    assert branch["recorded_success"] is True
    assert branch["success"] is False
    assert branch["raw_done_true_count"] == 5
    assert branch["terminal_step"] is None
    assert branch["action_count"] == 22


def test_ui_export_zip_is_imported_without_modifying_source(configured, tmp_path: Path):
    source = Path(configured.section("paths")["dataset_sources"][0])
    archive = tmp_path / "dataset.zip"
    with zipfile.ZipFile(archive, "w") as bundle:
        for path in source.rglob("*"):
            if path.is_file():
                bundle.write(path, path.relative_to(source))
    raw = yaml.safe_load(configured.path.read_text(encoding="utf-8"))
    raw["paths"]["dataset_sources"] = [str(archive)]
    raw["paths"]["work_dir"] = str(tmp_path / "zip-work")
    path = tmp_path / "zip-config.yaml"
    path.write_text(yaml.safe_dump(raw), encoding="utf-8")
    imported = load_train_config(path)
    prepare_dataset(imported)
    assert load_manifest(imported)["episode_count"] == 2
