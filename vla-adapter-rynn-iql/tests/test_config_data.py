from __future__ import annotations

from pathlib import Path
import json
import zipfile

import numpy as np
import yaml

import pytest

from vla_rynn_iql.config import load_train_config
from vla_rynn_iql.data import load_manifest, prepare_dataset
from vla_rynn_iql.io import sha256_file, stable_hash


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


def test_tensorboard_logging_configuration_is_strict(configured, tmp_path: Path):
    raw = yaml.safe_load(configured.path.read_text(encoding="utf-8"))
    raw["logging"]["tensorboard"] = "yes"
    path = tmp_path / "invalid-logging.yaml"
    path.write_text(yaml.safe_dump(raw), encoding="utf-8")
    with pytest.raises(TypeError, match="logging.tensorboard"):
        load_train_config(path)


@pytest.mark.parametrize("value", [0, 1.5, "10"])
def test_console_progress_interval_is_a_positive_integer(
    configured, tmp_path: Path, value
):
    raw = yaml.safe_load(configured.path.read_text(encoding="utf-8"))
    raw["logging"]["console_interval_steps"] = value
    path = tmp_path / "invalid-console-progress.yaml"
    path.write_text(yaml.safe_dump(raw), encoding="utf-8")
    with pytest.raises((TypeError, ValueError), match="console_interval_steps"):
        load_train_config(path)


def test_branch_keeps_full_trajectory_and_marks_interrupted_policy_prefix(configured):
    prepare_dataset(configured)
    manifest = load_manifest(configured)
    episodes = {episode["run_id"]: episode for episode in manifest["episodes"]}
    assert [chunk["start"] for chunk in episodes["root"]["chunks"]] == [0, 8, 16]
    assert [chunk["start"] for chunk in episodes["branch"]["chunks"]] == [0, 5, 13]
    assert episodes["branch"]["chunks"][0] == {
        "start": 0, "length": 5, "end": 5, "action_source": "policy",
        "transition_type": "policy_interrupted", "interrupted": True,
        "copied_prefix": True,
    }
    assert episodes["branch"]["chunks"][1]["action_source"] == "human"
    assert episodes["branch"]["reward_boundaries"] == [0, 5, 13, 18, 22]
    assert episodes["root"]["split"] == episodes["branch"]["split"]


def test_branch_reward_boundaries_include_natural_rollout_before_takeover(configured):
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
    branch = next(
        episode for episode in load_manifest(configured)["episodes"]
        if episode["run_id"] == "branch"
    )
    assert [
        (chunk["start"], chunk["end"], chunk["transition_type"])
        for chunk in branch["chunks"]
    ] == [
        (0, 8, "policy_prefix"),
        (8, 13, "policy_interrupted"),
        (13, 18, "human"),
    ]
    assert branch["reward_boundaries"] == [0, 8, 13, 18, 22]


def test_500_step_branch_evaluation_still_spans_full_25_seconds(configured):
    source = Path(configured.section("paths")["dataset_sources"][0])
    run_json = next(source.rglob("branch/run.json"))
    run = json.loads(run_json.read_text(encoding="utf-8"))
    run.update({"resume_step": 210, "success": False})
    run_json.write_text(json.dumps(run), encoding="utf-8")
    episode = run_json.parent / "episodes" / "episode_000"
    actions = np.zeros((500, 7), dtype=np.float32)
    actions[:, -1] = -1.0
    raw_actions = actions.copy()
    raw_actions[:, -1] = 1.0
    sources = np.asarray(["policy"] * 210 + ["human"] * 290, dtype="<U32")
    np.savez_compressed(
        episode / "trajectory.npz",
        time_seconds=np.arange(501, dtype=np.float64) / 20.0,
        eef_position=np.zeros((501, 3), np.float32),
        eef_axis_angle=np.zeros((501, 3), np.float32),
        gripper_qpos=np.zeros((501, 2), np.float32),
        env_action=actions,
        raw_action=raw_actions,
        reward=np.zeros(500, np.float32),
        done=np.zeros(500, dtype=bool),
        action_source=sources,
    )
    np.savez_compressed(
        episode / "trajectory_observations.npz",
        agentview_image=np.zeros((501, 16, 16, 3), np.uint8),
        wrist_image=np.zeros((501, 16, 16, 3), np.uint8),
    )

    prepare_dataset(configured)
    branch = next(
        episode for episode in load_manifest(configured)["episodes"]
        if episode["run_id"] == "branch"
    )
    assert branch["reward_boundaries"][0] == 0
    assert branch["reward_boundaries"][-1] == 500
    assert branch["reward_boundaries"][-1] / 20.0 == 25.0
    assert any(
        chunk["start"] == 208 and chunk["end"] == 210
        and chunk["transition_type"] == "policy_interrupted"
        for chunk in branch["chunks"]
    )


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
    assert branch["chunks"][-1] == {
        "start": 13, "length": 5, "end": 18, "action_source": "human",
        "transition_type": "human", "interrupted": False,
        "copied_prefix": False,
    }
    assert branch["reward_boundaries"][0] == 0
    assert branch["reward_boundaries"][-1] == 22
    assert branch["evaluation_chunks"][:len(branch["chunks"])] == branch["chunks"]
    assert branch["evaluation_chunks"][-1] == {
        "start": 18, "length": 4, "end": 22, "action_source": "human",
        "transition_type": "post_terminal_evaluation", "interrupted": False,
        "copied_prefix": False,
    }
    assert trajectory.read_bytes() == original_bytes


def test_action_source_change_is_a_hard_chunk_boundary(configured):
    source = Path(configured.section("paths")["dataset_sources"][0])
    trajectory = next(source.rglob("branch/episodes/episode_000/trajectory.npz"))
    with np.load(trajectory, allow_pickle=False) as archive:
        arrays = {key: archive[key] for key in archive.files}
    sources = arrays["action_source"].astype("<U32")
    sources[5:10] = "human"
    sources[10:] = "policy_requery"
    arrays["action_source"] = sources
    np.savez_compressed(trajectory, **arrays)

    prepare_dataset(configured)
    branch = next(
        episode for episode in load_manifest(configured)["episodes"]
        if episode["run_id"] == "branch"
    )
    assert [
        (chunk["start"], chunk["end"], chunk["action_source"])
        for chunk in branch["chunks"]
    ] == [
        (0, 5, "policy"),
        (5, 10, "human"),
        (10, 18, "policy_requery"),
    ]


def test_sibling_branches_keep_stable_interrupted_prefix_for_annotation(configured):
    source = Path(configured.section("paths")["dataset_sources"][0])
    branch_run = next(source.rglob("branch/run.json")).parent
    sibling_run = branch_run.parent / "branch-sibling"
    import shutil
    shutil.copytree(branch_run, sibling_run)
    run = json.loads((sibling_run / "run.json").read_text(encoding="utf-8"))
    run["id"] = "branch-sibling"
    (sibling_run / "run.json").write_text(json.dumps(run), encoding="utf-8")

    prepare_dataset(configured)
    branches = [
        episode for episode in load_manifest(configured)["episodes"]
        if episode["kind"] == "branch"
    ]
    interrupted = [
        chunk for episode in branches for chunk in episode["chunks"]
        if chunk["interrupted"]
    ]
    assert len(interrupted) == 2
    assert {(chunk["start"], chunk["end"]) for chunk in interrupted} == {(0, 5)}


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


def test_ui_selection_manifest_prepares_exact_members_and_split(configured, tmp_path: Path):
    source = Path(configured.section("paths")["dataset_sources"][0])
    run_json = next(source.rglob("branch/run.json"))
    episode = run_json.parent / "episodes" / "episode_000"
    members = [{
            "run_id": "branch", "split": "validation", "resume_step": 5,
            "end_step": 22,
            "artifacts": {
                name: {"path": str(path), "sha256": sha256_file(path), "size": path.stat().st_size}
                for name, path in {
                    "manifest": run_json,
                    "trajectory": episode / "trajectory.npz",
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
        "schema_version": 1, "id": "ds_exact", "project_id": "libero_x_vla",
        **immutable, "dataset_sha256": stable_hash(immutable),
    }
    selection_path = tmp_path / "dataset.json"
    selection_path.write_text(json.dumps(selection), encoding="utf-8")
    raw = yaml.safe_load(configured.path.read_text(encoding="utf-8"))
    raw["data"]["selection_manifest"] = str(selection_path)
    raw["data"]["task_ids"] = ["LEVEL1::task"]
    raw["paths"]["work_dir"] = str(tmp_path / "exact-work")
    config_path = tmp_path / "exact.yaml"
    config_path.write_text(yaml.safe_dump(raw), encoding="utf-8")

    selected = load_train_config(config_path)
    prepare_dataset(selected)
    manifest = load_manifest(selected)
    assert manifest["source_dataset_id"] == "ds_exact"
    assert [episode["run_id"] for episode in manifest["episodes"]] == ["branch"]
    assert manifest["episodes"][0]["split"] == "validation"
    assert manifest["episodes"][0]["chunks"][0]["start"] == 0
    assert manifest["episodes"][0]["chunks"][0]["end"] == 5
    assert manifest["episodes"][0]["chunks"][0]["transition_type"] == "policy_interrupted"
