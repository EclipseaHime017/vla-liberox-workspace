from __future__ import annotations

import json
import shutil
import threading
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest

from backend.app.core.config import load_ui_config
from backend.app.domain.run import SimulationSession
from backend.app.devices.spacemouse import SpaceMouseSnapshot
from backend.app.storage.files import scan_trajectories
from backend.app.workers.simulation_worker import SimulationManager
from trajectory_utils import TrajectoryRecorder, load_trajectory, save_trajectory_bundle


VALID = """\
host: 127.0.0.1
port: 8000
dataset_root: ./data
project_id: test_project
legacy_scan_roots: [./runs]
preview_width: 512
preview_height: 512
preview_fps: 10
jpeg_quality: 85
manual_translation_gain: 0.25
manual_rotation_gain: 0.25
additional_tasks: []
"""


TASK_ID = "LEVEL1::task_name"


class FakeCatalog:
    default_task_id = TASK_ID

    def metadata(self, task_id):
        assert task_id == TASK_ID
        return {
            "task_id": task_id,
            "level": "LEVEL1",
            "task_name": "task_name",
            "prompt": "place object",
        }


def test_ui_paths_are_relative_to_yaml(tmp_path: Path):
    path = tmp_path / "ui.yaml"
    path.write_text(VALID, encoding="utf-8")
    config = load_ui_config(path)
    assert config.dataset_root == (tmp_path / "data").resolve()
    assert config.output_root == (tmp_path / "data/projects/test_project/runs").resolve()
    assert config.legacy_scan_roots == ((tmp_path / "runs").resolve(),)


def test_duplicate_and_unknown_keys_fail(tmp_path: Path):
    duplicate = tmp_path / "duplicate.yaml"
    duplicate.write_text(VALID + "port: 9000\n", encoding="utf-8")
    with pytest.raises(Exception, match="duplicate key"):
        load_ui_config(duplicate)
    unknown = tmp_path / "unknown.yaml"
    unknown.write_text(VALID + "surprise: true\n", encoding="utf-8")
    with pytest.raises(ValueError, match="Unknown"):
        load_ui_config(unknown)


def test_ui_is_local_only(tmp_path: Path):
    path = tmp_path / "ui.yaml"
    path.write_text(VALID.replace("127.0.0.1", "0.0.0.0"), encoding="utf-8")
    with pytest.raises(ValueError, match="local-only"):
        load_ui_config(path)


def test_branch_progress_starts_at_resume_step(tmp_path: Path):
    manager = object.__new__(SimulationManager)
    manager.ui_config = SimpleNamespace(output_root=tmp_path)
    manager.catalog = FakeCatalog()
    record = manager._new_record(
        kind="branch",
        max_steps=300,
        open_loop_steps=1,
        parent={
            "id": "root",
            "root_session_id": "root",
            "trajectory": "/tmp/source.npz",
            "task_id": TASK_ID,
        },
        resume_step=137,
        control_mode="policy",
    )
    assert record.current_step == 137
    assert record.action_count == 137
    assert record.state_count == 138


def test_branch_record_keeps_manual_source_and_gains(tmp_path: Path):
    manager = object.__new__(SimulationManager)
    manager.ui_config = SimpleNamespace(output_root=tmp_path)
    manager.spacemouse_config = SimpleNamespace(stale_timeout_ms=250)
    manager.catalog = FakeCatalog()
    record = manager._new_record(
        kind="branch",
        max_steps=300,
        open_loop_steps=1,
        parent={
            "id": "root",
            "root_session_id": "root",
            "trajectory": "/tmp/source.npz",
            "task_id": TASK_ID,
        },
        resume_step=40,
        control_mode="manual",
        manual_translation_gain=0.25,
        manual_rotation_gain=0.08,
    )
    public = record.public()
    assert public["manual_source"] == "spacemouse"
    assert public["manual_translation_gain"] == 0.25
    assert public["manual_rotation_gain"] == 0.08
    assert public["spacemouse_deadman_ms"] == 250


def test_spacemouse_latency_and_live_gains_are_published(tmp_path: Path):
    class FakeSpaceMouse:
        def __init__(self):
            self.gains = None

        def set_gains(self, translation_gain, rotation_gain):
            self.gains = (translation_gain, rotation_gain)

        def latest_snapshot(self):
            return SpaceMouseSnapshot(
                sequence=9,
                captured_monotonic=10.025,
                event_monotonic=10.0,
                device_timestamp=4.0,
                raw_axes=(0.1, 0.2, 0.3, 0.4, 0.5, 0.6),
                corrected_axes=(0.1, 0.2, 0.3, 0.4, 0.5, 0.6),
                command_axes=(0.01, 0.02, 0.03, 0.04, 0.05, 0.06),
                action=(0.01, 0.02, 0.03, 0.04, 0.05, 0.06, -1.0),
                buttons=(1, 0),
                connected=True,
                stale=False,
                error=None,
            )

    manager = object.__new__(SimulationManager)
    manager.lock = threading.RLock()
    controller = FakeSpaceMouse()

    class FakeService:
        def set_gains(self, session_id, translation_gain, rotation_gain):
            assert session_id == "branch"
            controller.set_gains(translation_gain, rotation_gain)

        def snapshot(self, session_id):
            assert session_id == "branch"
            return controller.latest_snapshot()

    record = SimulationSession(
        id="branch",
        kind="branch",
        output_dir=tmp_path,
        max_steps=300,
        open_loop_steps=1,
        control_mode="manual",
        manual_source="spacemouse",
        manual_translation_gain=0.25,
        manual_rotation_gain=0.08,
        status="RUNNING",
    )
    manager.sessions = {record.id: record}
    manager.controller = FakeService()
    manager.manual_settings(record.id, 0.2, 0.1)
    action = manager._spacemouse_action(record, 42)
    assert controller.gains == (0.2, 0.1)
    assert action.tolist() == pytest.approx([0.01, 0.02, 0.03, 0.04, 0.05, 0.06, -1.0])
    public = record.public()
    assert public["spacemouse_latency_ms"] == pytest.approx(25.0)
    assert public["spacemouse_status"] == "ready"
    assert record.spacemouse_samples[0]["step"] == 42
    assert record.spacemouse_samples[0]["button_left"] == 1


def _write_minimal_trajectory(path: Path) -> None:
    metadata = {"schema_version": 2, "control_hz": 20.0, "seed": 7}
    np.savez_compressed(
        path,
        step=np.asarray([0, 1]),
        time_seconds=np.asarray([0.0, 0.05]),
        sim_state=np.zeros((2, 3)),
        eef_position=np.zeros((2, 3)),
        eef_quaternion=np.tile(np.asarray([0.0, 0.0, 0.0, 1.0]), (2, 1)),
        eef_axis_angle=np.zeros((2, 3)),
        gripper_qpos=np.zeros((2, 2)),
        raw_action=np.zeros((1, 7)),
        env_action=np.zeros((1, 7)),
        reward=np.zeros(1),
        done=np.zeros(1, dtype=bool),
        action_source=np.asarray(["policy"]),
        metadata_json=np.asarray(json.dumps(metadata)),
    )


def test_ui_bundle_embeds_metadata_without_duplicate_json(tmp_path: Path):
    recorder = TrajectoryRecorder(control_hz=20.0, capture_images=False)
    recorder.sim_states = [np.zeros(3)]
    recorder.eef_positions = [np.zeros(3)]
    recorder.eef_quaternions = [np.asarray([0.0, 0.0, 0.0, 1.0])]
    recorder.eef_axis_angles = [np.zeros(3)]
    recorder.gripper_qpos = [np.zeros(2)]
    paths = save_trajectory_bundle(
        recorder,
        tmp_path / "trajectory",
        {"session_id": "compact"},
        save_observations=False,
        create_plot=False,
        save_metadata_json=False,
    )
    assert set(paths) == {"trajectory", "trajectory_csv"}
    assert not (tmp_path / "trajectory.json").exists()
    _, metadata = load_trajectory(tmp_path / "trajectory.npz")
    assert metadata["session_id"] == "compact"


def test_branch_source_is_physical_and_survives_parent_deletion(tmp_path: Path):
    parent_dir = tmp_path / "parent"
    parent_dir.mkdir()
    source = parent_dir / "trajectory.npz"
    _write_minimal_trajectory(source)
    manager = object.__new__(SimulationManager)
    manager.ui_config = SimpleNamespace(output_root=tmp_path)
    manager.spacemouse_config = None
    manager.catalog = FakeCatalog()
    record = manager._new_record(
        kind="branch",
        max_steps=1,
        open_loop_steps=1,
        parent={
            "id": "parent",
            "root_session_id": "parent",
            "trajectory": str(source),
            "task_id": TASK_ID,
        },
        resume_step=0,
    )
    manager._copy_branch_source(record, {"trajectory": str(source)})
    copied = Path(record.source_trajectory)
    assert copied.name == "source_trajectory.npz"
    assert copied.is_file()
    assert not (record.output_dir / "source_trajectory.json").exists()
    shutil.rmtree(parent_dir)
    trajectory, metadata = load_trajectory(copied)
    assert len(trajectory["env_action"]) == 1
    assert metadata["seed"] == 7
    indexed = scan_trajectories([tmp_path])
    assert all(Path(item["trajectory"]).name != "source_trajectory.npz" for item in indexed)


def test_ui_manifest_and_summary_are_compact(tmp_path: Path):
    manager = object.__new__(SimulationManager)
    manager.eval_config = SimpleNamespace(
        level="LEVEL1",
        task_name="task_name",
        checkpoint="org/model",
    )
    record = SimulationSession(
        id="compact",
        kind="branch",
        output_dir=tmp_path,
        max_steps=300,
        open_loop_steps=1,
        task_id=TASK_ID,
        task_level="LEVEL1",
        task_name="task_name",
        task_prompt="place object",
        parent_session_id="parent",
        root_session_id="root",
        resume_step=40,
        control_mode="manual",
        manual_source="spacemouse",
        manual_translation_gain=0.25,
        manual_rotation_gain=0.08,
        status="COMPLETED",
        completed_at="2026-08-15T00:00:01+00:00",
        current_step=300,
        state_count=301,
        action_count=300,
        success=True,
        stopped_reason="source_horizon",
        measured_control_hz=19.9,
        simulated_duration_seconds=15.0,
        trajectory=str(tmp_path / "trajectory.npz"),
    )
    manager._persist_manifest(record)
    manifest = json.loads((tmp_path / "run.json").read_text(encoding="utf-8"))
    assert manifest["managed_by"] == "liberox_data_studio"
    assert manifest["trajectory"] == "trajectory.npz"
    assert "artifacts" not in manifest
    assert "output_dir" not in manifest
    assert "preparation_timing" not in manifest

    summary = manager._summary(
        record,
        {
            "simulated_duration_seconds": 15.0,
            "measured_control_hz": 19.9,
            "control_interval_mean_seconds": 0.0502,
            "control_interval_max_seconds": 0.054,
        },
        0.0,
        {
            "type": "spacemouse",
            "sample_count": 260,
            "translation_gain": 0.25,
            "rotation_gain": 0.08,
        },
    )
    assert summary["result"] == {
        "success": True,
        "steps": 300,
        "target_steps": 300,
        "policy_queries": 0,
        "stopped_reason": "source_horizon",
        "error": None,
    }
    assert summary["timing"]["mean_control_interval_ms"] == pytest.approx(50.2)
    assert summary["branch"]["resume_step"] == 40
    assert summary["controller"]["sample_count"] == 260
    for duplicate_key in ("artifacts", "output_dir", "state_count", "action_count"):
        assert duplicate_key not in summary

    public = manager._public_from_persisted(tmp_path, manifest, summary)
    assert public["id"] == "compact"
    assert public["action_count"] == 300
    assert public["resume_step"] == 40
    assert public["manual_source"] == "spacemouse"
    assert public["manual_rotation_gain"] == 0.08


def test_delete_requires_confirmation_and_managed_marker(tmp_path: Path):
    manager = object.__new__(SimulationManager)
    manager.ui_config = SimpleNamespace(output_root=tmp_path)
    manager.catalog = FakeCatalog()
    manager.lock = threading.RLock()
    manager.active_session_id = None
    manager.controller = None
    manager.spacemouse_config_error = "disabled"
    manager.legacy_sessions = {}
    manager._trajectory_cache = {}
    record = manager._new_record(kind="original", max_steps=1, open_loop_steps=1)
    record.status = "COMPLETED"
    manager.sessions = {record.id: record}
    (record.output_dir / "run.json").write_text(
        json.dumps({"id": record.id}), encoding="utf-8"
    )
    with pytest.raises(ValueError, match="confirm_session_id"):
        manager.delete_session(record.id, "wrong")
    manager.delete_session(record.id, record.id)
    assert not record.output_dir.exists()
    assert record.id not in manager.sessions


def test_delete_rejects_legacy_and_symlink_directories(tmp_path: Path):
    manager = object.__new__(SimulationManager)
    manager.ui_config = SimpleNamespace(output_root=tmp_path)
    manager.lock = threading.RLock()
    manager.active_session_id = None
    manager.controller = None
    manager.spacemouse_config_error = "disabled"
    manager.sessions = {}
    manager._trajectory_cache = {}
    manager.legacy_sessions = {
        "legacy": {
            "id": "legacy", "managed": False, "status": "COMPLETED",
            "output_dir": str(tmp_path / "external"),
        }
    }
    with pytest.raises(ValueError, match="UI-owned"):
        manager.delete_session("legacy", "legacy")

    target = tmp_path / "target"
    target.mkdir()
    (target / "run.json").write_text(json.dumps({"id": "linked"}), encoding="utf-8")
    link = tmp_path / "linked"
    link.symlink_to(target, target_is_directory=True)
    manager.sessions["linked"] = SimulationSession(
        id="linked", kind="original", output_dir=link, max_steps=1,
        open_loop_steps=1, status="COMPLETED", managed=True,
    )
    with pytest.raises(ValueError, match="symbolic-link"):
        manager.delete_session("linked", "linked")
