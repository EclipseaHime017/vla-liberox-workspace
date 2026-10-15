"""Promotion changes operation roles, never the physical record or reward goal."""
import hashlib
import json
import threading
from types import SimpleNamespace

import pytest

from backend.app.services.run_continuation import continuation_public, promote_run, prefix_segments, instruction_at
from backend.app.domain.run import SimulationSession
from backend.app.services.task_registry import freeze_task
from backend.app.workers.simulation_worker import SimulationManager
from test_task_registry import custom_catalog
from test_ui_config_storage import _write_minimal_trajectory


def test_promotion_is_persistent_idempotent_and_does_not_modify_evaluation_inputs(tmp_path):
    trajectory = tmp_path / "trajectory.npz"
    _write_minimal_trajectory(trajectory)
    manifest = tmp_path / "run.json"
    manifest.write_text('{"kind":"branch","task":"complete task"}')
    evaluation = tmp_path / "rynnvalue_evaluation.json"
    evaluation.write_text('{"status":"READY"}')
    before = [hashlib.sha256(path.read_bytes()).hexdigest() for path in (trajectory, manifest, evaluation)]
    run = dict(id="human", kind="branch", status="COMPLETED", output_dir=str(tmp_path),
               trajectory=str(trajectory), action_count=1, manual_source="spacemouse", root_session_id="root")
    assert continuation_public(run)["can_promote"]
    promoted = promote_run(run)
    assert promoted["role"] == "original" and promoted["kind"] == "branch"
    assert promoted["branchable"] and promoted["manual_source"] == "spacemouse"
    assert not promoted["can_promote"]
    assert continuation_public(run) == promote_run(run)
    assert before == [hashlib.sha256(path.read_bytes()).hexdigest() for path in (trajectory, manifest, evaluation)]


def test_last_state_can_continue_with_more_budget_and_local_instruction(tmp_path):
    trajectory = tmp_path / "trajectory.npz"
    _write_minimal_trajectory(trajectory)
    run = dict(id="human", kind="branch", status="COMPLETED", branchable=False,
               output_dir=str(tmp_path), trajectory=str(trajectory), action_count=1, max_steps=1,
               task_id="task", policy_id="trained", root_session_id="root")
    promote_run(run)
    manager = object.__new__(SimulationManager)
    manager.get_public = lambda _: continuation_public(run)
    manager.eval_config = SimpleNamespace(seed=0)
    manager.work_queue = object()
    created = []
    manager._new_record = lambda **kwargs: created.append(kwargs) or SimpleNamespace(id="next")
    manager._start_record = lambda record: {"id": record.id}
    with pytest.raises(ValueError, match="greater than resume_step"):
        manager.create_branch("human", 1, "policy", 1)
    assert manager.create_branch("human", 1, "policy", 1,
                                 max_steps=5, policy_instruction="close the drawer") == {"id": "next"}
    assert created[-1]["resume_step"] == 1 and created[-1]["max_steps"] == 5
    assert created[-1]["policy_instruction"] == "close the drawer"
    assert created[-1]["parent"]["root_session_id"] == "root"


def test_nested_prefix_keeps_each_original_instruction_and_origin():
    metadata = dict(task="complete task", policy_instruction="place bowl", trajectory_segments=[
        dict(start=0, end=3, origin_run_id="root", policy_instruction="open drawer"),
        dict(start=3, end=7, origin_run_id="human", policy_instruction="grasp bowl"),
        dict(start=7, end=12, origin_run_id="policy", policy_instruction="place bowl")])
    assert instruction_at(metadata, 12, 4) == "grasp bowl"
    clipped = prefix_segments(metadata, 12, 9)
    assert [segment["origin_run_id"] for segment in clipped] == ["root", "human", "policy"]
    assert clipped[-1]["end"] == 9
    assert metadata["trajectory_segments"][-1]["end"] == 12
    assert metadata["task"] == "complete task"


def test_declared_missing_snapshot_does_not_fall_back_to_current_registry(tmp_path):
    manager = object.__new__(SimulationManager)
    manager.catalog = object()
    with pytest.raises(FileNotFoundError, match="snapshot"):
        manager._task_catalog_for(dict(output_dir=str(tmp_path), task_snapshot="task_snapshot/task.json"))


def test_multi_generation_chunk_summary_counts_all_boundaries(tmp_path):
    record = SimulationSession(id="continued", kind="branch", output_dir=tmp_path,
        max_steps=16, open_loop_steps=8, resume_step=8, action_count=16, task_prompt="full goal",
        policy_instruction="finish task", handoff_steps=[4, 8], prefix_segments=[
            dict(start=0, end=4, origin_run_id="root", policy_instruction="first"),
            dict(start=4, end=8, origin_run_id="human", policy_instruction="second")])
    assert record.public()["training_chunk_count"] == 3
    assert record.trajectory_segments[-1] == dict(start=8, end=16,
        origin_run_id="continued", policy_instruction="finish task")


def test_restart_keeps_promoted_role_and_frozen_task_removed_from_registry(tmp_path, monkeypatch):
    catalog, _ = custom_catalog(tmp_path, monkeypatch)
    directory = tmp_path / "runs" / "human"
    frozen = freeze_task(catalog, catalog.default_task_id, directory / "task_snapshot")
    trajectory = directory / "trajectory.npz"
    _write_minimal_trajectory(trajectory)
    run = dict(id="human", kind="branch", status="COMPLETED", action_count=1,
        output_dir=str(directory), trajectory=str(trajectory), task_id=catalog.default_task_id,
        task="complete both steps", task_snapshot=str(frozen.manifest_path))
    (directory / "run.json").write_text(json.dumps(run))
    promote_run(run)
    manager = object.__new__(SimulationManager)
    manager.runtime, manager.lock, manager.sessions = catalog.runtime, threading.RLock(), {}
    manager.ui_config = SimpleNamespace(scan_roots=(), output_root=tmp_path / "runs")
    manager.catalog = SimpleNamespace(resolve_id=lambda *_: None)
    monkeypatch.setattr("backend.app.workers.simulation_worker.scan_trajectories", lambda _: [dict(run)])
    manager.refresh_history()
    restored = manager.get_public("human")
    assert restored["task_id"] == catalog.default_task_id
    assert restored["role"] == "original" and restored["kind"] == "branch" and restored["branchable"]
    assert manager._task_catalog_for(restored).paths(restored["task_id"]) == frozen.paths(restored["task_id"])
