import json
import threading
from pathlib import Path
from types import SimpleNamespace

import pytest
import numpy as np

from backend.app.core.exceptions import ConflictError
from backend.app.storage.repositories import RunRepository
from backend.app.workers.simulation_worker import SimulationManager
from backend.app.workers.simulation_worker import PreviewService
from backend.app.domain.run import SimulationDraft, SimulationSession
from test_ui_tasks_draft import _DraftCatalog


def test_missing_policy_blocks_requery_but_not_manual_record_or_metadata(tmp_path):
    manager = object.__new__(SimulationManager)
    manager.ui_config = SimpleNamespace(output_root=tmp_path)
    manager.eval_config = SimpleNamespace(control_hz=20)
    manager.catalog = _DraftCatalog()
    manager.catalog.paths = lambda _: (Path("task.bddl"), Path("task.init"))
    manager.spacemouse_config = None
    manager.controller = None
    def missing(_):
        raise ValueError("Policy no longer exists")
    manager.policy_catalog = SimpleNamespace(select=missing)
    manager.provider = SimpleNamespace(metadata=lambda: pytest.fail("manual queried deleted model"))
    parent = {"id": "parent", "trajectory": "source.npz", "task_id": "LEVEL1::task_a",
              "policy_id": "deleted", "policy_label": "Historical overlay"}
    with pytest.raises(ConflictError) as error:
        manager._new_record(kind="branch", max_steps=8, open_loop_steps=8, parent=parent, resume_step=0)
    assert error.value.detail()["severity"] == "warning"
    assert error.value.code == "POLICY_UNAVAILABLE"
    manual = manager._new_record(kind="branch", max_steps=8, open_loop_steps=8,
                                 parent=parent, resume_step=0, control_mode="manual")
    assert manual.policy_id == "deleted" and manual.manual_source == "spacemouse"
    assert manager._metadata(manual, {}, None)["policy_device"] is None


@pytest.mark.parametrize("status", ["QUEUED", "LOADING", "RUNNING", "READY"])
def test_restart_recovers_orphan_in_manifest_and_catalog(tmp_path, status):
    directory = tmp_path / "runs" / "interrupted"
    directory.mkdir(parents=True)
    manifest = directory / "run.json"
    manifest.write_text(json.dumps({"id": "orphan", "kind": "original", "status": status,
        "task_id": "LEVEL1::task_a", "created_at": "2026-10-05T00:00:00Z"}))
    (directory / "summary.json").write_text(json.dumps({"result": {"error": None}}))
    manager = object.__new__(SimulationManager)
    manager.ui_config = SimpleNamespace(scan_roots=(), output_root=tmp_path / "runs")
    manager.lock = threading.RLock()
    manager.sessions = {}
    manager.catalog = _DraftCatalog()
    manager.repository = RunRepository(tmp_path / "catalog.sqlite", "test")
    manager.refresh_history()
    assert manager.legacy_sessions["orphan"]["status"] == "ERROR"
    assert manager.repository.summary()["errors"] == 1
    assert json.loads(manifest.read_text())["stopped_reason"] == "backend_interrupted"
    manager.refresh_history()
    assert manager.legacy_sessions["orphan"]["error"] == "Backend interrupted before completion"


def test_busy_draft_retry_does_not_evict_session_first_frame(tmp_path, monkeypatch):
    blocked, proceed = threading.Event(), threading.Event()
    def busy(_):
        blocked.set()
        assert proceed.wait(3)
        raise ConflictError("Busy", code="WORK_RESOURCE_BUSY")
    monkeypatch.setattr("backend.app.workers.simulation_worker.ResourceLease", busy)
    draft = SimulationDraft(id="draft", task_id="task", max_steps=8, open_loop_steps=8)
    session = SimulationSession(id="sim", kind="original", task_id="task", output_dir=tmp_path,
                                max_steps=8, open_loop_steps=8, resource_owned=True)
    manager = SimpleNamespace(eval_config=object(), lock=threading.RLock(), _frame_lock=threading.Lock(),
        draft=draft, work_queue=SimpleNamespace(gpu_lock_path=tmp_path / "lease"),
        ui_config=SimpleNamespace(preview_fps=100, preview_width=4, preview_height=4, jpeg_quality=85),
        catalog=SimpleNamespace(paths=lambda _: (Path("task"), Path("init"))),
        simulator=SimpleNamespace(create=lambda *_, **__: object(), close=lambda _: None,
            restore=lambda *_: None, render_operator_preview=lambda *_: np.zeros((4, 4, 3), dtype=np.uint8)))
    preview = PreviewService(manager)
    try:
        preview.submit(draft, np.zeros(3), draft.preview_revision)
        assert blocked.wait(2)
        preview.submit(session, np.zeros(3))
        proceed.set()
        assert session.preview_event.wait(3)
        assert session.preview_error is None and session.preview_ready
    finally:
        proceed.set()
        preview.close()
