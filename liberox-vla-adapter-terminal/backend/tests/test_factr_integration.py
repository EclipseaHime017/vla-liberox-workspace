import asyncio
import json
import threading
import zipfile
from types import SimpleNamespace

import numpy as np
import pytest
from fastapi import FastAPI, Request
from pydantic import ValidationError

from backend.app.api import controller as api
from backend.app.api.models import ControllerCalibrationRequest, ControllerGravityRequest, CreateBranchRequest
from backend.app.domain.run import SimulationSession
from backend.app.devices.factr import FactrSnapshot
from backend.app.services.dataset_service import DatasetService
from backend.app.services.run_service import RunService
from backend.app.workers.simulation_worker import SimulationManager


class Controller:
    def __init__(self, state="READY"):
        self.state = state
        self.disarmed = False

    def status(self):
        return dict(state=self.state, connected=True, calibrated=True, stale=False, latency_ms=2.)

    def start_calibration(self):
        self.state = "CALIBRATING"
        return self.status()

    def snapshot(self, _session_id):
        raise RuntimeError("incomplete packet")

    def disarm(self, _session_id):
        self.disarmed = True

    def emergency_stop(self, reason):
        self.disarmed = True
        self.state = "ERROR"

    def set_gravity(self, enabled):
        return {**self.status(), "gravity_enabled": enabled}


def manager():
    worker = object.__new__(SimulationManager)
    worker.lock = threading.RLock()
    worker.active_session_id = None
    worker.controller = Controller()
    worker.factr_controller = Controller()
    return worker


def test_controller_api_default_and_factr_calibration_conflicts():
    worker = manager()
    app = FastAPI()
    app.include_router(api.router)
    app.state.run_service = RunService(worker)
    request = Request({"type": "http", "app": app})

    async def check():
        assert (await api.status(request))["controller_id"] == "spacemouse"
        assert [c["controller_id"] for c in (await api.controllers(request))["controllers"]] == ["spacemouse", "factr"]
        reply = await api.calibrate(request, ControllerCalibrationRequest(), "factr")
        assert reply["state"] == "CALIBRATING"
        with pytest.raises(Exception) as exc:
            await api.calibrate(request)
        assert exc.value.status_code == 409
        worker.active_session_id = "running"
        with pytest.raises(Exception) as exc:
            await api.calibrate(request, None, "factr")
        assert exc.value.status_code == 409

    asyncio.run(check())
    assert {r.path for r in api.router.routes} >= {"/api/controller", "/api/controllers", "/api/controller/calibrate"}


def test_gravity_api_works_during_takeover_and_rejects_invalid_inputs():
    worker = manager()
    worker.active_session_id = "running"
    app = FastAPI()
    app.state.run_service = RunService(worker)
    request = Request({"type": "http", "app": app})
    assert api.gravity(request, ControllerGravityRequest(enabled=True))["gravity_enabled"]
    assert not api.gravity(request, ControllerGravityRequest(enabled=False))["gravity_enabled"]
    with pytest.raises(ValidationError):
        ControllerGravityRequest(enabled="yes")
    with pytest.raises(ValidationError):
        ControllerCalibrationRequest(phase="gripper_open")
    with pytest.raises(ValueError):
        worker.set_controller_gravity("spacemouse", True)


def test_backend_shutdown_disables_factr_before_waiting_for_simulation():
    worker = manager()
    calls = []
    worker.factr_controller.close = lambda: calls.append("physical_off")
    worker.active_session_id = "running"
    worker.sessions = {"running": SimpleNamespace(
        stop_event=threading.Event(), thread=SimpleNamespace(join=lambda: calls.append("join")))}
    worker.preview = SimpleNamespace(close=lambda: calls.append("preview"))
    worker.controller.close = lambda: calls.append("spacemouse")
    worker._frame_lock = threading.RLock()
    worker._frame_env = None
    worker.provider = SimpleNamespace(unload=lambda: calls.append("provider"))
    worker.close()
    assert calls[:2] == ["physical_off", "join"]
    assert worker.sessions["running"].stop_event.is_set()


def test_policy_requests_reject_controller_and_manual_defaults_remain_compatible():
    body = dict(resume_step=3, open_loop_steps=1, control_mode="manual")
    assert CreateBranchRequest(**body).controller_id is None
    assert CreateBranchRequest(**body, controller_id="factr").controller_id == "factr"
    with pytest.raises(ValidationError):
        CreateBranchRequest(**{**body, "control_mode": "policy"}, controller_id="factr")
    with pytest.raises(ValidationError):
        CreateBranchRequest(**body, controller_id="browser")




def test_factr_metadata_and_export_preserve_manual_classification(tmp_path):
    episode = tmp_path / "episodes" / "episode_000"
    episode.mkdir(parents=True)
    samples = episode / "factr_samples.csv"
    samples.write_text("step,raw_joint_0,osc_action_0\n4,1,.2\n")
    record = SimulationSession(id="factr", kind="branch", output_dir=tmp_path,
                               max_steps=10, open_loop_steps=1, control_mode="manual", manual_source="factr")
    public = record.public({"episodes/episode_000/factr_samples.csv": str(samples)})
    public["task_id"] = "task"
    service = DatasetService(SimpleNamespace(list_runs=lambda: [public]))
    assert service._episode_category(public) == "manual_intervention"
    path, _ = service.export_task("task")
    try:
        with zipfile.ZipFile(path) as archive:
            assert "runs/factr/episodes/episode_000/factr_samples.csv" in archive.namelist()
            assert "factr" in archive.read("runs.csv").decode()
            assert "human" in archive.read("runs.csv").decode()
    finally:
        path.unlink()


def test_saved_factr_history_not_relabelled_spacemouse(tmp_path):
    manifest = {"id": "f", "manual_source": "factr", "control_mode": "manual", "kind": "branch"}
    (tmp_path / "run.json").write_text(json.dumps(manifest))
    (tmp_path / "summary.json").write_text(json.dumps({"controller": {"type": "factr"}}))
    # Use the same manifest history hydration path as a restarted UI.
    worker = manager()
    result = worker._public_from_persisted(tmp_path, manifest, {"controller": {"type": "factr"}})
    assert result["manual_source"] == "factr"


def test_factr_branch_record_locks_controller_and_parent_context(tmp_path):
    worker = manager()
    worker.ui_config = SimpleNamespace(output_root=tmp_path)
    worker.spacemouse_config = SimpleNamespace(stale_timeout_ms=250)
    worker.factr_controller.config = SimpleNamespace(stale_timeout_ms=200)
    worker.catalog = SimpleNamespace(
        initial_state=lambda *_: np.zeros(1),
        metadata=lambda task_id: dict(task_id=task_id, level="LEVEL1", task_name="bowl", prompt="place bowl"),
    )
    record = worker._new_record(
        kind="branch", max_steps=300, open_loop_steps=1, resume_step=42,
        parent=dict(id="root", task_id="LEVEL1::bowl", trajectory="/tmp/source.npz", seed=17),
        control_mode="manual", controller_id="factr", manual_translation_gain=.25, manual_rotation_gain=.25,
    )
    assert record.manual_source == "factr"
    assert record.seed == 17 and record.action_count == 42 and record.state_count == 43
    assert record.controller_deadman_ms == 200
    assert record.spacemouse_deadman_ms is None
    assert record.public()["controller_id"] == "factr"
    assert record.managed and not record.branchable
    worker._persist_manifest(record)
    assert (record.output_dir / "run.json").is_file()
    service = DatasetService(SimpleNamespace(list_runs=lambda: [record.public()]))
    assert service.list_runs()[0]["id"] == record.id
