import asyncio
import json
import threading
from concurrent.futures import ThreadPoolExecutor
from dataclasses import replace
from types import SimpleNamespace

import numpy as np
import pytest
import yaml
from pydantic import ValidationError

from backend.app.api.models import CreateBranchRequest
from backend.app.devices.spacemouse import SpaceMouseInput, load_spacemouse_config
from backend.app.devices.spacemouse_motion import SpaceMouseActionMapper
from backend.app.domain.run import SimulationSession
from backend.app.services.controller_service import SpaceMouseControllerService
from backend.app.workers.simulation_worker import SimulationManager


def test_branch_api_passes_frame_only_for_supported_controller():
    from backend.app.api.runs import branch
    request = dict(resume_step=1, control_mode="manual", open_loop_steps=1)
    assert CreateBranchRequest(**request).control_frame is None
    assert CreateBranchRequest(**request, control_frame="tool").controller_id is None
    for invalid in ({"control_frame": "camera"},
                    {"control_mode": "policy", "control_frame": "world"},
                    {"controller_id": "factr", "control_frame": "tool"}):
        with pytest.raises(ValidationError):
            CreateBranchRequest(**(request | invalid))
    calls = []
    service = SimpleNamespace(create_branch=lambda *args, **kwargs: calls.append((args, kwargs)))
    http = SimpleNamespace(app=SimpleNamespace(state=SimpleNamespace(run_service=service)))
    asyncio.run(branch("parent", CreateBranchRequest(**request, control_frame="tool"), http))
    assert calls[-1][1]["control_frame"] == "tool"
    asyncio.run(branch("parent", CreateBranchRequest(**request), http))
    assert "control_frame" not in calls[-1][1]


def test_live_settings_map_at_control_boundary_and_persist_actual_frame(tmp_path):
    worker = object.__new__(SimulationManager)
    worker.lock = threading.RLock()
    worker.spacemouse_config = replace(load_spacemouse_config(), deadzone=0, smoothing_alpha=1,
                                      translation_gain=.25, rotation_gain=.08)
    worker.ui_config = SimpleNamespace(project_id="test")
    worker.eval_config = SimpleNamespace(control_hz=20, checkpoint="base")
    worker.provider = SimpleNamespace(metadata=lambda: {"model_device": "cpu"})
    worker.catalog = SimpleNamespace(paths=lambda _: ("task.bddl", "task.init"))
    clock = [10.]
    mouse = SpaceMouseInput(worker.spacemouse_config, clock=lambda: clock[0])
    mouse._connected = True
    mouse.reset_for_arm(gripper=1)
    worker.controller = SimpleNamespace(snapshot=lambda _: mouse.latest_snapshot(),
        set_gains=lambda _, t, r: mouse.set_gains(t, r),
        request_control_frame=lambda _, frame: mouse.request_control_frame(frame))
    record = SimulationSession(id="s", kind="branch", output_dir=tmp_path, max_steps=10,
        open_loop_steps=1, control_mode="manual", manual_source="spacemouse", status="RUNNING",
        manual_control_frame="world", manual_requested_control_frame="world")
    worker.sessions = {"s": record}
    worker._persist_effective_config(record)
    ctrl = SimpleNamespace(use_delta=True, control_dim=6, input_min=-1, input_max=1,
        output_min=[-.05]*3+[-.5]*3, output_max=[.05]*3+[.5]*3, eef_name="eef")
    robot = SimpleNamespace(controller=ctrl, sim=SimpleNamespace(forward=lambda: None,
        data=SimpleNamespace(get_site_xmat=lambda _: np.array([[0, -1, 0], [1, 0, 0], [0, 0, 1]]))))
    mapper = SpaceMouseActionMapper(SimpleNamespace(robots=[robot]))
    mouse._accept_event(1, (0, .8, 0, 0, 0, 0), (0, 0))
    worker.manual_settings("s", .25, .08, control_frame="tool")
    assert record.public()["manual_requested_control_frame"] == "tool"
    before = worker._spacemouse_action(record, 0, mapper)
    assert np.allclose(before, [.2, 0, 0, 0, 0, 0, 1])
    assert record.manual_control_frame == "world"
    mouse._accept_event(2, (0,)*6, (0, 0))
    assert not worker._spacemouse_action(record, 1, mapper)[:6].any()
    # Old clients/reconnects only update gains, not the selected frame.
    worker.manual_settings("s", .5, .08)
    assert record.manual_requested_control_frame == "tool"
    mouse._accept_event(3, (0, .8, 0, 0, 0, 0), (0, 0))
    after = worker._spacemouse_action(record, 2, mapper)
    assert np.allclose(after, [0, .4, 0, 0, 0, 0, 1])
    row = record.spacemouse_samples[-1]
    assert row["control_frame"] == "tool"
    assert row["command_x"] == .4 and row["env_command_y"] == pytest.approx(.4)
    assert not row["frame_mapping_limited"]
    worker._persist_manifest(record)
    manifest = json.loads((tmp_path / "run.json").read_text())
    assert manifest["manual_control_frame"] == "tool"
    # Initial config remains immutable; actual frame and per-step changes are recorded separately.
    assert yaml.safe_load((tmp_path / "config.yaml").read_text())["controller"]["control_frame"] == "world"
    summary = worker._persist_spacemouse_samples(record)
    assert summary["control_frame"] == "tool"
    public = worker._public_from_persisted(tmp_path, manifest, {"controller": summary})
    assert public["manual_control_frame"] == public["manual_requested_control_frame"] == "tool"
    assert worker._metadata(record, {}, None)["manual_control_frame"] == "tool"
    for value in ("camera", True):
        with pytest.raises(ValueError, match="control_frame"):
            worker.manual_settings("s", .25, .08, control_frame=value)
    record.manual_source = "factr"
    with pytest.raises(ValueError, match="SpaceMouse"):
        worker.manual_settings("s", .25, .08, control_frame="world")


@pytest.mark.parametrize("source,expected", [("spacemouse", "world"), ("factr", None), (None, None)])
def test_old_history_defaults_world_only_for_spacemouse(tmp_path, source, expected):
    public = SimulationManager._public_from_persisted(tmp_path,
        {"id": "old", "manual_source": source}, {})
    assert public["manual_control_frame"] == expected


def test_frame_request_during_arm_is_not_lost_or_applied_before_neutral(tmp_path):
    config = load_spacemouse_config()
    mouse = SpaceMouseInput(config)
    mouse._connected = True
    service = SpaceMouseControllerService(config, start_monitor=False)
    service._input, service._state, service._calibration = mouse, "READY", object()
    worker = object.__new__(SimulationManager)
    worker.lock, worker.controller = threading.RLock(), service
    record = SimulationSession(id="s", kind="branch", output_dir=tmp_path, max_steps=10,
        open_loop_steps=1, control_mode="manual", manual_source="spacemouse", status="READY",
        manual_translation_gain=.25, manual_rotation_gain=.08,
        manual_control_frame="world", manual_requested_control_frame="world")
    worker.sessions = {"s": record}
    arm_entered, finish_arm, settings_started = (threading.Event() for _ in range(3))
    arm = service.arm

    def paused_arm(*args, **kwargs):
        arm_entered.set()
        assert finish_arm.wait(2)
        arm(*args, **kwargs)
        # A held joystick must still require neutral after this request.
        mouse._accept_event(1, (.8, 0, 0, 0, 0, 0), (0, 0))

    def change_frame():
        settings_started.set()
        worker.manual_settings("s", .5, .1, control_frame="tool")

    service.arm = paused_arm
    with ThreadPoolExecutor(max_workers=2) as pool:
        arming = pool.submit(worker._arm_spacemouse, record, 1.)
        try:
            assert arm_entered.wait(2)
            changing = pool.submit(change_frame)
            assert settings_started.wait(2)
        finally:
            finish_arm.set()
        arming.result(timeout=2)
        changing.result(timeout=2)
    # Session remains READY until the run loop advances, but controller is armed.
    assert record.manual_requested_control_frame == "tool"
    assert record.manual_control_frame == "world"
    assert mouse.latest_snapshot().pending_control_frame == "tool"
    mouse._accept_event(2, (0,) * 6, (0, 0))
    snapshot = mouse.latest_snapshot()
    assert snapshot.control_frame == "tool" and snapshot.pending_control_frame is None
    assert snapshot.action[-1] == 1.
    assert mouse.transform.gains == (.5, .1)
