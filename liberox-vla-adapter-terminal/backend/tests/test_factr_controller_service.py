"""Shared GUI/CLI lifecycle, no physical hardware."""
from dataclasses import replace
import time
import pytest
from backend.app.devices.factr import FactrSnapshot, load_factr_config
from backend.app.devices.factr_calibration import make_profile
from backend.app.services.factr_controller_service import FactrControllerService


class Client:
    def __init__(self, config):
        self.config, self.gravity, self.closed, self.stale = config, False, False, False
        self.calls = []
    def calibrate(self):
        self.calls.append("calibrate")
        return make_profile(self.config, [0]*7, 0., -.8, {"official": True})
    def status(self): return {"gravity_enabled": self.gravity}
    def start_alignment(self, owner, joints): self.calls.append("align")
    def finish_alignment(self, owner): self.calls.append("release")
    def latest_snapshot(self):
        now = time.monotonic()
        return FactrSnapshot(sequence=1, sample_monotonic=now-.001, captured_monotonic=now,
            connected=not self.closed, stale=self.stale, joint_positions=self.config.reference_joint_positions)
    def enable_gravity(self): self.calls.append("enable"); self.gravity = True
    def disable_gravity(self): self.calls.append("disable"); self.gravity = False
    def arm(self, *args, **kwargs): self.calls.append("arm"); return self.latest_snapshot()
    def disarm(self, *args): self.calls.append("disarm")
    def snapshot(self, *args): return self.latest_snapshot()
    def close(self): self.calls.append("close"); self.gravity = False; self.closed = True
    def diagnostics(self): return {"fake": True}


@pytest.fixture
def service(tmp_path):
    config = load_factr_config()
    config = replace(config, runtime={**config.runtime, "calibration_file": str(tmp_path/"calibration.json")})
    service = FactrControllerService(config, input_factory=Client, probe=lambda _: {"connected": True}, start_monitor=False)
    yield service
    service.close()


def calibrate(service):
    service.start_calibration()
    deadline = time.monotonic()+2
    while service.status()["state"] == "CALIBRATING" and time.monotonic() < deadline:
        time.sleep(.001)
    assert service.status()["state"] == "READY"


def test_one_capture_no_automatic_output_and_persistent_support(service):
    assert service.status()["state"] == "UNCALIBRATED" and service._input is None
    calibrate(service)
    client = service._input
    assert client.calls == ["calibrate"]
    service.set_gravity(True)
    for session in ("first", "rewind"):
        service.arm(session, .25, .25)
        service.snapshot(session)
        service.set_gains(session, .4, .4)
        service.disarm(session)
        assert service.status()["gravity_enabled"]
    assert client.calls.count("enable") == 1
    service.set_gravity(False)
    assert not service.status()["gravity_enabled"] and not client.closed


def test_error_and_close_remove_support_without_rearm(service):
    calibrate(service)
    client = service._input
    service.set_gravity(True)
    service.arm("s", .25, .25)
    service.emergency_stop("MuJoCo failed")
    assert client.closed and not client.gravity
    assert service.status()["state"] == "ERROR" and not service.status()["calibrated"]
    service._poll_once()
    assert service._input is None  # No implicit reopen/rearm after reconnect.
    calibrate(service)
    client2 = service._input
    assert client2 is not client and not client2.gravity
    service.set_gravity(True)
    service.close()
    assert client2.closed and not client2.gravity


def test_active_calibration_rejected_but_gravity_can_be_toggled(service):
    calibrate(service)
    service.arm("s", .25, .25)
    with pytest.raises(RuntimeError): service.start_calibration()
    service.set_gravity(True)
    assert service.status()["gravity_enabled"]
    service.set_gravity(False)
    assert service.status()["state"] == "ARMED"


def test_stale_fault_disables_support(service):
    calibrate(service)
    service.set_gravity(True)
    service.arm("s", .25, .25)
    client = service._input
    client.stale = True
    service._poll_once()
    assert client.closed and service.status()["state"] == "ERROR"


def test_standby_stale_preview_does_not_turn_off_healthy_physical_runtime(service):
    calibrate(service)
    service.set_gravity(True)
    client = service._input
    service.arm("first", .25, .25)
    service.disarm("first")
    client.stale = True
    service._poll_once()
    assert service.status()["state"] == "READY"
    assert service.status()["calibrated"] and service.status()["gravity_enabled"]
    assert not client.closed
    # Still cannot arm a real client until a fresh packet arrives.
    client.stale = False
    service._poll_once()
    service.arm("next", .25, .25)
    assert client.calls.count("enable") == 1


def test_runtime_disconnect_still_stops_even_when_not_armed(service):
    calibrate(service)
    service.set_gravity(True)
    client = service._input
    client.closed = True
    service._poll_once()
    assert not client.gravity and service.status()["state"] == "ERROR"
    original_error = service.status()["error"]
    service.emergency_stop("later cleanup")
    assert service.status()["error"] == original_error


def test_admin_transition_does_not_mistake_old_snapshot_for_runtime_fault(service):
    calibrate(service)
    client = service._input
    def enable():
        client.stale = True
        service._poll_once()  # Administrative ACK, not a stalled active loop.
        assert not client.closed
        client.stale = False
        client.gravity = True
    client.enable_gravity = enable
    service.set_gravity(True)
    assert service.status()["gravity_enabled"]


def test_calibration_and_gain_validation(service):
    with pytest.raises(RuntimeError): service.set_gravity(True)
    with pytest.raises(ValueError): service.set_gravity("yes")
    calibrate(service)
    service.set_gravity(True)
    with pytest.raises(RuntimeError, match="disable"): service.start_calibration()
    with pytest.raises(ValueError): service.set_gains(None, float("nan"), .25)
    service.arm("s", .25, .25)
    with pytest.raises(RuntimeError): service.set_gains("other", .25, .25)


def test_shutdown_errors_remain_visible(service):
    calibrate(service)
    client = service._input
    def fail(): raise RuntimeError("bus unplugged")
    client.close = fail
    service.emergency_stop("simulation error")
    assert "shutdown NOT verified" in service.status()["error"]
    assert service.status()["gravity_state"] == "unknown"


def test_alignment_owner_and_normal_release_keep_gravity(service):
    calibrate(service)
    service.set_gravity(True)
    client = service._input
    service.start_alignment("s", [0]*7)
    assert service.status()["state"] == "ALIGNING"
    with pytest.raises(RuntimeError): service.arm("s", .25, .25)
    with pytest.raises(RuntimeError): service.start_calibration()
    service.finish_alignment("other")
    assert "release" not in client.calls
    service.finish_alignment("s")
    assert service.status()["state"] == "READY" and client.gravity
    service.arm("s", .25, .25)
    service.disarm("s")
    assert client.gravity and not client.closed
