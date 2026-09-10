"""Shared client IPC/subprocess tests with a motor-free worker, no ROS needed."""
import dataclasses
import sys
import time
import pytest
from backend.app.devices.factr import load_factr_config
from backend.app.devices.factr_calibration import make_profile
from backend.app.devices.factr_client import FactrClient

FAKE_WORKER = r'''
import json, select, socket, sys, time
s = socket.socket(fileno=int(sys.argv[-1]))
cfg = json.loads(s.recv(65536))["config"]
ops, seq, calibrated, enabled = [], 0, False, False
def status():
    global seq
    seq += 1
    return {"state": "READY" if calibrated else "UNCALIBRATED", "calibrated": calibrated,
      "gravity_enabled": enabled, "fingerprint": {"fake": True}, "operations": ops,
      "sample": {"sequence": seq, "sample_monotonic": time.monotonic(),
        "raw_joints": [0]*7, "raw_gripper": 0., "joint_positions": cfg["reference_joint_positions"]}}
while True:
    if select.select([s], [], [], .02)[0]:
        data = s.recv(65536)
        if not data: break
        req = json.loads(data)
        op = req["operation"]
        if op == "heartbeat": continue
        if op == "close": break
        if op == "fail":
            enabled = False
            s.send(json.dumps({"fatal": "alignment failed", "shutdown": {
                "verified": req["value"], "error": None if req["value"] else "OFF readback failed"}}).encode())
            raise SystemExit(1)
        ops.append(op)
        error, result = None, None
        if op == "calibrate":
            result = {"offsets": [0.]*7, "gripper_open": 0., "gripper_closed": -.8,
                      "device_fingerprint": {"fake": True}, "config_hash": "", "created_at": "test"}
            calibrated = True
        if op == "profile": calibrated = True
        if op == "enable":
            if calibrated: enabled = True
            else: error = "calibrate first"
        if op == "disable": enabled = False
        msg = {"reply": req["id"], "result": result, "status": status()}
        if error: msg["error"] = error
        s.send(json.dumps(msg).encode())
    s.send(json.dumps({"status": status()}).encode())
'''

@pytest.fixture
def client(monkeypatch):
    monkeypatch.setattr("backend.app.devices.factr_client.verify_upstream", lambda _: {})
    service = FactrClient(load_factr_config(), worker_command=[sys.executable, "-c", FAKE_WORKER])
    yield service
    service.close()


def test_passive_start_and_explicit_commands(client):
    assert not client.status()["gravity_enabled"]
    assert client.status()["operations"] == []
    with pytest.raises(RuntimeError, match="calibrate"):
        client.enable_gravity()
    client.calibrate()
    client.arm("test", .25, .25, gripper=1.)
    assert client.snapshot("test").gripper_command == 1.
    assert not client.status()["gravity_enabled"]
    with pytest.raises(RuntimeError, match="Stop simulation"):
        client.calibrate()
    client.enable_gravity()
    with pytest.raises(RuntimeError, match="armed"):
        client.start_alignment("test", client.config.reference_joint_positions)
    client.disarm("test")
    assert client.status()["gravity_enabled"]  # Do not drop when simulation ends.
    client.start_alignment("next", client.config.reference_joint_positions)
    client.finish_alignment("next")
    assert client.status()["operations"][-2:] == ["align", "cancel_align"]
    assert client.status()["gravity_enabled"]
    client.disable_gravity()
    assert not client.status()["gravity_enabled"]


def test_stale_disarms_and_gripper_requires_deliberate_takeover(client):
    client.set_profile(make_profile(client.config, [0]*7, 0., .5, client.fingerprint))
    client.arm("test", .25, .25, gripper=1.)
    with client._lock:
        # Trigger started open but source gripper is closed: do not jump open.
        client._update_gripper(0.)
        assert client._gripper == 1.
        client._update_gripper(1.)
        client._update_gripper(0.)
        assert client._gripper == -1.
        client._snapshot = dataclasses.replace(client._snapshot, sample_monotonic=time.monotonic()-1)
        assert client.latest_snapshot().stale
        assert client._owner is None


def test_worker_exit_is_observable_without_terminal_input(client):
    client._process.terminate()
    client._process.wait(timeout=2)
    deadline = time.monotonic()+2
    while not client.latest_snapshot().error and time.monotonic() < deadline:
        time.sleep(.01)
    assert client.latest_snapshot().error and not client.latest_snapshot().connected
    with pytest.raises(RuntimeError, match="exited"):
        client.close()


def test_missing_runtime_exits_cleanly(monkeypatch):
    monkeypatch.setattr("backend.app.devices.factr_client.verify_upstream", lambda _: {})
    with pytest.raises(RuntimeError, match="disconnected|reset|Broken"):
        FactrClient(load_factr_config(), worker_command=[sys.executable, "-c", "raise SystemExit(2)"], startup_timeout=1)


def test_startup_traceback_survives_socket_reset(monkeypatch, capsys):
    monkeypatch.setattr("backend.app.devices.factr_client.verify_upstream", lambda _: {})
    worker = "raise RuntimeError('USB latency_timer=16 ms, requires 1 ms')"
    with pytest.raises(RuntimeError, match="USB latency_timer=16 ms, requires 1 ms"):
        FactrClient(load_factr_config(), worker_command=[sys.executable, "-c", worker], startup_timeout=2)
    assert "USB latency_timer=16 ms" in capsys.readouterr().err


def test_startup_fatal_packet_is_read_before_heartbeat(monkeypatch):
    monkeypatch.setattr("backend.app.devices.factr_client.verify_upstream", lambda _: {})
    worker = '''
import socket, sys, json
s = socket.socket(fileno=int(sys.argv[-1]))
s.recv(65536)
s.send(json.dumps({"fatal": "startup preflight rejected", "shutdown": {"verified": True}}).encode())
raise SystemExit(1)
'''
    with pytest.raises(RuntimeError, match="^startup preflight rejected$"):
        FactrClient(load_factr_config(), worker_command=[sys.executable, "-c", worker], startup_timeout=2)


@pytest.mark.parametrize("verified", [True, False])
def test_fatal_preserves_actual_off_verification(client, verified):
    with pytest.raises(RuntimeError, match="^alignment failed$"):
        client._call("fail", verified)
    assert client.latest_snapshot().error == "alignment failed"
    if verified:
        client.close()  # Nonzero exit is a fault, but not an unverified shutdown.
    else:
        with pytest.raises(RuntimeError, match="OFF readback failed"):
            client.close()
