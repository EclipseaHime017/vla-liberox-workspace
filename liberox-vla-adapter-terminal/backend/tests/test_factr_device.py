from __future__ import annotations

import ast
import inspect
import json
from types import SimpleNamespace

import pytest
import yaml

from backend.app.devices import factr
from backend.app.devices.factr import (
    FactrStartupProbe, MODEL_NUMBERS, load_factr_config, parse_factr_config,
)
from backend.app.devices.factr_discovery import FactrSerialDevice


def test_default_config_selects_usb_identity_not_fixed_port():
    config = load_factr_config()
    assert (config.vendor_id, config.product_id) == (0x0403, 0x6014)
    assert config.serial_number is None
    assert "device_path" not in config.metadata()
    assert config.motor_ids == tuple(range(1, 9))
    assert config.stale_timeout_ms == 250
    assert config.translation_gain == config.rotation_gain == 0.25
    assert config.reference_joint_positions == (0., -.7854, 0., -2.356, 0., 1.57, 0.)
    assert parse_factr_config(json.loads(json.dumps(config.metadata()))) == config


@pytest.mark.parametrize("change", [
    {"unknown": 1}, {"poll_hz": "100"}, {"poll_hz": float("nan")},
    {"motor_ids": [1]*8}, {"motor_models": ["XL330"]*8},
    {"joint_signs": [1]*6}, {"joint_signs": [True]*7},
    {"stale_timeout_ms": 251}, {"device_path": "ttyUSB0"},
    {"translation_gain": 0.01}, {"gripper_open_threshold": 0.8},
    {"gripper_min_travel_rad": 2}, {"reference_joint_positions": [10]*7},
    {"save_video": "yes"}, {"max_steps": True},
    {"vendor_id": "0403"}, {"vendor_id": True}, {"vendor_id": -1},
    {"product_id": 65536}, {"product_id": None},
    {"serial_number": " "}, {"serial_number": 123},
])
def test_strict_config_rejects_invalid_fields(tmp_path, change):
    raw = yaml.safe_load(factr.DEFAULT_FACTR_CONFIG.read_text())
    raw.update(change)
    path = tmp_path/"factr.yaml"
    path.write_text(yaml.safe_dump(raw))
    with pytest.raises((ValueError, TypeError)):
        load_factr_config(path)


def test_config_duplicate_keys_and_relative_output(tmp_path):
    original = factr.DEFAULT_FACTR_CONFIG.read_text()
    path = tmp_path/"factr.yaml"
    path.write_text(original+"\nmax_steps: 100\n")
    with pytest.raises(ValueError, match="duplicate"):
        load_factr_config(path)
    path.write_text(original)
    assert load_factr_config(path).runtime["calibration_file"].startswith(str(tmp_path.parent))


class FakePort:
    def __init__(self, path):
        self.path = path
        self.is_open = False

    def setBaudRate(self, value):
        self.baudrate = value
        self.is_open = True
        return True

    def closePort(self):
        self.is_open = False


class FakePacket:
    def __init__(self, config):
        self.config = config
        self.calls = []
        self.torque = self.hardware = self.servo_error = 0
        self.bad_model = False
        self.missing_id = None
        self.communication = 0
        self.ticks = -1024

    def ping(self, port, motor_id):
        self.calls.append(("ping", motor_id))
        model = self.config.motor_models[self.config.motor_ids.index(motor_id)]
        return (0 if self.bad_model else MODEL_NUMBERS[model]), 0, 0

    def syncReadTx(self, port, address, size, ids, length):
        self.calls.append(("syncReadTx", address, size, tuple(ids), length))
        return self.communication

    def readRx(self, port, motor_id, size):
        self.calls.append(("readRx", motor_id, size))
        data = [0]*size
        data[0], data[6] = self.torque, self.hardware
        data[68:72] = self.ticks.to_bytes(4, "little", signed=True)
        if motor_id == self.missing_id:
            return [], -3001, 0
        return data, 0, self.servo_error


def transport_fixture():
    config = load_factr_config()
    packet = FakePacket(config)
    sdk = SimpleNamespace(PortHandler=FakePort, PacketHandler=lambda _: packet, COMM_SUCCESS=0)
    device = FactrSerialDevice("/dev/fake", config.vendor_id, config.product_id, "fake-serial")
    return FactrStartupProbe(config, device=device, sdk=sdk,
                             owners_probe=lambda *a, **k: {"busy_pids": []}), packet


def test_startup_probe_checks_status_without_motor_write_operations():
    transport, packet = transport_fixture()
    details = transport.open()
    assert details["passive_only"]
    assert transport.port.path == "/dev/fake"
    assert details["serial_device"]["serial_number"] == "fake-serial"
    assert transport.check_status() is None
    assert set(call[0] for call in packet.calls) == {"ping", "syncReadTx", "readRx"}
    # Refuse future accidental expansion to SDK write/current/torque APIs.
    tree = ast.parse(inspect.getsource(FactrStartupProbe))
    packet_methods = {node.func.attr for node in ast.walk(tree) if isinstance(node, ast.Call)
                      and isinstance(node.func, ast.Attribute)
                      and isinstance(node.func.value, ast.Attribute)
                      and node.func.value.attr == "packet"}
    assert packet_methods == {"ping", "syncReadTx", "readRx"}
    transport.close()
    assert transport.port is None


def test_busy_port_is_rejected_before_any_sdk_packet():
    transport, packet = transport_fixture()
    transport._owners_probe = lambda _: {"busy_pids": [987]}
    with pytest.raises(RuntimeError, match="occupied"):
        transport.open()
    assert packet.calls == []
    assert transport.port is None


def test_proc_occupancy_checks_character_identity_not_only_path(tmp_path):
    # A symlink to /dev/null emulates a proc FD without touching serial devices.
    folder = tmp_path/"12345"/"fd"
    folder.mkdir(parents=True)
    (folder/"4").symlink_to("/dev/null")
    (folder/"5").symlink_to("/dev/zero")
    info = factr.serial_owners("/dev/null", proc_root=tmp_path)
    assert info["busy_pids"] == [12345]


@pytest.mark.parametrize("field,value,match", [
    ("torque", 1, "torque is enabled"), ("hardware", 4, "hardware error"),
    ("servo_error", 128, "servo_error"), ("bad_model", True, "expected"),
    ("missing_id", 5, "communication"), ("communication", -1000, "communication"),
])
def test_transport_refuses_invalid_full_pack(field, value, match):
    transport, packet = transport_fixture()
    setattr(packet, field, value)
    with pytest.raises(RuntimeError, match=match):
        transport.open()
    assert transport.port is None
