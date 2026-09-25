"""USB metadata discovery and idle status; never access a real serial device."""
from dataclasses import replace
from pathlib import Path
import stat
import sys
from types import ModuleType, SimpleNamespace
from unittest.mock import Mock

import pytest

from backend.app.devices import factr
from backend.app.devices.factr_discovery import discover_factr_device
from backend.app.services.factr_controller_service import FactrControllerService


def port(path="/dev/fakeFACTR0", *, serial="FACTR-A", vid=0x0403, pid=0x6014,
         location="1-2"):
    return SimpleNamespace(device=str(path), vid=vid, pid=pid,
                           serial_number=serial, location=location)


@pytest.fixture
def config(tmp_path):
    original = factr.load_factr_config()
    return replace(original, vendor_id=0x0403, product_id=0x6014, serial_number=None,
                   runtime={**original.runtime, "runtime_python": str(tmp_path/"missing-runtime")})


@pytest.fixture
def enumerated_ports(monkeypatch):
    """Exercise the normal import path without letting pyserial see the host."""
    ports = []
    serial = ModuleType("serial")
    serial_tools = ModuleType("serial.tools")
    list_ports = ModuleType("serial.tools.list_ports")
    sdk = ModuleType("dynamixel_sdk")

    def forbidden(*args, **kwargs):
        pytest.fail("USB discovery must not construct or open a serial transport")

    serial.Serial = forbidden
    sdk.PortHandler = forbidden
    list_ports.comports = Mock(side_effect=lambda: list(ports))
    serial.tools = serial_tools
    serial_tools.list_ports = list_ports
    for name, module in (("serial", serial), ("serial.tools", serial_tools),
                         ("serial.tools.list_ports", list_ports), ("dynamixel_sdk", sdk)):
        monkeypatch.setitem(sys.modules, name, module)
    monkeypatch.setattr(factr.FactrStartupProbe, "open", forbidden)
    return ports


def mock_character_port(monkeypatch, path, *, accessible=True):
    """Fake only this port's metadata; keep ordinary runtime-file checks real."""
    original_stat, original_access = Path.stat, factr.os.access

    def fake_stat(candidate, *args, **kwargs):
        if candidate == path:
            return SimpleNamespace(st_mode=stat.S_IFCHR | 0o660)
        return original_stat(candidate, *args, **kwargs)

    def fake_access(candidate, mode):
        if Path(candidate) == path:
            return accessible
        return original_access(candidate, mode)

    monkeypatch.setattr(Path, "stat", fake_stat)
    monkeypatch.setattr(factr.os, "access", fake_access)


def test_discovery_enumerates_metadata_without_opening_transport(config, enumerated_ports):
    enumerated_ports.append(port())
    result = discover_factr_device(config)
    assert result.path == "/dev/fakeFACTR0"
    assert result.serial_number == "FACTR-A"
    assert result.vendor_id == 0x0403 and result.product_id == 0x6014
    sys.modules["serial.tools.list_ports"].comports.assert_called_once_with()


@pytest.mark.parametrize("ports", [
    [], [port(vid=0x1234)], [port(pid=0x1234)], [port(vid=None, pid=None)],
])
def test_discovery_rejects_no_matching_usb_device(config, ports):
    with pytest.raises(RuntimeError, match="未发现 FACTR.*0403:6014"):
        discover_factr_device(config, ports=ports)


def test_discovery_ignores_unrelated_ports_when_one_matches(config):
    result = discover_factr_device(config, ports=[
        port("/dev/fakeOther", pid=0x1234), port("/dev/fakeFACTR1"),
    ])
    assert result.path == "/dev/fakeFACTR1"


def test_discovery_rejects_multiple_matches_with_actionable_diagnostics(config):
    with pytest.raises(RuntimeError, match="多个串口.*serial_number") as error:
        discover_factr_device(config, ports=[
            port("/dev/fakeFACTR0", serial="FACTR-A"),
            port("/dev/fakeFACTR1", serial="FACTR-B"),
        ])
    assert "/dev/fakeFACTR0" in str(error.value) and "FACTR-A" in str(error.value)
    assert "/dev/fakeFACTR1" in str(error.value) and "FACTR-B" in str(error.value)


def test_serial_selector_selects_exact_device_and_rejects_no_match(config):
    ports = [port(serial="FACTR-A"), port("/dev/fakeFACTR1", serial="FACTR-B")]
    selected = discover_factr_device(replace(config, serial_number="FACTR-B"), ports=ports)
    assert selected.path == "/dev/fakeFACTR1"
    with pytest.raises(RuntimeError, match="未发现 FACTR.*serial_number=FACTR-C"):
        discover_factr_device(replace(config, serial_number="FACTR-C"), ports=ports)


def test_duplicate_serial_numbers_remain_ambiguous(config):
    with pytest.raises(RuntimeError, match="多个串口"):
        discover_factr_device(replace(config, serial_number="FACTR-A"), ports=[
            port("/dev/fakeFACTR0"), port("/dev/fakeFACTR1"),
        ])


def test_aliases_of_one_port_are_deduplicated(config, tmp_path):
    target, alias = tmp_path/"ttyUSB0", tmp_path/"by-id-FACTR-A"
    alias.symlink_to(target)
    result = discover_factr_device(config, ports=[port(target), port(alias)])
    assert result.path == str(target)


def test_usb_serial_identity_survives_tty_renumbering_and_usb_location_change(config):
    before = discover_factr_device(config, ports=[port("/dev/fakeFACTR0", location="1-2")])
    after = discover_factr_device(config, ports=[port("/dev/fakeFACTR7", location="3-4")])
    assert before.path != after.path
    assert before.identity() == after.identity() == {
        "vendor_id": 0x0403, "product_id": 0x6014, "serial_number": "FACTR-A",
    }


@pytest.mark.parametrize("runtime_exists,accessible,error", [
    (False, True, "官方运行环境缺失"),
    (True, False, "Serial permission denied"),
    (True, True, None),
])
def test_probe_separates_usb_presence_from_runtime_readiness(
        config, enumerated_ports, monkeypatch, tmp_path, runtime_exists, accessible, error):
    if runtime_exists:
        Path(config.runtime["runtime_python"]).touch()
    path = tmp_path/"fakeFACTR0"
    enumerated_ports.append(port(path))
    mock_character_port(monkeypatch, path, accessible=accessible)
    result = factr.probe_factr(config)
    assert result["connected"] is True
    assert result["verified"] is False
    assert result["runtime_available"] is runtime_exists
    assert result["serial_device"]["path"] == str(path)
    if error is None:
        assert result["error"] is None
    else:
        assert error in result["error"]


def test_probe_rejects_regular_file_as_serial_device(config, enumerated_ports, tmp_path):
    path = tmp_path/"not-a-serial-device"
    path.touch()
    enumerated_ports.append(port(path))
    result = factr.probe_factr(config)
    assert result["connected"] is False
    assert "not a serial character device" in result["error"]


def test_idle_service_detects_unplug_and_replug_without_constructing_client(
        config, enumerated_ports, monkeypatch, tmp_path):
    path = tmp_path/"fakeFACTR0"
    enumerated_ports.append(port(path))
    mock_character_port(monkeypatch, path)
    factory = Mock(side_effect=AssertionError("Idle discovery must not start a FACTR client"))
    service = FactrControllerService(config, input_factory=factory, start_monitor=False)
    try:
        status = service.status()
        assert status["state"] == "UNCALIBRATED" and status["connected"] is True
        assert status["runtime_available"] is False
        assert "官方运行环境缺失" in status["error"]
        assert status["serial_device"]["serial_number"] == "FACTR-A"
        enumerated_ports.clear()
        service._poll_once()
        assert service.status()["state"] == "DISCONNECTED"
        assert service.status()["connected"] is False
        enumerated_ports.append(port(path))
        service._poll_once()
        status = service.status()
        assert status["state"] == "UNCALIBRATED" and status["connected"] is True
        assert status["calibrated"] is False and status["gravity_enabled"] is False
        assert status["armed_session_id"] is None
        factory.assert_not_called()
    finally:
        service.close()
