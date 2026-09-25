"""Restricted privileged helper tests, using fake sysfs and mocked reload only."""
import ast
import json
import os
from pathlib import Path
import stat
import subprocess

import pytest

from backend.app.devices import factr_usb_rule as helper


@pytest.fixture
def installer(tmp_path, monkeypatch):
    rules = tmp_path / "rules"
    rules.mkdir()
    monkeypatch.setattr(helper, "RULES_ROOT", rules)
    monkeypatch.setattr(helper.os, "geteuid", lambda: 0)
    calls = []
    def run(command, **kwargs):
        assert command == ["/usr/bin/udevadm", "control", "--reload-rules"]
        assert kwargs == {"check": True, "capture_output": True, "text": True, "timeout": 30}
        calls.append(command)
        return subprocess.CompletedProcess(command, 0)
    monkeypatch.setattr(helper.subprocess, "run", run)
    return rules, calls


def test_atomic_install_is_idempotent_and_does_not_touch_sysfs(installer, monkeypatch):
    rules, calls = installer
    monkeypatch.setattr(helper, "resolve_latency_path", lambda *a, **k: pytest.fail("no sysfs in rule-only mode"))
    result = helper.install_rule(0x0403, 0x6014)
    name, content = helper.build_rule(0x0403, 0x6014)
    target = rules / name
    assert target.read_text() == content
    assert stat.S_IMODE(target.stat().st_mode) == 0o644
    old_stat = target.stat()
    assert result["status"] == "installed" and result["current_applied"] is False
    assert result["rule_path"] == str(target)
    helper.install_rule(0x0403, 0x6014)
    assert target.stat().st_ino == old_stat.st_ino
    assert target.stat().st_mtime_ns == old_stat.st_mtime_ns
    assert len(calls) == 2
    assert list(rules.iterdir()) == [target]


def test_existing_setup_script_rule_can_be_replaced(installer):
    rules, _ = installer
    name, content = helper.build_rule(0x0403, 0x6014)
    (rules / name).write_text(content)
    helper.install_rule(0x0403, 0x6014, "NEW-SERIAL")
    assert (rules / name).read_text() == helper.build_rule(0x0403, 0x6014, "NEW-SERIAL")[1]


@pytest.mark.parametrize("existing", ["unmanaged", "symlink", "directory", "fifo"])
def test_refuses_nonowned_or_nonregular_targets(installer, existing):
    rules, calls = installer
    target = rules / helper.build_rule(0x0403, 0x6014)[0]
    if existing == "unmanaged":
        target.write_text("# Managed by someone else\n")
    elif existing == "symlink":
        target.symlink_to(rules / "missing")
    elif existing == "directory":
        target.mkdir()
    else:
        os.mkfifo(target)
    with pytest.raises(ValueError, match="Refusing"):
        helper.install_rule(0x0403, 0x6014)
    assert not calls


def test_requires_root_before_writing(installer, monkeypatch):
    rules, calls = installer
    monkeypatch.setattr(helper.os, "geteuid", lambda: 1000)
    with pytest.raises(PermissionError, match="administrator"):
        helper.install_rule(0x0403, 0x6014)
    assert not list(rules.iterdir()) and not calls


@pytest.mark.parametrize("vendor,product", [(True, 2), (1, False), (-1, 2), (65536, 2), (1, "0x6014")])
def test_invalid_ids_fail_before_privilege(vendor, product, installer):
    rules, calls = installer
    with pytest.raises(ValueError, match="VID/PID"):
        helper.install_rule(vendor, product)
    assert not list(rules.iterdir()) and not calls


def test_failed_atomic_replace_cleans_temp_and_preserves_old_rule(installer, monkeypatch):
    rules, calls = installer
    name, original = helper.build_rule(0x0403, 0x6014)
    (rules / name).write_text(original)
    def fail(*args, **kwargs):
        raise OSError("replace failed")
    monkeypatch.setattr(helper.os, "replace", fail)
    with pytest.raises(OSError, match="replace failed"):
        helper.install_rule(0x0403, 0x6014, "NEW")
    assert (rules / name).read_text() == original
    assert list(rules.iterdir()) == [rules / name] and not calls


def test_reload_failure_is_not_reported_as_success(installer, monkeypatch, capsys):
    def fail(*args, **kwargs):
        raise subprocess.CalledProcessError(1, args[0])
    monkeypatch.setattr(helper.subprocess, "run", fail)
    assert helper.main(["--vendor-id", "1027", "--product-id", "24596"]) == 1
    captured = capsys.readouterr()
    assert not captured.out
    assert "installation failed" in captured.err


def test_cli_returns_json(installer, capsys):
    assert helper.main(["--vendor-id", "1027", "--product-id", "24596"]) == 0
    result = json.loads(capsys.readouterr().out)
    assert result["status"] == "installed"


@pytest.mark.parametrize("arg", ["--path", "--command", "--rule", "--sysfs-root", "--vendor"])
def test_cli_has_no_arbitrary_privileged_inputs(arg):
    with pytest.raises(SystemExit) as exc:
        helper.main(["--vendor-id", "1027", "--product-id", "24596", arg, "/tmp/evil"])
    assert exc.value.code == 2


def _fake_device(root, index=0, serial="SERIAL", vendor="0403", product="6014", latency="16"):
    devices = root / "devices"
    devices.mkdir(exist_ok=True)
    parent = devices / f"usb1/1-{index + 1}"
    port = parent / f"1-{index + 1}:1.0/ttyUSB{index}"
    port.mkdir(parents=True)
    (parent / "idVendor").write_text(vendor)
    (parent / "idProduct").write_text(product)
    if serial is not None:
        (parent / "serial").write_text(serial)
    attribute = port / "latency_timer"
    attribute.write_text(latency)
    links = root / "bus/usb-serial/devices"
    links.mkdir(parents=True, exist_ok=True)
    (links / f"ttyUSB{index}").symlink_to(port)
    return attribute


def test_unique_device_apply_readback_and_other_device_untouched(tmp_path):
    selected = _fake_device(tmp_path)
    other = _fake_device(tmp_path, 1, vendor="1234")
    assert helper.resolve_latency_path(0x0403, 0x6014, sysfs_root=tmp_path) == selected
    assert helper.apply_current_latency(0x0403, 0x6014, sysfs_root=tmp_path) == selected
    assert selected.read_text().strip() == "1" and other.read_text() == "16"


def test_multiple_matches_require_serial(tmp_path):
    first = _fake_device(tmp_path, serial="FIRST")
    second = _fake_device(tmp_path, 1, serial="SECOND")
    with pytest.raises(RuntimeError, match="Multiple"):
        helper.apply_current_latency(0x0403, 0x6014, sysfs_root=tmp_path)
    assert first.read_text() == second.read_text() == "16"
    helper.apply_current_latency(0x0403, 0x6014, "SECOND", sysfs_root=tmp_path)
    assert second.read_text().strip() == "1" and first.read_text() == "16"


def test_no_match_or_disconnected_port_fails(tmp_path):
    attribute = _fake_device(tmp_path, vendor="1234")
    with pytest.raises(RuntimeError, match="No connected"):
        helper.resolve_latency_path(0x0403, 0x6014, sysfs_root=tmp_path)
    attribute.unlink()
    with pytest.raises(FileNotFoundError):
        helper.resolve_latency_path(0x1234, 0x6014, sysfs_root=tmp_path)


def test_attribute_cannot_escape_its_usb_device(tmp_path):
    attribute = _fake_device(tmp_path)
    attribute.unlink()
    external = tmp_path / "external"
    external.write_text("16")
    attribute.symlink_to(external)
    with pytest.raises(ValueError, match="escaped"):
        helper.apply_current_latency(0x0403, 0x6014, sysfs_root=tmp_path)
    assert external.read_text() == "16"


def test_readback_failure_is_reported(tmp_path, monkeypatch):
    attribute = _fake_device(tmp_path)
    monkeypatch.setattr(Path, "write_text", lambda self, data: len(data))
    with pytest.raises(RuntimeError, match="read-back"):
        helper.apply_current_latency(0x0403, 0x6014, sysfs_root=tmp_path)
    assert attribute.read_text() == "16"


def test_apply_failure_after_install_reports_persistent_state(installer, monkeypatch):
    rules, calls = installer
    monkeypatch.setattr(helper, "resolve_latency_path", lambda *a, **k: Path("/fake/latency_timer"))
    def fail(*args, **kwargs):
        raise OSError("disconnected")
    monkeypatch.setattr(helper, "apply_current_latency", fail)
    with pytest.raises(RuntimeError, match="rule installed, but current-device.*disconnected"):
        helper.install_rule(0x0403, 0x6014, apply_current=True)
    assert len(list(rules.iterdir())) == 1 and len(calls) == 1


def test_apply_current_cli_success(installer, monkeypatch, capsys):
    selected = Path("/fake/latency_timer")
    monkeypatch.setattr(helper, "resolve_latency_path", lambda *a, **k: selected)
    monkeypatch.setattr(helper, "apply_current_latency", lambda *a, **k: selected)
    assert helper.main(["--vendor-id", "1027", "--product-id", "24596", "--apply-current"]) == 0
    result = json.loads(capsys.readouterr().out)
    assert result["current_applied"] is True and result["latency_path"] == str(selected)


def test_helper_imports_only_standard_library():
    tree = ast.parse(Path(helper.__file__).read_text())
    imports = []
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            imports.extend(alias.name.split(".")[0] for alias in node.names)
        elif isinstance(node, ast.ImportFrom):
            assert node.level == 0
            imports.append(node.module.split(".")[0])
    assert set(imports) <= {"argparse", "json", "os", "pathlib", "re", "stat", "subprocess", "sys", "tempfile"}
