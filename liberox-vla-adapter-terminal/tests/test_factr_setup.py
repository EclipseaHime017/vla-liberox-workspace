"""Persistent USB setup tests: never write /etc/sysfs or run privileged commands."""
from dataclasses import replace
from pathlib import Path
import shutil
import subprocess
import sys

import pytest

import setup_factr as cli
from backend.app.devices.factr import load_factr_config


def test_default_rule_applies_to_all_matching_adapters_not_one_port():
    config = load_factr_config()
    name, rule = cli.usb_latency_rule(config)
    assert name == "99-factr-latency-0403-6014.rules"
    assert 'ACTION=="add|bind"' in rule
    assert 'SUBSYSTEM=="usb-serial"' in rule and 'TEST=="latency_timer"' in rule
    assert 'ATTRS{idVendor}=="0403"' in rule and 'ATTRS{idProduct}=="6014"' in rule
    assert 'ATTR{latency_timer}="1"' in rule
    assert "serial}" not in rule and "ttyUSB" not in rule and "/dev/" not in rule
    assert "RUN" not in rule and "MODE" not in rule and "DRIVERS" not in rule


def test_rule_uses_configured_ids_and_only_explicit_serial():
    config = replace(load_factr_config(), vendor_id=0x1234, product_id=0xabcd, serial_number="TEST-1")
    name, rule = cli.usb_latency_rule(config)
    assert name == "99-factr-latency-1234-abcd.rules"
    assert 'ATTRS{serial}=="TEST-1"' in rule
    assert 'ATTRS{idVendor}=="1234"' in rule and 'ATTRS{idProduct}=="abcd"' in rule
    # Stable name: switching from model-wide to serial-specific replaces its rule.
    assert name == cli.usb_latency_rule(replace(config, serial_number=None))[0]


@pytest.mark.parametrize("serial", ["*", "a?", "a[b]", 'a"', "a\nb", "a\\b", "../x", ""])
def test_rule_rejects_injection_or_wildcards(serial):
    with pytest.raises(ValueError, match="USB serial number"):
        cli.usb_latency_rule(replace(load_factr_config(), serial_number=serial))


@pytest.fixture
def installer(monkeypatch):
    from backend.app.devices import factr_discovery
    monkeypatch.setattr(cli.os, "geteuid", lambda: 1000)
    def forbidden(*args, **kwargs):
        pytest.fail("Rule installation must not enumerate/open devices or install runtime dependencies")
    monkeypatch.setattr(factr_discovery, "discover_factr_device", forbidden)
    commands = []
    def run(command, **kwargs):
        commands.append(command)
        assert kwargs == {"check": True}
        assert command[:4] == ["sudo", "--", "/usr/bin/python3", "-I"]
        assert command[4] == str(cli.USB_RULE_HELPER)
        assert command[5:9] == ["--vendor-id", "1027", "--product-id", "24596"]
        assert "--apply-current" not in command
        return subprocess.CompletedProcess(command, 0)
    monkeypatch.setattr(cli.subprocess, "run", run)
    return commands


def test_install_delegates_to_restricted_helper_without_device_or_motor_io(installer, capsys):
    commands = installer
    config = load_factr_config()
    assert cli.install_usb_rule(config) == 0
    assert len(commands) == 1
    assert len(commands[0]) == 9
    assert "ALL MATCHING DEVICES" in capsys.readouterr().out


def test_install_passes_serial_as_one_selector(installer):
    cli.install_usb_rule(replace(load_factr_config(), serial_number="TEST-1"))
    assert installer[0][-1] == "--serial-number=TEST-1"


def test_install_rejects_bad_selector_before_sudo(installer):
    with pytest.raises(ValueError):
        cli.install_usb_rule(replace(load_factr_config(), serial_number='bad";command'))
    assert not installer


def test_install_failure_does_not_reload_or_report_success(installer, monkeypatch, capsys):
    def fail(command, **kwargs):
        assert command[:2] == ["sudo", "--"]
        raise subprocess.CalledProcessError(1, command)
    monkeypatch.setattr(cli.subprocess, "run", fail)
    with pytest.raises(subprocess.CalledProcessError):
        cli.install_usb_rule(load_factr_config())
    assert "Rule installed" not in capsys.readouterr().out


def test_usb_command_does_not_run_runtime_setup(installer, monkeypatch):
    monkeypatch.setattr(sys, "argv", ["setup_factr.py", "--install-usb-rule"])
    assert cli.main() == 0  # Fixture permits ONLY the helper, not git/pip/check.


def test_root_cli_does_not_call_sudo(monkeypatch):
    monkeypatch.setattr(cli.os, "geteuid", lambda: 0)
    commands = []
    monkeypatch.setattr(cli.subprocess, "run", lambda command, **kwargs: commands.append(command))
    assert cli.install_usb_rule(load_factr_config()) == 0
    assert commands[0][:3] == ["/usr/bin/python3", "-I", str(cli.USB_RULE_HELPER)]


def test_check_cannot_be_combined_with_install(monkeypatch):
    monkeypatch.setattr(sys, "argv", ["setup_factr.py", "--check", "--install-usb-rule"])
    with pytest.raises(SystemExit) as exc:
        cli.main()
    assert exc.value.code == 2


def test_rule_syntax_with_udevadm_verify(tmp_path):
    if not shutil.which("udevadm"):
        pytest.skip("udevadm unavailable")
    version = int(subprocess.check_output(["udevadm", "--version"], text=True).strip())
    if version < 254:
        pytest.skip("udevadm verify needs systemd >=254")
    name, rule = cli.usb_latency_rule(load_factr_config())
    path = tmp_path/name
    path.write_text(rule)
    # verify is syntax-only, unlike udevadm test (which may write attributes).
    subprocess.run(["udevadm", "verify", str(path)], check=True)
