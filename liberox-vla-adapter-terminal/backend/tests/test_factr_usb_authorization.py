"""Mock desktop authorization only: never invoke pkexec or access a USB device."""
import asyncio
from dataclasses import replace
import json
from pathlib import Path
import subprocess
import threading
from types import SimpleNamespace

import pytest
from fastapi import FastAPI, Request

from backend.app.api import controller as api
from backend.app.devices.factr import load_factr_config
from backend.app.devices import factr_usb_rule as rule
from backend.app.services import factr_usb_authorization as auth


@pytest.fixture
def setup(tmp_path, monkeypatch):
    timer = tmp_path/"latency_timer"
    timer.write_text("16\n")
    pkexec = tmp_path/"pkexec"
    pkexec.touch()
    monkeypatch.setattr(auth, "PKEXEC", pkexec)
    monkeypatch.setenv("DISPLAY", ":fake")
    device = SimpleNamespace(path="/dev/fakeFACTR", serial_number="test-arm")
    monkeypatch.setattr(auth, "discover_factr_device", lambda _: device)
    monkeypatch.setattr(rule, "resolve_latency_path", lambda *a: timer)
    commands, messages, signals = [], [], []
    stop = threading.Event()
    class Process:
        pid = 987654
        returncode = None
        mode = "success"
        def communicate(self, timeout):
            if self.mode == "wait":
                raise subprocess.TimeoutExpired("fake", timeout)
            if self.mode == "stop":
                stop.set()
                raise subprocess.TimeoutExpired("fake", timeout)
            self.returncode = {"cancel": 126, "denied": 127, "failed": 1}.get(self.mode, 0)
            if self.mode == "success":
                timer.write_text("1\n")
            result = "invalid" if self.mode == "bad_json" else json.dumps({"current_applied": True})
            return result, "fixed helper failure" if self.mode == "failed" else ""
        def poll(self): return self.returncode
    process = Process()
    def launch(command, **kwargs):
        commands.append(command)
        assert kwargs == dict(stdin=subprocess.DEVNULL, stdout=subprocess.PIPE,
                              stderr=subprocess.PIPE, text=True, start_new_session=True)
        return process
    def signal(pid, signum):
        assert pid == process.pid
        signals.append(signum)
        process.returncode = -15
        process.mode = "killed"
    monkeypatch.setattr(auth.subprocess, "Popen", launch)
    monkeypatch.setattr(auth.os, "killpg", signal)
    def run(allowed=True, config=None):
        return auth.ensure_factr_usb_latency(config or load_factr_config(), allow_authorization=allowed,
                                            stop_event=stop, on_message=messages.append)
    return SimpleNamespace(timer=timer, commands=commands, messages=messages, process=process,
                           stop=stop, signals=signals, run=run)


def test_already_low_latency_never_prompts_even_without_authorization(setup):
    setup.timer.write_text("1\n")
    setup.run(allowed=False)
    assert setup.commands == setup.messages == []


def test_fix_uses_only_fixed_helper_and_continues_after_readback(setup):
    setup.run()
    command = setup.commands[0]
    assert command[1:5] == ["--disable-internal-agent", "/usr/bin/python3", "-I", str(auth.HELPER)]
    assert command[5:] == ["--vendor-id", "1027", "--product-id", "24596", "--apply-current"]
    assert len(setup.commands) == 1
    assert "系统授权窗口" in setup.messages[0]
    assert "正在调用官方整臂校准" in setup.messages[-1]
    assert setup.timer.read_text().strip() == "1"
    setup.run()
    assert len(setup.commands) == 1  # Subsequent calibration has no repeated prompt.


def test_serial_argument_is_not_interpreted_as_an_option(setup):
    setup.run(config=replace(load_factr_config(), serial_number="-FACTR"))
    assert setup.commands[0][-1] == "--serial-number=-FACTR"


@pytest.mark.parametrize("mode,error", [
    ("cancel", "已取消"), ("denied", "未获得系统授权"), ("failed", "fixed helper failure"),
    ("bad_json", "无效结果"), ("no_readback", "未读回 1 ms"),
])
def test_auth_failure_or_unverified_write_stops_calibration(setup, mode, error):
    setup.process.mode = mode
    with pytest.raises(RuntimeError, match=error):
        setup.run()
    assert len(setup.messages) == 1
    assert setup.timer.read_text().strip() == "16"


def test_remote_request_cannot_start_system_authorization(setup):
    with pytest.raises(RuntimeError, match="本机打开 UI"):
        setup.run(allowed=False)
    assert setup.commands == []


def test_headless_reports_actionable_error_without_terminal_password_prompt(setup, monkeypatch):
    monkeypatch.delenv("DISPLAY", raising=False)
    monkeypatch.delenv("WAYLAND_DISPLAY", raising=False)
    with pytest.raises(RuntimeError, match="SSH/无桌面"):
        setup.run()
    assert setup.commands == []


def test_cancel_on_service_shutdown_terminates_pending_authorization(setup):
    setup.process.mode = "stop"
    with pytest.raises(RuntimeError, match="校准已取消"):
        setup.run()
    assert setup.signals and len(setup.messages) == 1


def test_authorization_timeout_does_not_proceed(setup, monkeypatch):
    setup.process.mode = "wait"
    monkeypatch.setattr(auth, "AUTHORIZATION_TIMEOUT_SECONDS", 0)
    with pytest.raises(RuntimeError, match="授权超时"):
        setup.run()
    assert setup.signals and len(setup.messages) == 1


def request(*, peer="127.0.0.1", host="127.0.0.1:8000", origin="http://127.0.0.1:8000",
            marker="1", site="same-origin", app=None):
    headers = [(b"host", host.encode()), (b"origin", origin.encode()),
               (b"x-factr-usb-repair", marker.encode()), (b"sec-fetch-site", site.encode())]
    return Request(dict(type="http", method="POST", scheme="http", path="/api/controller/calibrate",
                        query_string=b"controller_id=factr", headers=headers, client=(peer, 10000), app=app))


@pytest.mark.parametrize("change", [
    {"peer": "192.168.1.20"}, {"origin": "http://evil.example"},
    {"host": "evil.example", "origin": "http://evil.example"},
    {"origin": ""}, {"marker": ""}, {"site": "cross-site"},
    {"origin": "http://127.0.0.1:8001"},
])
def test_only_local_same_origin_calibration_can_prompt(change):
    assert api.local_usb_authorization(request())
    assert not api.local_usb_authorization(request(**change))


def test_calibration_api_forwards_authorization_only_for_local_factr():
    calls = []
    app = FastAPI()
    app.state.run_service = SimpleNamespace(calibrate_controller=lambda *args, **kwargs: calls.append((args, kwargs)))
    asyncio.run(api.calibrate(request(app=app), None, "factr"))
    asyncio.run(api.calibrate(request(app=app), None, "spacemouse"))
    asyncio.run(api.calibrate(request(app=app, peer="192.168.1.20"), None, "factr"))
    assert calls == [(("factr", "reference"), {"allow_usb_authorization": True}),
                     (("spacemouse", "reference"), {}), (("factr", "reference"), {})]
