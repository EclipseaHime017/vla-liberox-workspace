"""Local desktop authorization before calibration. Never receives a password."""
import json
import os
from pathlib import Path
import signal
import subprocess
import time

from ..devices.factr_discovery import discover_factr_device

HELPER = Path(__file__).resolve().parents[1]/"devices/factr_usb_rule.py"
PKEXEC = Path("/usr/bin/pkexec")
AUTHORIZATION_TIMEOUT_SECONDS = 120.


def _stop_authorization(process):
    if process.poll() is not None:
        return
    try:
        os.killpg(process.pid, signal.SIGTERM)
    except (ProcessLookupError, PermissionError):
        pass
    try:
        process.communicate(timeout=2)
    except subprocess.TimeoutExpired:
        try:
            os.killpg(process.pid, signal.SIGKILL)
        except (ProcessLookupError, PermissionError):
            pass
        # A privileged helper already authorized may finish installing its
        # bounded rule operation; no calibration/torque follows cancellation.


def ensure_factr_usb_latency(config, *, allow_authorization, stop_event, on_message):
    from ..devices.factr_usb_rule import build_rule, resolve_latency_path
    if stop_event.is_set():
        raise RuntimeError("FACTR 校准已取消")
    device = discover_factr_device(config)
    path = resolve_latency_path(config.vendor_id, config.product_id, config.serial_number)
    if int(path.read_text().strip()) == 1:
        return  # No dialog / privileged process on the normal calibration path.
    if not allow_authorization:
        raise RuntimeError("USB 延迟需修复：请在部署设备本机打开 UI 后点击校准，或在终端安装 USB 规则")
    if not PKEXEC.is_file() or not (os.environ.get("DISPLAY") or os.environ.get("WAYLAND_DISPLAY")):
        raise RuntimeError("无法打开系统授权窗口：请从本机桌面启动 UI，并确保已安装 polkit 授权代理；SSH/无桌面环境请使用终端安装 USB 规则")
    build_rule(config.vendor_id, config.product_id, config.serial_number)  # Validate before privilege request.
    command = [str(PKEXEC), "--disable-internal-agent", "/usr/bin/python3", "-I", str(HELPER),
               "--vendor-id", str(config.vendor_id), "--product-id", str(config.product_id), "--apply-current"]
    if config.serial_number is not None:
        command.append(f"--serial-number={config.serial_number}")
    on_message("USB 延迟需修复，请在系统授权窗口输入密码；网页不会接收或保存密码")
    if stop_event.is_set():
        raise RuntimeError("FACTR 校准已取消")
    process = subprocess.Popen(command, stdin=subprocess.DEVNULL, stdout=subprocess.PIPE,
                               stderr=subprocess.PIPE, text=True, start_new_session=True)
    deadline = time.monotonic()+AUTHORIZATION_TIMEOUT_SECONDS
    try:
        while True:
            if stop_event.is_set():
                raise RuntimeError("FACTR 校准已取消")
            if time.monotonic() >= deadline:
                raise RuntimeError("系统授权超时，本次校准已停止；请重试并在系统窗口完成授权")
            try:
                stdout, stderr = process.communicate(timeout=.2)
                break
            except subprocess.TimeoutExpired:
                continue
        if stop_event.is_set():
            raise RuntimeError("FACTR 校准已取消")
        if process.returncode == 126:
            raise RuntimeError("已取消 USB 修复授权，本次校准未启动")
        if process.returncode == 127:
            raise RuntimeError("未获得系统授权：授权被拒绝或桌面授权代理不可用；本次校准未启动")
        if process.returncode:
            raise RuntimeError(f"USB 延迟修复失败：{stderr.strip()[-1000:] or '系统命令执行失败'}")
        try:
            result = json.loads(stdout)
        except (ValueError, TypeError) as exc:
            raise RuntimeError("USB 修复返回无效结果；请检查系统规则，本次校准未启动") from exc
        if not isinstance(result, dict) or not result.get("current_applied"):
            raise RuntimeError("USB 当前延迟未确认修复，本次校准未启动")
        if discover_factr_device(config) != device:
            raise RuntimeError("授权期间 USB 设备发生变化，请重新连接并校准")
        current = resolve_latency_path(config.vendor_id, config.product_id, config.serial_number)
        if current != path or int(current.read_text().strip()) != 1:
            raise RuntimeError("USB latency_timer 未读回 1 ms，本次校准未启动")
        on_message("USB 延迟已修复，正在调用官方整臂校准；保持参考构型并松开触发器")
    finally:
        _stop_authorization(process)
