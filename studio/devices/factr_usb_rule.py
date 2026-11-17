"""Restricted USB-latency installer, also runnable with system Python -I.

Only standard-library imports: the privileged process must not load the app,
Conda packages, configuration files, or device drivers. Its command line accepts
USB selectors only, never a command, destination, or rule body.
"""
import argparse
import json
import os
from pathlib import Path
import re
import stat
import subprocess
import sys
import tempfile


RULES_ROOT = Path("/etc/udev/rules.d")
RULE_HEADER = "# Managed by setup_factr.py --install-usb-rule.\n"
UDEVADM = "/usr/bin/udevadm"
SYSFS_ROOT = Path("/sys")


def build_rule(vendor_id, product_id, serial_number=None):
    """Build the one permitted rule, validating every interpolated value."""
    for value in (vendor_id, product_id):
        if type(value) is not int or not 0 <= value <= 0xffff:
            raise ValueError("Invalid USB VID/PID: expected an integer in [0, 65535]")
    if serial_number is not None and (
        not isinstance(serial_number, str)
        or not re.fullmatch(r"[A-Za-z0-9_.-]{1,128}", serial_number)
    ):
        raise ValueError("USB serial number must contain only letters/digits/._- for a udev rule")
    name = f"99-factr-latency-{vendor_id:04x}-{product_id:04x}.rules"
    # USB-serial latency_timer may appear only after driver binding. All ATTRS
    # match the same USB parent; do not bind this to a machine's ttyUSB number.
    rule = (RULE_HEADER + 'ACTION=="add|bind", SUBSYSTEM=="usb-serial", TEST=="latency_timer", '
            f'ATTRS{{idVendor}}=="{vendor_id:04x}", ATTRS{{idProduct}}=="{product_id:04x}", '
            + (f'ATTRS{{serial}}=="{serial_number}", ' if serial_number is not None else '')
            + 'ATTR{latency_timer}="1"\n')
    return name, rule


def usb_latency_rule(config):
    """Compatibility wrapper for the unprivileged setup CLI."""
    return build_rule(config.vendor_id, config.product_id, config.serial_number)


def resolve_latency_path(vendor_id, product_id, serial_number=None, *, sysfs_root=SYSFS_ROOT):
    """Resolve exactly one USB selector to a kernel-owned latency attribute.

    sysfs_root is injectable for unit tests only, never accepted by the CLI.
    Enumeration does not open the serial port or use a device driver.
    """
    build_rule(vendor_id, product_id, serial_number)
    root = Path(sysfs_root)
    devices_root = (root / "devices").resolve(strict=True)
    ports_root = root / "bus/usb-serial/devices"
    candidates = set()
    for port in sorted(ports_root.glob("ttyUSB*")):
        if not re.fullmatch(r"ttyUSB[0-9]+", port.name):
            continue
        try:
            device = port.resolve(strict=True)
        except FileNotFoundError:
            continue  # Unplugged during enumeration.
        if not device.is_relative_to(devices_root):
            raise ValueError(f"USB serial sysfs path escaped /sys/devices: {port}")
        for parent in (device, *device.parents):
            if parent == devices_root or not parent.is_relative_to(devices_root):
                break
            vendor_file, product_file = parent / "idVendor", parent / "idProduct"
            if not vendor_file.is_file() or not product_file.is_file():
                continue
            # VID/PID/serial must come from the SAME USB device parent.
            vendor = int(vendor_file.read_text().strip(), 16)
            product = int(product_file.read_text().strip(), 16)
            serial_file = parent / "serial"
            serial = serial_file.read_text().strip() if serial_file.is_file() else None
            if (vendor, product) == (vendor_id, product_id) and (
                serial_number is None or serial == serial_number
            ):
                attribute = (device / "latency_timer").resolve(strict=True)
                if not attribute.is_relative_to(devices_root) or attribute.parent != device:
                    raise ValueError(f"USB latency attribute escaped its device directory: {port}")
                if not attribute.is_file():
                    raise ValueError(f"USB latency attribute is not a regular file: {attribute}")
                candidates.add(attribute)
            break
    selector = f"{vendor_id:04x}:{product_id:04x}" + (
        f" serial={serial_number}" if serial_number is not None else ""
    )
    if not candidates:
        raise RuntimeError(f"No connected USB serial device with latency_timer matches {selector}")
    if len(candidates) != 1:
        raise RuntimeError(f"Multiple connected USB serial devices match {selector}; set serial_number")
    return next(iter(candidates))


def apply_current_latency(vendor_id, product_id, serial_number=None, *, sysfs_root=SYSFS_ROOT):
    """Set only the uniquely selected current adapter, and verify read-back."""
    attribute = resolve_latency_path(vendor_id, product_id, serial_number, sysfs_root=sysfs_root)
    if resolve_latency_path(vendor_id, product_id, serial_number, sysfs_root=sysfs_root) != attribute:
        raise RuntimeError("USB serial device changed during authorization; retry calibration")
    if attribute.read_text().strip() != "1":
        attribute.write_text("1\n")
    if attribute.read_text().strip() != "1":
        raise RuntimeError("USB latency_timer read-back was not 1 ms")
    return attribute


def _read_managed_rule(name, directory_fd):
    try:
        info = os.stat(name, dir_fd=directory_fd, follow_symlinks=False)
    except FileNotFoundError:
        return None
    if not stat.S_ISREG(info.st_mode):
        raise ValueError(f"Refusing to replace a symlink or non-regular udev rule: {name}")
    fd = os.open(name, os.O_RDONLY | os.O_NOFOLLOW | os.O_NONBLOCK, dir_fd=directory_fd)
    with os.fdopen(fd, "r", encoding="utf-8") as stream:
        opened = os.fstat(stream.fileno())
        if not stat.S_ISREG(opened.st_mode):
            raise ValueError(f"Refusing to read a non-regular udev rule: {name}")
        content = stream.read()
    if not content.startswith(RULE_HEADER):
        raise ValueError(f"Refusing to overwrite an unmanaged udev rule: {name}")
    return content


def install_rule(vendor_id, product_id, serial_number=None, *, apply_current=False):
    """Install atomically in the fixed system directory, then reload only.

    No global udev trigger, serial opening, or motor command. Explicit
    apply_current changes only the uniquely selected adapter's latency_timer.
    """
    name, rule = build_rule(vendor_id, product_id, serial_number)
    if os.geteuid() != 0:
        raise PermissionError("Installing the FACTR USB rule requires administrator authorization")
    if apply_current:
        # Reject ambiguous/missing devices before changing any system setting.
        resolve_latency_path(vendor_id, product_id, serial_number)
    directory_fd = os.open(RULES_ROOT, os.O_RDONLY | os.O_DIRECTORY | os.O_NOFOLLOW)
    temporary_path = None
    try:
        existing = _read_managed_rule(name, directory_fd)
        if existing != rule:
            fd, temporary_path = tempfile.mkstemp(prefix=".factr-latency-", suffix=".tmp", dir=RULES_ROOT)
            with os.fdopen(fd, "w", encoding="utf-8") as stream:
                stream.write(rule)
                stream.flush()
                os.fchmod(stream.fileno(), 0o644)
                os.fsync(stream.fileno())
            # Recheck before replacement, including existing rules from the
            # earlier setup_factr.py implementation, which used this header.
            _read_managed_rule(name, directory_fd)
            os.replace(Path(temporary_path).name, name,
                       src_dir_fd=directory_fd, dst_dir_fd=directory_fd)
            temporary_path = None
            os.fsync(directory_fd)
        subprocess.run([UDEVADM, "control", "--reload-rules"], check=True,
                       capture_output=True, text=True, timeout=30)
    finally:
        if temporary_path is not None:
            Path(temporary_path).unlink(missing_ok=True)
        os.close(directory_fd)
    applied_path = None
    if apply_current:
        try:
            applied_path = apply_current_latency(vendor_id, product_id, serial_number)
        except (OSError, ValueError, RuntimeError) as exc:
            raise RuntimeError(f"USB rule installed, but current-device latency application failed: {exc}") from exc
    return {
        "status": "installed",
        "rule_path": str(RULES_ROOT / name),
        "current_applied": apply_current,
        "latency_path": str(applied_path) if applied_path is not None else None,
        "message": ("USB 延迟已设为 1 ms，并安装永久规则；可继续校准。本次未发送任何电机指令。"
                    if apply_current else
                    "USB 延迟规则已安装。请停止 FACTR 控制、支撑机械臂后拔插 USB；"
                    "以后重插或重启自动设置为 1 ms。本次未修改当前设备或电机状态。"),
    }


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__, allow_abbrev=False)
    parser.add_argument("--vendor-id", type=int, required=True)
    parser.add_argument("--product-id", type=int, required=True)
    parser.add_argument("--serial-number")
    parser.add_argument("--apply-current", action="store_true")
    args = parser.parse_args(argv)
    try:
        result = install_rule(args.vendor_id, args.product_id, args.serial_number,
                              apply_current=args.apply_current)
    except (OSError, ValueError, RuntimeError, subprocess.SubprocessError) as exc:
        print(f"FACTR USB rule installation failed: {exc}", file=sys.stderr)
        return 1
    print(json.dumps(result, ensure_ascii=False), flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
