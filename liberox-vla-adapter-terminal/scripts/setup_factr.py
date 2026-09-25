#!/usr/bin/env python3
"""Install/check pinned official sources and isolated runtime; never access motors."""
import argparse
import json
import os
from pathlib import Path
import subprocess
import sys

PROJECT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT))

from backend.app.devices.factr_usb_rule import usb_latency_rule

USB_RULE_HELPER = PROJECT / "backend/app/devices/factr_usb_rule.py"


def install_usb_rule(config):
    # No attached device or official runtime is required for provisioning a PC.
    usb_latency_rule(config)  # Validate selectors before requesting sudo.
    print(f"USB {config.vendor_id:04x}:{config.product_id:04x}, serial={config.serial_number or 'ALL MATCHING DEVICES'}", flush=True)
    privilege = [] if os.geteuid() == 0 else ["sudo", "--"]
    command = [*privilege, "/usr/bin/python3", "-I", str(USB_RULE_HELPER),
               "--vendor-id", str(config.vendor_id), "--product-id", str(config.product_id)]
    if config.serial_number is not None:
        command.append(f"--serial-number={config.serial_number}")
    # Inherit the terminal, including sudo's password prompt. The helper also
    # prints its JSON status for non-interactive diagnostics.
    subprocess.run(command, check=True)
    return 0


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    operation = parser.add_mutually_exclusive_group()
    operation.add_argument("--check", action="store_true", help="Verify only; do not install anything")
    operation.add_argument("--install-usb-rule", action="store_true",
                           help="Install a persistent 1 ms USB latency rule for YAML VID/PID (uses sudo; device need not be connected)")
    args = parser.parse_args()
    from backend.app.devices.factr import load_factr_config
    config = load_factr_config()
    if args.install_usb_rule:
        return install_usb_rule(config)  # No runtime setup/check, calibration or torque.
    from backend.app.devices.factr_official import LOCK, verify_upstream, runtime_environment
    settings = config.runtime
    root, python = Path(settings["upstream_root"]), Path(settings["runtime_python"])
    lock = json.loads(LOCK.read_text())
    if not args.check:
        if not root.exists():
            root.parent.mkdir(parents=True, exist_ok=True)
            subprocess.run(["git", "clone", lock["repository"], str(root)], check=True)
            subprocess.run(["git", "-C", str(root), "checkout", "--detach", lock["commit"]], check=True)
        verify_upstream(root)  # Never reset existing user edits/checkouts.
        if not python.exists():
            if python.name != "python" or python.parent.name != "bin":
                raise ValueError("runtime_python must point to <venv>/bin/python")
            subprocess.run(["/usr/bin/python3", "-m", "venv", "--system-site-packages", str(python.parent.parent)], check=True)
        subprocess.run([str(python), "-m", "pip", "install", "dynamixel-sdk==3.7.31", "pyserial==3.5"], check=True)
    verify_upstream(root)
    result = subprocess.run([str(python), str(PROJECT/"scripts/factr_official_worker.py"), "--check"],
                            env=runtime_environment(root))
    if result.returncode:
        print("FACTR runtime check failed. Install/source ROS 2 and Pinocchio for the system Python first; "
              "do not install ROS into vla-liberox. No hardware was accessed.", file=sys.stderr)
    return result.returncode

if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except (OSError, ValueError, RuntimeError, subprocess.CalledProcessError) as exc:
        print(f"FACTR setup failed: {exc}", file=sys.stderr)
        raise SystemExit(1)
