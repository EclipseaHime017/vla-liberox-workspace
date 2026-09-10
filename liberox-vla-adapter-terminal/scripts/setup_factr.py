#!/usr/bin/env python3
"""Install/check pinned official sources and isolated runtime; never access motors."""
import argparse
import json
from pathlib import Path
import subprocess
import sys

PROJECT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT))

def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--check", action="store_true", help="Verify only; do not install anything")
    args = parser.parse_args()
    from backend.app.devices.factr import load_factr_config
    from backend.app.devices.factr_official import LOCK, verify_upstream, runtime_environment
    settings = load_factr_config().runtime
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
    raise SystemExit(main())
