"""Pinned-source verification and subprocess environment; no device access."""
import hashlib
import json
import os
from pathlib import Path
import subprocess
import sys

WORKSPACE = Path(__file__).resolve().parents[4]
LOCK = WORKSPACE / "configs/factr_official.lock.json"


def verify_upstream(root):
    root = Path(root).resolve()
    lock = json.loads(LOCK.read_text())
    result = subprocess.run(["git", "-C", str(root), "rev-parse", "HEAD"],
                            capture_output=True, text=True, check=True)
    if result.stdout.strip() != lock["commit"]:
        raise ValueError("FACTR checkout revision differs from dependency lock; run scripts/setup_factr.py")
    for name, expected in lock["files"].items():
        path = root/name
        if path.is_symlink() or hashlib.sha256(path.read_bytes()).hexdigest() != expected:
            raise ValueError(f"Modified/missing official FACTR source: {name}")
    return lock


def runtime_environment(root):
    env = os.environ.copy()
    env.pop("PYTHONHOME", None)
    env["PYTHONPATH"] = os.pathsep.join(p for p in env.get("PYTHONPATH", "").split(os.pathsep)
                                        if p.startswith("/opt/ros/"))
    env["LD_LIBRARY_PATH"] = os.pathsep.join(p for p in env.get("LD_LIBRARY_PATH", "").split(os.pathsep)
                                           if p.startswith(("/opt/ros/", "/usr/lib/")))
    if Path("/opt/ros/jazzy/lib").is_dir():
        env["LD_LIBRARY_PATH"] = "/opt/ros/jazzy/lib:"+env["LD_LIBRARY_PATH"]
    env["COLCON_PREFIX_PATH"] = str(Path(root).resolve()/"install")
    env["PYTHONUNBUFFERED"] = "1"
    env["PYTHONDONTWRITEBYTECODE"] = "1"
    return env


def import_upstream(root):
    """Import definitions only; never instantiate the auto-enabling base node."""
    verify_upstream(root)
    root = Path(root).resolve()
    os.environ["COLCON_PREFIX_PATH"] = str(root/"install")
    version = f"python{sys.version_info.major}.{sys.version_info.minor}"
    ros_site = Path("/opt/ros/jazzy/lib")/version/"site-packages"
    if ros_site.is_dir():
        sys.path.insert(0, str(ros_site))
    for package in ("factr_teleop", "python_utils"):
        sys.path.insert(0, str(root/"src"/package))
    from factr_teleop.factr_teleop import FACTRTeleop
    from factr_teleop.dynamixel.driver import DynamixelDriver
    return FACTRTeleop, DynamixelDriver
