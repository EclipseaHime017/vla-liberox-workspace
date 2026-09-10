"""Shared FACTR config/snapshot and read-only startup validation, not a control loop."""
from __future__ import annotations

import importlib
import math
import os
import stat
from dataclasses import asdict, dataclass, fields
from pathlib import Path
from typing import Any, Callable

import numpy as np
import yaml

from .spacemouse import UniqueKeyLoader


DEFAULT_FACTR_CONFIG = Path(__file__).resolve().parents[4] / "configs/factr_test_config.yaml"
# ROBOTIS eManual control tables. Restrict to the verified FACTR Franka motors.
MODEL_NUMBERS = {"XC330_T288_T": 1220, "XM430_W210_T": 1030}


@dataclass(frozen=True)
class FactrConfig:
    mode: str
    device_name: str
    device_path: str | None
    baudrate: int
    motor_ids: tuple[int, ...]
    motor_models: tuple[str, ...]
    joint_signs: tuple[int, ...]
    reference_joint_positions: tuple[float, ...]
    joint_limits_min: tuple[float, ...]
    joint_limits_max: tuple[float, ...]
    stale_timeout_ms: int
    max_joint_jump_rad: float
    gripper_min_travel_rad: float
    gripper_max_travel_rad: float
    gripper_open_threshold: float
    gripper_close_threshold: float
    gripper_takeover_delta: float
    translation_gain: float
    rotation_gain: float
    test_duration_seconds: float
    max_steps: int
    countdown_seconds: int
    runtime: dict[str, Any] | None = None

    def metadata(self) -> dict[str, Any]:
        return asdict(self)


def load_factr_config(path: Path = DEFAULT_FACTR_CONFIG) -> FactrConfig:
    path = Path(path).expanduser().resolve()
    try:
        raw = yaml.load(path.read_text(encoding="utf-8"), Loader=UniqueKeyLoader)
    except yaml.YAMLError as exc:
        raise ValueError(f"Invalid FACTR YAML: {exc}") from exc
    return parse_factr_config(raw, path)


def parse_factr_config(raw, path=DEFAULT_FACTR_CONFIG) -> FactrConfig:
    """Validate YAML or an IPC bootstrap with the same schema."""
    path = Path(path)
    if not isinstance(raw, dict) or any(not isinstance(key, str) for key in raw):
        raise TypeError("FACTR config must be a string-keyed mapping")
    expected = {field.name for field in fields(FactrConfig)}
    raw.setdefault("runtime", None)
    if set(raw) != expected:
        raise ValueError(f"FACTR keys: missing={sorted(expected-set(raw))}, unknown={sorted(set(raw)-expected)}")
    raw = dict(raw)
    for key in ("mode", "device_name"):
        if not isinstance(raw[key], str) or not raw[key].strip():
            raise TypeError(f"{key} must be a non-empty string")
    if raw["mode"] not in {"device", "simulation"}:
        raise ValueError("mode must be device or simulation")
    port = raw["device_path"]
    if port is not None:
        if not isinstance(port, str) or not port.strip() or not Path(port).expanduser().is_absolute():
            raise ValueError("device_path must be an explicit absolute serial path or null")
        raw["device_path"] = str(Path(port).expanduser())
    for key in ("baudrate", "stale_timeout_ms", "max_steps", "countdown_seconds"):
        if type(raw[key]) is not int:
            raise TypeError(f"{key} must be an integer")
    if raw["baudrate"] not in {57600, 115200, 1000000, 2000000, 3000000, 4000000}:
        raise ValueError("baudrate is unsupported by FACTR motors")
    if not 1 <= raw["stale_timeout_ms"] <= 250 or raw["max_steps"] < 1 or not 0 <= raw["countdown_seconds"] <= 30:
        raise ValueError("Invalid timeout, max_steps or countdown_seconds")
    for key, length in (("motor_ids", 8), ("motor_models", 8), ("joint_signs", 7),
                        ("reference_joint_positions", 7), ("joint_limits_min", 7), ("joint_limits_max", 7)):
        if not isinstance(raw[key], list) or len(raw[key]) != length:
            raise ValueError(f"{key} must have {length} entries")
        raw[key] = tuple(raw[key])
    if any(type(x) is not int or not 0 <= x <= 252 for x in raw["motor_ids"]) or len(set(raw["motor_ids"])) != 8:
        raise ValueError("motor_ids must contain eight distinct Protocol 2 IDs")
    if any(not isinstance(x, str) or x not in MODEL_NUMBERS for x in raw["motor_models"]):
        raise ValueError(f"motor_models must use verified models {list(MODEL_NUMBERS)}")
    if any(type(x) is not int or x not in {-1, 1} for x in raw["joint_signs"]):
        raise ValueError("joint_signs must be seven -1/+1 integers")
    numeric_vectors = ("reference_joint_positions", "joint_limits_min", "joint_limits_max")
    number_keys = {"max_joint_jump_rad", "gripper_min_travel_rad",
                   "gripper_max_travel_rad", "gripper_open_threshold", "gripper_close_threshold",
                   "gripper_takeover_delta", "translation_gain", "rotation_gain", "test_duration_seconds"}
    for key in number_keys | set(numeric_vectors):
        values = raw[key] if key in numeric_vectors else (raw[key],)
        if any(type(v) not in (int, float) or not math.isfinite(v) for v in values):
            raise TypeError(f"{key} must contain finite numbers")
        raw[key] = tuple(float(v) for v in values) if key in numeric_vectors else float(raw[key])
    if any(raw[key] <= 0 for key in number_keys):
        raise ValueError("FACTR numerical parameters must be positive")
    if not 0 < raw["max_joint_jump_rad"] < math.pi:
        raise ValueError("Invalid encoder discontinuity threshold")
    if not 0 < raw["gripper_open_threshold"] < raw["gripper_close_threshold"] < 1:
        raise ValueError("gripper hysteresis must satisfy 0 < open < close < 1")
    if not 0 < raw["gripper_min_travel_rad"] < raw["gripper_max_travel_rad"] < math.pi:
        raise ValueError("Invalid gripper travel limits")
    if raw["gripper_takeover_delta"] >= 1 or not all(0.05 <= raw[k] <= 1 for k in ("translation_gain", "rotation_gain")):
        raise ValueError("takeover delta and gains must not exceed their normalized range")
    lower, upper, reference = (np.asarray(raw[k]) for k in ("joint_limits_min", "joint_limits_max", "reference_joint_positions"))
    if np.any(lower >= upper) or np.any(reference < lower) or np.any(reference > upper):
        raise ValueError("Reference pose must lie inside strictly ordered joint limits")
    from .factr_runtime_config import serialized_options
    raw["runtime"] = serialized_options(raw["runtime"], path.parent)
    return FactrConfig(**raw)


@dataclass(frozen=True)
class FactrSnapshot:
    sequence: int = 0
    captured_monotonic: float = 0.0
    sample_monotonic: float | None = None
    raw_joints: tuple[float, ...] = (0.0,) * 7
    joint_positions: tuple[float, ...] = (0.0,) * 7
    raw_gripper: float = 0.0
    gripper_fraction: float | None = None
    gripper_command: float = -1.0
    connected: bool = False
    stale: bool = True
    error: str | None = None

    @property
    def sample_age_seconds(self) -> float | None:
        return None if self.sample_monotonic is None else max(0.0, self.captured_monotonic-self.sample_monotonic)

    def as_dict(self) -> dict[str, Any]:
        return {**asdict(self), "sample_age_seconds": self.sample_age_seconds}


def serial_owners(device_path: str, *, proc_root: Path = Path("/proc"),
                  ignore_fd: int | None = None) -> dict[str, Any]:
    """Read-only Linux occupancy check before the first Dynamixel instruction.

    flock cannot detect an older non-cooperating driver. Compare character
    device numbers through proc fds (including alternate symlinks), never kill
    processes. Linux may hide other users' descriptors; report that limitation.
    """
    target = Path(device_path).stat()
    if not stat.S_ISCHR(target.st_mode):
        raise RuntimeError("FACTR device_path is not a serial character device")
    busy, unreadable = set(), 0
    for process in proc_root.iterdir():
        if not process.name.isdecimal():
            continue
        try:
            descriptors = list((process/"fd").iterdir())
        except PermissionError:
            unreadable += 1
            continue
        except FileNotFoundError:
            continue
        for descriptor in descriptors:
            if int(process.name) == os.getpid() and ignore_fd is not None and descriptor.name == str(ignore_fd):
                continue
            try:
                opened = descriptor.stat()
            except (PermissionError, FileNotFoundError, OSError):
                continue
            if stat.S_ISCHR(opened.st_mode) and opened.st_rdev == target.st_rdev:
                busy.add(int(process.name))
    return {"busy_pids": sorted(busy), "unreadable_process_count": unreadable}


class FactrStartupProbe:
    """Read-only startup model/torque/error validation, not an input sampler.

    GroupSyncRead in SDK 3.7.31 discards servo error bytes. Explicit readRx
    validates them, as well as the Hardware Error Status at address70.
    """
    START_ADDRESS, DATA_LENGTH = 64, 72

    def __init__(self, config: FactrConfig, *, sdk: Any = None,
                 owners_probe: Callable[..., dict[str, Any]] = serial_owners) -> None:
        self.config = config
        self.sdk = sdk
        self.port = None
        self.packet = None
        self._exclusive_acquired = False
        self._owners_probe = owners_probe

    def open(self) -> dict[str, Any]:
        if self.config.device_path is None:
            raise RuntimeError("Set FACTR device_path to your explicit /dev/serial/by-id/... path")
        occupancy = self._owners_probe(self.config.device_path)
        if occupancy["busy_pids"]:
            raise RuntimeError(f"FACTR serial port is occupied by PID(s) {occupancy['busy_pids']}; close that controller manually")
        if self.sdk is None:
            try:
                self.sdk = importlib.import_module("dynamixel_sdk")
            except ImportError as exc:
                raise RuntimeError("Missing dynamixel-sdk; install requirements-factr.txt") from exc
        self.port = self.sdk.PortHandler(self.config.device_path)
        self.packet = self.sdk.PacketHandler(2.0)
        try:
            # setBaudRate configures the HOST serial port, not a motor register.
            if not self.port.setBaudRate(self.config.baudrate):
                raise RuntimeError("Unable to open FACTR serial port at configured baudrate")
            serial_port = getattr(self.port, "ser", None)
            if serial_port is not None:
                import fcntl
                import termios
                fcntl.flock(serial_port.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)
                fcntl.ioctl(serial_port.fileno(), termios.TIOCEXCL)
                self._exclusive_acquired = True
                # Recheck after obtaining host-side exclusion, before pinging.
                occupancy = self._owners_probe(self.config.device_path, ignore_fd=serial_port.fileno())
                if occupancy["busy_pids"]:
                    raise RuntimeError(f"FACTR serial port was already open in PID(s) {occupancy['busy_pids']}")
            models = []
            for motor_id, model_name in zip(self.config.motor_ids, self.config.motor_models):
                model, result, error = self.packet.ping(self.port, motor_id)
                self._validate_result(result, error, motor_id)
                if model != MODEL_NUMBERS[model_name]:
                    raise RuntimeError(f"FACTR motor {motor_id}: model {model}, expected {model_name} ({MODEL_NUMBERS[model_name]})")
                models.append({"id": motor_id, "model": model, "name": model_name})
            self.check_status()  # Verify torque OFF before official driver construction.
            return {"motors": models, "baudrate": self.config.baudrate, "passive_only": True,
                    "occupancy_check": occupancy}
        except BaseException:
            self.close()
            raise

    def _validate_result(self, result: int, error: int, motor_id: int | None = None) -> None:
        if result != self.sdk.COMM_SUCCESS or error:
            raise RuntimeError(f"FACTR read failed: motor={motor_id}, communication={result}, servo_error={error}")

    def check_status(self) -> None:
        if self.port is None or self.packet is None:
            raise RuntimeError("FACTR serial transport is not open")
        result = self.packet.syncReadTx(self.port, self.START_ADDRESS, self.DATA_LENGTH,
                                       list(self.config.motor_ids), len(self.config.motor_ids))
        self._validate_result(result, 0)
        for motor_id in self.config.motor_ids:
            data, result, error = self.packet.readRx(self.port, motor_id, self.DATA_LENGTH)
            self._validate_result(result, error, motor_id)
            if len(data) != self.DATA_LENGTH:
                raise RuntimeError(f"FACTR motor {motor_id}: incomplete packet")
            if data[0] != 0:
                raise RuntimeError(f"FACTR motor {motor_id}: torque is enabled; startup refused. Support the arm and close the active controller; no torque command was sent.")
            if data[6] != 0:
                raise RuntimeError(f"FACTR motor {motor_id}: hardware error {data[6]}")

    def close(self) -> None:
        port, self.port = self.port, None
        if port is not None and getattr(port, "is_open", False):
            serial_port = getattr(port, "ser", None)
            if serial_port is not None and self._exclusive_acquired:
                import fcntl
                import termios
                try:
                    fcntl.ioctl(serial_port.fileno(), termios.TIOCNXCL)
                except OSError:
                    pass
            port.closePort()
        self._exclusive_acquired = False


def probe_factr(config: FactrConfig) -> dict[str, Any]:
    """Read-only idle discovery; SDK and ROS live in the official environment."""
    if not config.device_path:
        return {"connected": False, "error": "请配置 FACTR device_path"}
    if not Path(config.runtime["runtime_python"]).is_file():
        return {"connected": False, "error": "官方运行环境缺失，请运行 setup_factr.py"}
    port = Path(config.device_path)
    try:
        if not stat.S_ISCHR(port.stat().st_mode):
            raise ValueError("FACTR path is not a serial character device")
        if not os.access(port, os.R_OK | os.W_OK):
            raise ValueError("Serial permission denied; check dialout/udev")
    except (OSError, ValueError) as exc:
        return {"connected": False, "error": str(exc)}
    return {"connected": True, "verified": False, "error": None}
