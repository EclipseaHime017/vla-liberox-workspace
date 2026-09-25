"""Persistent FACTR encoder calibration, independent of serial IO and UI.

One whole-arm reference capture follows the official FACTR pi/2 mounting
convention. The resulting physical calibration is shared by FK and gravity.
"""
from __future__ import annotations

import hashlib
import json
import math
import os
import tempfile
from dataclasses import asdict, dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Mapping, Sequence

import numpy as np

from .factr import FactrConfig

PROFILE_SCHEMA_VERSION = 2


def _json_bytes(value: Any) -> bytes:
    return json.dumps(value, sort_keys=True, separators=(",", ":"), allow_nan=False).encode("utf-8")


def config_hash(config: FactrConfig) -> str:
    """Ignore logging/gains; changes to physical mapping invalidate calibration."""
    names = ("motor_ids", "motor_models", "joint_signs", "reference_joint_positions",
             "joint_limits_min", "joint_limits_max",
             "gripper_min_travel_rad", "gripper_max_travel_rad")
    return hashlib.sha256(_json_bytes({key: getattr(config, key) for key in names})).hexdigest()


def _vector(value: Sequence[float], size: int, name: str) -> np.ndarray:
    array = np.asarray(value, dtype=float)
    if array.shape != (size,) or not np.all(np.isfinite(array)):
        raise ValueError(f"{name} must contain {size} finite numbers")
    return array


@dataclass(frozen=True)
class FactrCalibrationProfile:
    offsets: tuple[float, ...]
    gripper_open: float
    gripper_closed: float
    device_fingerprint: dict[str, Any]
    config_hash: str
    created_at: str
    schema_version: int = PROFILE_SCHEMA_VERSION

    def as_dict(self) -> dict[str, Any]:
        return asdict(self)


def make_profile(config: FactrConfig, offsets: Sequence[float], gripper_open: float,
                 gripper_closed: float, device_fingerprint: Mapping[str, Any]) -> FactrCalibrationProfile:
    offset_array = _vector(offsets, 7, "offsets")
    if not math.isfinite(gripper_open) or not math.isfinite(gripper_closed):
        raise ValueError("Gripper endpoints must be finite")
    travel = abs(gripper_closed-gripper_open)
    if not config.gripper_min_travel_rad <= travel <= config.gripper_max_travel_rad:
        raise ValueError(f"Gripper travel {travel:.4f} rad is outside configured calibration range")
    if not isinstance(device_fingerprint, Mapping) or not device_fingerprint:
        raise ValueError("A non-empty read-only device fingerprint is required")
    # Round-trip ensures only persistent JSON data, also rejecting NaN/Inf.
    fingerprint = json.loads(_json_bytes(dict(device_fingerprint)))
    return FactrCalibrationProfile(tuple(offset_array.tolist()), float(gripper_open), float(gripper_closed),
                                   fingerprint, config_hash(config), datetime.now(timezone.utc).isoformat())


def save_profile(path: Path, profile: FactrCalibrationProfile) -> None:
    """Atomic, checksummed local file. This never writes a servo register."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    payload = profile.as_dict()
    document = {"profile": payload, "sha256": hashlib.sha256(_json_bytes(payload)).hexdigest()}
    fd, temporary = tempfile.mkstemp(prefix=f".{path.name}.", suffix=".tmp", dir=path.parent)
    try:
        with os.fdopen(fd, "w", encoding="utf-8") as stream:
            json.dump(document, stream, indent=2, allow_nan=False)
            stream.write("\n")
            stream.flush()
            os.fsync(stream.fileno())
        os.replace(temporary, path)
    finally:
        if os.path.exists(temporary):
            os.unlink(temporary)


def load_profile(path: Path, config: FactrConfig,
                 device_fingerprint: Mapping[str, Any]) -> FactrCalibrationProfile:
    def unique(pairs: list[tuple[str, Any]]) -> dict[str, Any]:
        result: dict[str, Any] = {}
        for key, value in pairs:
            if key in result:
                raise ValueError(f"Duplicate calibration key: {key}")
            result[key] = value
        return result
    document = json.loads(Path(path).read_text(encoding="utf-8"), object_pairs_hook=unique)
    if not isinstance(document, dict) or set(document) != {"profile", "sha256"}:
        raise ValueError("Invalid FACTR calibration document")
    payload = document["profile"]
    expected_keys = set(FactrCalibrationProfile.__dataclass_fields__)
    if not isinstance(payload, dict) or set(payload) != expected_keys:
        raise ValueError("Invalid FACTR calibration profile fields")
    if hashlib.sha256(_json_bytes(payload)).hexdigest() != document["sha256"]:
        raise ValueError("FACTR calibration checksum mismatch; recalibrate")
    if type(payload["schema_version"]) is not int or payload["schema_version"] != PROFILE_SCHEMA_VERSION:
        raise ValueError("Old calibration method; run the whole-arm reference calibration (c) again")
    if payload["config_hash"] != config_hash(config):
        raise ValueError("FACTR kinematic configuration changed; recalibrate")
    usb_device = device_fingerprint.get("usb_device")
    if isinstance(usb_device, Mapping) and not usb_device.get("serial_number"):
        raise ValueError("FACTR has no USB serial number; saved calibration cannot identify this arm. Recalibrate")
    if _json_bytes(payload["device_fingerprint"]) != _json_bytes(dict(device_fingerprint)):
        raise ValueError("FACTR device fingerprint changed (model/ID/Homing Offset/Drive Mode); recalibrate")
    validated = make_profile(config, payload["offsets"], payload["gripper_open"],
                             payload["gripper_closed"], payload["device_fingerprint"])
    if not isinstance(payload["created_at"], str):
        raise ValueError("Invalid calibration creation time")
    return FactrCalibrationProfile(validated.offsets, validated.gripper_open, validated.gripper_closed,
                                   validated.device_fingerprint, validated.config_hash, payload["created_at"])


def gripper_fraction(raw: float, profile: FactrCalibrationProfile) -> float:
    if not math.isfinite(raw):
        raise ValueError("Non-finite gripper reading")
    # Endpoints can straddle the raw encoder's power-cycle boundary.
    distance = (raw-profile.gripper_open+math.pi) % (2*math.pi)-math.pi
    travel = profile.gripper_closed-profile.gripper_open
    if not math.isfinite(travel) or abs(travel) < 1e-9:
        raise ValueError("Invalid calibrated gripper travel")
    return float(np.clip(distance/travel, 0.0, 1.0))
