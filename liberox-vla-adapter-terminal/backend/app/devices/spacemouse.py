"""Threaded SpaceMouse input and strict configuration for teleoperation tests.

The HID dependency is imported lazily so the rest of the UI and its unit tests
do not require a SpaceMouse or a system HIDAPI installation.
"""

from __future__ import annotations

import importlib
import importlib.metadata
import os
import threading
import time
from dataclasses import dataclass, fields
from pathlib import Path
from typing import Any, Callable, Iterable

import numpy as np
import yaml


DEFAULT_SPACEMOUSE_CONFIG = Path(__file__).resolve().parents[3] / "spacemouse_test_config.yaml"
AXIS_NAMES = ("x", "y", "z", "roll", "pitch", "yaw")
VALID_MODES = frozenset({"device", "simulation"})
# This is the convention produced by *our* axis_order / axis_signs transform,
# not an option forwarded to PySpaceMouse.  PySpaceMouse 2.0.0 on PyPI predates
# the AxisConvention API currently documented on its main branch.
VALID_AXIS_CONVENTIONS = frozenset({"ros"})


class UniqueKeyLoader(yaml.SafeLoader):
    """Safe YAML loader which rejects duplicate keys."""


def _construct_unique_mapping(
    loader: UniqueKeyLoader,
    node: yaml.MappingNode,
    deep: bool = False,
) -> dict[Any, Any]:
    mapping: dict[Any, Any] = {}
    for key_node, value_node in node.value:
        key = loader.construct_object(key_node, deep=deep)
        if key in mapping:
            raise yaml.constructor.ConstructorError(
                "while constructing a mapping",
                node.start_mark,
                f"found duplicate key {key!r}",
                key_node.start_mark,
            )
        mapping[key] = loader.construct_object(value_node, deep=deep)
    return mapping


UniqueKeyLoader.add_constructor(
    yaml.resolver.BaseResolver.DEFAULT_MAPPING_TAG,
    _construct_unique_mapping,
)


@dataclass(frozen=True)
class SpaceMouseTestConfig:
    mode: str
    device_name: str
    device_index: int
    device_path: str | None
    expected_vendor_id: int
    expected_product_id: int
    axis_convention: str
    axis_order: tuple[str, ...]
    axis_signs: tuple[int, ...]
    translation_gain: float
    rotation_gain: float
    deadzone: float
    smoothing_alpha: float
    neutral_calibration_seconds: float
    neutral_calibration_timeout_seconds: float
    neutral_max_abs: float
    poll_interval_ms: float
    stale_timeout_ms: int
    test_duration_seconds: float
    max_steps: int
    countdown_seconds: int
    save_video: bool
    trajectory_plot: bool
    output_root: Path


def _strict_string(raw: dict[str, Any], key: str) -> str:
    value = raw[key]
    if not isinstance(value, str) or not value.strip():
        raise TypeError(f"SpaceMouse key {key!r} must be a non-empty string")
    return value.strip()


def _strict_int(raw: dict[str, Any], key: str) -> int:
    value = raw[key]
    if type(value) is not int:
        raise TypeError(f"SpaceMouse key {key!r} must be an integer")
    return value


def _strict_number(raw: dict[str, Any], key: str) -> float:
    value = raw[key]
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise TypeError(f"SpaceMouse key {key!r} must be a number")
    value = float(value)
    if not np.isfinite(value):
        raise ValueError(f"SpaceMouse key {key!r} must be finite")
    return value


def _strict_bool(raw: dict[str, Any], key: str) -> bool:
    value = raw[key]
    if type(value) is not bool:
        raise TypeError(f"SpaceMouse key {key!r} must be true or false")
    return value


def load_spacemouse_config(
    path: Path = DEFAULT_SPACEMOUSE_CONFIG,
) -> SpaceMouseTestConfig:
    """Load the fixed SpaceMouse test configuration with strict validation."""
    path = path.expanduser().resolve()
    if not path.is_file():
        raise FileNotFoundError(f"SpaceMouse configuration not found: {path}")
    try:
        raw = yaml.load(path.read_text(encoding="utf-8"), Loader=UniqueKeyLoader)
    except yaml.YAMLError as exc:
        raise ValueError(f"Invalid YAML in {path}: {exc}") from exc
    if not isinstance(raw, dict):
        raise TypeError("SpaceMouse configuration root must be a mapping")
    if any(not isinstance(key, str) for key in raw):
        raise TypeError("All SpaceMouse configuration keys must be strings")

    expected = {field.name for field in fields(SpaceMouseTestConfig)}
    missing = sorted(expected - set(raw))
    unknown = sorted(set(raw) - expected)
    if missing:
        raise ValueError(f"Missing SpaceMouse configuration keys: {missing}")
    if unknown:
        raise ValueError(f"Unknown SpaceMouse configuration keys: {unknown}")

    device_path = raw["device_path"]
    if device_path is not None:
        if not isinstance(device_path, str) or not device_path.strip():
            raise TypeError("SpaceMouse key 'device_path' must be an absolute path or null")
        candidate = Path(device_path).expanduser()
        if not candidate.is_absolute():
            raise ValueError("SpaceMouse key 'device_path' must be absolute when provided")
        device_path = str(candidate)

    axis_order = raw["axis_order"]
    if not isinstance(axis_order, list) or any(not isinstance(value, str) for value in axis_order):
        raise TypeError("SpaceMouse key 'axis_order' must be a list of six axis names")
    if len(axis_order) != 6 or set(axis_order) != set(AXIS_NAMES):
        raise ValueError(f"SpaceMouse key 'axis_order' must be a permutation of {AXIS_NAMES}")

    axis_signs = raw["axis_signs"]
    if not isinstance(axis_signs, list) or len(axis_signs) != 6:
        raise TypeError("SpaceMouse key 'axis_signs' must be a list of six -1/+1 integers")
    if any(type(value) is not int or value not in {-1, 1} for value in axis_signs):
        raise ValueError("Every SpaceMouse axis sign must be exactly -1 or +1")

    output_root_value = _strict_string(raw, "output_root")
    output_root = Path(output_root_value).expanduser()
    if not output_root.is_absolute():
        output_root = path.parent / output_root

    config = SpaceMouseTestConfig(
        mode=_strict_string(raw, "mode"),
        device_name=_strict_string(raw, "device_name"),
        device_index=_strict_int(raw, "device_index"),
        device_path=device_path,
        expected_vendor_id=_strict_int(raw, "expected_vendor_id"),
        expected_product_id=_strict_int(raw, "expected_product_id"),
        axis_convention=_strict_string(raw, "axis_convention"),
        axis_order=tuple(axis_order),
        axis_signs=tuple(axis_signs),
        translation_gain=_strict_number(raw, "translation_gain"),
        rotation_gain=_strict_number(raw, "rotation_gain"),
        deadzone=_strict_number(raw, "deadzone"),
        smoothing_alpha=_strict_number(raw, "smoothing_alpha"),
        neutral_calibration_seconds=_strict_number(raw, "neutral_calibration_seconds"),
        neutral_calibration_timeout_seconds=_strict_number(
            raw,
            "neutral_calibration_timeout_seconds",
        ),
        neutral_max_abs=_strict_number(raw, "neutral_max_abs"),
        poll_interval_ms=_strict_number(raw, "poll_interval_ms"),
        stale_timeout_ms=_strict_int(raw, "stale_timeout_ms"),
        test_duration_seconds=_strict_number(raw, "test_duration_seconds"),
        max_steps=_strict_int(raw, "max_steps"),
        countdown_seconds=_strict_int(raw, "countdown_seconds"),
        save_video=_strict_bool(raw, "save_video"),
        trajectory_plot=_strict_bool(raw, "trajectory_plot"),
        output_root=output_root.resolve(),
    )

    if config.mode not in VALID_MODES:
        raise ValueError(f"SpaceMouse key 'mode' must be one of {sorted(VALID_MODES)}")
    if config.device_index < 0:
        raise ValueError("SpaceMouse key 'device_index' must be >= 0")
    for key, value in (
        ("expected_vendor_id", config.expected_vendor_id),
        ("expected_product_id", config.expected_product_id),
    ):
        if not 0 <= value <= 0xFFFF:
            raise ValueError(f"SpaceMouse key {key!r} must be a 16-bit USB ID")
    if config.axis_convention not in VALID_AXIS_CONVENTIONS:
        raise ValueError(
            f"SpaceMouse key 'axis_convention' must be one of {sorted(VALID_AXIS_CONVENTIONS)}"
        )
    for key, value in (
        ("translation_gain", config.translation_gain),
        ("rotation_gain", config.rotation_gain),
    ):
        if not 0.0 < value <= 1.0:
            raise ValueError(f"SpaceMouse key {key!r} must be in (0, 1]")
    if not 0.0 <= config.deadzone < 1.0:
        raise ValueError("SpaceMouse key 'deadzone' must be in [0, 1)")
    if not 0.0 < config.smoothing_alpha <= 1.0:
        raise ValueError("SpaceMouse key 'smoothing_alpha' must be in (0, 1]")
    if config.neutral_calibration_seconds <= 0.0:
        raise ValueError("SpaceMouse key 'neutral_calibration_seconds' must be > 0")
    if config.neutral_calibration_timeout_seconds < config.neutral_calibration_seconds:
        raise ValueError(
            "SpaceMouse key 'neutral_calibration_timeout_seconds' must be >= "
            "neutral_calibration_seconds"
        )
    if not 0.0 < config.neutral_max_abs <= 1.0:
        raise ValueError("SpaceMouse key 'neutral_max_abs' must be in (0, 1]")
    if not 0.1 <= config.poll_interval_ms <= 20.0:
        raise ValueError("SpaceMouse key 'poll_interval_ms' must be in [0.1, 20]")
    if not 20 <= config.stale_timeout_ms <= 5000:
        raise ValueError("SpaceMouse key 'stale_timeout_ms' must be in [20, 5000]")
    if config.test_duration_seconds <= 0.0:
        raise ValueError("SpaceMouse key 'test_duration_seconds' must be > 0")
    if config.max_steps < 1:
        raise ValueError("SpaceMouse key 'max_steps' must be >= 1")
    if not 0 <= config.countdown_seconds <= 10:
        raise ValueError("SpaceMouse key 'countdown_seconds' must be in [0, 10]")
    return config


@dataclass(frozen=True)
class SpaceMouseSnapshot:
    sequence: int
    captured_monotonic: float
    event_monotonic: float | None
    device_timestamp: float
    raw_axes: tuple[float, ...]
    corrected_axes: tuple[float, ...]
    command_axes: tuple[float, ...]
    action: tuple[float, ...]
    buttons: tuple[int, ...]
    connected: bool
    stale: bool
    error: str | None

    @property
    def sample_age_seconds(self) -> float | None:
        if self.event_monotonic is None:
            return None
        return max(0.0, self.captured_monotonic - self.event_monotonic)


class SpaceMouseTransform:
    """Bias correction, deadzone, axis mapping, gain, and optional EMA."""

    def __init__(self, config: SpaceMouseTestConfig):
        self.config = config
        self.bias = np.zeros(6, dtype=np.float64)
        self.filtered = np.zeros(6, dtype=np.float64)
        self._axis_indices = np.asarray([AXIS_NAMES.index(name) for name in config.axis_order])
        self._translation_gain = config.translation_gain
        self._rotation_gain = config.rotation_gain

    @property
    def gains(self) -> tuple[float, float]:
        return self._translation_gain, self._rotation_gain

    def set_gains(self, translation_gain: float, rotation_gain: float) -> None:
        """Update gains without resetting calibration or the HID reader."""
        values = (float(translation_gain), float(rotation_gain))
        if not all(np.isfinite(value) and 0.0 < value <= 1.0 for value in values):
            raise ValueError("SpaceMouse gains must be finite and in (0, 1]")
        self._translation_gain, self._rotation_gain = values

    def calibrate(self, samples: Iterable[Iterable[float]]) -> dict[str, list[float] | int]:
        values = np.asarray(list(samples), dtype=np.float64)
        if values.ndim != 2 or values.shape[1] != 6 or len(values) < 2:
            raise ValueError("Neutral calibration requires at least two six-axis samples")
        if not np.isfinite(values).all():
            raise ValueError("Neutral calibration samples contain NaN or Inf")
        peak = np.max(np.abs(values), axis=0)
        span = np.ptp(values, axis=0)
        if float(np.max(peak)) > self.config.neutral_max_abs:
            raise RuntimeError(
                "SpaceMouse moved during neutral calibration: "
                f"maximum absolute axis value {float(np.max(peak)):.4f} exceeds "
                f"neutral_max_abs={self.config.neutral_max_abs:.4f}"
            )
        self.bias = np.median(values, axis=0)
        self.filtered.fill(0.0)
        return {
            "sample_count": int(len(values)),
            "bias": self.bias.tolist(),
            "peak_abs": peak.tolist(),
            "span": span.tolist(),
        }

    def reset_filter(self) -> None:
        self.filtered.fill(0.0)

    def apply(self, raw_axes: Iterable[float]) -> tuple[np.ndarray, np.ndarray]:
        raw = np.asarray(tuple(raw_axes), dtype=np.float64)
        if raw.shape != (6,) or not np.isfinite(raw).all():
            raise ValueError("SpaceMouse raw axes must be a finite six-vector")
        corrected = np.clip(raw - self.bias, -1.0, 1.0)
        magnitude = np.abs(corrected)
        deadzoned = np.zeros(6, dtype=np.float64)
        active = magnitude > self.config.deadzone
        deadzoned[active] = np.sign(corrected[active]) * (
            (magnitude[active] - self.config.deadzone) / (1.0 - self.config.deadzone)
        )
        mapped = deadzoned[self._axis_indices] * np.asarray(self.config.axis_signs)
        mapped[:3] *= self._translation_gain
        mapped[3:] *= self._rotation_gain
        alpha = self.config.smoothing_alpha
        self.filtered = alpha * mapped + (1.0 - alpha) * self.filtered
        return corrected, np.clip(self.filtered, -1.0, 1.0).copy()


class SpaceMouseInput:
    """Continuously read SpaceMouse HID reports and expose the latest action."""

    def __init__(
        self,
        config: SpaceMouseTestConfig,
        *,
        device_factory: Callable[[], Any] | None = None,
        clock: Callable[[], float] = time.monotonic,
        sleep: Callable[[float], None] = time.sleep,
    ) -> None:
        self.config = config
        self.transform = SpaceMouseTransform(config)
        self._device_factory = device_factory
        self._clock = clock
        self._sleep = sleep
        self._lock = threading.Lock()
        self._stop_event = threading.Event()
        self._thread: threading.Thread | None = None
        self._device: Any | None = None
        self._connected = False
        self._error: str | None = None
        self._sequence = 0
        self._last_device_timestamp = -1.0
        self._event_monotonic: float | None = None
        self._raw_axes = np.zeros(6, dtype=np.float64)
        self._corrected_axes = np.zeros(6, dtype=np.float64)
        self._command_axes = np.zeros(6, dtype=np.float64)
        self._buttons: tuple[int, ...] = ()
        self._previous_buttons: tuple[int, ...] = ()
        self._gripper = -1.0
        self._event_times: list[float] = []
        self._device_details: dict[str, Any] = {}

    def __enter__(self) -> "SpaceMouseInput":
        self.start()
        return self

    def __exit__(self, exc_type, exc_value, traceback) -> None:
        self.stop()

    @staticmethod
    def dependency_version() -> str | None:
        try:
            return importlib.metadata.version("pyspacemouse")
        except importlib.metadata.PackageNotFoundError:
            return None

    @classmethod
    def probe(cls, config: SpaceMouseTestConfig) -> dict[str, Any]:
        """Detect the configured device without opening or claiming it."""
        try:
            module = importlib.import_module("pyspacemouse")
        except ModuleNotFoundError:
            return {
                "connected": False,
                "dependency_version": None,
                "supported_connected": [],
                "matching_hid": [],
                "matching_hid_nodes": [],
                "error": "pyspacemouse is not installed",
            }
        try:
            connected = list(dict.fromkeys(module.get_connected_devices()))
            matching = {
                (product, manufacturer, int(vendor_id), int(product_id))
                for product, manufacturer, vendor_id, product_id in module.get_all_hid_devices()
                if int(vendor_id) == config.expected_vendor_id
                and int(product_id) == config.expected_product_id
            }
            matching_hid = [
                {
                    "product_name": product,
                    "manufacturer": manufacturer,
                    "vendor_id": vendor_id,
                    "product_id": product_id,
                }
                for product, manufacturer, vendor_id, product_id in sorted(matching)
            ]
            nodes = cls(config)._matching_hid_nodes()
            return {
                "connected": bool(matching_hid),
                "dependency_version": cls.dependency_version(),
                "supported_connected": connected,
                "matching_hid": matching_hid,
                "matching_hid_nodes": nodes,
                "error": None,
            }
        except Exception as exc:
            return {
                "connected": False,
                "dependency_version": cls.dependency_version(),
                "supported_connected": [],
                "matching_hid": [],
                "matching_hid_nodes": [],
                "error": f"{type(exc).__name__}: {exc}",
            }

    def _open_device(self) -> Any:
        if self._device_factory is not None:
            return self._device_factory()
        try:
            module = importlib.import_module("pyspacemouse")
        except ModuleNotFoundError as exc:
            raise RuntimeError(
                "pyspacemouse is not installed. Install requirements-spacemouse.txt "
                "and the system package libhidapi-dev."
            ) from exc
        try:
            connected = module.get_connected_devices()
            matches = {
                (product, manufacturer, int(vendor_id), int(product_id))
                for product, manufacturer, vendor_id, product_id in module.get_all_hid_devices()
                if int(vendor_id) == self.config.expected_vendor_id
                and int(product_id) == self.config.expected_product_id
            }
            matching_hid = [
                {
                    "product_name": product,
                    "manufacturer": manufacturer,
                    "vendor_id": vendor_id,
                    "product_id": product_id,
                }
                for product, manufacturer, vendor_id, product_id in sorted(matches)
            ]
        except Exception:
            connected = []
            matching_hid = []
        self._device_details = {
            "open_success": False,
            "supported_connected": list(dict.fromkeys(connected)),
            "matching_hid": matching_hid,
            "matching_hid_nodes": self._matching_hid_nodes(),
        }
        try:
            if self.config.device_path is not None:
                device = module.open_by_path(
                    self.config.device_path,
                    nonblocking=True,
                )
            else:
                device = module.open(
                    device=self.config.device_name,
                    device_index=self.config.device_index,
                    nonblocking=True,
                )
        except Exception as exc:
            inaccessible = [
                node
                for node in self._device_details["matching_hid_nodes"]
                if not (
                    node.get("permissions", {}).get("readable")
                    and node.get("permissions", {}).get("writable")
                )
            ]
            permission_hint = (
                f", inaccessible_hidraw={inaccessible!r}" if inaccessible else ""
            )
            raise RuntimeError(
                "Unable to open SpaceMouse. Check the USB cable, the scoped hidraw udev "
                "rule, and whether another driver owns the device. "
                f"supported_connected={connected!r}, matching_hid={matching_hid!r}"
                f"{permission_hint}, cause={exc}"
            ) from exc
        if device is None:
            raise RuntimeError("PySpaceMouse did not find a supported SpaceMouse device")
        return device

    def start(self) -> None:
        if self._thread is not None:
            raise RuntimeError("SpaceMouseInput has already been started")
        device = self._open_device()
        info = getattr(device, "info", None)
        vendor_id = getattr(info, "vendor_id", None)
        product_id = getattr(info, "product_id", None)
        if vendor_id is not None and product_id is not None and (
            int(vendor_id) != self.config.expected_vendor_id
            or int(product_id) != self.config.expected_product_id
        ):
            try:
                device.close()
            finally:
                raise RuntimeError(
                    "Unexpected SpaceMouse USB identity: "
                    f"found {int(vendor_id):04x}:{int(product_id):04x}, expected "
                    f"{self.config.expected_vendor_id:04x}:{self.config.expected_product_id:04x}"
                )
        self._device = device
        self._device_details.update({
            "name": str(getattr(device, "name", self.config.device_name)),
            "product_name": str(getattr(device, "product_name", "")),
            "vendor_name": str(getattr(device, "vendor_name", "")),
            "vendor_id": None if vendor_id is None else int(vendor_id),
            "product_id": None if product_id is None else int(product_id),
            "device_path": self.config.device_path or self._opened_device_path(device),
            "open_success": True,
        })
        self._connected = True
        self._stop_event.clear()
        self._thread = threading.Thread(
            target=self._reader_loop,
            daemon=True,
            name="spacemouse-hid-reader",
        )
        self._thread.start()

    @staticmethod
    def _opened_device_path(device: Any) -> str | None:
        """Return the selected hidraw path without exposing the serial number."""
        hid_device = getattr(device, "_device", None)
        path = getattr(hid_device, "path", None)
        return None if path is None else str(path)

    @staticmethod
    def path_permissions(path: str | None) -> dict[str, Any] | None:
        """Return non-sensitive access diagnostics for one selected hidraw node."""
        if path is None:
            return None
        candidate = Path(path)
        if not candidate.exists():
            return {"exists": False, "readable": False, "writable": False}
        mode = candidate.stat().st_mode & 0o777
        return {
            "exists": True,
            "mode": f"{mode:04o}",
            "readable": os.access(candidate, os.R_OK),
            "writable": os.access(candidate, os.W_OK),
        }

    def _matching_hid_nodes(self) -> list[dict[str, Any]]:
        """Discover matching node paths and permissions without reading reports."""
        try:
            easyhid = importlib.import_module("easyhid")
            devices = easyhid.Enumeration().find(
                vid=self.config.expected_vendor_id,
                pid=self.config.expected_product_id,
            )
        except Exception:
            return []
        nodes: dict[str, dict[str, Any]] = {}
        for device in devices:
            path = str(getattr(device, "path", ""))
            if path and path not in nodes:
                nodes[path] = {
                    "path": path,
                    "permissions": self.path_permissions(path),
                }
        return list(nodes.values())

    def stop(self) -> None:
        self._stop_event.set()
        if self._thread is not None:
            self._thread.join(timeout=2.0)
        device, self._device = self._device, None
        if device is not None:
            try:
                device.close()
            except Exception:
                pass
        with self._lock:
            self._connected = False
            self._command_axes.fill(0.0)
            self.transform.reset_filter()

    def _reader_loop(self) -> None:
        assert self._device is not None
        poll_seconds = self.config.poll_interval_ms / 1000.0
        while not self._stop_event.is_set():
            try:
                state = self._device.read()
                device_timestamp = float(getattr(state, "t", -1.0))
                if device_timestamp >= 0.0 and device_timestamp != self._last_device_timestamp:
                    raw_axes = tuple(float(getattr(state, name)) for name in AXIS_NAMES)
                    buttons = tuple(int(value) for value in getattr(state, "buttons", ()))
                    self._accept_event(device_timestamp, raw_axes, buttons)
                self._sleep(poll_seconds)
            except Exception as exc:
                with self._lock:
                    self._connected = False
                    self._error = f"{type(exc).__name__}: {exc}"
                    self._command_axes.fill(0.0)
                    self.transform.reset_filter()
                self._stop_event.set()
                break

    def _accept_event(
        self,
        device_timestamp: float,
        raw_axes: tuple[float, ...],
        buttons: tuple[int, ...],
    ) -> None:
        now = self._clock()
        with self._lock:
            corrected, command = self.transform.apply(raw_axes)
            previous = self._previous_buttons
            left_pressed = len(buttons) > 0 and buttons[0] and (len(previous) < 1 or not previous[0])
            right_pressed = len(buttons) > 1 and buttons[1] and (len(previous) < 2 or not previous[1])
            if right_pressed:
                self._gripper = 1.0
            if left_pressed:
                self._gripper = -1.0
            self._last_device_timestamp = device_timestamp
            self._event_monotonic = now
            self._raw_axes = np.asarray(raw_axes, dtype=np.float64)
            self._corrected_axes = corrected
            self._command_axes = command
            self._buttons = buttons
            self._previous_buttons = buttons
            self._sequence += 1
            self._event_times.append(now)

    def calibrate_neutral(self) -> dict[str, Any]:
        """Measure bias while untouched, including devices silent at neutral."""
        deadline = self._clock() + self.config.neutral_calibration_seconds
        initial_snapshot = self.latest_snapshot()
        last_sequence = initial_snapshot.sequence
        samples: list[tuple[float, ...]] = []
        while self._clock() < deadline:
            snapshot = self.latest_snapshot()
            if snapshot.error is not None:
                raise RuntimeError(f"SpaceMouse reader failed during calibration: {snapshot.error}")
            if snapshot.sequence != last_sequence:
                samples.append(snapshot.raw_axes)
                last_sequence = snapshot.sequence
            self._sleep(min(0.005, self.config.poll_interval_ms / 1000.0))
        observed_reports = len(samples)
        # SpaceMouse firmware is allowed to be silent while perfectly neutral.
        # A successful HID open already proves access; the subsequent coverage
        # test will still fail if the selected interface never produces reports.
        if not samples:
            samples = [initial_snapshot.raw_axes, initial_snapshot.raw_axes]
        elif len(samples) == 1:
            samples.append(samples[0])
        with self._lock:
            result = self.transform.calibrate(samples)
            corrected, command = self.transform.apply(self._raw_axes)
            self._corrected_axes = corrected
            self._command_axes = command
        result["hid_reports_observed"] = observed_reports
        result["used_initialized_neutral"] = observed_reports == 0
        return result

    def calibrate_until_stable(
        self,
        *,
        timeout_seconds: float | None = None,
        progress: Callable[[float, str, int], None] | None = None,
        cancelled: Callable[[], bool] | None = None,
    ) -> dict[str, Any]:
        """Wait for one continuous neutral window instead of failing on motion."""
        timeout = (
            self.config.neutral_calibration_timeout_seconds
            if timeout_seconds is None
            else float(timeout_seconds)
        )
        stable_required = self.config.neutral_calibration_seconds
        started = self._clock()
        stable_started = started
        initial = self.latest_snapshot()
        last_sequence = initial.sequence
        samples: list[tuple[float, ...]] = []
        movement_resets = 0
        reports_observed = 0
        while self._clock() - started < timeout:
            if cancelled is not None and cancelled():
                raise RuntimeError("SpaceMouse calibration cancelled")
            snapshot = self.latest_snapshot()
            if snapshot.error is not None or not snapshot.connected:
                raise RuntimeError(
                    "SpaceMouse disconnected during calibration"
                    if snapshot.error is None
                    else f"SpaceMouse reader failed during calibration: {snapshot.error}"
                )
            now = self._clock()
            moved = max(abs(value) for value in snapshot.raw_axes) > self.config.neutral_max_abs
            if moved:
                stable_started = now
                samples.clear()
                movement_resets += 1
                message = "检测到帽盖移动，请松开并保持静止"
                stable_fraction = 0.0
            else:
                if snapshot.sequence != last_sequence:
                    samples.append(snapshot.raw_axes)
                    reports_observed += 1
                stable_fraction = min(1.0, (now - stable_started) / stable_required)
                message = "保持帽盖静止，正在校准"
            last_sequence = snapshot.sequence
            if progress is not None:
                progress(stable_fraction, message, movement_resets)
            if not moved and now - stable_started >= stable_required:
                if not samples:
                    samples = [snapshot.raw_axes, snapshot.raw_axes]
                elif len(samples) == 1:
                    samples.append(samples[0])
                with self._lock:
                    result = self.transform.calibrate(samples)
                    corrected, command = self.transform.apply(self._raw_axes)
                    self._corrected_axes = corrected
                    self._command_axes = command
                result.update({
                    "hid_reports_observed": reports_observed,
                    "used_initialized_neutral": reports_observed == 0,
                    "movement_resets": movement_resets,
                    "elapsed_seconds": now - started,
                })
                return result
            self._sleep(min(0.01, self.config.poll_interval_ms / 1000.0))
        raise TimeoutError(
            f"SpaceMouse did not remain neutral for {stable_required:.1f}s "
            f"within the {timeout:.1f}s calibration timeout"
        )

    def latest_snapshot(self) -> SpaceMouseSnapshot:
        now = self._clock()
        with self._lock:
            age = None if self._event_monotonic is None else now - self._event_monotonic
            stale = age is None or age > self.config.stale_timeout_ms / 1000.0
            command = self._command_axes.copy()
            if stale or not self._connected:
                command.fill(0.0)
                self.transform.reset_filter()
            action = np.concatenate([command, np.asarray([self._gripper])])
            return SpaceMouseSnapshot(
                sequence=self._sequence,
                captured_monotonic=now,
                event_monotonic=self._event_monotonic,
                device_timestamp=self._last_device_timestamp,
                raw_axes=tuple(float(value) for value in self._raw_axes),
                corrected_axes=tuple(float(value) for value in self._corrected_axes),
                command_axes=tuple(float(value) for value in command),
                action=tuple(float(value) for value in action),
                buttons=self._buttons,
                connected=self._connected,
                stale=stale,
                error=self._error,
            )

    def latest_action(self) -> np.ndarray:
        return np.asarray(self.latest_snapshot().action, dtype=np.float32)

    def set_gains(self, translation_gain: float, rotation_gain: float) -> None:
        """Apply UI gain changes atomically; the next HID report uses them."""
        with self._lock:
            self.transform.set_gains(translation_gain, rotation_gain)

    def reset_for_arm(self, gripper: float = -1.0) -> None:
        """Discard idle input so a takeover always starts from zero motion."""
        if gripper not in {-1.0, 1.0}:
            raise ValueError("gripper must be -1.0 or +1.0")
        with self._lock:
            self._command_axes.fill(0.0)
            self.transform.reset_filter()
            self._gripper = gripper
            self._previous_buttons = self._buttons

    def diagnostics(self) -> dict[str, Any]:
        with self._lock:
            event_times = np.asarray(self._event_times, dtype=np.float64)
            return {
                **self._device_details,
                "pyspacemouse_version": self.dependency_version(),
                "axis_convention": self.config.axis_convention,
                "axis_order": list(self.config.axis_order),
                "axis_signs": list(self.config.axis_signs),
                "connected_at_shutdown": self._connected,
                "event_count": int(len(event_times)),
                "event_times": event_times.copy(),
                "reader_error": self._error,
                "path_permissions": self.path_permissions(
                    self._device_details.get("device_path")
                ),
            }
