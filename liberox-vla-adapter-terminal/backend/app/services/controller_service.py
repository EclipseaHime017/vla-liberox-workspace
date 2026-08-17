"""Long-lived, explicitly calibrated SpaceMouse service for the Web UI."""

from __future__ import annotations

import threading
import time
import uuid
from dataclasses import dataclass
from typing import Any, Callable

from ..devices.spacemouse import SpaceMouseInput, SpaceMouseSnapshot, SpaceMouseTestConfig


CONTROLLER_STATES = frozenset({
    "DISCONNECTED",
    "UNCALIBRATED",
    "CALIBRATING",
    "READY",
    "ARMED",
    "ERROR",
})


def latency_level(
    latency_ms: float | None,
    *,
    connected: bool,
    stale: bool,
    error: str | None,
) -> str:
    if error is not None or not connected or stale or latency_ms is None or latency_ms >= 250.0:
        return "red"
    if latency_ms >= 50.0:
        return "yellow"
    return "green"


@dataclass(frozen=True)
class CalibrationSnapshot:
    id: str
    completed_at: float
    result: dict[str, Any]


class SpaceMouseControllerService:
    """Own one HID handle across UI sessions and gate all robot commands."""

    def __init__(
        self,
        config: SpaceMouseTestConfig,
        *,
        input_factory: Callable[[SpaceMouseTestConfig], SpaceMouseInput] = SpaceMouseInput,
        probe: Callable[[SpaceMouseTestConfig], dict[str, Any]] = SpaceMouseInput.probe,
        monitor_interval_seconds: float = 0.5,
        start_monitor: bool = True,
    ) -> None:
        self.config = config
        self._input_factory = input_factory
        self._probe = probe
        self._monitor_interval = monitor_interval_seconds
        self._lock = threading.RLock()
        self._stop_event = threading.Event()
        self._calibration_cancel = threading.Event()
        self._input: SpaceMouseInput | None = None
        self._calibration_thread: threading.Thread | None = None
        self._monitor_thread: threading.Thread | None = None
        self._state = "DISCONNECTED"
        self._connected = False
        self._error: str | None = None
        self._message = "正在检测控制器"
        self._progress = 0.0
        self._movement_resets = 0
        self._armed_session_id: str | None = None
        self._calibration: CalibrationSnapshot | None = None
        self._probe_details: dict[str, Any] = {}
        self._poll_once()
        if start_monitor:
            self._monitor_thread = threading.Thread(
                target=self._monitor_loop,
                daemon=True,
                name="spacemouse-controller-monitor",
            )
            self._monitor_thread.start()

    def _monitor_loop(self) -> None:
        while not self._stop_event.wait(self._monitor_interval):
            self._poll_once()

    def _poll_once(self) -> None:
        with self._lock:
            controller = self._input
            state = self._state
        if controller is not None:
            snapshot = controller.latest_snapshot()
            if snapshot.error is not None or not snapshot.connected:
                self._invalidate_connection(snapshot.error or "SpaceMouse disconnected")
            return
        if state == "CALIBRATING":
            return
        details = self._probe(self.config)
        present = bool(details.get("connected"))
        with self._lock:
            was_connected = self._connected
            self._probe_details = details
            self._connected = present
            if not present:
                self._state = "DISCONNECTED"
                self._calibration = None
                self._armed_session_id = None
                self._message = "控制器未连接"
                if details.get("error"):
                    self._error = str(details["error"])
            elif self._state == "DISCONNECTED" or not was_connected:
                self._state = "UNCALIBRATED"
                self._error = None
                self._message = "控制器已连接，等待校准"

    def _invalidate_connection(self, error: str) -> None:
        with self._lock:
            controller, self._input = self._input, None
            self._state = "DISCONNECTED"
            self._connected = False
            self._calibration = None
            self._armed_session_id = None
            self._progress = 0.0
            self._error = error
            self._message = "控制器连接已断开"
        if controller is not None:
            controller.stop()

    def start_calibration(self) -> dict[str, Any]:
        self._poll_once()
        with self._lock:
            if self._state == "CALIBRATING":
                raise RuntimeError("SpaceMouse calibration is already running")
            if self._state == "ARMED":
                raise RuntimeError("SpaceMouse is armed by an active session")
            if not self._connected:
                raise RuntimeError("SpaceMouse is not connected")
            previous, self._input = self._input, None
            self._calibration = None
            self._calibration_cancel.clear()
            self._state = "CALIBRATING"
            self._progress = 0.0
            self._movement_resets = 0
            self._error = None
            self._message = "请松开帽盖并保持静止"
        if previous is not None:
            previous.stop()
        thread = threading.Thread(
            target=self._calibration_worker,
            daemon=True,
            name="spacemouse-calibration",
        )
        with self._lock:
            self._calibration_thread = thread
        thread.start()
        return self.status()

    def _calibration_worker(self) -> None:
        controller: SpaceMouseInput | None = None
        try:
            controller = self._input_factory(self.config)
            controller.start()
            with self._lock:
                self._input = controller
                self._connected = True

            def progress(value: float, message: str, movement_resets: int) -> None:
                with self._lock:
                    self._progress = value
                    self._message = message
                    self._movement_resets = movement_resets

            result = controller.calibrate_until_stable(
                timeout_seconds=self.config.neutral_calibration_timeout_seconds,
                progress=progress,
                cancelled=self._calibration_cancel.is_set,
            )
            controller.reset_for_arm()
            calibration = CalibrationSnapshot(
                id=uuid.uuid4().hex[:12],
                completed_at=time.time(),
                result=result,
            )
            with self._lock:
                if self._calibration_cancel.is_set():
                    raise RuntimeError("SpaceMouse calibration cancelled")
                self._calibration = calibration
                self._state = "READY"
                self._progress = 1.0
                self._message = "控制器已校准"
                self._error = None
        except Exception as exc:
            if controller is not None:
                controller.stop()
            with self._lock:
                if self._input is controller:
                    self._input = None
                self._state = "ERROR" if self._connected else "DISCONNECTED"
                self._calibration = None
                self._progress = 0.0
                self._error = f"{type(exc).__name__}: {exc}"
                self._message = "校准失败，请保持帽盖静止后重试"
        finally:
            with self._lock:
                self._calibration_thread = None

    def arm(
        self,
        session_id: str,
        translation_gain: float,
        rotation_gain: float,
    ) -> None:
        with self._lock:
            if self._state != "READY" or self._input is None or self._calibration is None:
                raise RuntimeError("SpaceMouse must be calibrated before takeover")
            controller = self._input
            controller.set_gains(translation_gain, rotation_gain)
            controller.reset_for_arm(gripper=-1.0)
            self._armed_session_id = session_id
            self._state = "ARMED"
            self._message = "SpaceMouse 接管中"

    def disarm(self, session_id: str) -> None:
        with self._lock:
            if self._armed_session_id != session_id:
                return
            controller = self._input
            self._armed_session_id = None
            if controller is not None:
                controller.reset_for_arm(gripper=-1.0)
                snapshot = controller.latest_snapshot()
                if snapshot.connected and snapshot.error is None and self._calibration is not None:
                    self._state = "READY"
                    self._connected = True
                    self._message = "控制器已校准"
                    return
            self._state = "DISCONNECTED"
            self._connected = False
            self._calibration = None
            self._message = "控制器未连接"

    def set_gains(self, session_id: str, translation_gain: float, rotation_gain: float) -> None:
        with self._lock:
            if self._state != "ARMED" or self._armed_session_id != session_id:
                return
            assert self._input is not None
            self._input.set_gains(translation_gain, rotation_gain)

    def snapshot(self, session_id: str) -> SpaceMouseSnapshot:
        with self._lock:
            if self._state != "ARMED" or self._armed_session_id != session_id:
                raise RuntimeError("SpaceMouse is not armed for this session")
            assert self._input is not None
            controller = self._input
        snapshot = controller.latest_snapshot()
        if snapshot.error is not None or not snapshot.connected:
            self._invalidate_connection(snapshot.error or "SpaceMouse disconnected")
            raise RuntimeError(snapshot.error or "SpaceMouse disconnected")
        return snapshot

    def calibration_snapshot(self) -> dict[str, Any] | None:
        with self._lock:
            if self._calibration is None:
                return None
            return {
                "id": self._calibration.id,
                "completed_at": self._calibration.completed_at,
                "result": self._calibration.result,
            }

    def diagnostics(self) -> dict[str, Any] | None:
        with self._lock:
            controller = self._input
        return None if controller is None else controller.diagnostics()

    def status(self) -> dict[str, Any]:
        self._poll_once()
        with self._lock:
            controller = self._input
            state = self._state
            connected = self._connected
            armed_session_id = self._armed_session_id
            calibration = self._calibration
            result = {
                "state": state,
                "connected": connected,
                "calibrated": calibration is not None,
                "calibration_id": None if calibration is None else calibration.id,
                "calibration_progress": self._progress,
                "movement_resets": self._movement_resets,
                "message": self._message,
                "error": self._error,
                "armed_session_id": armed_session_id,
                "device_name": self.config.device_name,
                "vendor_id": self.config.expected_vendor_id,
                "product_id": self.config.expected_product_id,
                "stale_timeout_ms": self.config.stale_timeout_ms,
                "green_latency_max_ms": 50,
                "probe": self._probe_details,
            }
        latency_ms = None
        stale = False
        snapshot_error = None
        if state == "ARMED" and controller is not None:
            snapshot = controller.latest_snapshot()
            latency_ms = (
                None
                if snapshot.sample_age_seconds is None
                else snapshot.sample_age_seconds * 1000.0
            )
            stale = snapshot.stale
            snapshot_error = snapshot.error
        result.update({
            "latency_ms": latency_ms,
            "latency_level": latency_level(
                latency_ms,
                connected=connected,
                stale=stale,
                error=snapshot_error,
            ) if state == "ARMED" else None,
            "stale": stale,
        })
        return result

    def close(self) -> None:
        self._stop_event.set()
        self._calibration_cancel.set()
        with self._lock:
            calibration_thread = self._calibration_thread
            monitor_thread = self._monitor_thread
            controller, self._input = self._input, None
        if calibration_thread is not None:
            calibration_thread.join(timeout=self.config.neutral_calibration_timeout_seconds + 2.0)
        if monitor_thread is not None:
            monitor_thread.join(timeout=2.0)
        if controller is not None:
            controller.stop()
