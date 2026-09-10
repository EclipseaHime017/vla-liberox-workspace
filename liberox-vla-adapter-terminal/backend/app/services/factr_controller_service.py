"""GUI lifecycle for the shared official FACTR client. No calibration/control math."""
import copy
import logging
from pathlib import Path
import threading
import uuid

import numpy as np

from ..devices.factr import probe_factr
from ..devices.factr_client import FactrClient
from ..devices.factr_calibration import save_profile
from .controller_service import latency_level

LOGGER = logging.getLogger(__name__)


class FactrControllerService:
    controller_id = "factr"

    def __init__(self, config, *, input_factory=FactrClient, probe=probe_factr,
                 monitor_interval_seconds=.05, start_monitor=True):
        self.config = config
        self._factory, self._probe = input_factory, probe
        self._lock = threading.RLock()
        self._operation = threading.RLock()
        self._stop = threading.Event()
        self._input = None
        self._calibration_thread = None
        self._gravity_busy = False
        self._shutdown_error = None
        self._state, self._error = "DISCONNECTED", None
        self._message = "FACTR 未连接"
        self._connected = False
        self._calibration = None
        self._armed_session_id = None
        self._alignment_session_id = None
        self._translation_gain, self._rotation_gain = config.translation_gain, config.rotation_gain
        self._monitor = None
        self._poll_once()
        if start_monitor:
            self._monitor = threading.Thread(target=self._monitor_loop, args=(monitor_interval_seconds,),
                                             name="factr-monitor", daemon=True)
            self._monitor.start()

    def _monitor_loop(self, interval):
        while not self._stop.wait(interval):
            try:
                self._poll_once()
            except Exception as exc:
                self.emergency_stop(str(exc))

    def _poll_once(self):
        with self._lock:
            client, state = self._input, self._state
            if state in {"CALIBRATING", "ERROR"} or self._gravity_busy or self._stop.is_set():
                return
        if client is not None:
            snap = client.latest_snapshot()
            if snap.error or not snap.connected:
                self.emergency_stop(snap.error or "FACTR runtime disconnected; recalibrate before reconnecting")
            elif snap.stale and state in {"ARMED", "ALIGNING"}:
                self.emergency_stop("FACTR takeover sample stale; recalibrate before reconnecting")
            # Standby/postprocessing does not use samples to command simulation.
            # A late GUI snapshot alone is not a physical-loop failure: the
            # isolated runtime still enforces bus health, watchdog and heartbeat.
        else:
            probe = self._probe(self.config)
            with self._lock:
                if self._input is None and self._state == state:
                    self._connected = bool(probe.get("connected"))
                    self._state = "UNCALIBRATED" if self._connected else "DISCONNECTED"
                    self._error = probe.get("error")
                    self._message = "FACTR 已连接·待校准" if self._connected else self._error or "FACTR 未连接"

    def start_calibration(self):
        with self._lock:
            if self._stop.is_set() or self._armed_session_id or self._alignment_session_id or self._gravity_busy or self._calibration_thread is not None:
                raise RuntimeError("Cannot calibrate while armed, closing or already calibrating")
            if self._input is not None and self._input.status().get("gravity_enabled"):
                raise RuntimeError("Support the arm and disable compensation before calibration")
            self._state, self._error = "CALIBRATING", None
            self._message = "正在调用官方整臂校准；保持参考构型并松开触发器"
            self._calibration_thread = threading.Thread(target=self._calibrate, name="factr-calibration", daemon=True)
            self._calibration_thread.start()
        return self.status()

    def _calibrate(self):
        try:
            with self._operation:
                if self._stop.is_set():
                    return
                if self._input is None:
                    client = self._factory(self.config)
                    with self._lock:
                        self._input = client
                else:
                    client = self._input
                profile = client.calibrate()
                if self._stop.is_set():
                    client.close()
                    return
                save_profile(Path(self.config.runtime["calibration_file"]), profile)
                with self._lock:
                    self._calibration = {"id": uuid.uuid4().hex[:12], "result": profile.as_dict(),
                                         "config": self.config.metadata()}
                    self._state, self._connected = "READY", True
                    self._shutdown_error = None
                    self._message = "FACTR 已校准；可手动开启重力补偿"
        except Exception as exc:
            self.emergency_stop(f"FACTR calibration failed: {exc}")
        finally:
            with self._lock:
                self._calibration_thread = None

    def set_gravity(self, enabled: bool):
        if type(enabled) is not bool:
            raise ValueError("enabled must be boolean")
        with self._operation:
            with self._lock:
                client = self._input
                if self._stop.is_set() or self._state == "CALIBRATING":
                    raise RuntimeError("Controller is closing or calibrating")
                if client is None or (enabled and self._state not in {"READY", "ARMED"}):
                    raise RuntimeError("Calibrate FACTR before enabling compensation")
                self._gravity_busy = True
            try:
                if enabled:
                    client.enable_gravity()
                else:
                    client.disable_gravity()
            finally:
                with self._lock:
                    self._gravity_busy = False
        return self.status()

    @staticmethod
    def _validate_gains(translation_gain, rotation_gain):
        if any(isinstance(g, bool) or not np.isfinite(g) or not .05 <= g <= 1 for g in (translation_gain, rotation_gain)):
            raise ValueError("FACTR gains must be in [.05,1]")

    def start_alignment(self, session_id, joints):
        with self._operation:
            with self._lock:
                if self._state != "READY" or self._input is None or self._gravity_busy:
                    raise RuntimeError("FACTR is not ready for alignment")
                self._input.start_alignment(session_id, joints)
                self._alignment_session_id = session_id
                self._state, self._message = "ALIGNING", "控制器正在缓慢对齐仿真关节姿态"

    def finish_alignment(self, session_id):
        with self._operation:
            with self._lock:
                if self._alignment_session_id == session_id:
                    if self._input is not None:
                        self._input.finish_alignment(session_id)
                    self._alignment_session_id = None
                    if self._state != "ERROR":
                        self._state, self._message = "READY", "控制器已校准"

    def arm(self, session_id, translation_gain, rotation_gain, *, gripper=-1.):
        self._validate_gains(translation_gain, rotation_gain)
        with self._lock:
            if self._state != "READY" or self._input is None or self._gravity_busy:
                raise RuntimeError("Calibrate FACTR before takeover")
            sample = self._input.arm(session_id, translation_gain, rotation_gain, gripper=gripper)
            self._armed_session_id = session_id
            self._translation_gain, self._rotation_gain = translation_gain, rotation_gain
            self._state, self._message = "ARMED", "FACTR 接管中"
            return sample

    def disarm(self, session_id):
        with self._lock:
            if self._armed_session_id == session_id:
                if self._input is not None:
                    self._input.disarm(session_id)
                self._armed_session_id = None
                self._state, self._message = "READY", "FACTR 已校准·未接管"
        # Normal completion/rewind/countdown does not switch physical support off.

    def snapshot(self, session_id):
        with self._lock:
            if self._state != "ARMED" or self._armed_session_id != session_id or self._input is None:
                raise RuntimeError(self._error or "FACTR is not armed")
            client = self._input
        return client.snapshot(session_id)

    def set_gains(self, session_id, translation_gain, rotation_gain):
        self._validate_gains(translation_gain, rotation_gain)
        with self._lock:
            if self._armed_session_id is not None and session_id != self._armed_session_id:
                raise RuntimeError("Controller belongs to another session")
            self._translation_gain, self._rotation_gain = translation_gain, rotation_gain

    def emergency_stop(self, reason):
        # Also used before expensive trajectory postprocessing and backend close.
        with self._lock:
            if self._state == "ERROR" and self._input is None:
                return  # Preserve the first failure through subsequent cleanup.
            self._state, self._error, self._message = "ERROR", reason, "FACTR 已停止，请检查设备并重新校准"
            self._armed_session_id = self._alignment_session_id = self._calibration = None
            self._connected = False
        LOGGER.error("FACTR output stop requested: %s", reason)
        with self._operation:
            with self._lock:
                client, self._input = self._input, None
            if client is not None:
                try:
                    client.close()
                except Exception as exc:
                    with self._lock:
                        self._shutdown_error = str(exc)
                        self._error = f"{reason}; shutdown NOT verified: {exc}"
                    LOGGER.error("FACTR shutdown NOT verified; support leader and inspect physical power: %s", exc)

    def status(self):
        # Never perform disk hashing, serial reads or process joins in HTTP polls.
        with self._lock:
            client = self._input
            raw = {} if client is None else client.status()
            snap = None if client is None else client.latest_snapshot()
            age = None if snap is None or snap.sample_age_seconds is None else snap.sample_age_seconds*1000
            connected = self._connected and (snap is None or snap.connected)
            stale = snap is None or snap.stale
            state = "CALIBRATING" if self._calibration_thread is not None else self._state
            return {"controller_id": "factr", "state": state, "connected": connected,
                "calibrated": self._calibration is not None, "calibration_progress": 1. if self._calibration else 0.,
                "movement_resets": 0, "message": self._message, "error": self._error,
                "armed_session_id": self._armed_session_id, "stale": stale, "latency_ms": age,
                "latency_level": latency_level(age, connected=connected, stale=stale, error=self._error),
                "translation_gain": self._translation_gain, "rotation_gain": self._rotation_gain,
                "gravity_supported": True, "gravity_enabled": bool(raw.get("gravity_enabled")),
                "gravity_state": "unknown" if self._shutdown_error else "on" if raw.get("gravity_enabled") else "off",
                "cycle_ms": raw.get("cycle_ms"), "official_commit": raw.get("official_commit"),
                "serial_read": raw.get("serial_read"),
                "alignment": raw.get("alignment"),
                "reference_joint_positions": list(self.config.reference_joint_positions)}

    def calibration_snapshot(self):
        with self._lock:
            return copy.deepcopy(self._calibration)

    def diagnostics(self):
        with self._lock:
            return {"controller_id": "factr", "status": self.status(), "calibration": self.calibration_snapshot(),
                    "device": None if self._input is None else self._input.diagnostics()}

    def close(self):
        self._stop.set()
        self.emergency_stop("Application closing; disabling FACTR")
        thread = self._calibration_thread
        if thread is not None and thread is not threading.current_thread():
            thread.join(timeout=25)
        if self._monitor is not None and self._monitor is not threading.current_thread():
            self._monitor.join(timeout=2)
