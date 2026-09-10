"""IPC client for pinned official FACTR; no motor control math or serial access."""
from concurrent.futures import Future
from collections import deque
from dataclasses import replace
import json
import select
import socket
import subprocess
import sys
import threading
import time
import numpy as np
from .factr import FactrSnapshot
from .factr_calibration import gripper_fraction
from .factr_official import WORKSPACE, runtime_environment, verify_upstream
from .factr_runtime_config import parse_runtime_options


class FactrClient:
    def __init__(self, config, *, worker_command=None, startup_timeout=20.):
        self.config = config
        self.options = parse_runtime_options(config.runtime, WORKSPACE/"configs")
        verify_upstream(self.options.upstream_root)
        self._lock = threading.RLock()
        self._send_lock = threading.Lock()
        self._operation_lock = threading.Lock()
        self._stop, self._ready = threading.Event(), threading.Event()
        self._pending, self._remote = {}, {}
        self._next_id = 0
        self._error = self._profile = self._owner = None
        self._shutdown_verified = False
        self._shutdown_error = None
        self._stderr_tail = deque(maxlen=32)
        self._snapshot = FactrSnapshot()
        self._gripper, self._gripper_anchor = -1., None
        self._gripper_aligned = self._gripper_changed = self._closed = False
        self._channel, child = socket.socketpair(socket.AF_UNIX, socket.SOCK_SEQPACKET)
        command = worker_command or [self.options.runtime_python, str(WORKSPACE/
            "liberox-vla-adapter-terminal/scripts/factr_official_worker.py")]
        try:
            self._process = subprocess.Popen([*command, "--fd", str(child.fileno())],
                pass_fds=(child.fileno(),), env=runtime_environment(self.options.upstream_root),
                stderr=subprocess.PIPE)
        except BaseException:
            self._channel.close()
            raise
        finally:
            child.close()
        self._channel.settimeout(.2)
        self._thread = None
        self._stderr_thread = threading.Thread(target=self._read_stderr, name="factr-stderr", daemon=True)
        self._stderr_thread.start()
        try:
            self._send({"config": config.metadata()})
        except OSError as exc:
            self._error = self._disconnect_reason(exc)
            try:
                self.close()
            finally:
                raise RuntimeError(self._error) from exc
        self._thread = threading.Thread(target=self._receive, name="factr-official-ipc", daemon=True)
        self._thread.start()
        if not self._ready.wait(startup_timeout):
            self._error = "Official FACTR startup timed out; check runtime dependencies and stderr"
        if self._error:
            try:
                self.close()
            finally:
                raise RuntimeError(self._error)

    def _send(self, message):
        with self._send_lock:
            self._channel.send(json.dumps(message, allow_nan=False).encode())

    def _read_stderr(self):
        # Drain continuously so a traceback cannot block the child. Keep only
        # a bounded in-memory tail; no device/test log files are created.
        with self._process.stderr as stream:
            for raw in stream:
                line = raw.decode("utf-8", errors="replace").rstrip()
                with self._lock:
                    self._stderr_tail.append(line[-2048:])
                try:
                    sys.stderr.write(line + "\n")
                except (OSError, ValueError):
                    pass

    def _disconnect_reason(self, exc):
        self._stderr_thread.join(timeout=.5)
        with self._lock:
            lines = [line for line in self._stderr_tail if line.strip()]
        errors = [line for line in lines if "Error:" in line or "Exception:" in line or "ERROR" in line]
        if errors:
            return f"FACTR worker stopped: {errors[-1]}"
        return str(exc) or type(exc).__name__

    def _accept_status(self, state):
        with self._lock:
            self._remote = state
            sample = state.get("sample")
            if sample is None:
                return
            raw, q = tuple(sample["raw_joints"]), tuple(sample["joint_positions"])
            if len(raw) != 7 or len(q) != 7 or not np.isfinite((*raw, *q, sample["raw_gripper"], sample["sample_monotonic"])).all():
                raise ValueError("Invalid official observation packet")
            fraction = None if self._profile is None else gripper_fraction(sample["raw_gripper"], self._profile)
            self._update_gripper(fraction)
            self._snapshot = FactrSnapshot(sequence=sample["sequence"], captured_monotonic=time.monotonic(),
                sample_monotonic=sample["sample_monotonic"], raw_joints=raw, joint_positions=q,
                raw_gripper=sample["raw_gripper"], gripper_fraction=fraction,
                gripper_command=self._gripper, connected=True, stale=False)
            self._ready.set()

    def _receive(self):
        last_heartbeat = 0.
        try:
            while not self._stop.is_set():
                # Read queued fatal/OFF diagnostics before writing a heartbeat
                # to a worker which may already have closed its socket.
                if not select.select([self._channel], [], [], 0)[0] and time.monotonic()-last_heartbeat >= .1:
                    try:
                        self._send({"operation": "heartbeat"})
                    except OSError:
                        if not select.select([self._channel], [], [], .1)[0]:
                            raise
                    last_heartbeat = time.monotonic()
                if not select.select([self._channel], [], [], .05)[0]:
                    continue
                packet = self._channel.recv(65536)
                if not packet:
                    raise RuntimeError("Official FACTR process disconnected")
                message = json.loads(packet)
                if "shutdown" in message:
                    self._shutdown_verified = message["shutdown"].get("verified") is True
                    self._shutdown_error = message["shutdown"].get("error")
                    if self._shutdown_verified:
                        with self._lock:
                            self._remote = {**self._remote, "gravity_enabled": False}
                if "fatal" in message:
                    raise RuntimeError(message["fatal"])
                if "status" in message:
                    self._accept_status(message["status"])
                if "reply" in message:
                    with self._lock:
                        future = self._pending.pop(message["reply"], None)
                    if future is not None and not future.done():
                        if "error" in message:
                            future.set_exception(RuntimeError(message["error"]))
                        else:
                            future.set_result(message.get("result"))
        except Exception as exc:
            if not self._closed:
                self._error = self._disconnect_reason(exc) if isinstance(exc, OSError) or str(exc) == "Official FACTR process disconnected" else str(exc) or type(exc).__name__
        finally:
            self._stop.set()
            self._ready.set()
            with self._lock:
                self._owner = None
                for future in self._pending.values():
                    if not future.done():
                        future.set_exception(RuntimeError(self._error or "FACTR closed"))
                self._pending.clear()

    def _call(self, operation, value=None):
        with self._operation_lock:
            if self._stop.is_set():
                raise RuntimeError(self._error or "FACTR process stopped")
            with self._lock:
                self._next_id += 1
                request_id = self._next_id
                future = Future()
                self._pending[request_id] = future
            try:
                self._send({"operation": operation, "value": value, "id": request_id})
            except OSError as exc:
                # The receiver may still be consuming the final fatal packet.
                try:
                    return future.result(timeout=1)
                except TimeoutError:
                    self._error = self._disconnect_reason(exc)
                    self.close()
                    raise RuntimeError(self._error) from exc
            try:
                return future.result(timeout=10)
            except TimeoutError:
                self._error = f"Official FACTR {operation} timed out"
                self.close()  # Never leave a delayed enable pending.
                raise

    @property
    def fingerprint(self):
        return self._remote.get("fingerprint", {})

    def calibrate(self):
        if self._owner is not None:
            raise RuntimeError("Stop simulation before calibration")
        from .factr_calibration import FactrCalibrationProfile
        profile = FactrCalibrationProfile(**self._call("calibrate"))
        with self._lock:
            self._profile = profile
            self._accept_status(self._remote)
        return profile

    def set_profile(self, profile):
        if self._owner is not None:
            raise RuntimeError("Stop simulation before calibration")
        self._call("profile", profile.as_dict())
        with self._lock:
            self._profile = profile
            self._accept_status(self._remote)

    def enable_gravity(self):
        return self._call("enable")

    def disable_gravity(self):
        return self._call("disable")

    def start_alignment(self, session_id, joints):
        if self._owner is not None:
            raise RuntimeError("Cannot align an armed controller")
        return self._call("align", np.asarray(joints, dtype=float).tolist())

    def finish_alignment(self, session_id):
        return self._call("cancel_align")

    def latest_snapshot(self):
        with self._lock:
            now = time.monotonic()
            stale = self._snapshot.sample_monotonic is None or now-self._snapshot.sample_monotonic >= self.config.stale_timeout_ms/1000
            if stale or self._error:
                self._owner = None
            return replace(self._snapshot, captured_monotonic=now, stale=stale,
                connected=not self._stop.is_set() and self._snapshot.connected, error=self._error)

    def arm(self, session_id, translation_gain, rotation_gain, *, gripper=-1.):
        with self._lock:
            sample = self.latest_snapshot()
            if self._profile is None or not self._remote.get("calibrated") or sample.stale or sample.error or not sample.connected:
                raise RuntimeError("Fresh official calibration required")
            if not session_id or self._owner is not None or gripper not in (-1., 1.):
                raise ValueError("Invalid session/gripper or already armed")
            if not all(np.isfinite(g) and .05 <= g <= 1 for g in (translation_gain, rotation_gain)):
                raise ValueError("Follow gains must be in [.05,1]")
            q = np.asarray(sample.joint_positions)
            if np.any(q < self.config.joint_limits_min) or np.any(q > self.config.joint_limits_max):
                raise ValueError("Current leader pose is outside physical joint limits")
            self._owner, self._gripper = session_id, gripper
            self._gripper_anchor = sample.gripper_fraction
            self._gripper_aligned = self._at_held_endpoint(sample.gripper_fraction)
            self._gripper_changed = False
            return replace(sample, gripper_command=gripper)

    def _at_held_endpoint(self, fraction):
        return fraction <= self.config.gripper_open_threshold if self._gripper < 0 else fraction >= self.config.gripper_close_threshold

    def _update_gripper(self, fraction):
        if self._owner is None or fraction is None:
            return
        if not self._gripper_aligned:
            if self._at_held_endpoint(fraction):
                self._gripper_aligned, self._gripper_anchor = True, fraction
        elif abs(fraction-self._gripper_anchor) >= self.config.gripper_takeover_delta:
            self._gripper_changed = True
        if self._gripper_changed:
            if fraction <= self.config.gripper_open_threshold:
                self._gripper = -1.
            elif fraction >= self.config.gripper_close_threshold:
                self._gripper = 1.

    def snapshot(self, session_id):
        sample = self.latest_snapshot()
        if self._owner != session_id or sample.stale or sample.error or not sample.connected:
            raise RuntimeError(sample.error or "FACTR not armed/fresh")
        return replace(sample, gripper_command=self._gripper)

    def disarm(self, session_id):
        with self._lock:
            if self._owner == session_id:
                self._owner = None
        # Simulation disarm does not unexpectedly remove physical support.

    def status(self):
        sample = self.latest_snapshot()
        return {**self._remote, "state": "ERROR" if self._error else (
            "DISCONNECTED" if not sample.connected else "ARMED" if self._owner else self._remote.get("state")),
            "connected": sample.connected, "stale": sample.stale, "error": self._error,
            "sample_age_ms": None if sample.sample_age_seconds is None else 1000*sample.sample_age_seconds}

    def calibration_snapshot(self):
        return {"physical": None if self._profile is None else self._profile.as_dict()}

    def diagnostics(self):
        return {"status": self.status(), "calibration": self.calibration_snapshot(),
                "runtime_python": self.options.runtime_python, "upstream_root": self.options.upstream_root}

    def close(self):
        if self._closed:
            return
        self._closed = True
        try:
            if self._process.poll() is None:
                try:
                    self._send({"operation": "close", "id": -1})
                except OSError:
                    pass
                try:
                    self._process.wait(timeout=5)
                except subprocess.TimeoutExpired:
                    self._process.terminate()
                    try:
                        self._process.wait(timeout=3)
                    except subprocess.TimeoutExpired:
                        self._process.kill()
                        self._process.wait(timeout=2)
                        raise RuntimeError("FACTR process killed; motor OFF unverified. Support leader and isolate power")
            # Drain the worker's final OFF verification before interpreting exit.
            if self._thread is not None and self._thread is not threading.current_thread():
                self._thread.join(timeout=1)
            if self._shutdown_error or (self._process.returncode != 0 and not self._shutdown_verified):
                raise RuntimeError(self._shutdown_error or f"Official FACTR exited {self._process.returncode}; motor OFF was not confirmed")
        finally:
            self._stop.set()
            if self._thread is not None:
                self._thread.join(timeout=1)
            self._stderr_thread.join(timeout=1)
            self._channel.close()
