"""Official ROS/Pinocchio control, isolated from the simulation Python process.

Only lifecycle, verified hardware limits and IPC are adapted. The upstream
calibration, dynamics, friction, null-space and limit-barrier methods are called
directly, not reimplemented. Importing this module never opens a serial port.
"""
import json
import logging
import os
from pathlib import Path
import signal
import socket
import time

import numpy as np

from .factr_official import import_upstream, verify_upstream

LOG = logging.getLogger(__name__)
WATCHDOG_MS = 100
READ_RECOVERY_MS = 75  # Full-read budget; never run the upstream 10-retry loop.


class ActivationRejected(RuntimeError):
    """No motor-output transition was attempted; operator may retry safely."""


def build_node_class(root):
    OfficialNode, OfficialDriver = import_upstream(root)
    from dynamixel_sdk import GroupSyncRead

    class ArmDriver(OfficialDriver):
        """Keep the trigger passive; preserve manufacturer current limits."""
        def get_positions_and_velocities(self, tries=1):
            # Keep upstream packet decoding/units. A single lost response need
            # not kill the worker, but ten upstream retries can outlive the
            # hardware watchdog. Never return cached or partially read states.
            started = time.monotonic()
            retries = min(max(int(tries), 0), 1)
            for attempt in range(retries + 1):
                try:
                    result = super().get_positions_and_velocities(tries=0)
                except RuntimeError as exc:
                    message = str(exc)
                    recoverable = message in {
                        "Warning, communication failed: -3001",
                        "Warning, communication failed: -3002",
                    }
                    if not recoverable:
                        raise
                    self.read_failures = getattr(self, "read_failures", 0) + 1
                    elapsed_ms = (time.monotonic()-started)*1000.
                    # SDK timeout currently includes ~34 ms USB allowance.
                    # Do not start another full read if it cannot fit the budget.
                    if attempt == retries or elapsed_ms >= READ_RECOVERY_MS / 2:
                        raise RuntimeError(
                            f"FACTR sync read failed ({message.rsplit(': ', 1)[-1]}): "
                            f"{attempt+1} attempt(s), {elapsed_ms:.1f} ms; "
                            "no fresh complete motor packet; stopping output"
                        ) from exc
                    # SDK clearPort() flushes TX, not RX. Discard remnants before
                    # requesting an entirely new eight-motor sync-read packet.
                    self._portHandler.ser.reset_input_buffer()
                    continue
                elapsed_ms = (time.monotonic()-started)*1000.
                self.last_read_ms = elapsed_ms
                if elapsed_ms >= READ_RECOVERY_MS:
                    raise RuntimeError(f"FACTR sync read exceeded recovery budget: {elapsed_ms:.1f} ms; stopping output")
                if attempt:
                    self.recovered_reads = getattr(self, "recovered_reads", 0) + 1
                    if self.recovered_reads == 1 or self.recovered_reads % 100 == 0:
                        LOG.warning("FACTR sync read recovered with a fresh full packet in %.1f ms (recovered=%d)",
                                    elapsed_ms, self.recovered_reads)
                return result

        def set_torque_mode(self, enable):
            if not getattr(self, "_claimed", False):
                if enable:
                    raise RuntimeError("Serial ownership has not been verified")
                import fcntl
                import termios
                from .factr import serial_owners
                from .factr_discovery import discover_factr_device
                serial = self._portHandler.ser
                # Verify identity after the SDK has opened the port but before
                # its constructor's first motor read/write. tty paths can be
                # reused between the read-only probe and driver construction.
                if (discover_factr_device(self._selector) != self._expected_device or
                    os.fstat(serial.fileno()).st_rdev != os.stat(self._expected_device.path).st_rdev):
                    raise RuntimeError("FACTR USB device changed while opening the official driver")
                fcntl.flock(serial.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)
                fcntl.ioctl(serial.fileno(), termios.TIOCEXCL)
                self._exclusive = True
                if serial_owners(serial.port, ignore_fd=serial.fileno())["busy_pids"]:
                    raise RuntimeError("Serial port acquired by another controller during startup")
                for motor_id in self._ids:
                    value, result, error = self._packetHandler.read1ByteTxRx(self._portHandler, motor_id, 64)
                    if value or result or error:
                        raise RuntimeError(f"Motor {motor_id}: not safely OFF at official driver startup")
                self._claimed = True
            ids = self._ids
            try:
                self._ids = ids[:7]
                return super().set_torque_mode(enable)
            finally:
                self._ids = ids

        def close(self):
            if getattr(self, "_exclusive", False):
                import fcntl
                import termios
                serial = self._portHandler.ser
                try:
                    fcntl.ioctl(serial.fileno(), termios.TIOCNXCL)
                finally:
                    super().close()
                    self._exclusive = False
            else:
                super().close()

        def set_current(self, currents):
            currents = np.asarray(currents, dtype=float).copy()
            if currents.shape != (8,) or not np.isfinite(currents).all():
                raise ValueError("Invalid official current output")
            currents[-1] = 0  # No physical gripper actuation in this adapter.
            # Upstream set_current uses raw register ticks, despite its docstring.
            currents[:7] = np.clip(currents[:7], -self.hardware_limits, self.hardware_limits)
            return super().set_current(currents)

    class Bridge(OfficialNode):
        driver_class = ArmDriver
        def __init__(self, settings):
            self.settings = settings
            self.enabled = False
            self.calibrated = False
            self.profile = None
            self.device = {}
            self.latest = None
            self.sequence = 0
            self.last_tick = None
            self.last_status = 0.
            self.watchdog_owned = False
            self.driver = None
            self.last_torque = np.zeros(7)
            self.cycle_ms = 0.
            self.alignment = None
            self.alignment_tick_paused = False
            from collections import deque
            self.recent_periods = deque(maxlen=50)
            super().__init__()  # Dispatches lifecycle overrides below, NOT auto-enable.
            self.joint_offsets = np.zeros(8)

        def _prepare_dynamixel(self):
            from .factr import FactrStartupProbe
            from .factr_discovery import discover_factr_device
            cfg = self.config
            reference = cfg["arm_teleop"]["initialization"]["calibration_joint_pos"]
            if (list(self.settings.motor_ids) != list(range(1, 9)) or
                list(self.settings.motor_models) != cfg["dynamixel"]["servo_types"] or
                list(self.settings.joint_signs) != cfg["dynamixel"]["joint_signs"][:7] or
                not np.allclose(self.settings.reference_joint_positions, reference, atol=1e-8) or
                not np.allclose(self.settings.joint_limits_min, cfg["arm_teleop"]["arm_joint_limits_min"]) or
                not np.allclose(self.settings.joint_limits_max, cfg["arm_teleop"]["arm_joint_limits_max"]) or
                self.settings.baudrate != 4000000):
                raise ValueError("Device/model/signs/reference must match the pinned official Franka configuration (Figure 1)")
            # Resolve again inside the serial-owning process, never reuse an
            # idle GUI probe's path after a hotplug / ttyUSB renumbering.
            serial_device = discover_factr_device(self.settings)
            port = Path(serial_device.path)
            latency = Path("/sys/bus/usb-serial/devices")/port.name/"latency_timer"
            latency_ms = int(latency.read_text().strip())
            if latency_ms != 1:
                raise RuntimeError(
                    f"FACTR startup blocked: USB latency_timer={latency_ms} ms, requires 1 ms: {latency}. "
                    "For persistence, run python liberox-vla-adapter-terminal/scripts/setup_factr.py "
                    "--install-usb-rule, then stop the controller and replug USB. No setting changed")
            # Read-only probe rejects torque owned by another program, wrong
            # firmware/mode/watchdog before official constructor writes torque OFF.
            probe = FactrStartupProbe(self.settings, device=serial_device)
            try:
                self.device = probe.open()
                limits, fingerprint = [], []
                for motor_id in self.settings.motor_ids:
                    values = {}
                    for name, address, width in (("mode", 11, 1), ("drive", 10, 1), ("homing", 20, 4),
                                                  ("limit", 38, 2), ("watchdog", 98, 1), ("firmware", 6, 1),
                                                  ("return_level", 68, 1), ("return_delay", 9, 1)):
                        value, result, error = getattr(probe.packet, f"read{width}ByteTxRx")(probe.port, motor_id, address)
                        probe._validate_result(result, error, motor_id)
                        values[name] = int(value)
                    if values["mode"] != 0 or values["drive"] != 0 or values["watchdog"] != 0 or values["firmware"] < 38 or values["return_level"] != 2 or values["return_delay"] != 0:
                        raise RuntimeError(f"Motor {motor_id}: inspect current mode/drive/watchdog/firmware/return level: {values}")
                    if values["limit"] <= 0:
                        raise ValueError(f"Motor {motor_id}: invalid hardware Current Limit")
                    limits.append(values["limit"])
                    fingerprint.append({"id": motor_id, **values})
            finally:
                probe.close()
            self.fingerprint = {"official_commit": verify_upstream(root)["commit"],
                                "usb_device": serial_device.identity(), "motors": fingerprint}
            self.device.update(passive_only=False, backend="official_factr", startup_torque_off=True)
            self.servo_types = cfg["dynamixel"]["servo_types"]
            self.num_motors = 8
            self.joint_signs = np.asarray(cfg["dynamixel"]["joint_signs"], dtype=float)
            if discover_factr_device(self.settings) != serial_device:
                raise RuntimeError("FACTR USB device changed during startup; reconnect and calibrate again")
            self.dynamixel_port = serial_device.path
            self.driver = self.driver_class.__new__(self.driver_class)
            self.driver._selector = self.settings
            self.driver._expected_device = serial_device
            self.driver.__init__(np.arange(1, 9), self.servo_types, self.dynamixel_port)
            if not getattr(self.driver, "_claimed", False):
                raise RuntimeError("Official driver could not establish exclusive OFF ownership")
            self.driver.hardware_limits = np.asarray(limits[:7])
            self.driver.verify_operating_mode(0)  # Do not write operating mode/EEPROM.
            self.status_reader = GroupSyncRead(self.driver._portHandler, self.driver._packetHandler, 64, 35)
            for motor_id in range(1, 9):
                if not self.status_reader.addParam(motor_id):
                    raise RuntimeError("Cannot configure motor status verification")
            self.verify_status(False)

        def _prepare_inverse_dynamics(self):
            # Upstream requires cwd at its workspace root for relative URDF paths.
            return super()._prepare_inverse_dynamics()

        def _get_dynamixel_offsets(self, verbose=True):
            pass  # Deferred until the explicit whole-arm calibration command.

        def _match_start_pos(self):
            pass  # Defer leader alignment until explicit simulation takeover.

        def set_up_communication(self):
            pass  # Local authenticated-by-inheritance socketpair is owned by main().

        def get_leader_arm_external_joint_torque(self):
            return np.zeros(7)

        def start_alignment(self, target):
            from .factr_alignment import LeaderAlignment
            if not self.enabled or not self.calibrated:
                raise ActivationRejected("Enable compensation before aligning the leader")
            # Upstream set_leader_joint_pos documents >=200 Hz for these PD
            # gains. This requirement is only for the preparation motion.
            if len(self.recent_periods) < 50 or np.mean(self.recent_periods) > .005:
                raise ActivationRejected("Leader alignment requires a verified >=200 Hz loop; inspect USB/control timing before retrying")
            target = np.asarray(target, dtype=float)
            if target.shape != (7,) or not np.isfinite(target).all() or np.any(target < self.arm_joint_limits_min) or np.any(target > self.arm_joint_limits_max):
                raise ValueError("Follower pose is outside FACTR joint limits; no alignment motion started")
            q, _, _, _ = self.get_leader_joint_states()
            self.alignment = LeaderAlignment(q, target, now=time.monotonic())

        def null_space_regulation(self, q, dq):
            if self.alignment is None:
                return super().null_space_regulation(q, dq)
            if self.alignment_tick_paused:
                # No catch-up reference advance or position PD on a late tick.
                # Keep the normal official compensation terms instead.
                self.alignment.last = time.monotonic()
                return super().null_space_regulation(q, dq)
            error = self.alignment.target-np.asarray(q) if self.alignment.done else self.alignment.update(q, dq, time.monotonic())
            gains = self.config["controller"]["joint_position_control"]
            # Same PD terms as official set_leader_joint_pos, executed one tick
            # at a time so OFF, heartbeat and watchdog remain responsive. Official
            # gravity/friction/limit terms are retained by control_loop_callback.
            return gains["kp"]*error - gains["kd"]*np.asarray(dq)

        def get_leader_gripper_feedback(self):
            return 0.

        def gripper_feedback(self, *args):
            return 0.

        def get_leader_joint_states(self):
            previous = self.latest
            result = super().get_leader_joint_states()
            if self.alignment is not None and getattr(self.driver, "last_read_ms", 0.) > 5.:
                self.alignment_tick_paused = True
            q, dq, grip, _ = result
            raw = np.asarray(self.driver._positions)/2048.*np.pi
            self.sequence += 1
            self.latest = {"sequence": self.sequence, "sample_monotonic": time.monotonic(),
                           "raw_joints": raw[:7].tolist(), "raw_gripper": float(raw[7]),
                           "joint_positions": q.tolist(), "joint_velocities": dq.tolist()}
            if not np.isfinite(np.r_[q, dq, raw]).all():
                raise RuntimeError("Invalid official joint state")
            if self.enabled and previous is not None and np.max(np.abs(q-np.asarray(previous["joint_positions"]))) > self.settings.max_joint_jump_rad:
                raise RuntimeError("Encoder discontinuity during compensation; stop and inspect whole-arm calibration")
            return result

        def update_communication(self, leader_arm_pos, leader_gripper_pos):
            pass  # Main process publishes the cached official observation.

        def set_leader_joint_torque(self, arm_torque, gripper_torque):
            # Check before sending current, not just after the callback returns:
            # scheduling/USB stalls must not turn a late read into late output.
            if self.enabled and self.last_tick is not None and (time.monotonic()-self.last_tick)*1000 >= WATCHDOG_MS:
                raise RuntimeError("Official control cycle exceeded watchdog interval before current output")
            self.last_torque = np.asarray(arm_torque).copy()
            # Direct upstream unit/sign conversion; no second gravity formula.
            return super().set_leader_joint_torque(arm_torque, 0.)

        def verify_status(self, enabled):
            if self.status_reader.txRxPacket() != 0:
                raise RuntimeError("FACTR status sync read failed")
            for i in range(1, 9):
                if not self.status_reader.isAvailable(i, 64, 35):
                    raise RuntimeError(f"Motor {i}: missing status")
                torque = self.status_reader.getData(i, 64, 1)
                hardware = self.status_reader.getData(i, 70, 1)
                wd = self.status_reader.getData(i, 98, 1)
                if torque != int(enabled and i <= 7) or hardware or (enabled and i <= 7 and wd != WATCHDOG_MS//20):
                    raise RuntimeError(f"Motor {i}: torque={torque}, hardware_error={hardware}, watchdog={wd}")

        def watchdog(self, value):
            packet, port = self.driver._packetHandler, self.driver._portHandler
            for i in range(1, 8):
                result, error = packet.write1ByteTxRx(port, i, 98, value)
                if result or error:
                    raise RuntimeError(f"Motor {i}: watchdog write failed ({result}, {error})")

        def capture_reference(self):
            if self.enabled:
                raise RuntimeError("Disable compensation before calibration")
            # This is the actual upstream method, including its ten warm-up reads.
            super()._get_dynamixel_offsets(verbose=False)
            self.calibrated = False  # Profile is committed after the same capture.
            return self.joint_offsets[:7].tolist()

        def apply_profile(self, payload):
            from .factr_calibration import FactrCalibrationProfile, make_profile, config_hash
            profile = FactrCalibrationProfile(**payload)
            if profile.schema_version != 2 or self.enabled or profile.device_fingerprint != self.fingerprint or profile.config_hash != config_hash(self.settings):
                raise ValueError("Profile does not match official device/configuration or compensation is active")
            make_profile(self.settings, profile.offsets, profile.gripper_open, profile.gripper_closed, self.fingerprint)
            self.profile = profile
            self.joint_offsets = np.r_[profile.offsets, profile.gripper_open]
            self.calibrated = True

        def calibrate(self):
            """One official whole-arm capture, with released trigger at zero."""
            from .factr_calibration import make_profile
            offsets = self.capture_reference()
            opened = float(self.joint_offsets[-1])
            closed = opened + self.config["gripper_teleop"]["actuation_range"]/self.joint_signs[-1]
            profile = make_profile(self.settings, offsets, opened, closed, self.fingerprint)
            self.apply_profile(profile.as_dict())
            return profile.as_dict()

        def enable_compensation(self):
            if not self.calibrated or self.enabled:
                raise ActivationRejected("Complete calibration first; cannot enable twice")
            self.verify_status(False)
            self.get_leader_joint_states()
            q = np.asarray(self.latest["joint_positions"])
            if np.any(q < np.asarray(self.settings.joint_limits_min)) or np.any(q > np.asarray(self.settings.joint_limits_max)):
                raise ActivationRejected("Support/reposition within physical limits before enabling")
            # Clear old goals while OFF, before any motor can produce torque.
            for i in range(1, 8):
                result, error = self.driver._packetHandler.write2ByteTxRx(self.driver._portHandler, i, 102, 0)
                if result or error:
                    raise RuntimeError(f"Motor {i}: cannot zero current before enable")
            self.watchdog_owned = True
            self.watchdog(WATCHDOG_MS//20)
            self.driver.set_torque_mode(True)
            self.enabled = True
            self.verify_status(True)
            self.last_tick = time.monotonic()
            self.control_loop_callback()  # Fresh official cycle before success reply.

        def disable_compensation(self):
            self.alignment = None
            errors = []
            if self.driver is not None and getattr(self.driver, "_claimed", False):
                # Try all owned motors even if one ACK is lost. Goal Current can
                # be read-only after watchdog expiry; torque OFF comes first.
                for i in range(1, 8):
                    try:
                        result, error = self.driver._packetHandler.write1ByteTxRx(self.driver._portHandler, i, 64, 0)
                        value, result2, error2 = self.driver._packetHandler.read1ByteTxRx(self.driver._portHandler, i, 64)
                        if value != 0 or result2 or error2:
                            raise RuntimeError(f"motor {i}: OFF unconfirmed ({result}, {error}, {result2}, {error2})")
                    except Exception as exc:
                        errors.append(str(exc))
                if not errors and self.watchdog_owned:
                    self.watchdog(0)
                    self.watchdog_owned = False
                self.driver._torque_enabled = False
            self.enabled = False
            self.last_tick = None
            self.recent_periods.clear()
            if errors:
                raise RuntimeError("Motor shutdown NOT verified: "+"; ".join(errors))

        def control_loop_callback(self):
            started = time.monotonic()
            self.alignment_tick_paused = False
            if self.enabled and self.last_tick is not None:
                period = started-self.last_tick
                self.recent_periods.append(period)
                self.alignment_tick_paused = period > .005
                if self.alignment is not None and len(self.recent_periods) == 50 and np.mean(self.recent_periods) > .005:
                    hz = 1./np.mean(self.recent_periods)
                    raise RuntimeError(f"FACTR alignment loop sustained {hz:.1f} Hz over 50 cycles; requires >=200 Hz")
            if self.enabled and self.last_tick is not None and started-self.last_tick >= WATCHDOG_MS/1000:
                raise RuntimeError("Official control loop stalled beyond hardware watchdog interval")
            self.last_tick = started
            if self.enabled:
                super().control_loop_callback()
                if started-self.last_status >= .02:
                    self.verify_status(True)
                    self.last_status = started
            else:
                self.get_leader_joint_states()
            self.cycle_ms = (time.monotonic()-started)*1000
            if self.enabled and self.cycle_ms >= WATCHDOG_MS:
                raise RuntimeError(f"Official control cycle too slow: {self.cycle_ms:.1f} ms")

        def status(self):
            return {"state": "READY" if self.calibrated else "UNCALIBRATED", "connected": True,
                    "calibrated": self.calibrated, "gravity_enabled": self.enabled,
                    "official_commit": self.fingerprint["official_commit"], "cycle_ms": self.cycle_ms,
                    "loop_hz": None if not self.recent_periods else 1./max(float(np.mean(self.recent_periods)), 1e-9),
                    "loop_max_period_ms": None if not self.recent_periods else 1000*max(self.recent_periods),
                    "alignment_tick_paused": self.alignment_tick_paused,
                    "serial_read": {
                        "failures": getattr(self.driver, "read_failures", 0),
                        "recovered": getattr(self.driver, "recovered_reads", 0),
                        "last_read_ms": getattr(self.driver, "last_read_ms", None),
                    },
                    "controller": self.config["controller"], "last_torque_nm": self.last_torque.tolist(),
                    "fingerprint": self.fingerprint, "device": self.device, "sample": self.latest,
                    "serial_device": self.device.get("serial_device"),
                    "alignment": None if self.alignment is None else self.alignment.status()}

    return Bridge


def serve(settings, fd):
    root = settings.runtime["upstream_root"]
    Bridge = build_node_class(root)
    import rclpy
    channel = socket.socket(fileno=fd)
    channel.setblocking(False)
    node = None
    shutdown_attempted = False
    heartbeat = time.monotonic()
    last_send = 0.
    def send(message):
        channel.send(json.dumps(message, allow_nan=False).encode())
    def stop_signal(*_):
        raise KeyboardInterrupt
    signal.signal(signal.SIGTERM, stop_signal)
    signal.signal(signal.SIGINT, stop_signal)
    os.chdir(root)
    rclpy.init(args=["--ros-args", "-p", "config_file:=grav_comp_demo.yaml",
                    "--disable-external-lib-logs", "--disable-rosout-logs"])
    try:
        node = Bridge.__new__(Bridge)
        node.__init__(settings)
        while True:
            for _ in range(16):
                try:
                    packet = channel.recv(65536)
                except BlockingIOError:
                    break
                if not packet:
                    raise RuntimeError("FACTR parent disconnected")
                request = json.loads(packet)
                operation = request["operation"]
                if operation == "heartbeat":
                    heartbeat = time.monotonic()
                    continue
                request_id = request["id"]
                transition_applied = False
                try:
                    if operation == "calibrate":
                        result = node.calibrate()
                    elif operation == "profile":
                        node.apply_profile(request["value"])
                        result = None
                    elif operation == "enable":
                        node.enable_compensation()
                        result = None
                    elif operation == "align":
                        node.start_alignment(request["value"])
                        result = None
                    elif operation == "cancel_align":
                        node.alignment = None
                        result = None
                    elif operation == "disable":
                        node.disable_compensation()
                        result = None
                    elif operation == "close":
                        return
                    else:
                        raise ValueError("Unknown official controller operation")
                    transition_applied = True
                    node.control_loop_callback()
                    send({"reply": request_id, "result": result, "status": node.status()})
                except (ValueError, RuntimeError) as exc:
                    # Any failed output transition is a fault, never continue on
                    # a potentially partially enabled arm.
                    if transition_applied or (operation in {"enable", "disable"} and not isinstance(exc, ActivationRejected)):
                        raise
                    send({"reply": request_id, "error": str(exc)})
            if time.monotonic()-heartbeat > 1.:
                raise RuntimeError("FACTR parent heartbeat lost")
            rclpy.spin_once(node, timeout_sec=.002)
            if node.latest is not None and time.monotonic()-last_send >= .025:
                send({"status": node.status()})
                last_send = time.monotonic()
    except BaseException as exc:
        # Attempt motor OFF before potentially blocking logging or parent IPC.
        shutdown_error = None
        if node is not None:
            shutdown_attempted = True
            try:
                node.disable_compensation()
            except Exception as cleanup:
                shutdown_error = str(cleanup)
        LOG.error("Official FACTR stopped: %s; shutdown=%s", exc, shutdown_error or "OFF confirmed/no owned driver")
        try:
            send({"fatal": str(exc) or type(exc).__name__,
                  "shutdown": {"verified": shutdown_error is None, "error": shutdown_error}})
        except (OSError, ValueError):
            pass
        raise
    finally:
        if node is not None:
            try:
                if not shutdown_attempted:
                    try:
                        node.disable_compensation()
                    except Exception as exc:
                        send({"shutdown": {"verified": False, "error": str(exc)}})
                        raise
                    send({"shutdown": {"verified": True, "error": None}})
            finally:
                if getattr(node, "driver", None) is not None:
                    node.driver.close()
                if hasattr(node, "_handle"):
                    node.destroy_node()
        channel.close()
        rclpy.shutdown()
