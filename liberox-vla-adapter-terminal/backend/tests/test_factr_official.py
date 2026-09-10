"""Real pinned algorithms, fake bus/ROS lifecycle. Never construct serial ports."""
from pathlib import Path
from types import SimpleNamespace
import json
import os
import subprocess
import sys
import time

import numpy as np
import pytest

from backend.app.devices.factr import load_factr_config
from backend.app.devices.factr_official import WORKSPACE, import_upstream, verify_upstream
from backend.app.devices.factr_calibration import make_profile


@pytest.fixture
def official_node(monkeypatch):
    pytest.importorskip("dynamixel_sdk")
    root = WORKSPACE/"third_party/FACTR_Teleop"
    if not root.exists():
        pytest.skip("Run setup_factr.py first")
    try:
        Official, Driver = import_upstream(root)
    except ImportError as exc:
        pytest.skip(f"Use third_party/factr-runtime/bin/python for official tests: {exc}")
    from backend.app.devices.factr_official_runtime import build_node_class
    import rclpy.node
    # All ROS and port construction explicitly disabled.
    monkeypatch.setattr(rclpy.node.Node, "__init__", lambda *a, **kw: None)
    monkeypatch.setattr(rclpy.node.Node, "declare_parameter", lambda *a, **kw:
        SimpleNamespace(get_parameter_value=lambda: SimpleNamespace(string_value="grav_comp_demo.yaml")))
    monkeypatch.setattr(rclpy.node.Node, "create_timer", lambda *a, **kw: None)
    import factr_teleop.dynamixel.driver as driver_module
    monkeypatch.setattr(driver_module, "PortHandler", lambda *a: pytest.fail("Real port access forbidden"))
    Bridge = build_node_class(root)
    settings = load_factr_config()
    writes = []
    torque = {i: 0 for i in range(1, 9)}
    watchdog = {i: 0 for i in range(1, 9)}

    class Packet:
        def write1ByteTxRx(self, port, i, address, value):
            assert address in {64, 98}
            assert i <= 7
            writes.append((i, address, value))
            (torque if address == 64 else watchdog)[i] = value
            return 0, 0
        def read1ByteTxRx(self, port, i, address):
            assert address == 64
            return torque[i], 0, 0
        def write2ByteTxRx(self, port, i, address, value):
            assert address == 102 and i <= 7
            writes.append((i, address, value))
            return 0, 0

    class Status:
        def txRxPacket(self): return 0
        def isAvailable(self, *a): return True
        def getData(self, i, address, width):
            return torque[i] if address == 64 else watchdog[i] if address == 98 else 0

    class Writer:
        def addParam(self, i, values):
            writes.append((i, 102, values))
            return True
        def txPacket(self): return 0
        def clearParam(self): pass

    class Reader:
        def txRxPacket(self): return 0
        def isAvailable(self, *a): return True
        def getData(self, i, address, width):
            # Encoder positions from a feasible physical reference; velocity is
            # deliberately nonzero to test measured, not finite-difference dq.
            if address == 128:
                return 2
            raw = np.r_[np.asarray(settings.reference_joint_positions)*settings.joint_signs, .2]
            return int(round(raw[i-1]/np.pi*2048)) % 2**32

    def prepare(self):
        import threading
        self.driver = self.driver_class.__new__(self.driver_class)
        driver = self.driver
        driver._ids = list(range(1, 9))
        driver._lock = threading.Lock()
        driver._portHandler = SimpleNamespace(closePort=lambda: None)
        driver._packetHandler = Packet()
        driver._groupSyncRead = Reader()
        driver._groupSyncWrite = Writer()
        driver._torque_enabled = False
        driver._claimed = True
        driver.hardware_limits = np.array([900]*7)
        driver.torque_to_current_map = np.ones(8)*100
        self.joint_signs = np.asarray(self.config["dynamixel"]["joint_signs"])
        self.fingerprint = {"official_commit": verify_upstream(root)["commit"]}
        self.status_reader = Status()
    monkeypatch.setattr(Bridge, "_prepare_dynamixel", prepare)
    monkeypatch.chdir(root)
    node = Bridge(settings)
    return node, Official, writes, torque, watchdog


def test_upstream_methods_are_inherited_and_calibration_is_official(official_node, monkeypatch):
    node, Official, writes, torque, watchdog = official_node
    for name in ("gravity_compensation", "friction_compensation", "joint_limit_barrier"):
        assert getattr(type(node), name) is getattr(Official, name)
    q = np.asarray(node.settings.reference_joint_positions)
    np.testing.assert_allclose(node.null_space_regulation(q, q*0),
                               Official.null_space_regulation(node, q, q*0))
    count = []
    original = node.driver.get_positions_and_velocities
    def read():
        count.append(1)
        return original()
    monkeypatch.setattr(node.driver, "get_positions_and_velocities", read)
    offsets = node.capture_reference()
    assert len(count) == 11
    np.testing.assert_allclose(offsets, 0., atol=1e-12)
    assert not writes and not node.enabled and not node.calibrated
    q, dq, _, _ = node.get_leader_joint_states()
    np.testing.assert_allclose(dq, np.array(node.settings.joint_signs)*2*.229*2*np.pi/60)


def test_official_full_torque_sum_and_explicit_enable_disable(official_node):
    node, Official, writes, torque, watchdog = official_node
    node.control_loop_callback()  # Passive read, never any output.
    assert writes == []
    with pytest.raises(RuntimeError, match="calibration"):
        node.enable_compensation()
    offsets = node.capture_reference()
    profile = make_profile(node.settings, offsets, 0., .5, node.fingerprint)
    node.apply_profile(profile.as_dict())
    assert writes == []
    node.enable_compensation()
    assert all(torque[i] == 1 for i in range(1, 8)) and torque[8] == 0
    assert all(watchdog[i] == 5 for i in range(1, 8))
    q, dq, grip, gripvel = node.get_leader_joint_states()
    # Test one complete actual upstream callback against its direct components.
    dither = node.stiction_dither_flag.copy()
    expected = (Official.joint_limit_barrier(node, q, dq, grip, gripvel)[0]
                + Official.null_space_regulation(node, q, dq)
                + Official.gravity_compensation(node, q, dq)
                + Official.friction_compensation(node, dq))
    node.stiction_dither_flag = dither
    node.control_loop_callback()
    np.testing.assert_allclose(node.last_torque, expected, rtol=1e-12, atol=1e-12)
    assert any(i == 8 and address == 102 and value == [0, 0] for i, address, value in writes)
    node.disable_compensation()
    assert not node.enabled and all(v == 0 for v in torque.values()) and all(v == 0 for v in watchdog.values())


def test_alignment_uses_official_pd_gains_and_release_preserves_support(official_node, monkeypatch):
    node, Official, writes, torque, watchdog = official_node
    profile = make_profile(node.settings, node.capture_reference(), 0., .5, node.fingerprint)
    node.apply_profile(profile.as_dict())
    node.enable_compensation()
    q, dq, _, _ = node.get_leader_joint_states()
    with pytest.raises(RuntimeError, match="200 Hz"):
        node.start_alignment(q)
    node.recent_periods.extend([.002]*50)
    node.start_alignment(q + .02)
    assert node.alignment is not None and node.enabled
    before = len(writes)
    now = node.alignment.last + .002
    monkeypatch.setattr("backend.app.devices.factr_official_runtime.time.monotonic", lambda: now)
    result = node.null_space_regulation(q, dq)
    gains = node.config["controller"]["joint_position_control"]
    np.testing.assert_allclose(result, gains["kp"] * .0003 - gains["kd"] * dq)
    assert len(writes) == before  # Computes torque only, actual output stays in official loop.
    node.alignment = None  # cancel_align protocol
    np.testing.assert_allclose(node.null_space_regulation(q, dq), Official.null_space_regulation(node, q, dq))
    assert node.enabled and all(torque[i] == 1 for i in range(1, 8))
    node.disable_compensation()
    assert node.alignment is None and all(v == 0 for v in torque.values())


def test_alignment_tolerates_one_late_cycle_but_rejects_sustained_low_rate(official_node, monkeypatch):
    node, _, _, _, _ = official_node
    profile = make_profile(node.settings, node.capture_reference(), 0., .5, node.fingerprint)
    node.apply_profile(profile.as_dict())
    node.enable_compensation()
    q, _, _, _ = node.get_leader_joint_states()
    node.recent_periods.extend([.002]*49 + [.006])
    node.start_alignment(q + .02)  # One late sample does not invalidate a fast window.
    reference = node.alignment.reference.copy()
    now = node.last_tick + .006
    monkeypatch.setattr("backend.app.devices.factr_official_runtime.time.monotonic", lambda: now)
    node.control_loop_callback()
    assert node.enabled and node.alignment_tick_paused
    np.testing.assert_array_equal(node.alignment.reference, reference)
    assert node.status()["loop_hz"] > 200
    now += .002
    node.control_loop_callback()
    assert not node.alignment_tick_paused
    assert np.all(node.alignment.reference > reference)
    node.recent_periods.clear()
    node.recent_periods.extend([.006]*50)
    now += .006
    with pytest.raises(RuntimeError, match="sustained .* Hz over 50 cycles"):
        node.control_loop_callback()
    node.disable_compensation()


def test_watchdog_expiry_is_not_reenabled(official_node):
    node, _, writes, torque, watchdog = official_node
    profile = make_profile(node.settings, node.capture_reference(), 0., .5, node.fingerprint)
    node.apply_profile(profile.as_dict())
    node.enable_compensation()
    watchdog[4] = 255
    with pytest.raises(RuntimeError, match="watchdog=255"):
        node.verify_status(True)
    writes.clear()
    node.disable_compensation()
    assert all(v == 0 for v in torque.values())
    assert not any(address == 64 and value == 1 for _, address, value in writes)


def test_stalled_loop_rejects_before_read_or_current(official_node):
    node, _, writes, _, _ = official_node
    node.enabled = True
    node.last_tick = time.monotonic()-.2
    with pytest.raises(RuntimeError, match="stalled"):
        node.control_loop_callback()
    assert writes == []


@pytest.mark.parametrize("code", [-3001, -3002])
def test_one_transient_read_failure_recovers_only_with_full_fresh_packet(official_node, monkeypatch, code):
    node, _, writes, _, _ = official_node
    from backend.app.devices import factr_official_runtime as runtime
    now = [100.]
    monkeypatch.setattr(runtime.time, "monotonic", lambda: now[0])
    calls = []
    def transaction():
        calls.append("read")
        now[0] += .034 if calls.count("read") == 1 else .002
        return code if calls.count("read") == 1 else 0
    node.driver._portHandler.ser = SimpleNamespace(reset_input_buffer=lambda: calls.append("discard_rx"))
    monkeypatch.setattr(node.driver._groupSyncRead, "txRxPacket", transaction)
    positions, velocities = node.driver.get_positions_and_velocities()
    assert calls == ["read", "discard_rx", "read"]
    assert len(positions) == len(velocities) == 8
    assert np.isfinite(positions).all()
    assert node.driver.recovered_reads == node.driver.read_failures == 1
    assert node.driver.last_read_ms == pytest.approx(36.)
    assert writes == []  # Retry itself only reads; no torque / re-enable.


@pytest.mark.parametrize("delay, expected_reads", [(.034, 2), (.04, 1)])
def test_persistent_timeout_is_bounded_and_never_returns_cached_state(official_node, monkeypatch, delay, expected_reads):
    node, _, writes, _, _ = official_node
    from backend.app.devices import factr_official_runtime as runtime
    node.get_leader_joint_states()
    previous = node.latest
    writes.clear()
    now = [100.]
    monkeypatch.setattr(runtime.time, "monotonic", lambda: now[0])
    calls = []
    def transaction():
        calls.append("read"); now[0] += delay; return -3001
    node.driver._portHandler.ser = SimpleNamespace(reset_input_buffer=lambda: None)
    monkeypatch.setattr(node.driver._groupSyncRead, "txRxPacket", transaction)
    with pytest.raises(RuntimeError, match="no fresh complete motor packet"):
        node.get_leader_joint_states()
    assert len(calls) == expected_reads
    assert node.latest is previous and writes == []
    assert getattr(node.driver, "recovered_reads", 0) == 0


def test_incomplete_packet_is_not_retried_or_published(official_node, monkeypatch):
    node, _, writes, _, _ = official_node
    monkeypatch.setattr(node.driver._groupSyncRead, "isAvailable", lambda *args: False)
    with pytest.raises(RuntimeError, match="Failed to get velocity"):
        node.get_leader_joint_states()
    assert node.latest is None and writes == []


def test_slow_successful_read_is_rejected_before_current(official_node, monkeypatch):
    node, _, writes, _, _ = official_node
    from backend.app.devices import factr_official_runtime as runtime
    now = [100.]
    monkeypatch.setattr(runtime.time, "monotonic", lambda: now[0])
    def transaction(): now[0] += .08; return 0
    monkeypatch.setattr(node.driver._groupSyncRead, "txRxPacket", transaction)
    with pytest.raises(RuntimeError, match="recovery budget"):
        node.get_leader_joint_states()
    assert node.latest is None and writes == []


def test_delayed_read_pauses_alignment_pd_and_rejects_late_current(official_node, monkeypatch):
    node, _, writes, _, _ = official_node
    from backend.app.devices import factr_official_runtime as runtime
    now = [100.]
    monkeypatch.setattr(runtime.time, "monotonic", lambda: now[0])
    def transaction(): now[0] += .006; return 0
    monkeypatch.setattr(node.driver._groupSyncRead, "txRxPacket", transaction)
    node.alignment = SimpleNamespace(last=100.)
    node.get_leader_joint_states()
    assert node.alignment_tick_paused
    node.enabled = True
    node.last_tick = now[0] - .101
    with pytest.raises(RuntimeError, match="before current output"):
        node.set_leader_joint_torque(np.zeros(7), 0.)
    assert writes == []


def test_unclaimed_driver_cleanup_does_not_take_over_another_controller(official_node):
    node, _, writes, _, _ = official_node
    node.driver._claimed = False
    node.disable_compensation()
    assert writes == []


def test_invalid_profile_never_enables(official_node):
    node, _, writes, _, _ = official_node
    profile = make_profile(node.settings, [0.]*7, 0., .5, node.fingerprint).as_dict()
    for invalid in ({**profile, "schema_version": 1}, {**profile, "device_fingerprint": {"wrong": True}}):
        with pytest.raises(ValueError, match="Profile"):
            node.apply_profile(invalid)
    assert not node.calibrated and writes == []


def test_partial_enable_shutdown_still_attempts_every_owned_motor(official_node, monkeypatch):
    node, _, writes, torque, watchdog = official_node
    original = node.driver._packetHandler.write1ByteTxRx
    def fail(port, motor, address, value):
        if address == 64 and value == 1 and motor == 4:
            return 1, 0
        return original(port, motor, address, value)
    monkeypatch.setattr(node.driver._packetHandler, "write1ByteTxRx", fail)
    profile = make_profile(node.settings, node.capture_reference(), 0., .5, node.fingerprint)
    node.apply_profile(profile.as_dict())
    try:
        with pytest.raises(RuntimeError):
            node.enable_compensation()
        assert torque[1] == 1 and not node.enabled
    finally:
        node.disable_compensation()  # Same cleanup used by serve's finally.
    assert all(v == 0 for v in torque.values()) and all(v == 0 for v in watchdog.values())


def test_raw_current_hardware_clamp_and_passive_trigger(official_node):
    node, _, writes, _, _ = official_node
    node.driver._torque_enabled = True
    node.driver.hardware_limits = np.arange(1, 8)*10
    node.driver.set_current(np.array([1000., -1000., 1000., -1000., 1000., -1000., 1000., 999.]))
    currents = []
    for _, address, value in writes:
        assert address == 102
        n = value[0]+256*value[1]
        currents.append(n-65536 if n >= 32768 else n)
    assert currents == [10, -20, 30, -40, 50, -60, 70, 0]


def test_configuration_mismatch_is_rejected_before_port_access(official_node):
    node, _, writes, _, _ = official_node
    from dataclasses import replace
    from backend.app.devices.factr_official_runtime import build_node_class
    UnmodifiedBridge = build_node_class(WORKSPACE/"third_party/FACTR_Teleop")
    node.settings = replace(node.settings, baudrate=1000000)
    with pytest.raises(ValueError, match="pinned official"):
        UnmodifiedBridge._prepare_dynamixel(node)
    assert writes == []


def test_one_click_calibration_uses_official_trigger_zero_and_range(official_node):
    node, _, writes, _, _ = official_node
    profile = node.calibrate()
    assert node.calibrated and not node.enabled and writes == []
    assert profile["gripper_closed"]-profile["gripper_open"] == pytest.approx(-.8)
    assert node.joint_offsets[-1] == profile["gripper_open"]
