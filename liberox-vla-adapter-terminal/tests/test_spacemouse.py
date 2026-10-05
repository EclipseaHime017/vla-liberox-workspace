from __future__ import annotations

import dataclasses
import time
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest
from backend.app.devices.spacemouse import (
    DEFAULT_SPACEMOUSE_CONFIG,
    SpaceMouseInput,
    SpaceMouseTransform,
    load_spacemouse_config,
)
from backend.app.devices.spacemouse_intent import SpaceMouseIntent


def base_config():
    return load_spacemouse_config(DEFAULT_SPACEMOUSE_CONFIG)


def test_default_config_is_valid_and_targets_cabled_wireless_device():
    config = base_config()
    # The same fixed file is intentionally switched between the two supported
    # standalone test modes by the operator.
    assert config.mode in {"device", "simulation"}
    assert config.expected_vendor_id == 0x256F
    assert config.expected_product_id == 0xC63A
    assert config.device_name == "SpaceMouseWirelessNew"
    assert config.axis_convention == "ros"
    assert config.axis_order == ("y", "x", "z", "roll", "pitch", "yaw")
    assert config.axis_signs == (1, -1, 1, 1, 1, -1)
    assert config.motion_mode == "exclusive"
    assert config.intent_switch_ratio == 1.25
    assert config.translation_gain == config.rotation_gain == 0.5


def test_config_rejects_duplicate_unknown_and_invalid_axis_mapping(tmp_path: Path):
    source = DEFAULT_SPACEMOUSE_CONFIG.read_text(encoding="utf-8")

    duplicate = tmp_path / "duplicate.yaml"
    duplicate.write_text(source + "\nmode: simulation\n", encoding="utf-8")
    with pytest.raises(ValueError, match="duplicate key"):
        load_spacemouse_config(duplicate)

    unknown = tmp_path / "unknown.yaml"
    unknown.write_text(source + "\nextra_setting: true\n", encoding="utf-8")
    with pytest.raises(ValueError, match="Unknown SpaceMouse configuration keys"):
        load_spacemouse_config(unknown)

    invalid = tmp_path / "invalid.yaml"
    invalid.write_text(
        source.replace(
            "axis_order: [y, x, z, roll, pitch, yaw]",
            "axis_order: [x, x, z, roll, pitch, yaw]",
        ),
        encoding="utf-8",
    )
    with pytest.raises(ValueError, match="permutation"):
        load_spacemouse_config(invalid)

    invalid_range = tmp_path / "invalid_range.yaml"
    invalid_range.write_text(source.replace("deadzone: 0.05", "deadzone: 1.0"), encoding="utf-8")
    with pytest.raises(ValueError, match="deadzone"):
        load_spacemouse_config(invalid_range)


def test_transform_applies_bias_deadzone_mapping_sign_gain_and_clipping():
    config = dataclasses.replace(
        base_config(),
        motion_mode="combined",
        deadzone=0.0,
        smoothing_alpha=1.0,
        translation_gain=0.5,
        rotation_gain=0.25,
        axis_order=("z", "y", "x", "yaw", "pitch", "roll"),
        axis_signs=(1, -1, 1, -1, 1, 1),
    )
    transform = SpaceMouseTransform(config)
    calibration = transform.calibrate([[0.1] * 6, [0.1] * 6])
    assert np.allclose(calibration["bias"], 0.1)
    corrected, command = transform.apply([1.1, -0.9, 0.6, 0.5, -0.3, 2.0])
    assert np.allclose(corrected, [1.0, -1.0, 0.5, 0.4, -0.4, 1.0])
    assert np.allclose(command, [0.25, 0.5, 0.5, -0.25, -0.1, 0.1])


def test_transform_gains_can_change_without_losing_calibration():
    config = dataclasses.replace(
        base_config(),
        motion_mode="combined",
        deadzone=0.0,
        smoothing_alpha=1.0,
        translation_gain=0.5,
        rotation_gain=0.5,
    )
    transform = SpaceMouseTransform(config)
    transform.bias = np.asarray([0.1, 0, 0, 0, 0, 0], dtype=np.float64)
    transform.set_gains(0.2, 0.1)
    corrected, command = transform.apply([0.6, 0, 0, 0.5, 0, 0])
    assert np.isclose(corrected[0], 0.5)
    assert np.isclose(command[1], -0.1)
    assert np.isclose(command[3], 0.05)
    assert transform.gains == (0.2, 0.1)


def test_default_mapping_converts_pyspacemouse_legacy_axes_to_ros():
    config = dataclasses.replace(
        base_config(),
        motion_mode="combined",
        deadzone=0.0,
        smoothing_alpha=1.0,
        translation_gain=1.0,
        rotation_gain=1.0,
    )
    transform = SpaceMouseTransform(config)
    transform.calibrate([[0.0] * 6, [0.0] * 6])
    _, command = transform.apply([0.1, 0.2, 0.3, 0.4, 0.5, 0.6])
    assert np.allclose(command, [0.2, -0.1, 0.3, 0.4, 0.5, -0.6])


def test_transform_deadzone_and_ema_are_stateful():
    config = dataclasses.replace(
        base_config(),
        deadzone=0.1,
        smoothing_alpha=0.5,
        translation_gain=1.0,
        rotation_gain=1.0,
        axis_order=("x", "y", "z", "roll", "pitch", "yaw"),
        axis_signs=(1, 1, 1, 1, 1, 1),
    )
    transform = SpaceMouseTransform(config)
    transform.calibrate([[0.0] * 6, [0.0] * 6])
    _, first = transform.apply([0.55, 0.05, 0, 0, 0, 0])
    _, second = transform.apply([0.55, 0.05, 0, 0, 0, 0])
    assert np.isclose(first[0], 0.25)
    assert np.isclose(second[0], 0.375)
    assert first[1] == second[1] == 0.0


@pytest.mark.parametrize("key,value,error", [
    ("motion_mode", "softmax", ValueError),
    ("motion_mode", "true", TypeError),
    ("intent_switch_ratio", "0.99", ValueError),
    ("intent_switch_ratio", "true", TypeError),
    ("intent_switch_ratio", ".nan", ValueError),
    ("intent_switch_ratio", ".inf", ValueError),
])
def test_intent_config_rejects_invalid_values(tmp_path, key, value, error):
    source = DEFAULT_SPACEMOUSE_CONFIG.read_text(encoding="utf-8")
    old = "exclusive" if key == "motion_mode" else "1.25"
    path = tmp_path / "config.yaml"
    path.write_text(source.replace(f"{key}: {old}", f"{key}: {value}"), encoding="utf-8")
    with pytest.raises(error, match=key):
        load_spacemouse_config(path)


def test_legacy_configs_default_to_exclusive_control(tmp_path):
    lines = DEFAULT_SPACEMOUSE_CONFIG.read_text(encoding="utf-8").splitlines()
    path = tmp_path / "config.yaml"
    path.write_text("\n".join(line for line in lines if not line.startswith(
        ("motion_mode:", "intent_switch_ratio:"))), encoding="utf-8")
    config = load_spacemouse_config(path)
    assert config.motion_mode == "exclusive"
    assert config.intent_switch_ratio == 1.25


def test_intent_hysteresis_does_not_mix_groups_or_lock_out_switches():
    selector = SpaceMouseIntent("exclusive", 1.25)

    def apply(translation, rotation):
        return selector.apply(np.array([translation, 0, 0, rotation, 0, 0]))

    assert not apply(.5, .5).any()  # No arbitrary winner for ambiguous idle input.
    assert not apply(.5, .4).any()  # Strict threshold.
    assert np.array_equal(apply(.6, .4), [.6, 0, 0, 0, 0, 0])
    for _ in range(10):
        assert np.array_equal(apply(.4, .49), [.4, 0, 0, 0, 0, 0])
        assert np.array_equal(apply(.49, .4), [.49, 0, 0, 0, 0, 0])
    assert np.array_equal(apply(.4, .6), [0, 0, 0, .6, 0, 0])
    assert np.array_equal(apply(.49, .4), [0, 0, 0, .4, 0, 0])
    assert np.array_equal(apply(.6, .4), [.6, 0, 0, 0, 0, 0])
    assert not apply(0, 0).any()
    assert selector.selected == "idle"
    assert not apply(.5, .5).any()


def test_group_norm_keeps_all_three_axes_and_ignores_output_gains():
    # Deliberately give rotation a much higher output gain than translation.
    transform = SpaceMouseTransform(dataclasses.replace(
        base_config(), deadzone=0, translation_gain=.05, rotation_gain=1,
    ))
    _, command = transform.apply([.4, -.4, .4, .3, .2, .1])
    assert transform.intent.selected == "translation"
    assert np.allclose(command[:3], [-.02, -.02, .02])
    assert not command[3:].any()
    transform.set_gains(1, .05)
    _, command = transform.apply([.4, -.4, .4, .3, .2, .1])
    assert transform.intent.selected == "translation"
    assert np.allclose(command[:3], [-.4, -.4, .4])
    transform.reset_filter()
    _, command = transform.apply([.1, .1, .1, -.4, .4, .4])
    assert transform.intent.selected == "rotation"
    assert not command[:3].any()
    assert np.allclose(command[3:], [-.02, .02, -.02])


def test_ema_never_leaks_losing_group_and_release_is_immediate():
    transform = SpaceMouseTransform(dataclasses.replace(
        base_config(), deadzone=0, smoothing_alpha=.25,
    ))
    for _ in range(10):
        _, command = transform.apply([.8, 0, 0, .1, 0, 0])
    assert command[:3].any()
    _, command = transform.apply([.1, 0, 0, .8, 0, 0])
    assert not command[:3].any()
    assert command[3:].any()
    _, command = transform.apply([0] * 6)
    assert not command.any()
    assert transform.intent.selected == "idle"


def test_exclusivity_and_bounds_hold_for_random_input_gain_changes_and_resets():
    rng = np.random.default_rng(42)
    transform = SpaceMouseTransform(dataclasses.replace(base_config(), smoothing_alpha=.3))
    selected = set()
    for index in range(2000):
        if index % 25 == 0:
            transform.reset_filter()
            transform.set_gains(*rng.uniform(.05, 1, size=2))
        _, command = transform.apply(rng.uniform(-1.5, 1.5, size=6))
        selected.add(transform.intent.selected)
        assert np.isfinite(command).all()
        assert np.abs(command).max() <= 1
        assert not (command[:3].any() and command[3:].any())
    assert selected == {"idle", "translation", "rotation"}


def test_neutral_calibration_rejects_motion():
    transform = SpaceMouseTransform(dataclasses.replace(base_config(), neutral_max_abs=0.1))
    with pytest.raises(RuntimeError, match="moved during neutral calibration"):
        transform.calibrate([[0.0] * 6, [0.0, 0.0, 0.2, 0.0, 0.0, 0.0]])


class FakeClock:
    def __init__(self):
        self.value = 10.0

    def __call__(self):
        return self.value

    def sleep(self, seconds):
        self.value += seconds


def test_neutral_calibration_accepts_a_connected_device_silent_at_rest():
    clock = FakeClock()
    config = dataclasses.replace(base_config(), neutral_calibration_seconds=0.01)
    controller = SpaceMouseInput(config, clock=clock, sleep=clock.sleep)
    controller._connected = True
    result = controller.calibrate_neutral()
    assert result["hid_reports_observed"] == 0
    assert result["used_initialized_neutral"] is True
    assert np.allclose(result["bias"], 0.0)


def test_stable_calibration_resets_instead_of_failing_when_cap_moves():
    clock = FakeClock()
    config = dataclasses.replace(
        base_config(),
        neutral_calibration_seconds=0.03,
        neutral_calibration_timeout_seconds=0.2,
        neutral_max_abs=0.15,
    )

    def sleep(seconds):
        clock.value += seconds
        if clock.value >= 10.02:
            controller._raw_axes.fill(0.0)

    controller = SpaceMouseInput(config, clock=clock, sleep=sleep)
    controller._connected = True
    controller._raw_axes[0] = 0.16
    result = controller.calibrate_until_stable()
    assert result["movement_resets"] > 0
    assert result["elapsed_seconds"] >= 0.03


def test_buttons_latch_gripper_and_stale_or_disconnect_zeroes_motion():
    clock = FakeClock()
    config = dataclasses.replace(
        base_config(),
        deadzone=0.0,
        translation_gain=1.0,
        rotation_gain=1.0,
        stale_timeout_ms=250,
        axis_order=("x", "y", "z", "roll", "pitch", "yaw"),
        axis_signs=(1, 1, 1, 1, 1, 1),
    )
    controller = SpaceMouseInput(config, clock=clock)
    controller._connected = True

    controller._accept_event(1.0, (0.5, 0, 0, 0, 0, 0), (0, 1))
    assert np.allclose(controller.latest_action(), [0.5, 0, 0, 0, 0, 0, 1])

    clock.value += 0.01
    controller._accept_event(2.0, (0.5, 0, 0, 0, 0, 0), (1, 1))
    assert controller.latest_action()[6] == -1.0

    clock.value += 0.3
    stale = controller.latest_snapshot()
    assert stale.stale
    assert np.allclose(stale.action[:6], 0.0)
    assert stale.action[6] == -1.0

    controller._connected = False
    disconnected = controller.latest_snapshot()
    assert not disconnected.connected
    assert np.allclose(disconnected.action[:6], 0.0)


def test_arm_stale_release_and_buttons_reset_intent_without_opening_gripper():
    clock = FakeClock()
    controller = SpaceMouseInput(dataclasses.replace(base_config(), smoothing_alpha=.3), clock=clock)
    controller._connected = True
    controller._accept_event(1, (.8, 0, 0, .1, 0, 0), (1, 0))
    assert controller.latest_snapshot().motion_intent == "translation"
    controller.reset_for_arm(gripper=1)
    armed = controller.latest_snapshot()
    assert armed.motion_intent == "idle"
    assert armed.action == (0., 0., 0., 0., 0., 0., 1.)
    # A button held before takeover must not retrigger on the next report.
    controller._accept_event(2, (.1, 0, 0, .8, 0, 0), (1, 0))
    assert controller.latest_action()[6] == 1
    assert controller.latest_snapshot().motion_intent == "rotation"
    clock.value += .25
    stale = controller.latest_snapshot()
    assert stale.stale and stale.motion_intent == "idle"
    assert stale.action == (0., 0., 0., 0., 0., 0., 1.)
    # Fresh but ambiguous input cannot inherit the old rotation latch.
    controller._accept_event(3, (.5, 0, 0, .5, 0, 0), (0, 0))
    assert not controller.latest_action()[:6].any()
    controller._accept_event(4, (0,) * 6, (1, 0))
    assert controller.latest_action()[6] == -1
    clock.value += 1  # Silent neutral firmware never repeats residual motion.
    assert not controller.latest_action()[:6].any()
    controller._accept_event(5, (.8, 0, 0, .1, 0, 0), (0, 0))
    assert controller.latest_snapshot().motion_intent == "translation"
    controller._connected = False
    assert controller.latest_snapshot().motion_intent == "idle"
    assert not controller.latest_action()[:6].any()


def test_diagnostic_event_window_is_bounded_but_total_count_is_preserved():
    clock = FakeClock()
    controller = SpaceMouseInput(base_config(), clock=clock)
    controller._connected = True
    for index in range(5000):
        clock.value += .001
        controller._accept_event(index + 1, (0,) * 6, (0, 0))
    diagnostics = controller.diagnostics()
    assert diagnostics["event_count"] == 5000
    assert diagnostics["event_window_count"] == 4096
    assert len(diagnostics["event_times"]) == 4096
    assert diagnostics["motion_mode"] == "exclusive"


def test_new_report_after_unsampled_stale_gap_starts_with_fresh_intent():
    clock = FakeClock()
    controller = SpaceMouseInput(dataclasses.replace(base_config(), smoothing_alpha=.5), clock=clock)
    controller._connected = True
    controller._accept_event(1, (1., 0, 0, 0, 0, 0), (0, 1))
    clock.value += .3  # No intervening latest_snapshot to detect staleness.
    controller._accept_event(2, (.2, 0, 0, .2, 0, 0), (0, 1))
    snapshot = controller.latest_snapshot()
    assert not snapshot.stale
    assert snapshot.motion_intent == "idle"
    assert snapshot.action == (0., 0., 0., 0., 0., 0., 1.)


class FakeDevice:
    def __init__(self):
        self.counter = 0
        self.closed = False
        self.info = SimpleNamespace(vendor_id=0x256F, product_id=0xC63A)
        self.name = "SpaceMouseWirelessNew"
        self.product_name = "Fake SpaceMouse"
        self.vendor_name = "3Dconnexion"

    def read(self):
        self.counter += 1
        return SimpleNamespace(
            t=float(self.counter),
            x=0.2,
            y=0.0,
            z=0.0,
            roll=0.0,
            pitch=0.0,
            yaw=0.0,
            buttons=[0, 0],
        )

    def close(self):
        self.closed = True


def test_threaded_reader_exposes_latest_state_without_waiting():
    device = FakeDevice()
    config = dataclasses.replace(base_config(), poll_interval_ms=0.5)
    controller = SpaceMouseInput(config, device_factory=lambda: device)
    controller.start()
    try:
        deadline = time.monotonic() + 0.2
        while controller.latest_snapshot().sequence < 2 and time.monotonic() < deadline:
            time.sleep(0.001)
        started = time.monotonic()
        snapshot = controller.latest_snapshot()
        elapsed = time.monotonic() - started
        assert snapshot.sequence >= 2
        assert elapsed < 0.01
        assert snapshot.connected
    finally:
        controller.stop()
    assert device.closed


def test_reader_error_disconnects_and_zeroes_motion():
    class FailingDevice(FakeDevice):
        def read(self):
            self.counter += 1
            if self.counter > 1:
                raise OSError("device removed")
            return SimpleNamespace(
                t=1.0,
                x=0.0,
                y=0.5,
                z=0.0,
                roll=0.0,
                pitch=0.0,
                yaw=0.0,
                buttons=[0, 1],
            )

    device = FailingDevice()
    controller = SpaceMouseInput(
        dataclasses.replace(base_config(), deadzone=0.0, translation_gain=1.0),
        device_factory=lambda: device,
    )
    controller.start()
    try:
        deadline = time.monotonic() + 0.2
        while controller.latest_snapshot().error is None and time.monotonic() < deadline:
            time.sleep(0.001)
        snapshot = controller.latest_snapshot()
        assert snapshot.error == "OSError: device removed"
        assert not snapshot.connected
        assert np.allclose(snapshot.action[:6], 0.0)
        assert snapshot.action[6] == 1.0
    finally:
        controller.stop()
