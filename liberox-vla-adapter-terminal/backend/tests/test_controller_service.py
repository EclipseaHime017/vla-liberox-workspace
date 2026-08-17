from __future__ import annotations

import time

from backend.app.services.controller_service import SpaceMouseControllerService, latency_level
from backend.app.devices.spacemouse import SpaceMouseSnapshot, load_spacemouse_config


class FakeInput:
    def __init__(self, config):
        self.config = config
        self.started = False
        self.stopped = False
        self.gains = None
        self.reset_count = 0

    def start(self):
        self.started = True

    def stop(self):
        self.stopped = True

    def calibrate_until_stable(self, *, timeout_seconds, progress, cancelled):
        assert timeout_seconds == self.config.neutral_calibration_timeout_seconds
        assert not cancelled()
        progress(0.0, "检测到帽盖移动，请松开并保持静止", 1)
        progress(1.0, "保持帽盖静止，正在校准", 1)
        return {"bias": [0.0] * 6, "movement_resets": 1}

    def reset_for_arm(self, gripper=-1.0):
        assert gripper == -1.0
        self.reset_count += 1

    def set_gains(self, translation_gain, rotation_gain):
        self.gains = (translation_gain, rotation_gain)

    def latest_snapshot(self):
        return SpaceMouseSnapshot(
            sequence=4,
            captured_monotonic=1.02,
            event_monotonic=1.0,
            device_timestamp=3.0,
            raw_axes=(0.0,) * 6,
            corrected_axes=(0.0,) * 6,
            command_axes=(0.0,) * 6,
            action=(0.0,) * 6 + (-1.0,),
            buttons=(0, 0),
            connected=True,
            stale=False,
            error=None,
        )

    def diagnostics(self):
        return {"event_times": [1.0, 1.01]}


def test_controller_uses_exact_uncalibrated_state_and_reuses_calibration():
    config = load_spacemouse_config()
    created = []

    def factory(value):
        device = FakeInput(value)
        created.append(device)
        return device

    service = SpaceMouseControllerService(
        config,
        input_factory=factory,
        probe=lambda _config: {"connected": True, "error": None},
        start_monitor=False,
    )
    try:
        assert service.status()["state"] == "UNCALIBRATED"
        service.start_calibration()
        deadline = time.monotonic() + 1.0
        while service.status()["state"] == "CALIBRATING" and time.monotonic() < deadline:
            time.sleep(0.001)
        status = service.status()
        assert status["state"] == "READY"
        assert status["movement_resets"] == 1
        assert len(created) == 1

        service.arm("branch", 0.2, 0.1)
        assert service.status()["state"] == "ARMED"
        assert created[0].gains == (0.2, 0.1)
        assert service.snapshot("branch").connected
        service.disarm("branch")
        assert service.status()["state"] == "READY"
        assert len(created) == 1
    finally:
        service.close()


def test_latency_levels_match_ui_contract():
    assert latency_level(49.9, connected=True, stale=False, error=None) == "green"
    assert latency_level(50.0, connected=True, stale=False, error=None) == "yellow"
    assert latency_level(249.0, connected=True, stale=False, error=None) == "yellow"
    assert latency_level(250.0, connected=True, stale=False, error=None) == "red"
    assert latency_level(1.0, connected=False, stale=False, error=None) == "red"
    assert latency_level(1.0, connected=True, stale=True, error=None) == "red"
