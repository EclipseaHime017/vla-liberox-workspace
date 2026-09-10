from pathlib import Path

import pytest

from backend.app.devices.factr_runtime_config import parse_runtime_options


@pytest.mark.parametrize("options", [
    {"torque": True}, {"gravity_gain": 2}, {"gravity_gain": True},
    {"gravity_gain": float("nan")}, {"gravity_watchdog_ms": 250},
    {"gravity_max_cycle_ms": 100}, {"gravity_ramp_seconds": 0},
    {"gravity_current_limits_ma": [100]*8}, {"gravity_current_limits_ma": [1000]*7},
    {"gravity_current_limits_ma": [True]*7}, {"gravity_vector": [0, 0, 0]},
    {"gravity_vector": [0, float("inf"), -9.81]}, {"gravity_max_seconds": 1000},
    {"save_joint_diagnostics": "false"}, {"gravity_usb_latency_ms": 21},
    {"gravity_usb_latency_ms": 0}, {"gravity_usb_latency_ms": True},
    {"gravity_usb_latency_ms": 1.5},
    {"calibration_file": None}, [], "bad",
])
def test_invalid_runtime_options_rejected(options):
    with pytest.raises((ValueError, TypeError)):
        parse_runtime_options(options, Path("/tmp/configs"))


def test_calibration_relative_to_yaml_and_not_ui(tmp_path):
    options = parse_runtime_options({"calibration_file": "../saved/physical.json"}, tmp_path/"configs")
    assert options.calibration_file == str(tmp_path/"saved/physical.json")
    assert options.runtime_python == str(tmp_path/"third_party/factr-runtime/bin/python")
    assert options.upstream_root == str(tmp_path/"third_party/FACTR_Teleop")


def test_python_symlink_preserved(tmp_path):
    python = tmp_path/"python"
    python.symlink_to("/usr/bin/python3")
    assert parse_runtime_options({"runtime_python": str(python)}, tmp_path).runtime_python == str(python)
