from dataclasses import replace
import json
import math

import numpy as np
import pytest

from backend.app.devices.factr import load_factr_config
from backend.app.devices.factr_calibration import (
    gripper_fraction, load_profile, make_profile, save_profile,
)


@pytest.fixture
def config():
    return load_factr_config()


@pytest.fixture
def fingerprint():
    return {"usb_device": {"vendor_id": 0x0403, "product_id": 0x6014, "serial_number": "test-arm"},
            "model_numbers": [1220, 1030, 1220, 1030, 1220, 1220, 1220, 1220],
            "motor_ids": list(range(1, 9)), "homing_offsets": [0]*8, "drive_modes": [0]*8}


def test_profile_roundtrip_checksum_and_fingerprint(tmp_path, config, fingerprint):
    profile = make_profile(config, np.arange(7)*math.pi/2, 1.2, 0.5, fingerprint)
    path = tmp_path/"calibration.json"
    save_profile(path, profile)
    assert load_profile(path, config, fingerprint) == profile
    assert len(list(tmp_path.iterdir())) == 1
    with pytest.raises(ValueError, match="fingerprint changed"):
        load_profile(path, config, {**fingerprint, "homing_offsets": [1]*8})
    with pytest.raises(ValueError, match="configuration changed"):
        load_profile(path, replace(config, joint_signs=(-1,)*7), fingerprint)
    changed = json.loads(path.read_text())
    changed["profile"]["offsets"][0] = 1.0
    path.write_text(json.dumps(changed))
    with pytest.raises(ValueError, match="checksum"):
        load_profile(path, config, fingerprint)


def test_profile_gains_do_not_invalidate_calibration(tmp_path, config, fingerprint):
    profile = make_profile(config, [0.]*7, 1., 0.5, fingerprint)
    path = tmp_path/"calibration.json"
    save_profile(path, profile)
    assert load_profile(path, replace(config, translation_gain=.15), fingerprint) == profile


def test_usb_identity_prevents_loading_another_arm_calibration(tmp_path, config, fingerprint):
    path = tmp_path/"calibration.json"
    save_profile(path, make_profile(config, [0.]*7, 0., .5, fingerprint))
    other = {**fingerprint, "usb_device": {**fingerprint["usb_device"], "serial_number": "other-arm"}}
    with pytest.raises(ValueError, match="fingerprint changed"):
        load_profile(path, config, other)
    anonymous = {**fingerprint, "usb_device": {**fingerprint["usb_device"], "serial_number": None}}
    # Even the same USB location cannot identify a device without a serial number.
    save_profile(path, make_profile(config, [0.]*7, 0., .5, anonymous))
    with pytest.raises(ValueError, match="no USB serial number"):
        load_profile(path, config, anonymous)


def test_gripper_endpoints_wrap_and_validation(config, fingerprint):
    profile = make_profile(config, [0.]*7, .1, -.7, fingerprint)
    assert gripper_fraction(2*math.pi+.1, profile) == pytest.approx(0.)
    assert gripper_fraction(2*math.pi-.7, profile) == pytest.approx(1.)
    assert gripper_fraction(-.3, profile) == pytest.approx(.5)
    with pytest.raises(ValueError, match="Gripper travel"):
        make_profile(config, [0.]*7, 0., .001, fingerprint)
    with pytest.raises(ValueError, match="fingerprint"):
        make_profile(config, [0.]*7, 0., .5, {})


def test_duplicate_keys_rejected(tmp_path, config, fingerprint):
    path = tmp_path/"invalid.json"
    path.write_text('{"profile": {}, "profile": {}}')
    with pytest.raises(ValueError, match="Duplicate"):
        load_profile(path, config, fingerprint)


def test_old_calibration_method_is_rejected(tmp_path, config, fingerprint):
    profile = replace(make_profile(config, [0.]*7, 0., .5, fingerprint), schema_version=1)
    path = tmp_path/"old.json"
    save_profile(path, profile)
    with pytest.raises(ValueError, match="whole-arm reference calibration"):
        load_profile(path, config, fingerprint)
