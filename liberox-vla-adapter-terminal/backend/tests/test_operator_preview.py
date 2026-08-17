from __future__ import annotations

import mujoco
import numpy as np

import eval_pickplace_direct as direct
from backend.app.simulators.libero_x import (
    OPERATOR_PREVIEW_CAMERAS,
    _look_at_quaternion,
    _oblique_camera_pose,
    compose_operator_preview,
)


def test_operator_preview_is_a_labelled_two_by_two_mosaic():
    frames = [np.full((32, 48, 3), value, dtype=np.uint8) for value in (20, 60, 100, 140)]
    mosaic = compose_operator_preview(frames, ("A", "B", "C", "D"))
    assert mosaic.shape == (64, 96, 3)
    assert mosaic.dtype == np.uint8
    assert int(mosaic[-1, -1, 0]) == 140


def test_camera_quaternion_looks_at_target():
    position = np.array([0.0, -1.2, 1.5])
    target = np.array([0.0, 0.0, 1.0])
    quaternion = _look_at_quaternion(position, target)
    matrix = np.empty(9, dtype=np.float64)
    mujoco.mju_quat2Mat(matrix, quaternion)
    actual_forward = -matrix.reshape(3, 3)[:, 2]
    expected_forward = target - position
    expected_forward /= np.linalg.norm(expected_forward)
    np.testing.assert_allclose(actual_forward, expected_forward, atol=1e-12)


def test_oblique_camera_positions_are_symmetric_and_distinct():
    reference_position = np.array([-0.05, 1.27, 1.49])
    reference_quaternion = np.array([0.0099, 0.0069, 0.5912, 0.8064])
    negative, negative_quaternion = _oblique_camera_pose(
        reference_position, reference_quaternion, -45.0
    )
    positive, positive_quaternion = _oblique_camera_pose(
        reference_position, reference_quaternion, 45.0
    )
    assert negative[1] < 0 < positive[1]
    np.testing.assert_allclose(negative[[0, 2]], positive[[0, 2]], atol=1e-12)
    assert not np.allclose(negative_quaternion, positive_quaternion)


def test_operator_only_cameras_do_not_change_vla_inputs():
    assert direct.VLA_OBSERVATION_CAMERAS == ("agentview", "robot0_eye_in_hand")
    assert [camera for camera, _ in OPERATOR_PREVIEW_CAMERAS] == [
        "agentview",
        "robot0_eye_in_hand",
        "oblique_minus_45",
        "oblique_plus_45",
    ]
