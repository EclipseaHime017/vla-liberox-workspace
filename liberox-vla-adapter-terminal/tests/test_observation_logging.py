import logging

import numpy as np
import pytest

from eval_pickplace_direct import validate_observation


def test_observation_validation_does_not_log_per_frame_info(caplog):
    obs = {"agentview_image": np.zeros((8, 8, 3)),
           "robot0_eye_in_hand_image": np.zeros((8, 8, 3)),
           "robot0_eef_pos": np.zeros(3), "robot0_eef_quat": np.array([0, 0, 0, 1]),
           "robot0_gripper_qpos": np.zeros(2)}
    with caplog.at_level(logging.INFO, logger="liberox_vla_adapter"):
        validate_observation(obs)
        validate_observation(obs)
    assert not caplog.records
    del obs["agentview_image"]
    with pytest.raises(KeyError, match="missing keys"):
        validate_observation(obs)
