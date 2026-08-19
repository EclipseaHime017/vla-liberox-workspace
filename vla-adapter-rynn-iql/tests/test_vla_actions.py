from __future__ import annotations

import numpy as np

from vla_rynn_iql.vla_adapter import dataset_to_env_actions, env_to_dataset_actions


def test_action_round_trip_preserves_motion_and_binary_gripper():
    stats = {
        "q01": [-1.0] * 6 + [0.0], "q99": [1.0] * 6 + [1.0],
        "mask": [True] * 6 + [False],
    }
    actions = np.asarray([[0.2, -0.3, 0.1, 0.0, 0.5, -0.2, -1.0]], dtype=np.float32)
    normalized = env_to_dataset_actions(actions, stats)
    restored = dataset_to_env_actions(normalized, stats)
    np.testing.assert_allclose(restored, actions, atol=1e-6)
