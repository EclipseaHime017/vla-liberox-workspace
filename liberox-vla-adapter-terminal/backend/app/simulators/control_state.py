"""Restore Panda controller targets alongside restored MuJoCo state."""
from __future__ import annotations

import numpy as np


def restore_control_state(env, env_actions, control_hz: float) -> float:
    base = getattr(env, "env", env)
    robot = base.robots[0]
    actions = np.asarray(env_actions, dtype=np.float64)
    if actions.shape == (0,):
        actions = np.empty((0, 7))
    if actions.ndim != 2 or actions.shape[1] != 7 or not np.isfinite(actions).all():
        raise ValueError("SpaceMouse takeover requires finite recorded seven-vector env actions")
    gripper = robot.gripper
    if type(gripper).__name__ != "PandaGripper":
        raise ValueError("SpaceMouse takeover state restoration requires PandaGripper")
    if not np.isfinite(control_hz) or control_hz <= 0:
        raise ValueError("SpaceMouse takeover requires a positive source control_hz")
    substeps = int((1. / control_hz) / base.model_timestep)
    if substeps < 1:
        raise ValueError("Source control_hz exceeds the simulation physics rate")

    controller = robot.controller
    controller.update(force=True)
    # Also re-anchor the nullspace posture. Neutral rotation otherwise keeps
    # OSC's pre-restore orientation goal, even with six zero input axes.
    controller.update_initial_joints(controller.joint_pos.copy())

    gripper.current_action = np.zeros(gripper.dof)
    for action in actions:
        for _ in range(substeps):
            gripper.format_action(action[6:7])
    robot.grip_action(gripper, np.zeros(gripper.dof))
    commands = actions[:, 6]
    nonzero = commands[commands != 0]
    return float(np.sign(nonzero[-1])) if nonzero.size else -1.
