"""SpaceMouse takeover and Cartesian frames; independent of UI, HID and policy."""
from __future__ import annotations

import numpy as np


def validate_control_frame(frame: str) -> str:
    if frame not in ("world", "tool"):
        raise ValueError("SpaceMouse control_frame must be world or tool")
    return frame


class SpaceMouseActionMapper:
    """Convert local tool increments to the environment's normalized world OSC.

    Rotation inputs are rotation vectors, not Euler angles. Both groups act at
    the same EEF origin; no camera offset or spatial-twist lever arm is applied.
    The control environment is read only at its own control boundary.
    """

    def __init__(self, env):
        base = getattr(env, "env", env)
        self.env = base
        self.robot = base.robots[0]
        controller = self.robot.controller
        if not controller.use_delta or controller.control_dim != 6:
            raise ValueError("SpaceMouse frame mapping requires delta OSC_POSE")
        limits = [np.broadcast_to(np.asarray(getattr(controller, name), dtype=float), (6,)).copy()
                  for name in ("input_min", "input_max", "output_min", "output_max")]
        lower, upper, out_lower, out_upper = limits
        if (not np.isfinite(limits).all() or not np.all(lower == -1)
                or not np.all(upper == 1) or not np.all(out_upper > 0)
                or not np.allclose(out_lower, -out_upper, rtol=0, atol=1e-12)):
            raise ValueError("SpaceMouse requires normalized, zero-centered OSC limits")
        self.scale = out_upper
        self.eef_name = controller.eef_name
        self.limited = False

    def prepare_takeover(self, env_actions, *, control_hz: float) -> float:
        """Align controller state without advancing or changing the restored physics.

        Recorded sim_state contains time/qpos/qvel, not OSC goals or the Panda
        gripper's accumulator. Replaying only its formatter preserves closing
        pressure, unlike deriving the target from finger positions in contact.
        """
        actions = np.asarray(env_actions, dtype=np.float64)
        if actions.shape == (0,):
            actions = np.empty((0, 7))
        if actions.ndim != 2 or actions.shape[1] != 7 or not np.isfinite(actions).all():
            raise ValueError("SpaceMouse takeover requires finite recorded seven-vector env actions")
        gripper = self.robot.gripper
        if type(gripper).__name__ != "PandaGripper":
            raise ValueError("SpaceMouse takeover state restoration requires PandaGripper")
        if not np.isfinite(control_hz) or control_hz <= 0:
            raise ValueError("SpaceMouse takeover requires a positive source control_hz")
        substeps = int((1. / control_hz) / self.env.model_timestep)
        if substeps < 1:
            raise ValueError("Source control_hz exceeds the simulation physics rate")

        controller = self.robot.controller
        controller.update(force=True)
        # Also re-anchor the nullspace posture. Neutral rotation otherwise keeps
        # OSC's pre-restore orientation goal, even with six zero input axes.
        controller.update_initial_joints(controller.joint_pos.copy())

        gripper.current_action = np.zeros(gripper.dof)
        for action in actions:
            for _ in range(substeps):
                gripper.format_action(action[6:7])
        self.robot.grip_action(gripper, np.zeros(gripper.dof))
        commands = actions[:, 6]
        nonzero = commands[commands != 0]
        return float(np.sign(nonzero[-1])) if nonzero.size else -1.

    def convert(self, action, frame: str) -> np.ndarray:
        validate_control_frame(frame)
        command = np.asarray(action, dtype=np.float64)
        if command.shape != (7,) or not np.isfinite(command).all():
            raise ValueError("SpaceMouse action must be a finite seven-vector")
        self.limited = False
        if frame == "world" or not command[:6].any():
            return command.astype(np.float32)
        # Final integration may leave site caches one physics tick behind qpos.
        # Forward updates kinematics, not simulation time or joint positions.
        self.robot.sim.forward()
        rotation = np.asarray(self.robot.sim.data.get_site_xmat(self.eef_name)).reshape(3, 3)
        if not np.isfinite(rotation).all():
            raise ValueError("Invalid tool orientation")
        physical = command[:6] * self.scale
        world = np.r_[rotation @ physical[:3], rotation @ physical[3:]] / self.scale
        # Rotating a diagonal normalized input can exceed a world-axis limit.
        # Scale each whole group down uniformly to preserve its direction.
        for group in (world[:3], world[3:]):
            limit = max(1., float(np.abs(group).max()))
            self.limited |= limit > 1.
            group /= limit
        return np.r_[world, command[6]].astype(np.float32)
