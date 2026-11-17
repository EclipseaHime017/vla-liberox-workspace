"""Panda joint following and measured end-effector action encoding."""
import time
import numpy as np


class EndEffectorActionEncoder:
    """Inverse of OSC scaling, applied to measured world-frame pose increments.

    The label describes achieved motion, not the joint controller's input.
    Keep the unclipped label in raw_action and the bounded label in env_action.
    """
    def __init__(self, controller):
        if not getattr(controller, "use_delta", False) or controller.control_dim != 6:
            raise ValueError("FACTR recording requires a delta OSC_POSE reference")
        for name in ("input_min", "input_max", "output_min", "output_max"):
            value = np.broadcast_to(np.asarray(getattr(controller, name), dtype=float), (6,)).copy()
            if not np.isfinite(value).all():
                raise ValueError("Invalid OSC action scaling")
            setattr(self, name, value)
        if (np.any(self.input_max <= self.input_min)
                or np.any(self.output_max <= self.output_min)
                or not np.all(self.input_min == -1.) or not np.all(self.input_max == 1.)):
            raise ValueError("FACTR recording requires normalized OSC input limits [-1, 1]")
        self.count = self.clipped_steps = 0
        self.clipped_axes = np.zeros(6, dtype=int)
        self.max_abs = np.zeros(6)

    def encode(self, before, after, gripper):
        from scipy.spatial.transform import Rotation
        position, rotation = before
        next_position, next_rotation = after
        # robosuite set_goal_orientation: R_goal = R_delta @ R_current.
        delta = np.r_[np.asarray(next_position) - position,
                      Rotation.from_matrix(next_rotation @ rotation.T).as_rotvec()]
        if not np.isfinite(delta).all() or gripper not in (-1., 1.):
            raise ValueError("Invalid measured end-effector motion/gripper")
        scale = (self.output_max-self.output_min)/(self.input_max-self.input_min)
        unbounded = (delta-(self.output_max+self.output_min)/2.)/scale + (self.input_max+self.input_min)/2.
        clipped = (unbounded < self.input_min) | (unbounded > self.input_max)
        self.count += 1
        self.clipped_steps += int(clipped.any())
        self.clipped_axes += clipped
        self.max_abs = np.maximum(self.max_abs, np.abs(unbounded))
        return (np.r_[unbounded, gripper].astype(np.float32),
                np.r_[np.clip(unbounded, self.input_min, self.input_max), gripper].astype(np.float32))

    def diagnostics(self):
        return {"method": "measured_eef_delta_inverse_osc_scale", "sample_count": self.count,
                "clipped_steps": self.clipped_steps,
                "clipped_fraction": self.clipped_steps/max(1, self.count),
                "clipped_axes": self.clipped_axes.tolist(),
                "max_abs_unclipped": self.max_abs.tolist(),
                "output_min": self.output_min.tolist(), "output_max": self.output_max.tolist()}


class JointFollower:
    def __init__(self, env):
        from robosuite.controllers.joint_pos import JointPositionController
        from robosuite.utils.buffers import DeltaBuffer

        base = getattr(env, "env", env)
        self.robot, self.env = base.robots[0], env
        if len(base.robots) != 1 or len(self.robot._ref_joint_pos_indexes) != 7:
            raise ValueError("FACTR requires one seven-joint Panda")
        self.encoder = EndEffectorActionEncoder(self.robot.controller)
        self.eef_name = self.robot.controller.eef_name

        class AbsoluteJointController(JointPositionController):
            def set_goal(self, action, set_qpos=None):
                # These are radians, not normalized OSC or incremental joint actions.
                goal = np.asarray(action, dtype=float) if set_qpos is None else set_qpos
                return super().set_goal(np.zeros(7), set_qpos=goal)

        parameters = dict(self.robot.controller_config)
        parameters.update(input_min=-np.pi, input_max=np.pi, output_min=-np.pi,
                          output_max=np.pi, impedance_mode="fixed", kp=50.,
                          damping_ratio=1., interpolator=None)
        self.robot.controller = AbsoluteJointController(**parameters)
        base._action_dim = self.robot.action_dim
        self.robot.recent_actions = DeltaBuffer(dim=self.robot.action_dim)
        if base.action_dim != 8:
            raise ValueError("FACTR joint mode requires seven joint targets plus gripper")
        self.robot.controller.set_goal(self.joints())

    def joints(self):
        return np.asarray(self.robot.sim.data.qpos[self.robot._ref_joint_pos_indexes]).copy()

    def pose(self):
        # Fresh FK after the physics step, not the controller's last cached pose.
        self.robot.sim.forward()
        data = self.robot.sim.data
        return (data.get_site_xpos(self.eef_name).copy(),
                data.get_site_xmat(self.eef_name).copy())

    def check_aligned(self, joints, tolerance=.1):
        if np.max(np.abs(np.asarray(joints)-self.joints())) > tolerance:
            raise RuntimeError("Leader/follower no longer aligned; simulation remains paused")

    def step(self, joints, gripper):
        joints = np.asarray(joints, dtype=float)
        if joints.shape != (7,) or not np.isfinite(joints).all() or gripper not in (-1., 1.):
            raise ValueError("Invalid FACTR joint/gripper target")
        _, reward, done, info = self.env.step(np.r_[joints, gripper])
        # MuJoCo's final integration can leave observable caches one internal
        # physics tick behind qpos. Align recorded proprio and FK with the state
        # used later to reconstruct images; do not advance physics again.
        self.robot.sim.forward()
        base = getattr(self.env, "env", self.env)
        observation = base._get_observations(force_update=True)
        return observation, reward, done, info


def align_for_takeover(controller, session_id, target, stop, on_status=lambda _: None):
    """Hold simulation fixed; only the leader's explicitly requested preparation moves."""
    if stop():
        return False
    controller.start_alignment(session_id, target)
    try:
        while not stop():
            status = controller.status()
            on_status(status)
            if status.get("error") or not status.get("connected"):
                raise RuntimeError(status.get("error") or "FACTR disconnected during alignment")
            if not status.get("gravity_enabled"):
                return False  # Explicit OFF cancels preparation, not a controller fault.
            alignment = status.get("alignment") or {}
            if alignment.get("state") == "COMPLETE":
                # Keep the leader at target during the visible countdown.
                return True
            time.sleep(.025)
        return False
    except BaseException:
        controller.finish_alignment(session_id)
        raise
