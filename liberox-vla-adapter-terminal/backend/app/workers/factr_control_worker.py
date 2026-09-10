"""FACTR preparation and recorded control, using the shared session lifecycle."""
import time

from ..devices.factr_joint_control import align_for_takeover


def prepare_factr(manager, record, follower, controller):
    """Align the leader while physics is paused. Hold it through countdown."""
    record.preparation_phase = "aligning_controller"
    record.preparation_message = "实体主臂对齐仿真姿态"
    manager._persist_manifest(record)

    def check_preview(_status):
        if record.preview_error:
            raise RuntimeError(record.preview_error)

    return align_for_takeover(controller, record.id, follower.joints(),
                              record.stop_event.is_set, check_preview)


def run_factr_loop(*, record, controller, follower, recorder, initial_observation,
                   rate_limiter, on_transition):
    from simulation_core import ControlLoopResult
    observation = initial_observation
    success = bool(recorder.dones[-1]) if recorder.dones else False
    start_count = recorder.action_count
    times = []
    reason = "max_steps"
    while recorder.action_count < record.max_steps:
        if record.stop_event.is_set():
            reason = "user_stop"
            break
        rate_limiter.wait_before_step()
        if record.stop_event.is_set():
            reason = "user_stop"
            break
        sample = controller.snapshot(record.id)
        if record.preview_error:
            raise RuntimeError(record.preview_error)
        if not sample.connected or sample.stale or sample.error:
            raise RuntimeError(sample.error or "FACTR disconnected or sample is stale")
        record.controller_status, record.controller_connected = "armed", True
        record.controller_stale = False
        record.controller_latency_ms = sample.sample_age_seconds * 1000.
        before = follower.pose()
        times.append(time.monotonic())
        observation, reward, done, _ = follower.step(sample.joint_positions, sample.gripper_command)
        raw, action = follower.encoder.encode(before, follower.pose(), sample.gripper_command)
        recorder.record_transition(follower.env, observation, raw, action, reward, done,
                                   action_source="human")
        success = success or bool(done)
        on_transition(observation, recorder.action_count, success)
    return ControlLoopResult(success, recorder.action_count-start_count, 0, reason, times, observation)
