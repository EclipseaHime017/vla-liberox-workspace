from dataclasses import replace
from types import SimpleNamespace

import numpy as np
import pytest
from scipy.spatial.transform import Rotation

from backend.app.devices.spacemouse import SpaceMouseInput, load_spacemouse_config
from backend.app.devices.spacemouse_motion import SpaceMouseActionMapper


def mapper_for(rotation, scale=(.05, .05, .05, .5, .5, .5)):
    controller = SimpleNamespace(use_delta=True, control_dim=6, input_min=-1, input_max=1,
        output_min=-np.array(scale), output_max=np.array(scale), eef_name="grip_site")
    state = SimpleNamespace(rotation=rotation, forwards=0)
    def forward():
        state.forwards += 1
    def site_rotation(name):
        assert name == "grip_site"
        return state.rotation
    robot = SimpleNamespace(controller=controller, sim=SimpleNamespace(forward=forward,
        data=SimpleNamespace(get_site_xmat=site_rotation)))
    return SpaceMouseActionMapper(SimpleNamespace(env=SimpleNamespace(robots=[robot]))), state


def test_world_is_unchanged_and_identity_tool_has_same_action():
    mapper, state = mapper_for(np.eye(3))
    action = np.array([.1, -.2, .3, -.4, .5, -.6, 1], dtype=np.float32)
    assert np.array_equal(mapper.convert(action, "world"), action)
    assert state.forwards == 0
    assert np.allclose(mapper.convert(action, "tool"), action)
    assert state.forwards == 1
    assert np.array_equal(mapper.convert([0, 0, 0, 0, 0, 0, -1], "tool"), [0, 0, 0, 0, 0, 0, -1])
    assert state.forwards == 1  # Neutral/stale doesn't invoke physics.


@pytest.mark.parametrize("axis", range(6))
def test_tool_axes_use_current_orientation_and_real_osc_scales(axis):
    rotation = Rotation.from_euler("xyz", [35, -42, 90], degrees=True).as_matrix()
    scales = np.array([.02, .05, .08, .2, .4, .6])
    mapper, state = mapper_for(rotation, scales)
    action = np.zeros(7)
    action[axis], action[6] = .1, -1
    out = mapper.convert(action, "tool")
    group = slice(0, 3) if axis < 3 else slice(3, 6)
    assert np.allclose(out[:3] * scales[:3], rotation @ (action[:3] * scales[:3]), atol=1e-9)
    assert np.allclose(out[3:6] * scales[3:], rotation @ (action[3:6] * scales[3:]), atol=1e-9)
    assert out[6] == -1
    assert not mapper.limited
    # Local rotation is a right multiplication; world OSC is left multiplication.
    assert np.allclose(Rotation.from_rotvec(out[3:6] * scales[3:]).as_matrix() @ rotation,
                       rotation @ Rotation.from_rotvec(action[3:6] * scales[3:]).as_matrix())
    state.rotation = np.eye(3)
    assert np.allclose(mapper.convert(action, "tool")[group], action[group])
    assert state.forwards == 2


def test_rotated_input_limits_preserve_direction_and_exclusivity():
    mapper, _ = mapper_for(Rotation.from_euler("z", 45, degrees=True).as_matrix())
    output = mapper.convert([1, 1, 0, 0, 0, 0, 1], "tool")
    assert mapper.limited
    assert np.allclose(output, [0, 1, 0, 0, 0, 0, 1], atol=1e-7)
    assert np.max(np.abs(output)) <= 1
    for mode in ("world", "tool"):
        with pytest.raises(ValueError, match="finite seven-vector"):
            mapper.convert([float("nan")] * 7, mode)
    with pytest.raises(ValueError, match="control_frame"):
        mapper.convert(np.zeros(7), "camera")


def test_frame_switch_requires_real_neutral_not_ambiguous_or_stale_zero():
    clock = [10.]
    mouse = SpaceMouseInput(replace(load_spacemouse_config(), smoothing_alpha=.5), clock=lambda: clock[0])
    mouse._connected = True
    mouse.reset_for_arm(gripper=1)
    mouse._accept_event(1, (.5, 0, 0, .5, 0, 0), (0, 0))
    assert not any(mouse.latest_snapshot().command_axes)  # Ambiguous, not neutral!
    mouse.request_control_frame("tool")
    snapshot = mouse.latest_snapshot()
    assert snapshot.control_frame == "world" and snapshot.pending_control_frame == "tool"
    clock[0] += .3
    assert mouse.latest_snapshot().stale
    assert mouse.latest_snapshot().control_frame == "world"
    mouse._accept_event(2, (0,) * 6, (0, 0))
    snapshot = mouse.latest_snapshot()
    assert snapshot.control_frame == "tool" and snapshot.pending_control_frame is None
    assert snapshot.motion_intent == "idle"
    assert snapshot.action == (0., 0., 0., 0., 0., 0., 1.)
    mouse._accept_event(3, (.8, 0, 0, .1, 0, 0), (0, 0))
    assert mouse.latest_snapshot().motion_intent == "translation"
    mouse.request_control_frame("world")
    assert mouse.latest_snapshot().control_frame == "tool"
    mouse.request_control_frame("tool")  # Cancel a pending request without resetting input.
    assert mouse.latest_snapshot().pending_control_frame is None
    assert any(mouse.latest_snapshot().command_axes)
    mouse.reset_for_arm(gripper=1, control_frame="world")
    assert mouse.latest_snapshot().control_frame == "world"
    assert not any(mouse.latest_snapshot().command_axes)


def test_frame_switch_accepts_observed_silent_neutral_and_keeps_calibration():
    clock = [10.]
    mouse = SpaceMouseInput(load_spacemouse_config(), clock=lambda: clock[0])
    mouse._connected = True
    mouse.transform.bias[:] = .02
    mouse._accept_event(1, (.02,) * 6, (0, 1))
    clock[0] += 1  # Neutral devices need not keep sending reports.
    mouse.request_control_frame("tool")
    snapshot = mouse.latest_snapshot()
    assert snapshot.control_frame == "tool" and snapshot.action[-1] == 1
    assert np.all(mouse.transform.bias == .02)


def test_config_frame_is_strict_and_defaults_world_for_old_files(tmp_path):
    from backend.app.devices.spacemouse import DEFAULT_SPACEMOUSE_CONFIG
    source = DEFAULT_SPACEMOUSE_CONFIG.read_text()
    path = tmp_path / "mouse.yaml"
    for invalid in ("camera", "true", "null"):
        path.write_text(source.replace("control_frame: world", f"control_frame: {invalid}"))
        with pytest.raises((TypeError, ValueError), match="control_frame"):
            load_spacemouse_config(path)
    path.write_text(source.replace("control_frame: world", ""))
    assert load_spacemouse_config(path).control_frame == "world"
