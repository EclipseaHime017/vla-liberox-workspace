from pathlib import Path
import sys
from types import ModuleType, SimpleNamespace

import numpy as np
import pytest

from vla_rynn_iql.base_models import registered_defaults
from vla_rynn_iql.config import LoadedConfig
from vla_rynn_iql import vla_adapter, upstream_environment


def install_module(monkeypatch, name, **items):
    module = ModuleType(name)
    module.__dict__.update(items)
    monkeypatch.setitem(sys.modules, name, module)


def observation():
    image = np.arange(18, dtype=np.uint8).reshape(2, 3, 3)
    return {"agentview_image": image, "robot0_eye_in_hand_image": image + 20,
            "robot0_eef_pos": np.ones(3), "robot0_eef_quat": np.array([0, 0, 0, 1]),
            "robot0_gripper_qpos": np.zeros(2)}


@pytest.mark.parametrize("base,axes", [("base", (0, 1)), ("vla-adapter-pro-liberox-base", (0,))])
def test_vla_inference_compensates_upstream_flip_once(monkeypatch, base, axes):
    settings = registered_defaults("vla_adapter", base)
    config = LoadedConfig(Path("fixture"), {"model": settings, "paths": {"vla_adapter_root": "fixture"}})
    cfg = SimpleNamespace(model_family="openvla", use_film=False, use_minivlm=True)
    components = SimpleNamespace(cfg=cfg, model=None, processor=None, action_head=None, proprio_projector=None)
    monkeypatch.setattr(upstream_environment, "verify_vla_adapter_environment", lambda _: None)
    monkeypatch.setattr(vla_adapter, "load_components", lambda *a, **kw: components)
    captured = []
    def prepare(obs, _):
        # Official LIBERO helpers rotate both raw cameras by 180 degrees.
        return {k: v[::-1, ::-1].copy() for k, v in obs.items() if k.endswith("_image")}, None
    def predict(cfg, model, obs, prompt, **kwargs):
        captured.append(obs)
        return np.zeros((8, 7))
    install_module(monkeypatch, "experiments.robot.libero.run_libero_eval", prepare_observation=prepare,
                   process_action=lambda a, family: np.r_[a[:6], -np.sign(2 * a[6] - 1)])
    install_module(monkeypatch, "experiments.robot.robot_utils", get_action=predict, get_image_resize_size=lambda _: 224)
    predict, action = vla_adapter.evaluation_policy(config, Path("fixture/policy.yaml"), None)
    obs = observation()
    original = obs["agentview_image"].copy()
    assert np.asarray(predict(obs, "task")).shape == (8, 7)
    np.testing.assert_array_equal(captured[0]["agentview_image"], np.flip(original, axes))
    np.testing.assert_array_equal(obs["agentview_image"], original)
    assert action(np.r_[np.zeros(6), 0.])[-1] == 1
    assert action(np.r_[np.zeros(6), 1.])[-1] == -1


@pytest.mark.parametrize("base,axes", [("pi05-libero-base", (0, 1)), ("pi05-liberox-base", (0,))])
def test_pi05_preserves_native_actions_and_variant_images(monkeypatch, base, axes):
    from vla_rynn_iql import pi05
    settings = registered_defaults("pi05", base)
    config = LoadedConfig(Path("fixture"), {"model": settings})
    components = SimpleNamespace(model=None, input_transform=None, output_transform=None)
    monkeypatch.setattr(pi05, "load_components", lambda *a, **kw: components)
    restored = []
    monkeypatch.setattr(pi05, "restore_actor", lambda *args: restored.append(args))
    captured = []
    class Policy:
        def __init__(self, *args, **kwargs):
            assert kwargs["sample_kwargs"]["num_steps"] == settings["num_inference_steps"]
        def infer(self, obs):
            captured.append(obs)
            return {"actions": np.zeros((10, 7))}
    install_module(monkeypatch, "openpi.policies.policy", Policy=Policy)
    predict, action = pi05.evaluation_policy(config, Path("fixture/policy.yaml"), None)
    obs = observation()
    predict(obs, "task")
    np.testing.assert_array_equal(captured[0]["observation/image"], np.flip(obs["agentview_image"], axes))
    np.testing.assert_array_equal(captured[0]["observation/wrist_image"], np.flip(obs["robot0_eye_in_hand_image"], axes))
    np.testing.assert_array_equal(captured[0]["observation/state"], [1, 1, 1, 0, 0, 0, 0, 0])
    assert restored and captured[0]["prompt"] == "task"
    assert action(np.r_[np.zeros(6), -.7])[-1] == -.7
