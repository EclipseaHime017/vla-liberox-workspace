from types import SimpleNamespace
import threading

import numpy as np
import pytest

import eval_pickplace_direct as direct
from backend.app.policies.catalog import PolicyCatalog
from backend.app.policies.registry import model_module
from backend.app.policies.vla_adapter import VLAAdapterPolicyProvider


@pytest.mark.parametrize("orientation,strides", [
    ("none", (1, 1)), ("flip_vertical", (-1, 1)),
    ("flip_horizontal", (1, -1)), ("rotate_180", (-1, -1)),
])
def test_vla_provider_final_pixels_after_official_fixed_rotation(tmp_path, orientation, strides):
    raw = np.arange(45, dtype=np.uint8).reshape(3, 5, 3)
    obs = {"agentview_image": raw, "robot0_eye_in_hand_image": raw + 50}
    captured = {}

    def prepare(observation, size):
        # Official prepare_observation rotates both raw frames before resize.
        return {key: value[::-1, ::-1].copy() for key, value in observation.items()}, None

    def get_action(cfg, model, observation, prompt, **kwargs):
        captured.update(observation)
        return np.zeros((8, 7), np.float32)

    settings = model_module("models").base_model_config("base")
    settings["contract"] = model_module("base_models").model_contract(settings)
    settings["contract"]["io"].update(image_flip_vertical=strides[0] == -1, image_flip_horizontal=strides[1] == -1)
    provider = object.__new__(VLAAdapterPolicyProvider)
    provider._lock = threading.RLock()
    provider.runtime = SimpleNamespace(prepare_observation=prepare, get_action=get_action)
    provider.cfg = SimpleNamespace(use_film=False, use_minivlm=True)
    provider.components = SimpleNamespace(resize_size=224, model=None, processor=None,
        action_head=None, proprio_projector=None, noisy_action_projector=None)
    provider.current_policy_entry = PolicyCatalog(tmp_path, base_models={"base": settings}).entry("base")
    assert len(provider.predict(obs, "task")) == 8
    np.testing.assert_array_equal(captured["agentview_image"], raw[::strides[0], ::strides[1]])
    np.testing.assert_array_equal(captured["robot0_eye_in_hand_image"], (raw + 50)[::strides[0], ::strides[1]])
    np.testing.assert_array_equal(obs["agentview_image"], raw)
    provider.predict(obs, "task", ("robot0_eye_in_hand",))
    assert not captured["robot0_eye_in_hand_image"].any()


def test_default_direct_vla_input_is_passed_to_upstream_unchanged():
    obs = {"agentview_image": np.zeros((3, 5, 3), np.uint8)}
    seen = []
    runtime = SimpleNamespace(prepare_observation=lambda value, size: (seen.append(value) or {}, None),
                              get_action=lambda *args, **kwargs: np.zeros((8, 7)))
    components = SimpleNamespace(resize_size=224, model=None, processor=None,
                                 action_head=None, proprio_projector=None, noisy_action_projector=None)
    direct.checked_action_chunk(runtime, SimpleNamespace(use_film=False, use_minivlm=True), components, obs, "task")
    assert seen == [obs] and seen[0] is obs
