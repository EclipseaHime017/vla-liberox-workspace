import numpy as np
import pytest

from vla_rynn_iql.image_transforms import orient_image, policy_view, vla_adapter_observation
from vla_rynn_iql.rewards import reward_view


MODES = [("none", (1, 1)), ("flip_vertical", (-1, 1)),
         ("flip_horizontal", (1, -1)), ("rotate_180", (-1, -1))]


@pytest.mark.parametrize("target,strides", MODES)
@pytest.mark.parametrize("source,source_strides", MODES)
def test_orientation_conversion_is_exact_contiguous_and_non_mutating(target, strides, source, source_strides):
    raw = np.arange(45, dtype=np.uint8).reshape(3, 5, 3)
    stored = raw[::source_strides[0], ::source_strides[1]].copy()
    before = stored.copy()
    result = orient_image(stored, target, source=source)
    np.testing.assert_array_equal(result, raw[::strides[0], ::strides[1]])
    np.testing.assert_array_equal(stored, before)
    assert result.flags.c_contiguous
    assert not np.shares_memory(result, stored)


@pytest.mark.parametrize("target,strides", MODES)
def test_legacy_storage_and_official_vla_flip_produce_the_same_model_pixels(target, strides):
    raw = np.arange(45, dtype=np.uint8).reshape(3, 5, 3)
    expected = raw[::strides[0], ::strides[1]]
    np.testing.assert_array_equal(policy_view(raw, "libero_raw", target), expected)
    np.testing.assert_array_equal(policy_view(raw[::-1, ::-1], "vla_policy", target), expected)
    obs = {"agentview_image": raw, "robot0_eye_in_hand_image": raw + 50, "state": object()}
    compensated = vla_adapter_observation(obs, target)
    for key in ("agentview_image", "robot0_eye_in_hand_image"):
        np.testing.assert_array_equal(compensated[key][::-1, ::-1], obs[key][::strides[0], ::strides[1]])
    assert compensated["state"] is obs["state"]
    if target == "rotate_180":
        assert compensated is obs  # Default official preprocessing is untouched.
    np.testing.assert_array_equal(reward_view(raw, "libero_raw"), raw[::-1])


@pytest.mark.parametrize("value", ["mirror", "", None, True, [], 180])
def test_invalid_model_orientation_is_rejected(value):
    with pytest.raises(ValueError, match="image_orientation"):
        orient_image(np.zeros((3, 5, 3), np.uint8), value)


def test_invalid_storage_and_images_are_rejected():
    with pytest.raises(ValueError, match="observation orientation"):
        policy_view(np.zeros((3, 5, 3), np.uint8), "unknown")
    with pytest.raises(ValueError, match="HxWx3"):
        orient_image(np.zeros((3, 5), np.uint8), "none")
