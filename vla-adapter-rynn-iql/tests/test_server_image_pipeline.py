from pathlib import Path
import sys
from types import SimpleNamespace

import numpy as np
import pytest
from PIL import Image

from vla_rynn_iql import vla_adapter
from vla_rynn_iql.upstream_environment import verify_vla_adapter_environment


@pytest.mark.parametrize("crop", [False, True])
def test_policy_pipeline_uses_official_resize_then_optional_crop(monkeypatch, crop):
    calls = []

    def resize(image, size):
        assert image.flags.c_contiguous and size == 224
        calls.append(("resize", int(image[0, 0, 0])))
        return image

    def center_crop(image):
        calls.append(("crop", int(np.asarray(image)[0, 0, 0])))
        return image

    monkeypatch.setitem(sys.modules, "experiments.robot.openvla_utils", SimpleNamespace(
        resize_image_for_policy=resize, center_crop_image=center_crop))
    monkeypatch.setitem(sys.modules, "experiments.robot.robot_utils", SimpleNamespace(get_image_resize_size=lambda _: 224))
    monkeypatch.setattr(vla_adapter, "restrict_tensorflow_to_cpu", lambda: None)
    images = np.stack([np.full((4, 5, 3), i, np.uint8) for i in (1, 2)])[:, ::-1]
    output = vla_adapter.policy_image_pipeline(SimpleNamespace(center_crop=crop))(images)
    assert all(isinstance(image, Image.Image) and image.mode == "RGB" for image in output)
    assert [int(np.asarray(image)[0, 0, 0]) for image in output] == [1, 2]
    assert calls == ([('resize', 1), ('crop', 1), ('resize', 2), ('crop', 2)] if crop
                     else [('resize', 1), ('resize', 2)])


@pytest.mark.parametrize("initialized,visible", [(False, []), (True, []), (True, ["GPU:0"])])
def test_tensorflow_cannot_keep_a_visible_gpu(monkeypatch, initialized, visible):
    calls = []

    def hide(devices, kind):
        calls.append((devices, kind))
        if initialized:
            raise RuntimeError("initialized")

    monkeypatch.setitem(sys.modules, "tensorflow", SimpleNamespace(config=SimpleNamespace(
        set_visible_devices=hide, get_visible_devices=lambda _: visible)))
    if initialized and visible:
        with pytest.raises(RuntimeError, match="restart the training worker"):
            vla_adapter.restrict_tensorflow_to_cpu()
    else:
        vla_adapter.restrict_tensorflow_to_cpu()
    assert calls == [([], "GPU")]


@pytest.fixture
def official_preprocessing(monkeypatch):
    project = Path(__file__).resolve().parents[1]
    root = project.parent / "third_party/VLA-Adapter"
    if not (root / "experiments/robot/openvla_utils.py").is_file():
        pytest.skip("VLA-Adapter checkout is unavailable")
    verify_vla_adapter_environment(root)
    vla_adapter.restrict_tensorflow_to_cpu()
    monkeypatch.syspath_prepend(str(root))
    from experiments.robot.openvla_utils import prepare_images_for_vla, resize_image_for_policy
    from experiments.robot.robot_utils import get_image_resize_size
    return prepare_images_for_vla, resize_image_for_policy, get_image_resize_size


@pytest.mark.parametrize("source_size", [224, 256])
@pytest.mark.parametrize("orientation", ["flip_vertical", "rotate_180"])
def test_training_images_match_the_official_inference_preprocessing(official_preprocessing,
                                                                     source_size, orientation):
    from vla_rynn_iql.image_transforms import policy_view
    prepare_images, resize, get_size = official_preprocessing
    cfg = SimpleNamespace(model_family="openvla", center_crop=True)
    raw = np.random.default_rng(0).integers(0, 256, (source_size, source_size, 3), dtype=np.uint8)
    oriented = raw[::-1, ::-1] if orientation == "rotate_180" else raw[::-1]
    expected = prepare_images([resize(np.ascontiguousarray(oriented), get_size(cfg))], cfg)[0]
    actual = vla_adapter.policy_image_pipeline(cfg)([policy_view(raw, "libero_raw", orientation)])[0]
    assert actual.size == expected.size == (224, 224)
    np.testing.assert_array_equal(np.asarray(actual), np.asarray(expected))
