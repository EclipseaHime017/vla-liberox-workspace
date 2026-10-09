from contextlib import nullcontext
from types import SimpleNamespace

import numpy as np
import pytest

from keyframe_annotation.config import Config
from keyframe_annotation.runtime import QwenAnnotator


class Batch(dict):
    input_ids = np.array([[1, 2]])

    def to(self, device):
        self.device = device
        return self


class Processor:
    def apply_chat_template(self, messages, **kwargs):
        self.content = messages[0]["content"]
        return "template"

    def __call__(self, **kwargs):
        self.kwargs = kwargs
        self.batch = Batch(input_ids=Batch.input_ids)
        if "videos" in kwargs:
            self.batch.update(pixel_values_videos="native-video", video_grid_thw="temporal-grid")
        elif "images" in kwargs:
            self.batch.update(pixel_values="still-image", image_grid_thw="image-grid")
        return self.batch

    def decode(self, tokens, **kwargs):
        assert tokens.tolist() == [7, 8]
        return "output"


class Model:
    def generate(self, **kwargs):
        self.kwargs = kwargs
        return np.array([[1, 2, 7, 8]])


class Recording:
    hz, count = 20., 500

    def __init__(self):
        self.reads = []

    def image(self, camera, step):
        self.reads.append((camera, step))
        return np.full((32, 32, 3), step + (100 if camera == "wrist_image" else 0), dtype=np.uint8)


def annotator(cameras=("agentview_image", "wrist_image")):
    model = object.__new__(QwenAnnotator)
    model.config = Config(cameras=cameras)
    model.processor, model.model = Processor(), Model()
    model.torch = SimpleNamespace(inference_mode=nullcontext)
    return model


def test_contract_remains_text_only():
    model, recording = annotator(), Recording()
    assert model.generate("contract", recording, []) == "output"
    assert recording.reads == []
    assert model.processor.content == [{"type": "text", "text": "contract"}]
    assert not {"images", "videos", "videos_kwargs"} & model.processor.kwargs.keys()


def test_plan_remains_independent_first_frame_images():
    model, recording = annotator(), Recording()
    model.generate("plan", recording, [0])
    assert recording.reads == [("agentview_image", 0), ("wrist_image", 0)]
    assert [v["type"] for v in model.processor.content] == ["text", "image", "text", "image", "text"]
    assert len(model.processor.kwargs["images"]) == 2
    assert "videos" not in model.processor.kwargs
    assert model.model.kwargs["pixel_values"] == "still-image"


@pytest.mark.parametrize("cameras", [("agentview_image",), ("agentview_image", "wrist_image")])
@pytest.mark.parametrize("steps", [list(range(40, 81, 4)), [40, 44], [40, 44, 48, 49]])
def test_localization_native_video_preserves_camera_and_time(cameras, steps):
    model, recording = annotator(cameras), Recording()
    original = list(steps)
    assert model.generate("same judgment prompt", recording, steps) == "output"
    kwargs = model.processor.kwargs
    assert "images" not in kwargs and "images_kwargs" not in kwargs
    assert [v["type"] for v in model.processor.content] == ["text", "video"] * len(cameras) + ["text"]
    assert model.processor.content[-1]["text"] == "same judgment prompt"
    videos = kwargs["videos"]
    options = kwargs["videos_kwargs"]
    assert options["do_sample_frames"] is False
    assert options["input_data_format"] == "channels_last"
    assert options["size"] == dict(shortest_edge=len(steps)*262144, longest_edge=len(steps)*262144)
    for camera, video, metadata in zip(cameras, videos, options["video_metadata"]):
        assert video.shape == (len(steps), 32, 32, 3) and video.dtype == np.uint8
        assert video[:, 0, 0, 0].tolist() == [step + (100 if camera == "wrist_image" else 0) for step in steps]
        assert metadata == {"total_num_frames": 501, "fps": 20., "frames_indices": original}
        assert metadata["frames_indices"] is not steps
    # Native odd-frame timestamp padding must not leak into the next camera or saved steps.
    options["video_metadata"][0]["frames_indices"].append(steps[-1])
    assert steps == original
    if len(cameras) > 1:
        assert options["video_metadata"][1]["frames_indices"] == original
    assert model.model.kwargs["pixel_values_videos"] == "native-video"
    assert model.model.kwargs["video_grid_thw"] == "temporal-grid"
    assert "pixel_values" not in model.model.kwargs
    assert model.processor.batch.device == model.config.device
    assert model.model.kwargs["do_sample"] is False
    assert model.model.kwargs["repetition_penalty"] == 1.05
