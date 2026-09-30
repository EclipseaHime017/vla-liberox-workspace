from __future__ import annotations

import numpy as np
import pytest
import sys
import torch
import types
from types import SimpleNamespace

from vla_rynn_iql.vla_adapter import (
    dataset_to_env_actions,
    env_to_dataset_actions,
    extract_action_hidden_states,
    predict_normalized,
    processor_inputs,
)


class FakeBatch(dict):
    def to(self, device, dtype):
        for key, value in self.items():
            if torch.is_floating_point(value):
                self[key] = value.to(device=device, dtype=dtype)
            else:
                self[key] = value.to(device=device)
        return self


class FakeProcessor:
    def __call__(self, text, images, padding, return_tensors):
        assert padding is True
        assert return_tensors == "pt"
        assert isinstance(text, list) and isinstance(images, list)
        pixels = torch.stack([
            torch.from_numpy(np.asarray(image).copy()).permute(2, 0, 1).float()
            for image in images
        ])
        return FakeBatch({
            "input_ids": torch.tensor([[1, 2, 3]] * len(text), dtype=torch.int64),
            "attention_mask": torch.ones((len(text), 3), dtype=torch.int64),
            "pixel_values": pixels,
        })


class FakeVisionBackbone:
    def __call__(self, pixels):
        return torch.zeros((len(pixels), 4, 6), dtype=pixels.dtype)

    @staticmethod
    def get_num_patches():
        return 2

    @staticmethod
    def get_num_images_in_input():
        return 2


class FakeLanguageModel:
    def __call__(self, *, inputs_embeds, **_kwargs):
        return SimpleNamespace(hidden_states=(inputs_embeds, inputs_embeds + 1))


class FakePrismaticModel(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.device = torch.device("cpu")
        self.dtype = torch.float32
        self.vision_backbone = FakeVisionBackbone()
        self.projector = torch.nn.Identity()
        self.action_queries = torch.nn.Embedding(2, 6)
        self.language_model = FakeLanguageModel()
        self.embeddings = torch.nn.Embedding(16, 6)

    def _prepare_input_for_action_prediction(self, input_ids, attention_mask):
        extension = torch.ones((len(input_ids), 3), dtype=input_ids.dtype)
        return (
            torch.cat((input_ids, extension), dim=1),
            torch.cat((attention_mask, torch.ones_like(extension)), dim=1),
        )

    @staticmethod
    def _prepare_labels_for_action_prediction(labels, input_ids):
        extension = torch.full(
            (len(labels), input_ids.shape[1] - labels.shape[1]), 9,
            dtype=labels.dtype,
        )
        extension[:, -1] = 10
        return torch.cat((labels, extension), dim=1)

    @staticmethod
    def _process_action_masks(labels):
        return labels.eq(9)

    def get_input_embeddings(self):
        return self.embeddings

    def _process_vision_features(self, pixel_values, _language, _film):
        return self.projector(self.vision_backbone(pixel_values))

    @staticmethod
    def _replace_input_embeddings(input_embeddings, action_mask, queries):
        result = input_embeddings.clone()
        for batch_index in range(len(result)):
            result[batch_index, action_mask[batch_index]] = queries[batch_index]
        return result

    @staticmethod
    def _build_multimodal_attention(input_embeddings, patches, attention_mask):
        embeddings = torch.cat(
            (input_embeddings[:, :1], patches, input_embeddings[:, 1:]), dim=1,
        )
        patch_mask = torch.ones(
            (len(attention_mask), patches.shape[1]), dtype=attention_mask.dtype,
        )
        mask = torch.cat(
            (attention_mask[:, :1], patch_mask, attention_mask[:, 1:]), dim=1,
        )
        return embeddings, mask


class FakeActionHead:
    @staticmethod
    def predict_action(hidden, proprio, proprio_projector, phase):
        assert phase == "Inference"
        assert len(hidden) == len(proprio)
        assert proprio_projector is not None
        return torch.zeros((len(hidden), 8, 7), dtype=torch.float32)


def test_action_round_trip_preserves_motion_and_binary_gripper():
    stats = {
        "q01": [-1.0] * 6 + [0.0], "q99": [1.0] * 6 + [1.0],
        "mask": [True] * 6 + [False],
    }
    actions = np.asarray([[0.2, -0.3, 0.1, 0.0, 0.5, -0.2, -1.0]], dtype=np.float32)
    normalized = env_to_dataset_actions(actions, stats)
    restored = dataset_to_env_actions(normalized, stats)
    np.testing.assert_allclose(restored, actions, atol=1e-6)


def test_processor_inputs_batches_two_views_without_changing_sample_order():
    components = SimpleNamespace(
        processor=FakeProcessor(),
        model=SimpleNamespace(device=torch.device("cpu"), dtype=torch.float32),
    )
    agent = np.zeros((3, 4, 5, 3), dtype=np.uint8)
    wrist = np.stack([
        np.full((4, 5, 3), fill_value=index + 1, dtype=np.uint8)
        for index in range(3)
    ])
    inputs = processor_inputs(
        components, ["place the bowl"] * 3, agent, wrist,
    )
    assert inputs["input_ids"].shape == (3, 3)
    assert inputs["pixel_values"].shape == (3, 6, 4, 5)
    torch.testing.assert_close(inputs["pixel_values"][:, :3], torch.zeros((3, 3, 4, 5)))
    assert inputs["pixel_values"][:, 3:].mean(dim=(1, 2, 3)).tolist() == [1.0, 2.0, 3.0]


def test_processor_inputs_keeps_legacy_single_sample_and_rejects_mixed_prompts():
    components = SimpleNamespace(
        processor=FakeProcessor(),
        model=SimpleNamespace(device=torch.device("cpu"), dtype=torch.float32),
    )
    image = np.zeros((4, 5, 3), dtype=np.uint8)
    assert processor_inputs(components, "task", image, image)["input_ids"].shape[0] == 1
    with pytest.raises(ValueError, match="one task prompt"):
        processor_inputs(
            components,
            ["first task", "second task"],
            np.stack([image, image]),
            np.stack([image, image]),
        )


@pytest.mark.parametrize("trainable", [False, True])
def test_frozen_hidden_and_action_helpers_preserve_batch_dimension(monkeypatch, trainable):
    constants = types.ModuleType("prismatic.vla.constants")
    constants.IGNORE_INDEX = -100
    constants.NUM_TOKENS = 2
    constants.STOP_INDEX = 10
    prismatic = types.ModuleType("prismatic")
    prismatic.__path__ = []
    vla = types.ModuleType("prismatic.vla")
    vla.__path__ = []
    monkeypatch.setitem(sys.modules, "prismatic", prismatic)
    monkeypatch.setitem(sys.modules, "prismatic.vla", vla)
    monkeypatch.setitem(sys.modules, "prismatic.vla.constants", constants)

    components = SimpleNamespace(
        processor=FakeProcessor(),
        model=FakePrismaticModel().requires_grad_(trainable),
        action_head=FakeActionHead(),
        proprio_projector=object(),
    )
    images = np.zeros((3, 4, 5, 3), dtype=np.uint8)
    inputs = processor_inputs(components, ["task"] * 3, images, images)
    hidden = extract_action_hidden_states(components, inputs)
    output = predict_normalized(
        components, hidden, torch.zeros((3, 8), dtype=torch.bfloat16),
    )
    individual_hidden = torch.cat([
        extract_action_hidden_states(
            components,
            processor_inputs(components, "task", images[index], images[index]),
        )
        for index in range(3)
    ])
    assert hidden.shape == (3, 2, 6, 6)
    assert output.shape == (3, 8, 7)
    torch.testing.assert_close(hidden, individual_hidden)
    assert hidden.requires_grad is trainable
    if trainable:
        hidden.sum().backward()
        assert components.model.action_queries.weight.grad.abs().sum() > 0
