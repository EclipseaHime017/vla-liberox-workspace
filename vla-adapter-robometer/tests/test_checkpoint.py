from types import SimpleNamespace

from vla_adapter_robometer.checkpoint import remap_checkpoint_state_dict


def tensor(*shape):
    return SimpleNamespace(shape=shape)


def test_maps_self_contained_peft_prefix_without_extra_model_layer():
    checkpoint = {
        "model.model.language_model.layers.0.self_attn.q_proj.lora_A.default.weight": tensor(8, 16),
        "progress_head.0.weight": tensor(1, 16),
    }
    model = {
        "model.base_model.model.language_model.layers.0.self_attn.q_proj.lora_A.default.weight": tensor(8, 16),
        "progress_head.0.weight": tensor(1, 16),
    }
    remapped, mapping, unmatched = remap_checkpoint_state_dict(checkpoint, model)
    assert set(remapped) == set(model)
    assert mapping[next(iter(checkpoint))].startswith("model.base_model.model.language_model")
    assert unmatched == []


def test_suffix_mapping_requires_matching_shape():
    checkpoint = {"legacy.wrapper.layers.0.weight": tensor(2, 3)}
    model = {"new.wrapper.layers.0.weight": tensor(3, 2)}
    remapped, _, unmatched = remap_checkpoint_state_dict(checkpoint, model)
    assert remapped == {}
    assert unmatched == ["legacy.wrapper.layers.0.weight"]
