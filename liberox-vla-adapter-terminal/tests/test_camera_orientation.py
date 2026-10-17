from types import SimpleNamespace
from dataclasses import replace
import ast
import base64
import copy
import io
from pathlib import Path
import threading

import numpy as np
import pytest
import torch

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
    settings["contract"]["observation"].update(image_flip_vertical=strides[0] == -1, image_flip_horizontal=strides[1] == -1)
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


def test_default_direct_vla_pixels_are_unchanged_but_owned_by_the_provider():
    obs = {"agentview_image": np.arange(45, dtype=np.uint8).reshape(3, 5, 3)}
    seen = []
    runtime = SimpleNamespace(prepare_observation=lambda value, size: (seen.append(value) or {}, None),
                              get_action=lambda *args, **kwargs: np.zeros((8, 7)))
    components = SimpleNamespace(resize_size=224, model=None, processor=None,
                                 action_head=None, proprio_projector=None, noisy_action_projector=None)
    direct.checked_action_chunk(runtime, SimpleNamespace(use_film=False, use_minivlm=True), components, obs, "task")
    np.testing.assert_array_equal(seen[0]["agentview_image"], obs["agentview_image"])
    assert seen[0] is not obs
    assert not np.shares_memory(seen[0]["agentview_image"], obs["agentview_image"])


def test_upstream_preprocessing_cannot_mutate_shared_environment_images():
    raw = np.arange(45, dtype=np.uint8).reshape(3, 5, 3)
    obs = {"agentview_image": raw.copy(), "robot0_eye_in_hand_image": raw.copy() + 50}
    before = {name: value.copy() for name, value in obs.items()}

    def prepare(value, size):
        for name in value:
            value[name][:] = 0
        return value, None

    runtime = SimpleNamespace(prepare_observation=prepare, get_action=lambda *args, **kwargs: np.zeros((8, 7)))
    components = SimpleNamespace(resize_size=224, model=None, processor=None,
                                 action_head=None, proprio_projector=None, noisy_action_projector=None)
    direct.checked_action_chunk(runtime, SimpleNamespace(use_film=False, use_minivlm=True), components, obs, "task")
    for name in before:
        np.testing.assert_array_equal(obs[name], before[name])


def test_actual_upstream_camera_helpers_match_each_selected_orientation():
    workspace = Path(__file__).resolve().parents[2]
    root = direct.load_config(workspace / "configs/config.yaml").vla_root
    sources = ((root / "experiments/robot/libero/libero_utils.py", {"get_libero_image", "get_libero_wrist_image"}),
               (root / "experiments/robot/libero/run_libero_eval.py", {"prepare_observation"}))
    # Execute the installed helper bodies without importing their GPU/TF dependencies.
    scope = {"np": np, "resize_image_for_policy": lambda image, size: image.copy(),
             "quat2axisangle": lambda value: np.zeros(3)}
    for path, names in sources:
        if not path.is_file():
            pytest.skip("VLA-Adapter checkout is not installed")
        functions = [node for node in ast.parse(path.read_text()).body
                     if isinstance(node, ast.FunctionDef) and node.name in names]
        assert {node.name for node in functions} == names
        exec(compile(ast.Module(body=functions, type_ignores=[]), str(path), "exec"), scope)
    frames = {"agentview_image": np.arange(45, dtype=np.uint8).reshape(3, 5, 3),
              "robot0_eye_in_hand_image": np.arange(45, dtype=np.uint8).reshape(3, 5, 3) + 50,
              "robot0_eef_pos": [0, 0, 0], "robot0_eef_quat": [0, 0, 0, 1], "robot0_gripper_qpos": [0, 0]}
    transforms = model_module("image_transforms")
    for target, (vertical, horizontal) in transforms.ORIENTATIONS.items():
        prepared, _ = scope["prepare_observation"](transforms.vla_adapter_observation(frames, target), 224)
        rows, columns = -1 if vertical else 1, -1 if horizontal else 1
        np.testing.assert_array_equal(prepared["full_image"], frames["agentview_image"][::rows, ::columns])
        np.testing.assert_array_equal(prepared["wrist_image"], frames["robot0_eye_in_hand_image"][::rows, ::columns])


def test_cross_family_switches_keep_each_variants_pixels_and_identity(tmp_path, monkeypatch):
    from backend.app.policies import pi05, vla_adapter
    from backend.app.policies.router import PolicyProvider

    registry, models = model_module("base_models"), model_module("models")
    variants, configurations = {}, {}
    for family in registry.MODEL_FAMILIES.values():
        if family.adapter not in ("vla_adapter", "pi05"):
            continue
        originals = [base for base in registry.BASE_MODELS.values() if base.family == family.id]
        variants[family.adapter] = []
        for index, original in enumerate(originals):
            identifier = f"{family.id}-fixture-{index}"
            contract = copy.deepcopy(original.contract)
            contract["observation"].update(image_flip_vertical=True,
                image_flip_horizontal=family.adapter == "vla_adapter" or index == 0)
            base = replace(original, id=identifier, checkpoint=f"fixture/{identifier}", contract=contract)
            monkeypatch.setitem(registry.BASE_MODELS, identifier, base)
            configurations[identifier] = models.base_model_config(identifier)
            variants[family.adapter].append(identifier)
    catalog = PolicyCatalog(tmp_path, base_models=configurations,
                            base_revisions={identifier: "a" * 64 for identifier in configurations})
    # Weight loading/RPC are replaced; real selection, provider cache and pixel paths run on CPU.
    monkeypatch.setattr(catalog, "select", catalog.entry)
    monkeypatch.setattr(vla_adapter.direct, "load_policy_runtime", lambda _: None)
    monkeypatch.setattr(vla_adapter, "replace", lambda value, **kwargs: SimpleNamespace(**kwargs))
    monkeypatch.setattr(vla_adapter, "checkpoint_view",
                        lambda entry: SimpleNamespace(name=entry.base_checkpoint, cleanup=lambda: None))
    captures = {}

    def build(runtime, config):
        return SimpleNamespace(num_open_loop_steps=config.open_loop_steps, use_film=False, use_minivlm=True), SimpleNamespace(
            resize_size=224, model=torch.nn.Linear(1, 1), action_head=torch.nn.Linear(1, 1),
            proprio_projector=torch.nn.Linear(1, 1), processor=None, noisy_action_projector=None)

    def prepare(observation, size):
        return {name: value[::-1, ::-1].copy() for name, value in observation.items()
                if name.endswith("_image")}, None

    def get_action(config, model, observation, prompt, **kwargs):
        captures["vla_adapter"] = observation
        return np.zeros((8, 7), np.float32)

    def send(provider, request, timeout=120):
        if "model" in request:
            return {"identity": request["content_sha256"]}
        with np.load(io.BytesIO(base64.b64decode(request["arrays"]))) as arrays:
            captures["pi05"] = {"agentview_image": arrays["observation/image"],
                                "robot0_eye_in_hand_image": arrays["observation/wrist_image"]}
        horizon = registry.model_contract(provider.current_policy_entry.settings)["io"]["native_action_horizon"]
        return {"actions": np.zeros((horizon, 7)).tolist()}

    monkeypatch.setattr(vla_adapter.direct, "build_model", build)
    monkeypatch.setattr(pi05.Pi05PolicyProvider, "_send", send)
    monkeypatch.setattr(pi05.subprocess, "Popen", lambda *args, **kwargs:
        SimpleNamespace(pid=1, stdin=io.StringIO(), stdout=io.StringIO(), poll=lambda: None, wait=lambda **kwargs: None))
    monkeypatch.setattr(pi05.os, "killpg", lambda *args: None)
    provider = PolicyProvider(SimpleNamespace(torch=torch, prepare_observation=prepare, get_action=get_action), None, catalog)
    raw = np.arange(45, dtype=np.uint8).reshape(3, 5, 3)
    observation = {"agentview_image": raw.copy(), "robot0_eye_in_hand_image": raw.copy() + 50,
                   "robot0_eef_pos": [0, 0, 0], "robot0_eef_quat": [0, 0, 0, 1], "robot0_gripper_qpos": [0, 0]}
    identities = {identifier: catalog.entry(identifier).content_sha256 for identifier in configurations}
    try:
        for vla_id in variants["vla_adapter"]:
            for pi_id in variants["pi05"]:
                for identifier in (vla_id, pi_id, pi_id, vla_id, vla_id):
                    entry = catalog.entry(identifier)
                    provider.load(None, identifier, expected_content_sha256=identities[identifier])
                    provider.predict(observation, "task")
                    settings = registry.model_observation(entry.settings)
                    horizontal = -1 if settings["image_flip_horizontal"] else 1
                    adapter = registry.MODEL_FAMILIES[entry.family].adapter
                    np.testing.assert_array_equal(captures[adapter]["agentview_image"], raw[::-1, ::horizontal])
                    np.testing.assert_array_equal(captures[adapter]["robot0_eye_in_hand_image"], (raw + 50)[::-1, ::horizontal])
                    assert provider.current_policy_id == identifier
                    assert provider.current_policy_entry.stats_key == entry.stats_key
                    assert provider.metadata()["variant_id"] == identifier
                    assert provider.metadata()["observation"] == settings
                    assert provider.metadata()["policy_content_sha256"] == identities[identifier]
        np.testing.assert_array_equal(observation["agentview_image"], raw)
        np.testing.assert_array_equal(observation["robot0_eye_in_hand_image"], raw + 50)
        assert {identifier: catalog.entry(identifier).content_sha256 for identifier in configurations} == identities
    finally:
        provider.unload()


@pytest.mark.parametrize("mismatch", ["family", "policy_id", "content", "observation"])
def test_router_refuses_a_provider_that_loaded_a_different_model(tmp_path, monkeypatch, mismatch):
    from backend.app.policies import router
    registry, models = model_module("base_models"), model_module("models")
    identifier = next(base.id for base in registry.BASE_MODELS.values() if registry.MODEL_FAMILIES[base.family].adapter == "vla_adapter")
    catalog = PolicyCatalog(tmp_path, base_models={identifier: models.base_model_config(identifier)})
    entry = catalog.entry(identifier)
    if mismatch == "observation":
        contract = copy.deepcopy(entry.contract)
        contract["observation"]["image_flip_horizontal"] = not contract["observation"]["image_flip_horizontal"]
        wrong = replace(entry, contract=contract)
    else:
        wrong = replace(entry, **{"family": {"family": "pi05"}, "policy_id": {"policy_id": "other"},
                                 "content": {"base_revision": "b" * 64}}[mismatch])
    events = []
    stub = SimpleNamespace(loaded=True, current_policy_entry=wrong, current_policy_id=wrong.policy_id,
                           load=lambda *args, **kwargs: None, unload=lambda: events.append("unloaded"))
    monkeypatch.setattr(router, "provider_factory", lambda family: lambda *args: stub)
    provider = router.PolicyProvider(None, None, catalog)
    with pytest.raises(ValueError, match="does not match the selected policy"):
        provider.load(None, identifier)
    assert events == ["unloaded"]
