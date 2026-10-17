"""Model image orientation, independent of recording and reward display formats."""
from __future__ import annotations

import copy
import numpy as np


ORIENTATIONS = {
    "none": (False, False),
    "flip_vertical": (True, False),
    "flip_horizontal": (False, True),
    "rotate_180": (True, True),
}
STORED_ORIENTATIONS = {"libero_raw": "none", "vla_policy": "rotate_180"}
CAMERA_KEYS = ("agentview_image", "robot0_eye_in_hand_image")
FLIP_KEYS = {"image_flip_vertical", "image_flip_horizontal"}


def validate_orientation(value: str) -> str:
    if not isinstance(value, str) or value not in ORIENTATIONS:
        raise ValueError(f"image_orientation must be one of {', '.join(ORIENTATIONS)}")
    return value


def image_orientation(observation: dict) -> str:
    """Resolve independent YAML flips; retain frozen contracts from older exports."""
    keys = ("image_flip_vertical", "image_flip_horizontal")
    if "image_orientation" in observation:
        if any(key in observation for key in keys):
            raise ValueError("Do not combine image_orientation with image_flip_* settings")
        return validate_orientation(observation["image_orientation"])
    if any(type(observation.get(key)) is not bool for key in keys):
        raise ValueError("image_flip_vertical and image_flip_horizontal must be booleans")
    flips = tuple(observation[key] for key in keys)
    return next(name for name, value in ORIENTATIONS.items() if value == flips)


def validate_observation(value: dict) -> dict:
    if not isinstance(value, dict) or set(value) != FLIP_KEYS:
        raise ValueError("Variant observation requires image_flip_vertical and image_flip_horizontal")
    image_orientation(value)
    return dict(value)


def observation_config(contract: dict) -> dict:
    """Read current observation settings or an unchanged legacy frozen contract."""
    if "observation" in contract:
        if (FLIP_KEYS | {"image_orientation"}) & contract["io"].keys():
            raise ValueError("Observation settings cannot also appear in io")
        return validate_observation(contract["observation"])
    vertical, horizontal = ORIENTATIONS[image_orientation(contract["io"])]
    return {"image_flip_vertical": vertical, "image_flip_horizontal": horizontal}


def canonical_contract(contract: dict) -> dict:
    value = copy.deepcopy(contract)
    value["observation"] = observation_config(value)
    for key in FLIP_KEYS | {"image_orientation"}:
        value["io"].pop(key, None)
    return value


def orient_image(frame: np.ndarray, target: str, *, source: str = "none") -> np.ndarray:
    """Convert HWC pixels between orientations relative to the raw framebuffer."""
    vertical, horizontal = ORIENTATIONS[validate_orientation(target)]
    source_vertical, source_horizontal = ORIENTATIONS[validate_orientation(source)]
    array = np.asarray(frame, dtype=np.uint8)
    if array.ndim != 3 or array.shape[-1] != 3:
        raise ValueError("Model camera image must be HxWx3")
    return array[::(-1 if vertical != source_vertical else 1),
                 ::(-1 if horizontal != source_horizontal else 1)].copy()


def policy_view(frame: np.ndarray, orientation: str, target: str = "rotate_180") -> np.ndarray:
    try:
        source = STORED_ORIENTATIONS[orientation]
    except KeyError as exc:
        raise ValueError(f"Unknown observation orientation: {orientation}") from exc
    return orient_image(frame, target, source=source)


def vla_adapter_observation(observation: dict, target: str) -> dict:
    """Compensate upstream's 180° flip using privately owned camera buffers."""
    validate_orientation(target)
    return {**observation, **{key: orient_image(observation[key], target, source="rotate_180")
                              for key in CAMERA_KEYS if key in observation}}
