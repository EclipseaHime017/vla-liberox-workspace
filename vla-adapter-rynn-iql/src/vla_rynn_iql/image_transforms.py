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


def validate_orientation(value: str) -> str:
    if not isinstance(value, str) or value not in ORIENTATIONS:
        raise ValueError(f"image_orientation must be one of {', '.join(ORIENTATIONS)}")
    return value


def image_orientation(io: dict) -> str:
    """Resolve independent YAML flips; retain frozen contracts from older exports."""
    keys = ("image_flip_vertical", "image_flip_horizontal")
    if "image_orientation" in io:
        if any(key in io for key in keys):
            raise ValueError("Do not combine image_orientation with image_flip_* settings")
        return validate_orientation(io["image_orientation"])
    if any(type(io.get(key)) is not bool for key in keys):
        raise ValueError("image_flip_vertical and image_flip_horizontal must be booleans")
    flips = tuple(io[key] for key in keys)
    return next(name for name, value in ORIENTATIONS.items() if value == flips)


def canonical_contract(contract: dict) -> dict:
    value = copy.deepcopy(contract)
    vertical, horizontal = ORIENTATIONS[image_orientation(value["io"])]
    value["io"].pop("image_orientation", None)
    value["io"].update(image_flip_vertical=vertical, image_flip_horizontal=horizontal)
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
    """Compensate for the official VLA-Adapter prepare function's fixed 180° flip."""
    validate_orientation(target)
    if target == "rotate_180":
        return observation
    return {**observation, **{key: orient_image(observation[key], target, source="rotate_180")
                              for key in CAMERA_KEYS}}
