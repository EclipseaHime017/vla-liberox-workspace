from __future__ import annotations

from dataclasses import asdict, dataclass, fields
from pathlib import Path
import re

import yaml


class UniqueLoader(yaml.SafeLoader):
    pass


def _mapping(loader, node, deep=False):
    result = {}
    for key_node, value_node in node.value:
        key = loader.construct_object(key_node, deep=deep)
        if key in result:
            raise ValueError(f"Duplicate configuration key: {key}")
        result[key] = loader.construct_object(value_node, deep=deep)
    return result


UniqueLoader.add_constructor(yaml.resolver.BaseResolver.DEFAULT_MAPPING_TAG, _mapping)


@dataclass(frozen=True)
class Config:
    model_id: str = "Qwen/Qwen3-VL-4B-Instruct"
    revision: str = "ebb281ec70b05090aa6165b016eac8ec08e71b17"
    environment: str = "keyframe-vlm"
    cache_dir: str = "evaluator/keyframe-vlm"
    device: str = "cuda:0"
    local_files_only: bool = True
    seed: int = 7
    mode: str = "localize"
    cameras: tuple[str, ...] = ("agentview_image", "wrist_image")
    coarse_fps: float = 5.0
    max_coarse_samples: int = 600
    window_seconds: float = 2.0
    max_new_tokens: int = 1024
    repetition_penalty: float = 1.05
    image_max_pixels: int = 262144
    success_consecutive_steps: int = 5

    @classmethod
    def from_dict(cls, raw, base: Path):
        if not isinstance(raw, dict):
            raise ValueError("Annotation configuration must be a mapping")
        unknown = set(raw) - {field.name for field in fields(cls)}
        if unknown:
            names = ", ".join(sorted(repr(key) for key in unknown))
            raise ValueError(f"Unknown annotation configuration keys: {names}. "
                             "Check YAML fields; restart the UI/backend after updating annotation code.")
        values = {**asdict(cls()), **raw}
        if not isinstance(values["mode"], str) or values["mode"] not in {"plan_only", "localize"}:
            raise ValueError("mode must be plan_only or localize")
        if values["model_id"] != cls.model_id or not re.fullmatch(r"[a-f0-9]{40}", str(values["revision"])):
            raise ValueError("Use the official Qwen3-VL-4B-Instruct with a pinned commit revision")
        if not re.fullmatch(r"[A-Za-z0-9_-]+", str(values["environment"])):
            raise ValueError("Invalid Conda environment name")
        if not re.fullmatch(r"cuda:\d+", str(values["device"])):
            raise ValueError("Annotation requires an explicit CUDA device")
        if type(values["local_files_only"]) is not bool:
            raise ValueError("local_files_only must be boolean")
        for name, low, high in (("seed", 0, 2**32-1), ("max_coarse_samples", 8, 2000),
                ("max_new_tokens", 128, 4096), ("image_max_pixels", 3136, 1048576),
                ("success_consecutive_steps", 1, 100)):
            if type(values[name]) is not int or not low <= values[name] <= high:
                raise ValueError(f"Invalid {name}: expected integer in [{low}, {high}]")
        penalty = values["repetition_penalty"]
        if isinstance(penalty, bool) or not isinstance(penalty, (int, float)) or not 1 <= penalty <= 1.3:
            raise ValueError("repetition_penalty must be in [1, 1.3]")
        for name, low, high in (("coarse_fps", .25, 10.), ("window_seconds", .5, 4.)):
            value = values[name]
            if isinstance(value, bool) or not isinstance(value, (int, float)) or not low <= value <= high:
                raise ValueError(f"{name} must be in [{low}, {high}]")
        if values["coarse_fps"] * values["window_seconds"] < 2:
            raise ValueError("A window must contain at least two sampling intervals")
        cameras = values["cameras"]
        if not isinstance(cameras, (list, tuple)) or not cameras or len(set(cameras)) != len(cameras) or set(cameras) - {"agentview_image", "wrist_image"}:
            raise ValueError("cameras must select agentview_image and/or wrist_image")
        values["cameras"] = tuple(cameras)
        cache = Path(values["cache_dir"]).expanduser()
        values["cache_dir"] = str((base / cache).resolve() if not cache.is_absolute() else cache.resolve())
        return cls(**values)


def load_config(path: Path) -> Config:
    try:
        return Config.from_dict(yaml.load(path.read_text(), Loader=UniqueLoader), path.resolve().parent)
    except ValueError as exc:
        raise ValueError(f"{path.resolve()}: {exc}") from exc
