from __future__ import annotations

import hashlib
import json
import os
from pathlib import Path

import numpy as np
from PIL import Image


def atomic_json(path: Path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    with temporary.open("w", encoding="utf-8") as stream:
        json.dump(value, stream, ensure_ascii=False, indent=2, allow_nan=False)
        stream.flush()
        os.fsync(stream.fileno())
    temporary.replace(path)


def sha256(path: Path):
    value = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            value.update(block)
    return value.hexdigest()


class Recording:
    def __init__(self, source, config):
        self.source = source
        trajectory, observations = Path(source["trajectory_path"]), Path(source["observations_path"])
        self.hashes = {"trajectory": sha256(trajectory), "observations": sha256(observations)}
        if source.get("manifest_path"):
            self.hashes["manifest"] = sha256(Path(source["manifest_path"]))
        with np.load(trajectory, allow_pickle=False) as archive:
            metadata = json.loads(str(archive["metadata_json"].item())) if "metadata_json" in archive else {}
            self.hz = float(metadata.get("control_hz") or source.get("control_hz") or 0)
            if not np.isfinite(self.hz) or self.hz <= 0:
                raise ValueError("Missing or invalid recorded control_hz")
            if source.get("control_hz") is not None and not np.isclose(self.hz, source["control_hz"]):
                raise ValueError("Source and recorded control_hz disagree")
            if metadata.get("task_id", source["task_id"]) != source["task_id"]:
                raise ValueError("Source and recorded task identity disagree")
            self.done = archive["done"]
            if self.done.ndim != 1 or self.done.dtype.kind != "b" or not len(self.done):
                raise ValueError("Expected nonempty primitive-step boolean done array")
            self.count = len(self.done)
            self.times = np.arange(self.count + 1) / self.hz
            if "time_seconds" in archive and (archive["time_seconds"].shape != self.times.shape or
                    not np.allclose(archive["time_seconds"], self.times, atol=1e-5)):
                raise ValueError("Observation timeline is inconsistent with recorded control_hz")
        self.images = {}
        with np.load(observations, allow_pickle=False) as archive:
            for camera in config.cameras:
                images = archive[camera]
                if images.ndim != 4 or images.shape[0] != self.count + 1 or images.shape[-1] != 3 or images.dtype != np.uint8:
                    raise ValueError(f"{camera} must contain N+1 uint8 RGB observations")
                self.images[camera] = images
        self.success_step = None
        streak = 0
        for i, success in enumerate(self.done):
            streak = streak + 1 if success else 0
            if streak >= config.success_consecutive_steps:
                self.success_step = i + 1
                break

    def image(self, camera, step):
        pixels = self.images[camera][step]
        orientation = self.source.get("orientation", "libero_raw")
        if orientation == "libero_raw":
            pixels = pixels[::-1]
        elif orientation == "vla_policy":
            pixels = pixels[:, ::-1]
        elif orientation != "upright":
            raise ValueError(f"Unknown observation orientation: {orientation}")
        return Image.fromarray(np.ascontiguousarray(pixels))

    def coarse_steps(self, config):
        if config.coarse_fps > self.hz:
            raise ValueError("coarse_fps cannot exceed the recorded control_hz")
        ticks = np.arange(int(np.floor(self.count / self.hz * config.coarse_fps)) + 1)
        steps = sorted(set(np.rint(ticks * self.hz / config.coarse_fps).astype(int).tolist() + [self.count]))
        if len(steps) > config.max_coarse_samples:
            raise ValueError(f"Coarse timeline needs {len(steps)} samples; raise max_coarse_samples or lower coarse_fps")
        return steps

    def evidence(self, root: Path, steps):
        for step in steps:
            for camera in self.images:
                path = root / f"{camera}_{step}.jpg"
                if not path.exists():
                    path.parent.mkdir(parents=True, exist_ok=True)
                    self.image(camera, step).save(path, quality=90)
