"""Random-access replay frames, decoded from each compressed recording only once.

``trajectory_observations.npz`` is a deflate archive: reading one frame
decompresses both complete camera arrays. Replay only needs the frames at
chunk boundaries, so each episode's boundary frames are written once as
uncompressed ``.npy`` files and memory-mapped afterwards. The cache key binds
the source content hash and the frame index set; source files are never
modified.
"""
from __future__ import annotations

import fcntl
import json
import os
import shutil
import tempfile
from collections import OrderedDict
from pathlib import Path
from typing import Iterable

import numpy as np

from .io import stable_hash
from .storage_paths import storage_path

CACHE_VERSION = 1
CAMERAS = ("agentview_image", "wrist_image")


class EpisodeFrames:
    """Frames of one episode addressed by their original recording index."""

    def __init__(self, rows: dict[int, int], arrays: dict[str, np.ndarray]):
        self.rows = rows
        self.arrays = arrays

    def frame(self, camera: str, index: int) -> np.ndarray:
        try:
            row = self.rows[int(index)]
        except KeyError as exc:
            raise KeyError(f"Frame {index} is not part of the replay frame cache") from exc
        return self.arrays[camera][row]


class FrameCache:
    """Per-dataset cache; opened episodes are kept in a small LRU of mmaps."""

    def __init__(self, directory: Path | None, *, max_open: int = 64):
        self.directory = None if directory is None else Path(directory)
        self.max_open = max_open
        self._open: OrderedDict[str, EpisodeFrames] = OrderedDict()

    @staticmethod
    def key(episode: dict, indices: Iterable[int]) -> str:
        ordered = sorted({int(value) for value in indices})
        # Size and mtime also invalidate the entry if a recording is rewritten
        # in place without re-running Prepare.
        stat = storage_path(episode["observations_path"]).stat()
        return stable_hash({"version": CACHE_VERSION, "observations_sha256": episode["observations_sha256"],
                            "size": stat.st_size, "mtime_ns": stat.st_mtime_ns, "frames": ordered})[:32]

    def episode(self, episode: dict, indices: Iterable[int]) -> EpisodeFrames:
        ordered = sorted({int(value) for value in indices})
        key = self.key(episode, ordered)
        cached = self._open.get(key)
        if cached is not None:
            self._open.move_to_end(key)
            return cached
        frames = self._load(key, ordered) if self.directory is not None else None
        if frames is None:
            frames = self._build(key, episode, ordered)
        self._open[key] = frames
        while len(self._open) > self.max_open:
            self._open.popitem(last=False)
        return frames

    def _load(self, key: str, ordered: list[int]) -> EpisodeFrames | None:
        folder = self.directory / key
        try:
            meta = json.loads((folder / "frames.json").read_text(encoding="utf-8"))
            if not isinstance(meta, dict) or meta.get("version") != CACHE_VERSION or meta.get("frames") != ordered:
                return None
            arrays = {name: np.load(folder / f"{name}.npy", mmap_mode="r", allow_pickle=False)
                      for name in CAMERAS}
        except (OSError, ValueError, EOFError):
            return None
        # Every camera must hold exactly the requested HxWx3 uint8 frames.
        shapes = {array.shape for array in arrays.values()}
        if (len(shapes) != 1 or any(array.dtype != np.uint8 for array in arrays.values())):
            return None
        shape = next(iter(shapes))
        if len(shape) != 4 or shape[0] != len(ordered) or shape[-1] != 3:
            return None
        return EpisodeFrames({index: row for row, index in enumerate(ordered)}, arrays)

    def _decode(self, episode: dict, ordered: list[int]) -> dict[str, np.ndarray]:
        selection = np.asarray(ordered, dtype=np.int64)
        with np.load(storage_path(episode["observations_path"]), allow_pickle=False) as source:
            return {name: np.ascontiguousarray(source[name][selection]) for name in CAMERAS}

    def _build(self, key: str, episode: dict, ordered: list[int]) -> EpisodeFrames:
        rows = {index: row for row, index in enumerate(ordered)}
        if self.directory is None:
            return EpisodeFrames(rows, self._decode(episode, ordered))
        self.directory.mkdir(parents=True, exist_ok=True)
        folder = self.directory / key
        # DataLoader workers and concurrent runs share this directory. Only the
        # holder of the per-key lock may judge an entry damaged and replace it;
        # otherwise one process could delete an entry another just published.
        with (self.directory / f".{key}.lock").open("a") as lock:
            fcntl.flock(lock.fileno(), fcntl.LOCK_EX)
            try:
                frames = self._load(key, ordered)
                if frames is not None:
                    return frames
                arrays = self._decode(episode, ordered)
                # A previous writer was interrupted or the files were damaged.
                shutil.rmtree(folder, ignore_errors=True)
                temporary = Path(tempfile.mkdtemp(prefix=f".{key}.", dir=self.directory))
                try:
                    for name, array in arrays.items():
                        np.save(temporary / f"{name}.npy", array, allow_pickle=False)
                    (temporary / "frames.json").write_text(json.dumps({
                        "version": CACHE_VERSION, "run_id": episode.get("run_id"),
                        "observations_sha256": episode["observations_sha256"], "frames": ordered,
                    }), encoding="utf-8")
                    os.replace(temporary, folder)
                except OSError:
                    # The disk is full or read-only; the decoded arrays remain correct.
                    pass
                finally:
                    shutil.rmtree(temporary, ignore_errors=True)
            finally:
                fcntl.flock(lock.fileno(), fcntl.LOCK_UN)
        return self._load(key, ordered) or EpisodeFrames(rows, arrays)
