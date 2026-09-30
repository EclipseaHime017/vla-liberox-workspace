"""Read-only mmap samples assembled by the same replay implementation as the UI."""
from __future__ import annotations

import fcntl
import json
import os
import shutil
import tempfile
import time
from functools import lru_cache
from pathlib import Path

import numpy as np
import torch
from numpy.lib.format import open_memmap
from torch.utils.data import Dataset, Sampler

from .io import atomic_json, sha256_file, stable_hash
from .methods import training_method
from .replay import ActionDataset, ReplayDataset
from .rewards import reward_manifest_digest


class _EpisodeCache:
    # Only one trajectory resident during cache construction; never all videos.
    @lru_cache(maxsize=1)
    def _trajectory(self, path):
        return super()._trajectory(path)

    @lru_cache(maxsize=1)
    def _observations(self, path):
        return super()._observations(path)


class _CachedActions(_EpisodeCache, ActionDataset):
    pass


class _CachedTransitions(_EpisodeCache, ReplayDataset):
    pass


def validate_cache(directory: Path, fingerprint: str | None = None, *, hashes: bool = True) -> dict:
    metadata = json.loads((directory / "cache.json").read_text())
    if metadata.get("schema_version") != 1 or (fingerprint and metadata["fingerprint"] != fingerprint):
        raise ValueError("Server replay cache identity changed")
    for name, description in metadata["arrays"].items():
        if not name.isidentifier():
            raise ValueError("Invalid cache field")
        path = directory / f"{name}.npy"
        if path.is_symlink() or (hashes and sha256_file(path) != description["sha256"]):
            raise ValueError(f"Corrupted server replay cache: {name}")
        array = np.load(path, mmap_mode="r", allow_pickle=False)
        if list(array.shape) != description["shape"] or str(array.dtype) != description["dtype"]:
            raise ValueError(f"Server replay shape mismatch: {name}")
        if len(array) != metadata["count"]:
            raise ValueError("Server replay count mismatch")
    if len(metadata["records"]) != metadata["count"]:
        raise ValueError("Server replay metadata count mismatch")
    return metadata


def build_replay_cache(config, root: Path, action_stats: dict, proprio_stats: dict,
                       reward_index: dict | None, *, rebuild: bool = False) -> tuple[Path, bool]:
    transitions = training_method(config.raw).requires_transitions
    source = (_CachedTransitions(config, action_stats, proprio_stats, reward_index=reward_index)
              if transitions else _CachedActions(config, action_stats, proprio_stats))
    identity = {
        "schema_version": 1, "dataset": source.manifest["dataset_sha256"],
        "reward": reward_manifest_digest(reward_index) if reward_index else None,
        "action_stats": action_stats, "proprio_stats": proprio_stats,
        "include_post_success": source.include_post_success, "transitions": transitions,
        "image_size": config.section("iql")["critic_image_size"] if transitions else None,
        "implementation": {name: sha256_file(Path(__file__).with_name(name)) for name in
                           ("server_replay.py", "replay.py", "data.py", "vla_adapter.py")},
    }
    # Source bytes, not timestamps or the original PC's absolute paths, bind the cache.
    for episode in source.manifest["episodes"]:
        for name in ("trajectory", "observations"):
            if sha256_file(Path(episode[f"{name}_path"])) != episode[f"{name}_sha256"]:
                raise ValueError(f"Source {name} changed: {episode['run_id']}")
    fingerprint = stable_hash(identity)
    root.mkdir(parents=True, exist_ok=True)
    target = root / fingerprint
    with (root / f".{fingerprint}.lock").open("a") as lock:
        fcntl.flock(lock, fcntl.LOCK_EX)
        if target.is_dir() and not rebuild:
            try:
                validate_cache(target, fingerprint)
                return target, True
            except (ValueError, OSError, KeyError):
                pass
        temporary = Path(tempfile.mkdtemp(prefix=f".{fingerprint}.", dir=root))
        arrays, records = {}, []
        started = time.monotonic()
        try:
            for index in range(len(source)):
                sample = source[index]
                record = {}
                for name, value in sample.items():
                    if not isinstance(value, torch.Tensor):
                        record[name] = value
                        continue
                    value = value.cpu().numpy()
                    if index == 0:
                        arrays[name] = open_memmap(temporary / f"{name}.npy", mode="w+",
                            dtype=value.dtype, shape=(len(source), *value.shape))
                    if arrays[name].shape[1:] != value.shape:
                        raise ValueError(f"Server replay requires consistent {name} shapes")
                    arrays[name][index] = value
                records.append(record)
                if index == 0 or (index + 1) % 128 == 0 or index + 1 == len(source):
                    elapsed = time.monotonic() - started
                    eta = elapsed * (len(source) - index - 1) / (index + 1)
                    print(f"CACHE {index + 1}/{len(source)} chunks | elapsed={elapsed:.1f}s | ETA={eta:.1f}s", flush=True)
            descriptions = {}
            for name, array in arrays.items():
                array.flush()
                descriptions[name] = {"shape": list(array.shape), "dtype": str(array.dtype),
                                       "sha256": sha256_file(temporary / f"{name}.npy")}
            metadata = {"schema_version": 1, "fingerprint": fingerprint, "identity": identity,
                        "count": len(source), "arrays": descriptions, "records": records}
            atomic_json(temporary / "cache.json", metadata)
            validate_cache(temporary, fingerprint)
            if target.exists():
                # Open mmap readers retain their old inode; construction is serialized.
                retired = root / f".{fingerprint}.retired"
                if retired.exists():
                    shutil.rmtree(retired)
                target.rename(retired)
                os.replace(temporary, target)
                shutil.rmtree(retired)
            else:
                os.replace(temporary, target)
        finally:
            arrays.clear()
            source._observations.cache_clear()
            source._trajectory.cache_clear()
            if temporary.exists():
                shutil.rmtree(temporary)
    return target, False


class CachedReplayDataset(Dataset):
    def __init__(self, directory: Path):
        self.directory = directory
        self.metadata = validate_cache(directory, hashes=False)
        self.arrays = None

    def __len__(self):
        return self.metadata["count"]

    def __getstate__(self):
        return {**self.__dict__, "arrays": None}

    def __getitem__(self, index):
        if self.arrays is None:
            self.arrays = {name: np.load(self.directory / f"{name}.npy", mmap_mode="c", allow_pickle=False)
                           for name in self.metadata["arrays"]}
        return {**self.metadata["records"][index], **{
            name: torch.as_tensor(value[index]) for name, value in self.arrays.items()}}


class DistributedBatchSampler(Sampler):
    """Same global replacement-sampled batches at any world size, including resume."""
    def __init__(self, size: int, batch: int, rank: int, world: int, seed: int, start: int, end: int):
        if size < 1 or batch < world or batch % world or not 0 <= rank < world or not 0 <= start < end:
            raise ValueError("Invalid distributed batch sampler")
        self.size, self.batch, self.rank, self.world = size, batch, rank, world
        self.seed, self.start, self.end = seed, start, end

    def __len__(self):
        return self.end - self.start

    def __iter__(self):
        generator = torch.Generator().manual_seed(self.seed)
        local = self.batch // self.world
        for step in range(self.end):
            batch = torch.randint(self.size, (self.batch,), generator=generator)
            if step >= self.start:
                yield batch[self.rank * local:(self.rank + 1) * local].tolist()
