"""Read-only mmap samples assembled by the same replay implementation as the UI."""
from __future__ import annotations

from core.paths import ROOT
import fcntl
import json
import os
import shutil
import tempfile
import time
from pathlib import Path

import numpy as np
import torch
from numpy.lib.format import open_memmap
from torch.utils.data import Dataset, Sampler

from core.io import atomic_json, sha256_file, stable_hash
from core.methods import training_method
from core.model_config import model_config
from core.registry import model_contract
from core.frame_cache import FrameCache
from train.replay import ActionDataset, ReplayDataset
from rewards.manifest import reward_manifest_digest


# Sources whose behaviour shapes cached samples; any edit rebuilds the cache.
IMPLEMENTATION_FILES = ("train/replay_cache.py", "train/replay.py", "train/encoding.py", "core/data.py", "core/execution.py",
                        "core/image_transforms.py", "train/models/vla_adapter.py", "train/models/pi05.py",
                        "rewards/pbrs.py")


class _EpisodeCache:
    """Build the mmap cache from boundary frames decoded in memory, one recording at a time."""

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self._frames = FrameCache(None, max_open=1)

    def _trajectory(self, episode):
        if getattr(self, "_resident_run", None) != episode["run_id"]:
            self._trajectories.clear()
            self._resident_run = episode["run_id"]
        return super()._trajectory(episode)


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
        "model_contract": model_contract(model_config(config.raw)),
        "include_post_success": source.include_post_success, "transitions": transitions,
        "next_actor_inputs": source.next_actor_inputs if transitions else False,
        "bootstrap_failed_end": config.section("data").get("bootstrap_failed_end", False),
        "image_size": (config.section(training_method(config.raw).config_section)["critic_image_size"]
                       if transitions else None),
        "implementation": {Path(name).name: sha256_file(ROOT / name) for name in IMPLEMENTATION_FILES},
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
                        "count": len(source), "arrays": descriptions, "records": records,
                        "prompt_groups": source.prompt_groups}
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
            source._trajectories.clear()
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

    def _open(self):
        if self.arrays is None:
            self.arrays = {name: np.load(self.directory / f"{name}.npy", mmap_mode="c", allow_pickle=False)
                           for name in self.metadata["arrays"]}

    def reward_range(self) -> tuple[float, float]:
        """Lowest and highest chunk reward of the cached transitions."""
        self._open()
        return float(self.arrays["reward"].min()), float(self.arrays["reward"].max())

    def __getitem__(self, index):
        self._open()
        return {**self.metadata["records"][index], **{
            name: torch.as_tensor(value[index]) for name, value in self.arrays.items()}}


class DistributedBatchSampler(Sampler):
    """Same global replacement-sampled batches at any world size, including resume."""
    def __init__(self, size: int, batch: int, rank: int, world: int, seed: int, start: int, end: int,
                 prompt_groups: dict | None = None):
        if size < 1 or batch < world or batch % world or not 0 <= rank < world or not 0 <= start < end:
            raise ValueError("Invalid distributed batch sampler")
        self.size, self.batch, self.rank, self.world = size, batch, rank, world
        self.seed, self.start, self.end = seed, start, end
        self.groups = {}
        if batch > 1 and prompt_groups and len(prompt_groups) > 1:
            if sorted(index for group in prompt_groups.values() for index in group) != list(range(size)):
                raise ValueError("Prompt groups must partition the replay samples")
            for group in prompt_groups.values():
                pool = torch.tensor(group, dtype=torch.int64)
                self.groups.update({index: pool for index in group})

    def draw_batch(self, generator):
        if self.groups:
            anchor = int(torch.randint(self.size, (1,), generator=generator))
            pool = self.groups[anchor]
            return pool[torch.randint(len(pool), (self.batch,), generator=generator)]
        return torch.randint(self.size, (self.batch,), generator=generator)

    def __len__(self):
        return self.end - self.start

    def __iter__(self):
        generator = torch.Generator().manual_seed(self.seed)
        local = self.batch // self.world
        for step in range(self.end):
            batch = self.draw_batch(generator)
            if step >= self.start:
                yield batch[self.rank * local:(self.rank + 1) * local].tolist()
