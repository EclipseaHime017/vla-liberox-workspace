from __future__ import annotations

import json
import os
import shutil
import uuid
from collections import defaultdict
from pathlib import Path
from typing import Any, Iterator

import numpy as np
import torch
from numpy.lib.format import open_memmap
from torch.nn import functional as F
from torch.utils.data import Dataset, Sampler

from .config import LoadedConfig
from .data import iter_unique_replay_chunks, load_manifest
from .io import atomic_json, stable_hash
from .rewards import load_reward_index, policy_view
from .server_config import ReplayCacheConfig
from .vla_adapter import env_to_dataset_actions, normalize_with_stats, proprio_from_trajectory


CACHE_SCHEMA_VERSION = 1
ARRAY_FILES = {
    "agent_image": "agent_image.npy",
    "wrist_image": "wrist_image.npy",
    "pixels": "pixels.npy",
    "next_pixels": "next_pixels.npy",
}


def _cache_items(manifest: dict[str, Any]) -> list[tuple[dict[str, Any], int]]:
    return list(iter_unique_replay_chunks(manifest["episodes"], split="train"))


def replay_cache_fingerprint(config: LoadedConfig, manifest: dict[str, Any]) -> str:
    reward_index = load_reward_index(config)
    return stable_hash({
        "schema_version": CACHE_SCHEMA_VERSION,
        "dataset_sha256": manifest["dataset_sha256"],
        "reward_sha256": stable_hash(reward_index),
        "critic_image_size": int(config.section("iql")["critic_image_size"]),
        "view_transform": "policy_view_v1",
    })


def replay_cache_path(
    config: LoadedConfig,
    cache_config: ReplayCacheConfig,
    manifest: dict[str, Any] | None = None,
) -> Path:
    manifest = manifest or load_manifest(config)
    return cache_config.root / replay_cache_fingerprint(config, manifest)


def _pixels(agent: np.ndarray, wrist: np.ndarray, image_size: int) -> np.ndarray:
    value = torch.from_numpy(np.concatenate((agent, wrist), axis=-1).copy()).permute(2, 0, 1)
    resized = F.interpolate(
        value.unsqueeze(0).float(),
        size=(image_size, image_size),
        mode="bilinear",
        align_corners=False,
    )
    return resized.squeeze(0).clamp(0, 255).to(torch.uint8).numpy()


def _expected_item_metadata(items: list[tuple[dict[str, Any], int]]) -> list[dict[str, Any]]:
    result = []
    for episode, chunk_index in items:
        chunk = episode["chunks"][chunk_index]
        result.append({
            "run_id": str(episode["run_id"]),
            "chunk_index": int(chunk_index),
            "start": int(chunk["start"]),
            "end": int(chunk["end"]),
        })
    return result


def validate_replay_cache(
    directory: Path,
    config: LoadedConfig,
    manifest: dict[str, Any] | None = None,
) -> dict[str, Any]:
    manifest = manifest or load_manifest(config)
    reward_index = load_reward_index(config)
    metadata_path = directory / "cache.json"
    if not metadata_path.is_file():
        raise FileNotFoundError(f"Server replay cache metadata is missing: {metadata_path}")
    metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
    expected_items = _expected_item_metadata(_cache_items(manifest))
    expected = {
        "schema_version": CACHE_SCHEMA_VERSION,
        "fingerprint": replay_cache_fingerprint(config, manifest),
        "dataset_sha256": manifest["dataset_sha256"],
        "reward_sha256": stable_hash(reward_index),
        "critic_image_size": int(config.section("iql")["critic_image_size"]),
        "item_count": len(expected_items),
        "items_sha256": stable_hash(expected_items),
    }
    mismatches = {
        key: (metadata.get(key), value)
        for key, value in expected.items()
        if metadata.get(key) != value
    }
    if mismatches:
        raise ValueError(f"Server replay cache metadata mismatch: {mismatches}")
    arrays = metadata.get("arrays")
    if not isinstance(arrays, dict) or set(arrays) != set(ARRAY_FILES):
        raise ValueError("Server replay cache has an invalid array manifest")
    for name, filename in ARRAY_FILES.items():
        path = directory / filename
        if not path.is_file():
            raise FileNotFoundError(f"Server replay cache array is missing: {path}")
        value = np.load(path, mmap_mode="r", allow_pickle=False)
        description = arrays[name]
        if (
            list(value.shape) != description.get("shape")
            or str(value.dtype) != description.get("dtype")
            or value.shape[0] != len(expected_items)
        ):
            raise ValueError(f"Server replay cache array metadata mismatch: {path}")
    return metadata


def build_replay_cache(
    config: LoadedConfig,
    cache_config: ReplayCacheConfig,
    *,
    rebuild: bool = False,
) -> tuple[Path, bool]:
    manifest = load_manifest(config)
    reward_index = load_reward_index(config)
    items = _cache_items(manifest)
    if not items:
        raise RuntimeError("Cannot build a server replay cache for an empty training split")
    target = replay_cache_path(config, cache_config, manifest)
    force = bool(rebuild or cache_config.rebuild)
    if target.is_dir() and not force:
        try:
            validate_replay_cache(target, config, manifest)
            return target, True
        except (FileNotFoundError, ValueError, OSError):
            pass

    cache_config.root.mkdir(parents=True, exist_ok=True)
    temporary = cache_config.root / f".{target.name}.{uuid.uuid4().hex}.tmp"
    temporary.mkdir(parents=False, exist_ok=False)
    try:
        first_episode, first_chunk_index = items[0]
        first_chunk = first_episode["chunks"][first_chunk_index]
        with np.load(first_episode["observations_path"], allow_pickle=False) as source:
            first_agent = policy_view(
                source["agentview_image"][int(first_chunk["start"])],
                first_episode["observation_orientation"],
            )
            first_wrist = policy_view(
                source["wrist_image"][int(first_chunk["start"])],
                first_episode["observation_orientation"],
            )
        if first_agent.shape != first_wrist.shape or first_agent.ndim != 3:
            raise ValueError("Server replay cache requires matching HxWxC agent/wrist images")
        count = len(items)
        image_size = int(config.section("iql")["critic_image_size"])
        agent_array = open_memmap(
            temporary / ARRAY_FILES["agent_image"], mode="w+", dtype=np.uint8,
            shape=(count, *first_agent.shape),
        )
        wrist_array = open_memmap(
            temporary / ARRAY_FILES["wrist_image"], mode="w+", dtype=np.uint8,
            shape=(count, *first_wrist.shape),
        )
        pixels_array = open_memmap(
            temporary / ARRAY_FILES["pixels"], mode="w+", dtype=np.uint8,
            shape=(count, 6, image_size, image_size),
        )
        next_pixels_array = open_memmap(
            temporary / ARRAY_FILES["next_pixels"], mode="w+", dtype=np.uint8,
            shape=(count, 6, image_size, image_size),
        )

        grouped: dict[str, list[tuple[int, dict[str, Any], int]]] = defaultdict(list)
        for index, (episode, chunk_index) in enumerate(items):
            grouped[str(episode["run_id"])].append((index, episode, chunk_index))
        for entries in grouped.values():
            episode = entries[0][1]
            with np.load(episode["observations_path"], allow_pickle=False) as source:
                agent_raw = source["agentview_image"]
                wrist_raw = source["wrist_image"]
                for index, _, chunk_index in entries:
                    chunk = episode["chunks"][chunk_index]
                    start, end = int(chunk["start"]), int(chunk["end"])
                    agent = policy_view(agent_raw[start], episode["observation_orientation"])
                    wrist = policy_view(wrist_raw[start], episode["observation_orientation"])
                    next_agent = policy_view(agent_raw[end], episode["observation_orientation"])
                    next_wrist = policy_view(wrist_raw[end], episode["observation_orientation"])
                    if agent.shape != first_agent.shape or wrist.shape != first_wrist.shape:
                        raise ValueError(
                            "All server replay images in one training run must have identical shapes"
                        )
                    agent_array[index] = agent
                    wrist_array[index] = wrist
                    pixels_array[index] = _pixels(agent, wrist, image_size)
                    next_pixels_array[index] = _pixels(next_agent, next_wrist, image_size)
        for array in (agent_array, wrist_array, pixels_array, next_pixels_array):
            array.flush()

        expected_items = _expected_item_metadata(items)
        arrays = {}
        for name, filename in ARRAY_FILES.items():
            value = np.load(temporary / filename, mmap_mode="r", allow_pickle=False)
            arrays[name] = {"shape": list(value.shape), "dtype": str(value.dtype)}
        metadata = {
            "schema_version": CACHE_SCHEMA_VERSION,
            "fingerprint": replay_cache_fingerprint(config, manifest),
            "dataset_sha256": manifest["dataset_sha256"],
            "reward_sha256": stable_hash(reward_index),
            "critic_image_size": image_size,
            "item_count": len(items),
            "items_sha256": stable_hash(expected_items),
            "items": expected_items,
            "arrays": arrays,
        }
        atomic_json(temporary / "cache.json", metadata)
        validate_replay_cache(temporary, config, manifest)
        if target.exists():
            shutil.rmtree(target)
        os.replace(temporary, target)
    finally:
        if temporary.exists():
            shutil.rmtree(temporary)
    return target, False


class CachedReplayDataset(Dataset):
    def __init__(
        self,
        config: LoadedConfig,
        cache_directory: Path,
        action_stats: dict[str, Any],
        proprio_stats: dict[str, Any],
    ):
        self.config = config
        self.manifest = load_manifest(config)
        self.metadata = validate_replay_cache(cache_directory, config, self.manifest)
        reward_index = load_reward_index(config)
        if reward_index.get("complete") is False:
            raise ValueError("Reward annotation manifest is incomplete")
        if reward_index["dataset_sha256"] != self.manifest["dataset_sha256"]:
            raise ValueError("Reward annotations were generated for a different dataset manifest")
        annotations = {
            item["run_id"]: Path(item["annotation_path"])
            for item in reward_index["episodes"]
        }
        self.items = _cache_items(self.manifest)
        self.agent_images = np.load(
            cache_directory / ARRAY_FILES["agent_image"], mmap_mode="c", allow_pickle=False
        )
        self.wrist_images = np.load(
            cache_directory / ARRAY_FILES["wrist_image"], mmap_mode="c", allow_pickle=False
        )
        self.pixels = np.load(
            cache_directory / ARRAY_FILES["pixels"], mmap_mode="c", allow_pickle=False
        )
        self.next_pixels = np.load(
            cache_directory / ARRAY_FILES["next_pixels"], mmap_mode="c", allow_pickle=False
        )
        horizon = int(self.manifest["action_horizon"])
        action_dim = int(self.manifest["action_dim"])
        self.numeric: list[dict[str, Any]] = [None] * len(self.items)  # type: ignore[list-item]
        grouped: dict[str, list[tuple[int, dict[str, Any], int]]] = defaultdict(list)
        for index, (episode, chunk_index) in enumerate(self.items):
            grouped[str(episode["run_id"])].append((index, episode, chunk_index))
        for run_id, entries in grouped.items():
            episode = entries[0][1]
            with np.load(episode["trajectory_path"], allow_pickle=False) as source:
                required = (
                    "eef_position", "eef_axis_angle", "gripper_qpos", "env_action"
                )
                missing = [key for key in required if key not in source.files]
                if missing:
                    raise KeyError(
                        f"Server replay trajectory {run_id} is missing arrays {missing}"
                    )
                # Do not materialize sim_state, raw camera observations, or
                # diagnostic arrays in every rank. Only compact training fields
                # are retained after this one-time startup pass.
                trajectory = {key: source[key] for key in required}
            proprio = normalize_with_stats(
                proprio_from_trajectory(trajectory), proprio_stats
            )
            annotation_path = annotations.get(run_id)
            if annotation_path is None:
                raise KeyError(f"Reward annotation is missing for {run_id}")
            with np.load(annotation_path, allow_pickle=False) as rewards:
                chunk_rewards = np.asarray(rewards["pbrs_chunk_reward"], dtype=np.float32)
            for index, _, chunk_index in entries:
                chunk = episode["chunks"][chunk_index]
                start, end, length = (
                    int(chunk["start"]), int(chunk["end"]), int(chunk["length"])
                )
                actions = np.zeros((horizon, action_dim), dtype=np.float32)
                action_mask = np.zeros(horizon, dtype=bool)
                actions[:length] = env_to_dataset_actions(
                    trajectory["env_action"][start:end], action_stats
                )
                action_mask[:length] = True
                self.numeric[index] = {
                    "proprio": torch.from_numpy(proprio[start].copy()),
                    "next_proprio": torch.from_numpy(proprio[end].copy()),
                    "actions": torch.from_numpy(actions),
                    "action_mask": torch.from_numpy(action_mask),
                    "reward": torch.tensor(float(chunk_rewards[chunk_index]), dtype=torch.float32),
                    "bootstrap_mask": torch.tensor(
                        0.0 if end == int(episode["action_count"]) else 1.0
                    ),
                    "chunk_length": torch.tensor(length, dtype=torch.int64),
                    "prompt": str(episode["prompt"]),
                    "run_id": run_id,
                    "start": start,
                    "action_source": str(chunk["action_source"]),
                    "transition_type": str(chunk["transition_type"]),
                    "interrupted": bool(chunk["interrupted"]),
                }

    def __len__(self) -> int:
        return len(self.items)

    def __getitem__(self, index: int) -> dict[str, Any]:
        result = dict(self.numeric[index])
        # mmap_mode="c" exposes writable copy-on-write views to torch without
        # modifying the shared .npy files. DataLoader performs the only batch
        # copy while all workers retain the OS page-cache benefit.
        result.update({
            "agent_image": torch.from_numpy(self.agent_images[index]),
            "wrist_image": torch.from_numpy(self.wrist_images[index]),
            "pixels": torch.from_numpy(self.pixels[index]),
            "next_pixels": torch.from_numpy(self.next_pixels[index]),
        })
        return result


class DeterministicDistributedBatchSampler(Sampler[list[int]]):
    """Replay the same global random batches independent of DDP world size."""

    def __init__(
        self,
        *,
        dataset_size: int,
        global_batch_size: int,
        rank: int,
        world_size: int,
        seed: int,
        start_step: int,
        end_step: int,
    ):
        if dataset_size < 1 or global_batch_size < 1:
            raise ValueError("dataset_size and global_batch_size must be positive")
        if global_batch_size % world_size:
            raise ValueError("global_batch_size must be divisible by world_size")
        if not 0 <= rank < world_size:
            raise ValueError("rank must be inside the distributed world")
        if not 0 <= start_step < end_step:
            raise ValueError("Sampler requires 0 <= start_step < end_step")
        self.dataset_size = dataset_size
        self.global_batch_size = global_batch_size
        self.rank = rank
        self.world_size = world_size
        self.seed = seed
        self.start_step = start_step
        self.end_step = end_step

    def __iter__(self) -> Iterator[list[int]]:
        generator = torch.Generator(device="cpu")
        generator.manual_seed(self.seed)
        if self.start_step:
            torch.randint(
                self.dataset_size,
                (self.start_step * self.global_batch_size,),
                generator=generator,
            )
        local = self.global_batch_size // self.world_size
        offset = self.rank * local
        for _ in range(self.start_step, self.end_step):
            indices = torch.randint(
                self.dataset_size,
                (self.global_batch_size,),
                generator=generator,
            )
            yield indices[offset:offset + local].tolist()

    def __len__(self) -> int:
        return self.end_step - self.start_step
