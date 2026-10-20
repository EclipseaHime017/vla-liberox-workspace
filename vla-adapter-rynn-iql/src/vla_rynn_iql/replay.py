from __future__ import annotations

import json
from pathlib import Path

from .storage_paths import storage_path
from typing import Any

import numpy as np
import torch
from torch.nn import functional as F
from torch.utils.data import Dataset

from .config import LoadedConfig
from .base_models import model_observation, model_contract
from .data import iter_unique_replay_chunks, load_manifest, replay_chunks, replay_end_step
from .frame_cache import FrameCache
from .image_transforms import policy_view, image_orientation
from .rewards import load_reward_index
from .vla_adapter import env_to_dataset_actions, normalize_with_stats, proprio_from_trajectory


class ActionDataset(Dataset):
    """All selected replay chunks, independent of rewards and critic inputs."""
    def __init__(self, config: LoadedConfig, action_stats: dict[str, Any],
                 proprio_stats: dict[str, Any], split: str = "train"):
        self.config = config
        self.image_orientation = image_orientation(model_observation(config.section("model")))
        self.actor_horizon = model_contract(config.section("model"))["io"]["replay_horizon"]
        self.include_post_success = config.section("data").get("include_post_success", True)
        self.manifest = load_manifest(config)
        self.action_stats, self.proprio_stats = action_stats, proprio_stats
        self.chunks = {
            episode["run_id"]: replay_chunks(episode)
            for episode in self.manifest["episodes"]
        }
        self.items = [(episode, index, None) for episode, index in
                      iter_unique_replay_chunks(self.manifest["episodes"], split=split,
                                               include_post_success=self.include_post_success)]
        if split == "train" and not self.items:
            raise RuntimeError("Training replay is empty")
        # Only chunk boundaries are ever read: the start frame for the actor
        # and the end frame for the critic's next observation.
        self._frame_indices: dict[str, set[int]] = {}
        for episode, chunk_index, _ in self.items:
            chunk = self.chunks[episode["run_id"]][chunk_index]
            self._frame_indices.setdefault(episode["run_id"], set()).update(
                (int(chunk["start"]), int(chunk["end"])))
        # Prepared work directories may be frozen dataset inputs; the decoded
        # frames belong to the shared, writable cache root instead.
        cache_root = config.section("paths").get("annotation_cache")
        self._frames = FrameCache(Path(cache_root) / "replay-frames" if cache_root else None)
        self._trajectories: dict[str, dict[str, np.ndarray]] = {}

    def __len__(self) -> int:
        return len(self.items)

    def _pixels(self, agent: np.ndarray, wrist: np.ndarray) -> torch.Tensor:
        value = torch.from_numpy(np.concatenate((agent, wrist), axis=-1).copy()).permute(2, 0, 1)
        value = F.interpolate(value.unsqueeze(0).float(), size=(self.image_size, self.image_size), mode="bilinear", align_corners=False)
        return value.squeeze(0).clamp(0, 255).to(torch.uint8)

    def __getitem__(self, item: int) -> dict[str, Any]:
        return self._sample(item, transitions=False)

    def _trajectory(self, episode: dict[str, Any]) -> dict[str, np.ndarray]:
        # Control arrays are small; keep each decoded recording for the run.
        run_id = episode["run_id"]
        cached = self._trajectories.get(run_id)
        if cached is None:
            with np.load(storage_path(episode["trajectory_path"]), allow_pickle=False) as source:
                cached = {key: source[key] for key in source.files}
            self._trajectories[run_id] = cached
        return cached

    def _sample(self, item: int, *, transitions: bool) -> dict[str, Any]:
        episode, chunk_index, annotation_path = self.items[item]
        chunk = self.chunks[episode["run_id"]][chunk_index]
        start, end, length = int(chunk["start"]), int(chunk["end"]), int(chunk["length"])
        trajectory = self._trajectory(episode)
        # The actor learns a model-sized window from the same state. Ordinary
        # re-planning boundaries end a critic transition, not actor supervision.
        actor_end = min(start + self.actor_horizon, replay_end_step(episode, self.include_post_success))
        hard_boundaries = {int(s["end"]) for s in episode["recorded_action_source_segments"]}
        hard_boundaries.update(int(s["end"]) for s in (episode.get("trajectory_segments") or []))
        hard_boundaries.update(int(step) for step in episode.get("handoff_steps", []))
        if episode.get("resume_step") is not None:
            hard_boundaries.add(int(episode["resume_step"]))
        actor_end = min([actor_end] + [point for point in hard_boundaries if start < point < actor_end])
        actor_length = actor_end - start
        frames = self._frames.episode(episode, self._frame_indices[episode["run_id"]])
        orientation = episode["observation_orientation"]
        agent = policy_view(frames.frame("agentview_image", start), orientation, self.image_orientation)
        wrist = policy_view(frames.frame("wrist_image", start), orientation, self.image_orientation)
        if transitions:
            next_agent = policy_view(frames.frame("agentview_image", end), orientation, self.image_orientation)
            next_wrist = policy_view(frames.frame("wrist_image", end), orientation, self.image_orientation)
        raw_proprio = proprio_from_trajectory(trajectory)
        if self.action_stats.get("codec") == "openpi_quantile":
            from .pi05 import normalize
            proprio = normalize(raw_proprio, self.proprio_stats)
            encoded_actions = normalize(trajectory["env_action"][start:end], self.action_stats)
            actor_encoded = normalize(trajectory["env_action"][start:actor_end], self.action_stats)
        else:
            proprio = normalize_with_stats(raw_proprio, self.proprio_stats)
            encoded_actions = env_to_dataset_actions(trajectory["env_action"][start:end], self.action_stats)
            actor_encoded = env_to_dataset_actions(trajectory["env_action"][start:actor_end], self.action_stats)
        actions = np.zeros((self.manifest["action_horizon"], self.manifest["action_dim"]), dtype=np.float32)
        action_mask = np.zeros(self.manifest["action_horizon"], dtype=bool)
        actions[:length] = encoded_actions
        raw_actions = np.zeros_like(actions)
        raw_actions[:length] = trajectory["env_action"][start:end]
        action_mask[:length] = True
        actor_actions = np.zeros((self.actor_horizon, self.manifest["action_dim"]), dtype=np.float32)
        actor_raw_actions = np.zeros_like(actor_actions)
        actor_mask = np.zeros(self.actor_horizon, dtype=bool)
        actor_actions[:actor_length] = actor_encoded
        actor_raw_actions[:actor_length] = trajectory["env_action"][start:actor_end]
        actor_mask[:actor_length] = True
        # The selected replay endpoint is terminal in either mode. Reward arrays
        # retain full-recording indices and never need a model re-evaluation.
        terminal = end == replay_end_step(episode, self.include_post_success)
        sample = {
            "raw_proprio": torch.from_numpy(raw_proprio[start].astype(np.float32)),
            "raw_actions": torch.from_numpy(raw_actions),
            "proprio": torch.from_numpy(proprio[start]),
            "actions": torch.from_numpy(actions),
            "action_mask": torch.from_numpy(action_mask),
            "actor_actions": torch.from_numpy(actor_actions),
            "actor_raw_actions": torch.from_numpy(actor_raw_actions),
            "actor_action_mask": torch.from_numpy(actor_mask),
            "chunk_length": torch.tensor(length, dtype=torch.int64),
            "agent_image": torch.from_numpy(agent.copy()),
            "wrist_image": torch.from_numpy(wrist.copy()),
            "prompt": episode["prompt"],
            "run_id": episode["run_id"],
            "start": start,
            "action_source": str(chunk["action_source"]),
            "transition_type": str(chunk["transition_type"]),
            "interrupted": bool(chunk["interrupted"]),
        }
        if transitions:
            sample.update(
                pixels=self._pixels(agent, wrist), next_pixels=self._pixels(next_agent, next_wrist),
                next_proprio=torch.from_numpy(proprio[end]),
                bootstrap_mask=torch.tensor(0.0 if terminal else 1.0),
            )
        return sample


class ReplayDataset(ActionDataset):
    """RL view of the same samples, with validated rewards and next observations."""
    def __init__(self, config: LoadedConfig, action_stats: dict[str, Any],
                 proprio_stats: dict[str, Any], split: str = "train",
                 reward_index: dict[str, Any] | None = None):
        super().__init__(config, action_stats, proprio_stats, split)
        self.image_size = int(config.section("iql")["critic_image_size"])
        index = load_reward_index(config) if reward_index is None else reward_index
        if index.get("complete") is False:
            raise ValueError("Reward annotation manifest is incomplete")
        if index["dataset_sha256"] != self.manifest["dataset_sha256"]:
            raise ValueError("Reward annotations were generated for a different dataset manifest")
        annotations = {item["run_id"]: storage_path(item["annotation_path"]) for item in index["episodes"]}
        checked = set()
        for episode, _, _ in self.items:
            run_id = episode["run_id"]
            if run_id not in checked:
                with np.load(annotations[run_id], allow_pickle=False) as rewards:
                    key = "final_reward" if "final_reward" in rewards else "pbrs_chunk_reward"
                    expected = len(self.chunks[run_id])
                    if rewards[key].shape != (expected,):
                        raise ValueError(
                            f"Run {run_id} reward array does not cover all {expected} "
                            "full-recording replay chunks; rematerialize rewards for the complete recording"
                        )
                checked.add(run_id)
        self.items = [(episode, chunk_index, annotations[episode["run_id"]])
                      for episode, chunk_index, _ in self.items]
        self._rewards: dict[Path, np.ndarray] = {}

    def __getitem__(self, item: int) -> dict[str, Any]:
        sample = self._sample(item, transitions=True)
        _, chunk_index, annotation = self.items[item]
        rewards = self._rewards.get(annotation)
        if rewards is None:
            with np.load(annotation, allow_pickle=False) as source:
                key = "final_reward" if "final_reward" in source else "pbrs_chunk_reward"
                rewards = np.asarray(source[key], dtype=np.float32)
            self._rewards[annotation] = rewards
        sample["reward"] = torch.tensor(float(rewards[chunk_index]), dtype=torch.float32)
        return sample
