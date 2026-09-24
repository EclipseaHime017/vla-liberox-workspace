from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import numpy as np
import torch
from torch.nn import functional as F
from torch.utils.data import Dataset

from .config import LoadedConfig
from .data import iter_unique_replay_chunks, load_manifest, replay_chunks
from .rewards import load_reward_index, policy_view
from .vla_adapter import env_to_dataset_actions, normalize_with_stats, proprio_from_trajectory


class ActionDataset(Dataset):
    """All selected replay chunks, independent of rewards and critic inputs."""
    def __init__(self, config: LoadedConfig, action_stats: dict[str, Any],
                 proprio_stats: dict[str, Any], split: str = "train"):
        self.config = config
        self.manifest = load_manifest(config)
        self.action_stats, self.proprio_stats = action_stats, proprio_stats
        self.image_size = int(config.section("iql")["critic_image_size"])
        self.chunks = {
            episode["run_id"]: replay_chunks(episode)
            for episode in self.manifest["episodes"]
        }
        self.items = [(episode, index, None) for episode, index in
                      iter_unique_replay_chunks(self.manifest["episodes"], split=split)]
        if split == "train" and not self.items:
            raise RuntimeError("Training replay is empty")

    def __len__(self) -> int:
        return len(self.items)

    def _pixels(self, agent: np.ndarray, wrist: np.ndarray) -> torch.Tensor:
        value = torch.from_numpy(np.concatenate((agent, wrist), axis=-1).copy()).permute(2, 0, 1)
        value = F.interpolate(value.unsqueeze(0).float(), size=(self.image_size, self.image_size), mode="bilinear", align_corners=False)
        return value.squeeze(0).clamp(0, 255).to(torch.uint8)

    def __getitem__(self, item: int) -> dict[str, Any]:
        return self._sample(item, transitions=False)

    def _sample(self, item: int, *, transitions: bool) -> dict[str, Any]:
        episode, chunk_index, annotation_path = self.items[item]
        chunk = self.chunks[episode["run_id"]][chunk_index]
        start, end, length = int(chunk["start"]), int(chunk["end"]), int(chunk["length"])
        with np.load(episode["trajectory_path"], allow_pickle=False) as source:
            trajectory = {key: source[key] for key in source.files}
        with np.load(episode["observations_path"], allow_pickle=False) as source:
            agent_raw, wrist_raw = source["agentview_image"], source["wrist_image"]
            agent = policy_view(agent_raw[start], episode["observation_orientation"])
            wrist = policy_view(wrist_raw[start], episode["observation_orientation"])
            if transitions:
                next_agent = policy_view(agent_raw[end], episode["observation_orientation"])
                next_wrist = policy_view(wrist_raw[end], episode["observation_orientation"])
        proprio = normalize_with_stats(proprio_from_trajectory(trajectory), self.proprio_stats)
        actions = np.zeros((self.manifest["action_horizon"], self.manifest["action_dim"]), dtype=np.float32)
        action_mask = np.zeros(self.manifest["action_horizon"], dtype=bool)
        actions[:length] = env_to_dataset_actions(trajectory["env_action"][start:end], self.action_stats)
        action_mask[:length] = True
        # Success confirmation affects rewards but replay continues through the
        # recorded tail. Only the final recorded observation ends bootstrapping.
        terminal = end == episode["recorded_action_count"]
        sample = {
            "proprio": torch.from_numpy(proprio[start]),
            "actions": torch.from_numpy(actions),
            "action_mask": torch.from_numpy(action_mask),
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
        index = load_reward_index(config) if reward_index is None else reward_index
        if index.get("complete") is False:
            raise ValueError("Reward annotation manifest is incomplete")
        if index["dataset_sha256"] != self.manifest["dataset_sha256"]:
            raise ValueError("Reward annotations were generated for a different dataset manifest")
        annotations = {item["run_id"]: Path(item["annotation_path"]) for item in index["episodes"]}
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

    def __getitem__(self, item: int) -> dict[str, Any]:
        sample = self._sample(item, transitions=True)
        _, chunk_index, annotation = self.items[item]
        with np.load(annotation, allow_pickle=False) as rewards:
            key = "final_reward" if "final_reward" in rewards else "pbrs_chunk_reward"
            sample["reward"] = torch.tensor(float(rewards[key][chunk_index]), dtype=torch.float32)
        return sample
