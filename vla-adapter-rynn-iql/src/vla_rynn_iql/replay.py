from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import numpy as np
import torch
from torch.nn import functional as F
from torch.utils.data import Dataset

from .config import LoadedConfig
from .data import load_manifest
from .rewards import load_reward_index, policy_view
from .vla_adapter import env_to_dataset_actions, normalize_with_stats, proprio_from_trajectory


class ReplayDataset(Dataset):
    def __init__(self, config: LoadedConfig, action_stats: dict[str, Any],
                 proprio_stats: dict[str, Any], split: str = "train"):
        self.config = config
        self.manifest = load_manifest(config)
        reward_index = load_reward_index(config)
        if reward_index["dataset_sha256"] != self.manifest["dataset_sha256"]:
            raise ValueError("Reward annotations were generated for a different dataset manifest")
        annotations = {item["run_id"]: item["annotation_path"] for item in reward_index["episodes"]}
        self.action_stats, self.proprio_stats = action_stats, proprio_stats
        self.image_size = int(config.section("iql")["critic_image_size"])
        self.items: list[tuple[dict[str, Any], int, Path]] = []
        for episode in self.manifest["episodes"]:
            if episode["split"] != split:
                continue
            annotation = Path(annotations[episode["run_id"]])
            for chunk_index in range(len(episode["chunks"])):
                self.items.append((episode, chunk_index, annotation))
        if split == "train" and not self.items:
            raise RuntimeError("Training replay is empty")

    def __len__(self) -> int:
        return len(self.items)

    def _pixels(self, agent: np.ndarray, wrist: np.ndarray) -> torch.Tensor:
        value = torch.from_numpy(np.concatenate((agent, wrist), axis=-1).copy()).permute(2, 0, 1)
        value = F.interpolate(value.unsqueeze(0).float(), size=(self.image_size, self.image_size), mode="bilinear", align_corners=False)
        return value.squeeze(0).clamp(0, 255).to(torch.uint8)

    def __getitem__(self, item: int) -> dict[str, Any]:
        episode, chunk_index, annotation_path = self.items[item]
        chunk = episode["chunks"][chunk_index]
        start, end, length = int(chunk["start"]), int(chunk["end"]), int(chunk["length"])
        with np.load(episode["trajectory_path"], allow_pickle=False) as source:
            trajectory = {key: source[key] for key in source.files}
        with np.load(episode["observations_path"], allow_pickle=False) as source:
            agent_raw, wrist_raw = source["agentview_image"], source["wrist_image"]
            agent = policy_view(agent_raw[start], episode["observation_orientation"])
            wrist = policy_view(wrist_raw[start], episode["observation_orientation"])
            next_agent = policy_view(agent_raw[end], episode["observation_orientation"])
            next_wrist = policy_view(wrist_raw[end], episode["observation_orientation"])
        proprio = normalize_with_stats(proprio_from_trajectory(trajectory), self.proprio_stats)
        actions = np.zeros((self.manifest["action_horizon"], self.manifest["action_dim"]), dtype=np.float32)
        action_mask = np.zeros(self.manifest["action_horizon"], dtype=bool)
        actions[:length] = env_to_dataset_actions(trajectory["env_action"][start:end], self.action_stats)
        action_mask[:length] = True
        with np.load(annotation_path, allow_pickle=False) as rewards:
            reward = float(rewards["chunk_reward"][chunk_index])
        # Raw done may flicker before the configured confirmation streak. Only the
        # effective endpoint selected during preparation terminates a replay chunk.
        terminal = end == episode["action_count"]
        return {
            "pixels": self._pixels(agent, wrist),
            "next_pixels": self._pixels(next_agent, next_wrist),
            "proprio": torch.from_numpy(proprio[start]),
            "next_proprio": torch.from_numpy(proprio[end]),
            "actions": torch.from_numpy(actions),
            "action_mask": torch.from_numpy(action_mask),
            "reward": torch.tensor(reward, dtype=torch.float32),
            "bootstrap_mask": torch.tensor(0.0 if terminal else 1.0),
            "chunk_length": torch.tensor(length, dtype=torch.int64),
            "agent_image": torch.from_numpy(agent.copy()),
            "wrist_image": torch.from_numpy(wrist.copy()),
            "prompt": episode["prompt"],
            "run_id": episode["run_id"],
            "start": start,
        }
