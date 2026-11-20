from __future__ import annotations

from pathlib import Path

from core.storage_paths import storage_path
from typing import Any

import numpy as np
import torch
from torch.utils.data import Dataset

from core.config import LoadedConfig
from core.registry import model_observation, model_contract
from core.data import iter_unique_replay_chunks, load_manifest, replay_chunks, replay_end_step, transition_end
from core.methods import training_method
from core.frame_cache import FrameCache
from core.image_transforms import policy_view, image_orientation
from rewards.manifest import load_reward_index
from rewards.pbrs import discounted_sum
from train.encoding import critic_pixels, encode_actions, encode_proprio
from train.models.vla_adapter import proprio_from_trajectory


class ActionDataset(Dataset):
    """All selected replay chunks, independent of rewards and critic inputs.

    Each sample is the model chunk planned at a re-planning point: ``horizon`` (the model's
    replay horizon) actions from ``start``, cut at hard boundaries and the replay endpoint
    (``core.data.transition_end``). The actor imitates it and, for RL methods, the critic
    scores it as one action whose transition ends at its last step.

    ``items`` are ``(episode, chunk_index, start)``. Starts are the recorded re-planning
    points, plus every ``horizon`` steps inside a recorded interval longer than ``horizon``,
    so every action belongs to a transition.
    """
    def __init__(self, config: LoadedConfig, action_stats: dict[str, Any],
                 proprio_stats: dict[str, Any], split: str = "train"):
        self.config = config
        self.image_orientation = image_orientation(model_observation(config.section("model")))
        self.horizon = model_contract(config.section("model"))["io"]["replay_horizon"]
        self.include_post_success = config.section("data").get("include_post_success", True)
        self.manifest = load_manifest(config)
        self.action_stats, self.proprio_stats = action_stats, proprio_stats
        # Value Flows samples next chunks from the actor: transitions carry the s' actor inputs.
        self.next_actor_inputs = training_method(config.raw).actor_samples_in_training
        self.chunks = {
            episode["run_id"]: replay_chunks(episode)
            for episode in self.manifest["episodes"]
        }
        self.items: list[tuple[dict[str, Any], int, int]] = []
        for episode, index in iter_unique_replay_chunks(self.manifest["episodes"], split=split,
                                                        include_post_success=self.include_post_success):
            chunk = self.chunks[episode["run_id"]][index]
            self.items.extend((episode, index, start)
                              for start in range(int(chunk["start"]), int(chunk["end"]), self.horizon))
        self.prompt_groups: dict[str, list[int]] = {}
        for index, (episode, _, _) in enumerate(self.items):
            self.prompt_groups.setdefault(episode["prompt"], []).append(index)
        if split == "train" and not self.items:
            raise RuntimeError("Training replay is empty")
        # Only transition boundaries are ever read: the start frame for both networks
        # and the end frame for the critic's next observation.
        self._frame_indices: dict[str, set[int]] = {}
        for episode, _, start in self.items:
            self._frame_indices.setdefault(episode["run_id"], set()).update(
                (start, self.transition_end(episode, start)))
        # Prepared work directories may be frozen dataset inputs; the decoded
        # frames belong to the shared, writable cache root instead.
        cache_root = config.section("paths").get("annotation_cache")
        self._frames = FrameCache(Path(cache_root) / "replay-frames" if cache_root else None)
        self._trajectories: dict[str, dict[str, np.ndarray]] = {}

    def __len__(self) -> int:
        return len(self.items)

    def transition_end(self, episode: dict[str, Any], start: int) -> int:
        return transition_end(episode, start, self.horizon, self.include_post_success)

    def _pixels(self, agent: np.ndarray, wrist: np.ndarray) -> torch.Tensor:
        return critic_pixels(agent, wrist, self.image_size)

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
        episode, chunk_index, start = self.items[item]
        chunk = self.chunks[episode["run_id"]][chunk_index]
        end = self.transition_end(episode, start)
        length = end - start
        trajectory = self._trajectory(episode)
        frames = self._frames.episode(episode, self._frame_indices[episode["run_id"]])
        orientation = episode["observation_orientation"]
        agent = policy_view(frames.frame("agentview_image", start), orientation, self.image_orientation)
        wrist = policy_view(frames.frame("wrist_image", start), orientation, self.image_orientation)
        if transitions:
            next_agent = policy_view(frames.frame("agentview_image", end), orientation, self.image_orientation)
            next_wrist = policy_view(frames.frame("wrist_image", end), orientation, self.image_orientation)
        raw_proprio = proprio_from_trajectory(trajectory)
        proprio = encode_proprio(raw_proprio, self.action_stats, self.proprio_stats)
        actions = np.zeros((self.horizon, self.manifest["action_dim"]), dtype=np.float32)
        raw_actions = np.zeros_like(actions)
        action_mask = np.zeros(self.horizon, dtype=bool)
        actions[:length] = encode_actions(trajectory["env_action"][start:end], self.action_stats)
        raw_actions[:length] = trajectory["env_action"][start:end]
        action_mask[:length] = True
        # The selected replay endpoint is terminal in either mode. Reward arrays
        # retain full-recording indices and never need a model re-evaluation.
        terminal = end == replay_end_step(episode, self.include_post_success)
        if self.config.section("data").get("bootstrap_failed_end", False) and episode.get("terminal_step") is None:
            terminal = False
        sample = {
            "raw_proprio": torch.from_numpy(raw_proprio[start].astype(np.float32)),
            "raw_actions": torch.from_numpy(raw_actions),
            "proprio": torch.from_numpy(proprio[start]),
            "actions": torch.from_numpy(actions),
            "action_mask": torch.from_numpy(action_mask),
            "chunk_length": torch.tensor(length, dtype=torch.int64),
            "agent_image": torch.from_numpy(agent.copy()),
            "wrist_image": torch.from_numpy(wrist.copy()),
            "prompt": episode["prompt"],
            "run_id": episode["run_id"],
            "start": start,
            # Hard boundaries end every transition at a source change.
            "action_source": str(chunk["action_source"]),
        }
        if transitions:
            sample.update(
                pixels=self._pixels(agent, wrist), next_pixels=self._pixels(next_agent, next_wrist),
                next_proprio=torch.from_numpy(proprio[end]),
                bootstrap_mask=torch.tensor(0.0 if terminal else 1.0),
            )
            if self.next_actor_inputs:
                sample.update(next_agent_image=torch.from_numpy(next_agent.copy()),
                              next_wrist_image=torch.from_numpy(next_wrist.copy()),
                              next_raw_proprio=torch.from_numpy(raw_proprio[end].astype(np.float32)))
        return sample


class ReplayDataset(ActionDataset):
    """RL view of the same samples, with rewards and next observations of each transition."""
    def __init__(self, config: LoadedConfig, action_stats: dict[str, Any],
                 proprio_stats: dict[str, Any], split: str = "train",
                 reward_index: dict[str, Any] | None = None):
        super().__init__(config, action_stats, proprio_stats, split)
        self.image_size = int(config.section(training_method(config.raw).config_section)["critic_image_size"])
        index = load_reward_index(config) if reward_index is None else reward_index
        if index.get("complete") is False:
            raise ValueError("Reward annotation manifest is incomplete")
        if index["dataset_sha256"] != self.manifest["dataset_sha256"]:
            raise ValueError("Reward annotations were generated for a different dataset manifest")
        paths = {item["run_id"]: storage_path(item["annotation_path"]) for item in index["episodes"]}
        self.gamma = float(config.section("reward")["gamma"])
        # Per-action rewards of each recording in this split.
        self.action_rewards: dict[str, np.ndarray] = {}
        for episode, _, _ in self.items:
            run_id = episode["run_id"]
            if run_id in self.action_rewards:
                continue
            with np.load(paths[run_id], allow_pickle=False) as source:
                rewards = np.asarray(source["final_reward"], dtype=np.float64)
            if rewards.shape != (int(episode["recorded_action_count"]),) or not np.isfinite(rewards).all():
                raise ValueError(f"Run {run_id} rewards do not cover every recorded action; "
                                 "rematerialize rewards for the complete recording")
            self.action_rewards[run_id] = rewards

    def transition_reward(self, item: int) -> float:
        """Per-action rewards over the planned chunk, summed with the per-step discount (float32, as stored)."""
        episode, _, start = self.items[item]
        return float(np.float32(discounted_sum(self.action_rewards[episode["run_id"]], start,
                                               self.transition_end(episode, start), self.gamma)))

    def __getitem__(self, item: int) -> dict[str, Any]:
        sample = self._sample(item, transitions=True)
        sample["reward"] = torch.tensor(self.transition_reward(item), dtype=torch.float32)
        return sample

    def reward_range(self) -> tuple[float, float]:
        """Lowest and highest transition reward of this split."""
        values = [self.transition_reward(item) for item in range(len(self.items))]
        return float(min(values)), float(max(values))
