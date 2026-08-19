from __future__ import annotations

import json
import os
import sys
import time
import uuid
from collections import deque
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import numpy as np

from .config import LoadedConfig
from .io import atomic_json
from .vla_adapter import load_components, load_overlay


def _training_view(config: LoadedConfig, base_checkpoint: str, stats_key: str) -> LoadedConfig:
    ev, paths = config.section("evaluation"), config.section("paths")
    return LoadedConfig(config.path, {
        "paths": {"vla_adapter_root": paths["vla_adapter_root"]},
        "vla": {"base_checkpoint": base_checkpoint, "stats_key": stats_key,
                "use_pro_version": True, "freeze_backbone": True},
        "iql": {"seed": ev["seed"]},
    })


def _save_video(path: Path, frames: list[np.ndarray], fps: int = 20) -> None:
    if not frames:
        return
    import imageio.v2 as imageio
    imageio.mimwrite(path, frames, fps=fps, codec="libx264", quality=8)


def _run_episode(env: Any, initial_state: Any, prompt: str, components: Any,
                 max_steps: int, open_loop_steps: int) -> dict[str, Any]:
    from experiments.robot.libero.run_libero_eval import (
        prepare_observation, process_action, quat2axisangle,
    )
    from experiments.robot.robot_utils import get_action, get_image_resize_size

    observation = env.regenerate_obs_from_state(
        initial_state.numpy() if hasattr(initial_state, "numpy") else np.asarray(initial_state)
    )
    queue: deque[np.ndarray] = deque()
    raw_actions, env_actions, rewards, dones = [], [], [], []
    agent_frames, mosaics = [], []
    sim_states, eef_position, eef_axis_angle, gripper_qpos = [], [], [], []
    inference_query_steps, query_count = [], 0
    period, previous = 1.0 / 20.0, None
    success = False

    def record_state(obs: dict[str, Any]) -> None:
        sim_states.append(np.asarray(env.get_sim_state()).copy())
        eef_position.append(np.asarray(obs["robot0_eef_pos"], dtype=np.float32))
        eef_axis_angle.append(
            np.asarray(quat2axisangle(obs["robot0_eef_quat"].copy()), dtype=np.float32)
        )
        gripper_qpos.append(np.asarray(obs["robot0_gripper_qpos"], dtype=np.float32))

    record_state(observation)
    for step in range(max_steps):
        if not queue:
            policy_obs, _ = prepare_observation(observation, get_image_resize_size(components.cfg))
            chunk = get_action(
                components.cfg, components.model, policy_obs, prompt,
                processor=components.processor, action_head=components.action_head,
                proprio_projector=components.proprio_projector,
                use_film=False, use_minivlm=True,
            )
            queue.extend(np.asarray(action, dtype=np.float32) for action in chunk[:open_loop_steps])
            query_count += 1
            inference_query_steps.append(step)
        now = time.monotonic()
        if previous is not None and now < previous + period:
            time.sleep(previous + period - now)
        previous = time.monotonic()
        raw = queue.popleft()
        action = np.asarray(process_action(raw, "openvla"), dtype=np.float32)
        agent = np.asarray(observation["agentview_image"], dtype=np.uint8)
        wrist = np.asarray(observation["robot0_eye_in_hand_image"], dtype=np.uint8)
        agent_frames.append(agent[::-1].copy())
        mosaics.append(np.concatenate((agent[::-1, ::-1], wrist[::-1, ::-1]), axis=1).copy())
        observation, reward, done, _ = env.step(action.tolist())
        record_state(observation)
        raw_actions.append(raw)
        env_actions.append(action)
        rewards.append(float(reward))
        dones.append(bool(done))
        success = success or bool(done)
        if done:
            break
    return {
        "success": success, "steps": len(env_actions), "policy_queries": query_count,
        "raw_action": np.asarray(raw_actions, dtype=np.float32),
        "env_action": np.asarray(env_actions, dtype=np.float32),
        "reward": np.asarray(rewards, dtype=np.float32), "done": np.asarray(dones, dtype=bool),
        "sim_state": np.asarray(sim_states),
        "eef_position": np.asarray(eef_position, dtype=np.float32),
        "eef_axis_angle": np.asarray(eef_axis_angle, dtype=np.float32),
        "gripper_qpos": np.asarray(gripper_qpos, dtype=np.float32),
        "action_source": np.asarray(["policy"] * len(env_actions)),
        "inference_query_step": np.asarray(inference_query_steps, dtype=np.int64),
        "agent_frames": agent_frames,
        "vla_mosaics": mosaics,
    }


def evaluate(config: LoadedConfig) -> Path:
    ev, paths = config.section("evaluation"), config.section("paths")
    os.environ["MUJOCO_GL"] = ev["mujoco_gl"]
    for root in (paths["libero_x_root"], paths["vla_adapter_root"]):
        if root not in sys.path:
            sys.path.insert(0, root)
    import torch
    from libero.libero.envs import OffScreenRenderEnv
    from libero.libero.utils.parse_bddl import parse_bddl_file

    if not torch.cuda.is_available():
        raise RuntimeError("evaluation.device=cuda:0 was requested but CUDA is unavailable")
    torch.cuda.set_device(torch.device(ev["device"]))

    overlay = load_overlay(Path(config.section("policy")["overlay"]))
    policies = [("iql", Path(config.section("policy")["overlay"]))]
    if ev["compare_base"]:
        policies.insert(0, ("base", None))
    output = Path(paths["output_dir"]) / (
        datetime.now(timezone.utc).strftime("%Y-%m-%d_%H%M%S") + "__" + uuid.uuid4().hex[:8]
    )
    output.mkdir(parents=True, exist_ok=False)
    bddl = Path(paths["libero_x_root"]) / "libero" / "libero_x" / "bddl" / ev["level"] / f"{ev['task_name']}.bddl"
    init_path = Path(paths["libero_x_root"]) / "libero" / "libero_x" / "init" / ev["level"] / f"{ev['task_name']}.init"
    if not bddl.is_file() or not init_path.is_file():
        raise FileNotFoundError(f"Task assets are missing: {bddl}, {init_path}")
    prompt = str(parse_bddl_file(str(bddl))["language"])
    initial_states = torch.load(init_path, map_location="cpu", weights_only=False)
    summary: dict[str, Any] = {"schema_version": 1, "task": prompt, "policies": {}}
    for policy_name, overlay_path in policies:
        training_config = _training_view(config, overlay.base_checkpoint, overlay.stats_key)
        components = load_components(training_config, overlay_path, training=False)
        components.cfg.num_open_loop_steps = int(ev["open_loop_steps"])
        policy_dir = output / policy_name
        policy_dir.mkdir()
        successes = 0
        results = []
        for episode_id in range(int(ev["trials"])):
            env = OffScreenRenderEnv(
                bddl_file_name=str(bddl), use_camera_obs=True,
                camera_names=["agentview", "robot0_eye_in_hand"],
                camera_heights=[256, 256], camera_widths=[256, 256],
                horizon=int(ev["max_steps"]) + 1, control_freq=20,
            )
            env.seed(int(ev["seed"]) + episode_id)
            env.reset()
            try:
                result = _run_episode(
                    env, initial_states[episode_id % len(initial_states)], prompt, components,
                    int(ev["max_steps"]), int(ev["open_loop_steps"]),
                )
            finally:
                env.close()
            successes += int(result["success"])
            episode = policy_dir / f"episode_{episode_id:03d}"
            episode.mkdir()
            np.savez_compressed(
                episode / "trajectory.npz", raw_action=result["raw_action"],
                env_action=result["env_action"], reward=result["reward"],
                done=result["done"], sim_state=result["sim_state"],
                eef_position=result["eef_position"],
                eef_axis_angle=result["eef_axis_angle"],
                gripper_qpos=result["gripper_qpos"],
                action_source=result["action_source"],
                inference_query_step=result["inference_query_step"],
            )
            _save_video(episode / "agentview.mp4", result["agent_frames"])
            _save_video(episode / "vla_views.mp4", result["vla_mosaics"])
            results.append({key: result[key] for key in ("success", "steps", "policy_queries")})
        summary["policies"][policy_name] = {
            "episodes": len(results), "successes": successes,
            "success_rate": successes / len(results), "results": results,
            "overlay": None if overlay_path is None else str(overlay_path),
        }
        del components
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
    atomic_json(output / "summary.json", summary)
    return output
