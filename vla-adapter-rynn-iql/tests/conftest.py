from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pytest
import yaml

from vla_rynn_iql.config import DEFAULT_TRAIN_CONFIG, load_train_config


def _episode(root: Path, run_id: str, *, kind: str = "original", resume: int | None = None,
             action_source: str = "policy", action_count: int = 17, success: bool = False,
             success_from: int | None = None):
    run_dir = root / "projects" / "libero_x_vla" / "runs" / "task" / "2026-01-01" / run_id
    episode = run_dir / "episodes" / "episode_000"
    episode.mkdir(parents=True)
    actions = np.zeros((action_count, 7), dtype=np.float32)
    actions[:, -1] = -1
    raw_actions = actions.copy()
    raw_actions[:, -1] = 1
    sources = np.asarray(["policy"] * action_count)
    if kind == "branch":
        sources[int(resume):] = action_source
    done = np.zeros(action_count, dtype=bool)
    if success:
        terminal = action_count - 1 if success_from is None else success_from
        if not 0 <= terminal < action_count:
            raise ValueError("success_from must identify a recorded action")
        done[terminal:] = True
    np.savez_compressed(
        episode / "trajectory.npz",
        time_seconds=np.arange(action_count + 1, dtype=np.float64) / 20.0,
        eef_position=np.zeros((action_count + 1, 3), np.float32),
        eef_axis_angle=np.zeros((action_count + 1, 3), np.float32),
        gripper_qpos=np.zeros((action_count + 1, 2), np.float32),
        env_action=actions, raw_action=raw_actions,
        reward=np.zeros(action_count, np.float32),
        done=done,
        action_source=sources,
    )
    np.savez_compressed(
        episode / "trajectory_observations.npz",
        agentview_image=np.zeros((action_count + 1, 16, 16, 3), np.uint8),
        wrist_image=np.zeros((action_count + 1, 16, 16, 3), np.uint8),
    )
    (run_dir / "run.json").write_text(json.dumps({
        "id": run_id, "kind": kind, "task_id": "LEVEL1::task", "task_name": "task",
        "task": "do the task", "status": "COMPLETED", "error": None,
        "root_session_id": "root", "parent_session_id": "root" if kind == "branch" else None,
        "resume_step": resume, "control_mode": "manual" if action_source == "human" else "policy",
        "success": success,
    }), encoding="utf-8")


@pytest.fixture
def configured(tmp_path: Path):
    dataset = tmp_path / "dataset"
    _episode(dataset, "root", action_count=17)
    _episode(
        dataset, "branch", kind="branch", resume=5, action_source="human",
        action_count=17, success=True, success_from=13,
    )
    raw = yaml.safe_load(DEFAULT_TRAIN_CONFIG.read_text(encoding="utf-8"))
    raw["paths"].update({
        "dataset_sources": [str(dataset)], "work_dir": str(tmp_path / "work"),
        "output_dir": str(tmp_path / "output"), "vla_adapter_root": str(tmp_path / "vla"),
        "libero_x_root": str(tmp_path / "libero"),
        "rynnvalue_root": str(tmp_path / "RynnValue"),
        "policy_registry": str(tmp_path / "registry"),
    })
    config_path = tmp_path / "config.yaml"
    config_path.write_text(yaml.safe_dump(raw), encoding="utf-8")
    return load_train_config(config_path)
