from __future__ import annotations

import hashlib
import json
from pathlib import Path

import numpy as np

from backend.app.services.trajectory_evaluation_service import (
    EVALUATION_SCHEMA_VERSION,
    SIDECAR_NAME,
    TrajectoryEvaluationService,
    VALUES_NAME,
)


def sha(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


class Runs:
    def __init__(self, run):
        self.run = run

    def get_run(self, run_id):
        if run_id != self.run["id"]:
            raise KeyError(run_id)
        return self.run


def test_reward_values_are_copied_and_bound_to_trajectory(tmp_path: Path):
    episode = tmp_path / "run" / "episodes" / "episode_000"
    episode.mkdir(parents=True)
    trajectory = episode / "trajectory.npz"
    observations = episode / "trajectory_observations.npz"
    np.savez_compressed(
        trajectory,
        time_seconds=np.asarray([0.0, 0.05, 0.1]),
        env_action=np.zeros((2, 7), np.float32),
        raw_action=np.zeros((2, 7), np.float32),
        eef_position=np.zeros((3, 3), np.float32),
        eef_axis_angle=np.zeros((3, 3), np.float32),
        gripper_qpos=np.zeros((3, 2), np.float32),
    )
    np.savez_compressed(observations, agentview_image=np.zeros((3, 2, 2, 3), np.uint8))
    annotation = tmp_path / "cache.npz"
    np.savez_compressed(
        annotation,
        boundary_steps=np.asarray([0, 2]),
        absolute_temporal_distance_seconds=np.asarray([[2.0], [0.0]]),
        absolute_value_entropy_nats=np.asarray([[0.1], [0.2]]),
        absolute_value_logits=np.zeros((2, 1, 256), np.float32),
        relative_temporal_distance_seconds=np.asarray([0.0, 0.1]),
        relative_value_logits=np.zeros((2, 256), np.float32),
        # Raw shape +2.0; sparse -1.0 plus kappa=0.1 gives final -0.8.
        pbrs_shaping_reward=np.asarray([2.0]),
        pbrs_chunk_reward=np.asarray([-0.8]),
    )
    prepared = tmp_path / "prepared.json"
    prepared.write_text(json.dumps({
        "dataset_sha256": "dataset",
        "episodes": [{
            "run_id": "run-1", "trajectory_path": str(trajectory),
            "trajectory_sha256": sha(trajectory),
            "observations_sha256": sha(observations),
        }],
    }), encoding="utf-8")
    rewards = tmp_path / "rewards.json"
    rewards.write_text(json.dumps({
        "schema_version": 1,
        "kind": "derived_iql_reward",
        "complete": True, "dataset_sha256": "dataset",
        "annotator": {"model": "RynnValue-4B", "revision": "fixed"},
        "reward_config": {"gamma": 0.99, "shaping_weight": 0.1},
        "episodes": [{
            "run_id": "run-1", "source_key": "key",
            "annotation_path": str(annotation),
            "annotation_sha256": sha(annotation),
            "environment_success": True,
            "official_outputs": {
                "inference_method": "prefix_uniform_last_slot",
                "prefix_image_slots": 4,
                "analysis": {
                    "generated_text": "- Match: Yes\n- Success: Yes",
                    "generated_token_ids": [1, 2],
                    "parsed_for_display": {"match": "Yes", "success": "Yes"},
                },
            },
            "pbrs_reward": {
                "array_keys": ["pbrs_shaping_reward", "pbrs_chunk_reward"],
            },
        }],
    }), encoding="utf-8")
    run = {
        "id": "run-1", "trajectory": str(trajectory), "artifacts": {},
    }
    service = TrajectoryEvaluationService(Runs(run), tmp_path)
    result = service.bind(prepared, rewards, overwrite=False)
    assert result["bound"] == ["run-1"]
    assert (episode / SIDECAR_NAME).is_file()
    assert (episode / VALUES_NAME).is_file()
    assert service.status(run)["status"] == "READY"
    assert service.bind(prepared, rewards, overwrite=False)["skipped"] == ["run-1"]
    detail = service.detail("run-1")["evaluation"]
    np.testing.assert_allclose(
        detail["official_outputs"]["relative_temporal_distance_seconds"], [0.0, 0.1]
    )
    assert detail["official_outputs"]["analysis"]["parsed_for_display"]["match"] == "Yes"
    assert detail["pbrs_reward"] == {
        "sparse_reward": [-1.0],
        "dense_reward": [0.2],
        "shape_reward": [2.0],
        "final_reward": [-0.8],
        "chunk_start_steps": [0],
            "chunk_end_steps": [2],
            "chunk_lengths": [2],
            "accumulate_primitive_steps": False,
            "description": None,
    }
