from __future__ import annotations

import hashlib
import json
from pathlib import Path

import numpy as np

from backend.app.services.robometer_evaluation_service import (
    RobometerEvaluationService, SIDECAR_NAME, VALUES_NAME,
)


def sha(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


class Runs:
    def __init__(self, run): self.run = run
    def get_run(self, run_id):
        assert run_id == self.run["id"]
        return self.run


def test_bind_and_source_hash_invalidation(tmp_path: Path):
    episode = tmp_path / "episode_000"
    episode.mkdir()
    trajectory = episode / "trajectory.npz"
    observations = episode / "trajectory_observations.npz"
    np.savez_compressed(trajectory, time_seconds=np.arange(5) / 20, env_action=np.zeros((4, 7)))
    np.savez_compressed(observations, agentview_image=np.zeros((5, 2, 2, 3), np.uint8))
    run_manifest = tmp_path / "run.json"
    run_manifest.write_text(json.dumps({"id": "run", "task": "do task"}))
    run = {"id": "run", "trajectory": str(trajectory), "output_dir": str(tmp_path)}
    source = tmp_path / "result.npz"
    np.savez_compressed(
        source, observation_steps=[0, 4], time_seconds=[0, .2],
        progress_pred=[0.1, 0.8], success_probs=[0.0, 0.9],
    )
    manifest = tmp_path / "robometer_manifest.json"
    manifest.write_text(json.dumps({
        "schema_version": 1, "complete": True,
        "annotator": {"model": "Robometer", "revision": "rev"},
        "evaluation_config": {"fps": 3},
        "episodes": [{
            "run_id": "run", "annotation_path": str(source),
            "values_sha256": sha(source), "trajectory_sha256": sha(trajectory),
            "observations_sha256": sha(observations), "manifest_sha256": sha(run_manifest),
            "source_key": "key", "sample_count": 2,
        }],
    }))
    service = RobometerEvaluationService(Runs(run), tmp_path)
    assert service.bind(manifest, overwrite=False)["bound"] == ["run"]
    assert (episode / SIDECAR_NAME).is_file() and (episode / VALUES_NAME).is_file()
    assert service.status(run)["status"] == "READY"
    np.savez_compressed(observations, agentview_image=np.ones((5, 2, 2, 3), np.uint8))
    # Catalog status is deliberately sidecar-only; strict consumers still
    # detect source changes without making every list request decompress data.
    assert service.status(run)["status"] == "READY"
    assert not service.exists(run)
    assert service.detail(run) is None


def test_catalog_status_does_not_hash_or_open_observations(tmp_path: Path, monkeypatch):
    episode = tmp_path / "episode_000"
    episode.mkdir()
    (episode / "trajectory.npz").write_bytes(b"trajectory")
    (episode / "trajectory_observations.npz").write_bytes(b"large observation archive")
    (episode / VALUES_NAME).write_bytes(b"values")
    (episode / SIDECAR_NAME).write_text(json.dumps({
        "schema_version": 1, "run_id": "run", "sample_count": 2,
        "annotator": {"model": "Robometer", "revision": "rev"},
    }))
    (tmp_path / "run.json").write_text(json.dumps({"id": "run"}))
    run = {"id": "run", "trajectory": str(episode / "trajectory.npz"), "output_dir": str(tmp_path)}
    service = RobometerEvaluationService(Runs(run), tmp_path)
    monkeypatch.setattr(
        "backend.app.services.robometer_evaluation_service._sha256",
        lambda _path: (_ for _ in ()).throw(AssertionError("list status must not hash files")),
    )
    assert service.status(run)["status"] == "READY"
