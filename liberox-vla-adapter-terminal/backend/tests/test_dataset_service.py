from __future__ import annotations

import zipfile
from pathlib import Path

from backend.app.services.dataset_service import DatasetService


class FakeRunService:
    def __init__(self, runs):
        self.runs = runs

    def list_runs(self):
        return self.runs


def test_dataset_service_filters_by_task_and_exports_offline_rl_bundle(tmp_path: Path):
    summary = tmp_path / "summary.json"
    trajectory = tmp_path / "trajectory.csv"
    agentview_video = tmp_path / "agentview.mp4"
    vla_views_video = tmp_path / "vla_views.mp4"
    summary.write_text("{}\n", encoding="utf-8")
    trajectory.write_text("step,success\n0,false\n", encoding="utf-8")
    agentview_video.write_bytes(b"external-video")
    vla_views_video.write_bytes(b"policy-views-video")
    service = DatasetService(FakeRunService([
        {
            "id": "run-a", "task_id": "task-a", "task_name": "A",
            "created_at": "2026-08-17T01:00:00Z", "success": False,
            "kind": "original", "control_mode": "policy", "status": "COMPLETED",
            "artifacts": {
                "summary.json": str(summary),
                "episodes/episode_000/trajectory.csv": str(trajectory),
                "episodes/episode_000/agentview.mp4": str(agentview_video),
                "episodes/episode_000/vla_views.mp4": str(vla_views_video),
            },
        },
        {
            "id": "run-b", "task_id": "task-b", "task_name": "B",
            "created_at": "2026-08-17T02:00:00Z", "success": True,
            "kind": "original", "control_mode": "policy", "status": "COMPLETED",
            "artifacts": {},
        },
    ]))

    assert [run["id"] for run in service.list_runs("task-a")] == ["run-a"]
    export_path, filename = service.export_task("task-a")
    try:
        assert filename == "liberox_task-a.zip"
        with zipfile.ZipFile(export_path) as archive:
            names = set(archive.namelist())
            assert "runs.csv" in names
            assert "export.json" in names
            assert "DATA_FORMAT.md" in names
            assert "runs/run-a/summary.json" in names
            assert "runs/run-a/episodes/episode_000/trajectory.csv" in names
            assert "runs/run-a/episodes/episode_000/agentview.mp4" in names
            assert "runs/run-a/episodes/episode_000/vla_views.mp4" in names
            assert archive.read("runs/run-a/episodes/episode_000/agentview.mp4") == b"external-video"
            assert archive.getinfo("runs/run-a/episodes/episode_000/agentview.mp4").compress_type == zipfile.ZIP_STORED
            assert "action_source" in archive.read("DATA_FORMAT.md").decode("utf-8")
            manifest = archive.read("runs.csv").decode("utf-8")
            assert "episode_category" in manifest
            assert "unassisted_failure" in manifest
    finally:
        export_path.unlink(missing_ok=True)
