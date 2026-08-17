from __future__ import annotations

import zipfile
from pathlib import Path

from backend.app.services.dataset_service import DatasetService


class FakeRunService:
    def __init__(self, runs):
        self.runs = runs

    def list_runs(self):
        return self.runs


def test_dataset_service_filters_by_task_and_exports_lightweight_bundle(tmp_path: Path):
    summary = tmp_path / "summary.json"
    trajectory = tmp_path / "trajectory.csv"
    video = tmp_path / "agentview.mp4"
    summary.write_text("{}\n", encoding="utf-8")
    trajectory.write_text("step,success\n0,false\n", encoding="utf-8")
    video.write_bytes(b"video")
    service = DatasetService(FakeRunService([
        {
            "id": "run-a", "task_id": "task-a", "task_name": "A",
            "created_at": "2026-08-17T01:00:00Z", "success": False,
            "artifacts": {
                "summary.json": str(summary),
                "episodes/episode_000/trajectory.csv": str(trajectory),
                "episodes/episode_000/agentview.mp4": str(video),
            },
        },
        {
            "id": "run-b", "task_id": "task-b", "task_name": "B",
            "created_at": "2026-08-17T02:00:00Z", "success": True,
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
            assert "runs/run-a/summary.json" in names
            assert "runs/run-a/trajectory.csv" in names
            assert all(not name.endswith(".mp4") for name in names)
    finally:
        export_path.unlink(missing_ok=True)

