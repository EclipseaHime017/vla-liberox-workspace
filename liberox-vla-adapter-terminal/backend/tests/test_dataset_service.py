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
    rynn_metadata = tmp_path / "rynnvalue_evaluation.json"
    rynn_values = tmp_path / "rynnvalue_evaluation.npz"
    stage_marks = tmp_path / "stage_annotation.json"
    summary.write_text("{}\n", encoding="utf-8")
    trajectory.write_text("step,success\n0,false\n", encoding="utf-8")
    agentview_video.write_bytes(b"external-video")
    vla_views_video.write_bytes(b"policy-views-video")
    rynn_metadata.write_text("{}\n", encoding="utf-8")
    rynn_values.write_bytes(b"reward-values")
    stage_marks.write_text('{"schema_version": 1}\n', encoding="utf-8")
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
                "episodes/episode_000/rynnvalue_evaluation.json": str(rynn_metadata),
                "episodes/episode_000/rynnvalue_evaluation.npz": str(rynn_values),
                "episodes/episode_000/stage_annotation.json": str(stage_marks),
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
            assert "runs/run-a/episodes/episode_000/rynnvalue_evaluation.json" in names
            assert "runs/run-a/episodes/episode_000/rynnvalue_evaluation.npz" in names
            assert archive.read("runs/run-a/episodes/episode_000/stage_annotation.json") == stage_marks.read_bytes()
            assert archive.read("runs/run-a/episodes/episode_000/agentview.mp4") == b"external-video"
            assert archive.getinfo("runs/run-a/episodes/episode_000/agentview.mp4").compress_type == zipfile.ZIP_STORED
            assert "action_source" in archive.read("DATA_FORMAT.md").decode("utf-8")
            manifest = archive.read("runs.csv").decode("utf-8")
            assert "episode_category" in manifest
            assert "unassisted_failure" in manifest
    finally:
        export_path.unlink(missing_ok=True)


def test_task_export_skips_test_labelled_runs(tmp_path: Path):
    artifact = tmp_path / "summary.json"
    artifact.write_text("{}\n", encoding="utf-8")
    runs = [
        {
            "id": "train", "task_id": "task", "created_at": "2026-08-17T01:00:00Z",
            "artifacts": {"summary.json": str(artifact)},
        },
        {
            "id": "test", "task_id": "task", "created_at": "2026-08-17T02:00:00Z",
            "artifacts": {"summary.json": str(artifact)},
        },
    ]
    service = DatasetService(
        FakeRunService(runs), is_test=lambda run_id: run_id == "test",
    )
    export_path, _ = service.export_task("task")
    try:
        with zipfile.ZipFile(export_path) as archive:
            manifest = archive.read("runs.csv").decode("utf-8")
            assert "train" in manifest
            assert "test" not in manifest
            assert not any(name.startswith("runs/test/") for name in archive.namelist())
    finally:
        export_path.unlink(missing_ok=True)
