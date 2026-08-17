from pathlib import Path

from backend.app.storage.repositories import RunRepository


def test_catalog_upsert_aggregate_and_delete(tmp_path: Path):
    repository = RunRepository(tmp_path / "catalog.sqlite3", "libero_x_vla")
    run = {
        "id": "run-a", "kind": "original", "task_id": "LEVEL1::task",
        "task_name": "task", "level": "LEVEL1", "status": "COMPLETED",
        "success": True, "action_count": 42, "created_at": "2026-08-15T00:00:00Z",
    }
    repository.upsert(run, tmp_path / "projects/libero_x_vla/runs/task/2026-08-15/run-a")
    summary = repository.summary()
    assert summary["runs"] == 1
    assert summary["successes"] == 1
    assert summary["success_rate"] == 1.0
    assert summary["tasks"][0]["runs"] == 1
    repository.delete("run-a")
    assert repository.summary()["runs"] == 0
