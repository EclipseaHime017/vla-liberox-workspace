"""Repository for managed run metadata and aggregate metrics."""

from __future__ import annotations

from pathlib import Path
from typing import Any

from .database import connect, migrate


class RunRepository:
    def __init__(self, database_path: Path, project_id: str):
        self.database_path = database_path
        self.project_id = project_id
        migrate(database_path)

    def upsert(self, run: dict[str, Any], run_path: Path) -> None:
        with connect(self.database_path) as database:
            database.execute(
                """
                INSERT INTO runs (
                    id, project_id, kind, task_id, task_name, level, status,
                    success, action_count, parent_session_id, created_at,
                    completed_at, run_path, updated_at
                ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, CURRENT_TIMESTAMP)
                ON CONFLICT(id) DO UPDATE SET
                    kind=excluded.kind,
                    task_id=excluded.task_id,
                    task_name=excluded.task_name,
                    level=excluded.level,
                    status=excluded.status,
                    success=excluded.success,
                    action_count=excluded.action_count,
                    parent_session_id=excluded.parent_session_id,
                    completed_at=excluded.completed_at,
                    run_path=excluded.run_path,
                    updated_at=CURRENT_TIMESTAMP
                """,
                (
                    run["id"], self.project_id, run.get("kind", "original"),
                    run.get("task_id"), run.get("task_name"), run.get("level"),
                    run.get("status", "ERROR"), int(bool(run.get("success"))),
                    int(run.get("action_count", 0) or 0), run.get("parent_session_id"),
                    run.get("created_at"), run.get("completed_at"), str(run_path.resolve()),
                ),
            )

    def delete(self, run_id: str) -> None:
        with connect(self.database_path) as database:
            database.execute(
                "DELETE FROM runs WHERE id = ? AND project_id = ?",
                (run_id, self.project_id),
            )

    def summary(self) -> dict[str, Any]:
        with connect(self.database_path) as database:
            totals = database.execute(
                """
                SELECT COUNT(*) AS total,
                       SUM(CASE WHEN status = 'COMPLETED' THEN 1 ELSE 0 END) AS completed,
                       SUM(CASE WHEN status = 'ERROR' THEN 1 ELSE 0 END) AS errors,
                       SUM(CASE WHEN success = 1 THEN 1 ELSE 0 END) AS successes
                FROM runs WHERE project_id = ?
                """,
                (self.project_id,),
            ).fetchone()
            tasks = database.execute(
                """
                SELECT task_id, task_name, level, COUNT(*) AS runs,
                       SUM(CASE WHEN success = 1 THEN 1 ELSE 0 END) AS successes
                FROM runs WHERE project_id = ?
                GROUP BY task_id, task_name, level ORDER BY runs DESC, task_name
                """,
                (self.project_id,),
            ).fetchall()
        total = int(totals["total"] or 0)
        successes = int(totals["successes"] or 0)
        return {
            "project_id": self.project_id,
            "runs": total,
            "completed": int(totals["completed"] or 0),
            "errors": int(totals["errors"] or 0),
            "successes": successes,
            "success_rate": successes / total if total else 0.0,
            "tasks": [
                {
                    **dict(row),
                    "runs": int(row["runs"]),
                    "successes": int(row["successes"] or 0),
                    "success_rate": int(row["successes"] or 0) / int(row["runs"]),
                }
                for row in tasks
            ],
        }
