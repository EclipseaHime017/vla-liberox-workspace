"""Repository for managed run metadata and aggregate metrics."""

from __future__ import annotations

import json
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


class TrainingDatasetRepository:
    """SQLite index for immutable dataset manifests and their run references."""

    def __init__(self, database_path: Path, project_id: str):
        self.database_path = database_path
        self.project_id = project_id
        migrate(database_path)

    def upsert(self, dataset: dict[str, Any], manifest_path: Path) -> None:
        members = dataset.get("members") or []
        with connect(self.database_path) as database:
            database.execute(
                """
                INSERT INTO training_datasets (
                    id, project_id, task_id, name, status, integrity_status,
                    annotation_status, member_count, parent_dataset_id,
                    manifest_path, created_at, updated_at
                ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, CURRENT_TIMESTAMP)
                ON CONFLICT(id) DO UPDATE SET
                    name=excluded.name,
                    status=excluded.status,
                    integrity_status=excluded.integrity_status,
                    annotation_status=excluded.annotation_status,
                    member_count=excluded.member_count,
                    manifest_path=excluded.manifest_path,
                    updated_at=CURRENT_TIMESTAMP
                """,
                (
                    dataset["id"], self.project_id, dataset["task_id"], dataset["name"],
                    dataset["status"], dataset["integrity_status"],
                    dataset["annotation_status"], len(members),
                    dataset.get("parent_dataset_id"), str(manifest_path.resolve()),
                    dataset["created_at"],
                ),
            )
            database.execute(
                "DELETE FROM training_dataset_members WHERE dataset_id = ?",
                (dataset["id"],),
            )
            database.executemany(
                """
                INSERT INTO training_dataset_members (
                    dataset_id, run_id, source_type, outcome
                ) VALUES (?, ?, ?, ?)
                """,
                [
                    (
                        dataset["id"], member["run_id"], member["source_type"],
                        member["outcome"],
                    )
                    for member in members
                ],
            )

    def references_for_run(self, run_id: str) -> list[dict[str, Any]]:
        with connect(self.database_path) as database:
            rows = database.execute(
                """
                SELECT d.id, d.name, d.status, d.integrity_status
                FROM training_dataset_members m
                JOIN training_datasets d ON d.id = m.dataset_id
                WHERE m.run_id = ? AND d.project_id = ?
                ORDER BY d.created_at DESC
                """,
                (run_id, self.project_id),
            ).fetchall()
        return [dict(row) for row in rows]

    def delete(self, dataset_id: str) -> None:
        with connect(self.database_path) as database:
            database.execute(
                "DELETE FROM training_datasets WHERE id = ? AND project_id = ?",
                (dataset_id, self.project_id),
            )


class OfflineJobRepository:
    """Small recoverable index; job.json remains the durable source of truth."""

    def __init__(self, database_path: Path, project_id: str):
        self.database_path = database_path
        self.project_id = project_id
        migrate(database_path)

    def upsert(self, job: dict[str, Any], job_path: Path) -> None:
        with connect(self.database_path) as database:
            database.execute(
                """
                INSERT INTO offline_jobs (
                    id, project_id, kind, status, dataset_id, job_path, pid,
                    created_at, completed_at, updated_at
                ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, CURRENT_TIMESTAMP)
                ON CONFLICT(id) DO UPDATE SET
                    status=excluded.status, pid=excluded.pid,
                    completed_at=excluded.completed_at,
                    updated_at=CURRENT_TIMESTAMP
                """,
                (
                    job["id"], self.project_id, job["kind"], job["status"],
                    job.get("dataset_id"), str(job_path.resolve()), job.get("pid"),
                    job["created_at"], job.get("completed_at"),
                ),
            )
            if job["kind"] == "annotation":
                output = Path(str(job.get("output_path") or ""))
                reward_manifest = output / "work" / "rewards" / "reward_manifest.json"
                database.execute(
                    """
                    INSERT INTO annotation_runs (
                        id, project_id, dataset_id, job_id, status,
                        manifest_path, created_at, completed_at
                    ) VALUES (?, ?, ?, ?, ?, ?, ?, ?)
                    ON CONFLICT(id) DO UPDATE SET
                        status=excluded.status,
                        manifest_path=excluded.manifest_path,
                        completed_at=excluded.completed_at
                    """,
                    (
                        job["id"], self.project_id, job["dataset_id"], job["id"],
                        job["status"],
                        str(reward_manifest.resolve()) if reward_manifest.is_file() else None,
                        job["created_at"],
                        job.get("completed_at"),
                    ),
                )

            elif job["kind"] == "training":
                overlay = None
                output = Path(str(job.get("output_path") or ""))
                if output.is_dir():
                    candidates = sorted(output.glob("*/summary.json"))
                    if candidates:
                        try:
                            summary = json.loads(candidates[-1].read_text(encoding="utf-8"))
                            overlay = summary.get("policy_overlay")
                        except Exception:
                            overlay = None
                database.execute(
                    """
                    INSERT INTO training_runs (
                        id, project_id, dataset_id, annotation_id, job_id,
                        status, output_path, overlay_path, created_at, completed_at
                    ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                    ON CONFLICT(id) DO UPDATE SET
                        status=excluded.status,
                        output_path=excluded.output_path,
                        overlay_path=excluded.overlay_path,
                        completed_at=excluded.completed_at
                    """,
                    (
                        job["id"], self.project_id, job["dataset_id"],
                        (job.get("parameters") or {}).get("annotation_id", ""),
                        job["id"], job["status"], job.get("output_path"), overlay,
                        job["created_at"], job.get("completed_at"),
                    ),
                )

    def references_for_dataset(self, dataset_id: str) -> list[dict[str, Any]]:
        with connect(self.database_path) as database:
            rows = database.execute(
                """
                SELECT id, kind, status
                FROM offline_jobs
                WHERE project_id = ? AND dataset_id = ?
                ORDER BY created_at DESC
                """,
                (self.project_id, dataset_id),
            ).fetchall()
        return [dict(row) for row in rows]
