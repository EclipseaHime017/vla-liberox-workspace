"""SQLite catalog connection and schema migration.

The catalog is an index. Run files remain the durable source of truth, which
makes experiments portable and recoverable without the database.
"""

from __future__ import annotations

import sqlite3
from pathlib import Path


SCHEMA_VERSION = 3


def connect(path: Path) -> sqlite3.Connection:
    path.parent.mkdir(parents=True, exist_ok=True)
    connection = sqlite3.connect(path, timeout=10.0)
    connection.row_factory = sqlite3.Row
    connection.execute("PRAGMA foreign_keys = ON")
    connection.execute("PRAGMA journal_mode = WAL")
    return connection


def migrate(path: Path) -> None:
    with connect(path) as database:
        database.executescript(
            """
            CREATE TABLE IF NOT EXISTS schema_info (
                version INTEGER NOT NULL
            );
            CREATE TABLE IF NOT EXISTS runs (
                id TEXT PRIMARY KEY,
                project_id TEXT NOT NULL,
                kind TEXT NOT NULL,
                task_id TEXT,
                task_name TEXT,
                level TEXT,
                status TEXT NOT NULL,
                success INTEGER NOT NULL DEFAULT 0,
                action_count INTEGER NOT NULL DEFAULT 0,
                parent_session_id TEXT,
                created_at TEXT,
                completed_at TEXT,
                run_path TEXT NOT NULL UNIQUE,
                updated_at TEXT NOT NULL DEFAULT CURRENT_TIMESTAMP
            );
            CREATE INDEX IF NOT EXISTS idx_runs_project_created
                ON runs(project_id, created_at DESC);
            CREATE INDEX IF NOT EXISTS idx_runs_project_task
                ON runs(project_id, task_id);
            CREATE INDEX IF NOT EXISTS idx_runs_status
                ON runs(status);
            CREATE TABLE IF NOT EXISTS training_datasets (
                id TEXT PRIMARY KEY,
                project_id TEXT NOT NULL,
                task_id TEXT NOT NULL,
                name TEXT NOT NULL,
                status TEXT NOT NULL,
                integrity_status TEXT NOT NULL,
                annotation_status TEXT NOT NULL,
                member_count INTEGER NOT NULL,
                parent_dataset_id TEXT,
                manifest_path TEXT NOT NULL UNIQUE,
                created_at TEXT NOT NULL,
                updated_at TEXT NOT NULL DEFAULT CURRENT_TIMESTAMP
            );
            CREATE INDEX IF NOT EXISTS idx_training_datasets_project_task
                ON training_datasets(project_id, task_id, created_at DESC);
            CREATE TABLE IF NOT EXISTS training_dataset_members (
                dataset_id TEXT NOT NULL,
                run_id TEXT NOT NULL,
                source_type TEXT NOT NULL,
                outcome TEXT NOT NULL,
                PRIMARY KEY(dataset_id, run_id),
                FOREIGN KEY(dataset_id) REFERENCES training_datasets(id) ON DELETE CASCADE
            );
            CREATE INDEX IF NOT EXISTS idx_training_dataset_members_run
                ON training_dataset_members(run_id);
            CREATE TABLE IF NOT EXISTS offline_jobs (
                id TEXT PRIMARY KEY,
                project_id TEXT NOT NULL,
                kind TEXT NOT NULL,
                status TEXT NOT NULL,
                dataset_id TEXT,
                job_path TEXT NOT NULL UNIQUE,
                pid INTEGER,
                created_at TEXT NOT NULL,
                completed_at TEXT,
                updated_at TEXT NOT NULL DEFAULT CURRENT_TIMESTAMP
            );
            CREATE INDEX IF NOT EXISTS idx_offline_jobs_project_created
                ON offline_jobs(project_id, created_at DESC);
            CREATE TABLE IF NOT EXISTS annotation_runs (
                id TEXT PRIMARY KEY,
                project_id TEXT NOT NULL,
                dataset_id TEXT NOT NULL,
                job_id TEXT NOT NULL,
                status TEXT NOT NULL,
                manifest_path TEXT,
                created_at TEXT NOT NULL,
                completed_at TEXT
            );
            CREATE TABLE IF NOT EXISTS training_runs (
                id TEXT PRIMARY KEY,
                project_id TEXT NOT NULL,
                dataset_id TEXT NOT NULL,
                annotation_id TEXT NOT NULL,
                job_id TEXT NOT NULL,
                status TEXT NOT NULL,
                output_path TEXT,
                overlay_path TEXT,
                created_at TEXT NOT NULL,
                completed_at TEXT
            );
            CREATE TABLE IF NOT EXISTS evaluation_runs (
                id TEXT PRIMARY KEY,
                project_id TEXT NOT NULL,
                task_id TEXT NOT NULL,
                policy_id TEXT NOT NULL,
                job_id TEXT NOT NULL,
                status TEXT NOT NULL,
                trials INTEGER NOT NULL,
                attempted INTEGER NOT NULL DEFAULT 0,
                successes INTEGER NOT NULL DEFAULT 0,
                result_path TEXT NOT NULL UNIQUE,
                created_at TEXT NOT NULL,
                completed_at TEXT
            );
            CREATE INDEX IF NOT EXISTS idx_evaluation_runs_project_created
                ON evaluation_runs(project_id, created_at DESC);
            CREATE INDEX IF NOT EXISTS idx_evaluation_runs_project_task
                ON evaluation_runs(project_id, task_id, created_at DESC);
            CREATE INDEX IF NOT EXISTS idx_evaluation_runs_project_policy
                ON evaluation_runs(project_id, policy_id, created_at DESC);
            """
        )
        row = database.execute("SELECT version FROM schema_info LIMIT 1").fetchone()
        if row is None:
            database.execute("INSERT INTO schema_info(version) VALUES (?)", (SCHEMA_VERSION,))
        elif int(row["version"]) in {1, 2}:
            database.execute("UPDATE schema_info SET version = ?", (SCHEMA_VERSION,))
        elif int(row["version"]) != SCHEMA_VERSION:
            raise RuntimeError(
                f"Unsupported catalog schema {row['version']}; expected {SCHEMA_VERSION}"
            )
