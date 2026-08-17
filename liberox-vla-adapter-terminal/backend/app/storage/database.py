"""SQLite catalog connection and schema migration.

The catalog is an index. Run files remain the durable source of truth, which
makes experiments portable and recoverable without the database.
"""

from __future__ import annotations

import sqlite3
from pathlib import Path


SCHEMA_VERSION = 1


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
            """
        )
        row = database.execute("SELECT version FROM schema_info LIMIT 1").fetchone()
        if row is None:
            database.execute("INSERT INTO schema_info(version) VALUES (?)", (SCHEMA_VERSION,))
        elif int(row["version"]) != SCHEMA_VERSION:
            raise RuntimeError(
                f"Unsupported catalog schema {row['version']}; expected {SCHEMA_VERSION}"
            )
