"""Shared ownership, artifact and lifecycle rules of queue-backed analysis tools."""
from __future__ import annotations

import json
from pathlib import Path
from typing import Any


def read_json(path: Path, *, limit: int = 2_000_000) -> dict[str, Any]:
    if path.is_symlink() or not path.is_file() or path.stat().st_size > limit:
        raise ValueError(f"Missing, unsafe, or oversized tool metadata: {path.name}")
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise ValueError(f"Tool metadata must be an object: {path.name}")
    return value


class ToolJobService:
    """A tool's jobs live in the work queue; results are whitelisted files of its job directory."""

    KIND: str
    ROUTE: str
    ARTIFACTS: frozenset[str]
    # Small JSON results attached to the job detail: file name -> detail key.
    DETAILS: dict[str, str] = {}
    JOB_FILES = frozenset({"effective_config.yaml"})  # beside the artifacts directory

    def __init__(self, jobs):
        self.jobs = jobs

    def _owned_job(self, job_id: str) -> tuple[Path, dict]:
        path, job = self.jobs._load_job(job_id)
        directory = self.jobs.jobs_root / job_id
        if (job.get("kind") != self.KIND or directory.is_symlink()
                or directory.resolve().parent != self.jobs.jobs_root.resolve()):
            raise ValueError(f"This is not an owned {self.KIND} job")
        return path, job

    def list(self) -> dict[str, list]:
        return {"jobs": [job for job in self.jobs.list() if job.get("kind") == self.KIND]}

    def artifact(self, job_id: str, filename: str) -> Path:
        self._owned_job(job_id)
        if filename not in self.ARTIFACTS:
            raise ValueError(f"Unknown {self.KIND} artifact")
        directory = self.jobs.jobs_root / job_id
        path = directory / filename if filename in self.JOB_FILES else directory / "artifacts" / filename
        if (path.is_symlink() or path.parent.is_symlink() or not path.is_file()
                or not path.resolve().is_relative_to(directory.resolve())):
            raise FileNotFoundError(f"Tool artifact is missing or unsafe: {filename}")
        return path

    def get(self, job_id: str) -> dict[str, Any]:
        self._owned_job(job_id)
        result = self.jobs.get(job_id)
        result["artifacts"] = {}
        for filename in sorted(self.ARTIFACTS):
            try:
                self.artifact(job_id, filename)
            except FileNotFoundError:
                continue
            result["artifacts"][filename] = f"{self.ROUTE}/{job_id}/artifacts/{filename}"
        for filename, key in self.DETAILS.items():
            path = self.jobs.jobs_root / job_id / "artifacts" / filename
            if path.exists() and not path.parent.is_symlink():
                try:
                    result[key] = read_json(path)
                except (ValueError, OSError, json.JSONDecodeError):
                    continue
        return result

    def stop(self, job_id: str) -> dict:
        self._owned_job(job_id)
        return self.jobs.stop(job_id)

    def delete(self, job_id: str, confirmation: str) -> dict:
        self._owned_job(job_id)
        return self.jobs.delete_job(job_id, confirmation)
