"""Task-oriented run listing and portable lightweight exports."""

from __future__ import annotations

import csv
import io
import json
import re
import tempfile
import zipfile
from pathlib import Path
from typing import Any


EXPORT_ARTIFACT_NAMES = frozenset({"run.json", "config.yaml", "summary.json"})


class DatasetService:
    def __init__(self, run_service: Any):
        self.run_service = run_service

    def list_runs(self, task_id: str | None = None) -> list[dict[str, Any]]:
        runs = self.run_service.list_runs()
        if task_id:
            runs = [run for run in runs if run.get("task_id") == task_id]
        return sorted(
            runs,
            key=lambda run: str(run.get("created_at") or ""),
            reverse=True,
        )

    @staticmethod
    def _manifest_csv(runs: list[dict[str, Any]]) -> str:
        output = io.StringIO(newline="")
        fields = [
            "id", "task_id", "task_name", "task", "kind", "control_mode",
            "status", "success", "action_count", "max_steps", "created_at",
            "completed_at", "parent_session_id",
        ]
        writer = csv.DictWriter(output, fieldnames=fields)
        writer.writeheader()
        for run in runs:
            writer.writerow({key: run.get(key) for key in fields})
        return output.getvalue()

    @staticmethod
    def _exportable_artifacts(run: dict[str, Any]):
        for logical_name, value in (run.get("artifacts") or {}).items():
            path = Path(str(value)).resolve()
            basename = path.name
            if basename not in EXPORT_ARTIFACT_NAMES and basename != "trajectory.csv":
                continue
            if path.is_file():
                yield str(logical_name), path

    def export_task(self, task_id: str) -> tuple[Path, str]:
        if not task_id.strip():
            raise ValueError("task_id is required")
        runs = self.list_runs(task_id)
        if not runs:
            raise FileNotFoundError(f"No runs found for task {task_id!r}")
        slug = re.sub(r"[^A-Za-z0-9._-]+", "_", task_id).strip("_.") or "task"
        handle = tempfile.NamedTemporaryFile(
            prefix=f"liberox_{slug}_",
            suffix=".zip",
            delete=False,
        )
        export_path = Path(handle.name)
        handle.close()
        try:
            with zipfile.ZipFile(
                export_path,
                mode="w",
                compression=zipfile.ZIP_DEFLATED,
                compresslevel=6,
            ) as archive:
                archive.writestr("runs.csv", self._manifest_csv(runs))
                archive.writestr(
                    "export.json",
                    json.dumps(
                        {
                            "schema_version": 1,
                            "task_id": task_id,
                            "run_count": len(runs),
                            "contents": [
                                "run/config/summary metadata when available",
                                "trajectory.csv when available",
                            ],
                            "excluded": [
                                "videos",
                                "trajectory.npz",
                                "observation images",
                                "plots",
                            ],
                        },
                        ensure_ascii=False,
                        indent=2,
                    )
                    + "\n",
                )
                for run in runs:
                    seen: set[str] = set()
                    for logical_name, artifact in self._exportable_artifacts(run):
                        archive_name = f"runs/{run['id']}/{Path(logical_name).name}"
                        if archive_name in seen:
                            continue
                        seen.add(archive_name)
                        archive.write(artifact, archive_name)
        except Exception:
            export_path.unlink(missing_ok=True)
            raise
        return export_path, f"liberox_{slug}.zip"

