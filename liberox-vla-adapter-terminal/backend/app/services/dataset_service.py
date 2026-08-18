"""Task-oriented run listing and portable offline-RL exports."""

from __future__ import annotations

import csv
import io
import json
import re
import tempfile
import zipfile
from pathlib import Path, PurePosixPath
from typing import Any


EXPORT_ARTIFACT_NAMES = frozenset({"run.json", "config.yaml", "summary.json"})
EXPORT_TRAJECTORY_NAMES = frozenset({"trajectory.csv", "trajectory_inference.csv"})
EXPORT_VIDEO_NAMES = frozenset({"agentview.mp4", "vla_views.mp4"})

DATA_FORMAT_MARKDOWN = """# LIBERO-X offline RL export

Each run is self-contained below `runs/<run_id>/`. `runs.csv` is the catalog;
`run.json`, `config.yaml`, and `summary.json` carry identity, configuration, and
outcome. Episode data remains under `episodes/episode_000/`.

## Transition alignment

`trajectory.csv` contains `N+1` state rows for `N` actions. For every
`i < N`, state row `i` and its action columns describe
`state[i] -- action[i] --> state[i+1]`. The final row has empty action fields.
`vla_action_*` is the raw command before environment conversion; despite the
legacy column name it contains SpaceMouse input on `action_source=human` rows.
`action_*` is the normalized OSC_POSE command sent to LIBERO-X.

## Segments and outcomes

- `policy`: action executed by the original VLA rollout.
- `policy_requery`: VLA action generated after restoring a branch point.
- `human`: SpaceMouse action after manual takeover.

A branch trajectory is already merged: rows before `resume_step` are a physical
copy of the parent prefix and rows from `resume_step` onward are the new suffix.
Use `action_source` to segment transitions; do not infer segments from filenames
or action values. An unassisted failed rollout is
`kind=original`, `control_mode=policy`, and `success=false` in `runs.csv`.
`episode_category`, `prefix_action_source`, and `suffix_action_source` provide
run-level filters; transition-level `action_source` remains authoritative.

## Videos

Both videos contain one frame per executed action and use the control-step frame
rate. `agentview.mp4` is the high-resolution external view. `vla_views.mp4` is
the exact synchronized policy input mosaic: its left half is `agentview` and its
right half is `robot0_eye_in_hand`. Frame `i` aligns with action row `i`.
"""


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
    def _episode_category(run: dict[str, Any]) -> str:
        if run.get("status") == "ERROR" or run.get("stopped_reason") in {
            "user_stop", "controller_stop", "error",
        }:
            return "error_or_incomplete"
        if run.get("kind") == "branch":
            return (
                "manual_intervention"
                if run.get("control_mode") == "manual"
                else "policy_requery_branch"
            )
        if run.get("control_mode") == "policy":
            return "unassisted_success" if run.get("success") else "unassisted_failure"
        return "other"

    @staticmethod
    def _manifest_csv(runs: list[dict[str, Any]]) -> str:
        output = io.StringIO(newline="")
        fields = [
            "id", "task_id", "task_name", "task", "kind", "control_mode",
            "status", "success", "action_count", "max_steps", "created_at",
            "completed_at", "parent_session_id", "root_session_id", "resume_step",
            "manual_source", "stopped_reason", "error", "episode_category", "prefix_action_source",
            "suffix_action_source",
        ]
        writer = csv.DictWriter(output, fieldnames=fields)
        writer.writeheader()
        for run in runs:
            row = {key: run.get(key) for key in fields}
            row["episode_category"] = DatasetService._episode_category(run)
            row["prefix_action_source"] = "policy"
            row["suffix_action_source"] = (
                "human"
                if run.get("kind") == "branch" and run.get("control_mode") == "manual"
                else "policy_requery"
                if run.get("kind") == "branch"
                else "policy"
            )
            writer.writerow(row)
        return output.getvalue()

    @staticmethod
    def _exportable_artifacts(run: dict[str, Any]):
        for logical_name, value in (run.get("artifacts") or {}).items():
            path = Path(str(value)).resolve()
            basename = path.name
            if basename not in (
                EXPORT_ARTIFACT_NAMES | EXPORT_TRAJECTORY_NAMES | EXPORT_VIDEO_NAMES
            ):
                continue
            logical_path = PurePosixPath(str(logical_name))
            if logical_path.is_absolute() or ".." in logical_path.parts:
                continue
            if path.is_file():
                yield logical_path.as_posix(), path

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
                archive.writestr("DATA_FORMAT.md", DATA_FORMAT_MARKDOWN)
                archive.writestr(
                    "export.json",
                    json.dumps(
                        {
                            "schema_version": 2,
                            "task_id": task_id,
                            "run_count": len(runs),
                            "contents": [
                                "run/config/summary metadata when available",
                                "trajectory.csv when available",
                                "trajectory_inference.csv when available",
                                "agentview.mp4 and vla_views.mp4 when available",
                            ],
                            "excluded": [
                                "trajectory.npz",
                                "observation images",
                                "plots",
                                "raw SpaceMouse diagnostic samples",
                            ],
                            "action_source": {
                                "policy": "original VLA action",
                                "policy_requery": "VLA action after branch restore",
                                "human": "SpaceMouse takeover action",
                            },
                            "episode_categories": [
                                "unassisted_success",
                                "unassisted_failure",
                                "manual_intervention",
                                "policy_requery_branch",
                                "error_or_incomplete",
                            ],
                            "video_alignment": "frame[i] aligns with trajectory action row i",
                        },
                        ensure_ascii=False,
                        indent=2,
                    )
                    + "\n",
                )
                for run in runs:
                    seen: set[str] = set()
                    for logical_name, artifact in self._exportable_artifacts(run):
                        archive_name = f"runs/{run['id']}/{logical_name}"
                        if archive_name in seen:
                            continue
                        seen.add(archive_name)
                        archive.write(
                            artifact,
                            archive_name,
                            compress_type=(
                                zipfile.ZIP_STORED
                                if artifact.suffix.lower() == ".mp4"
                                else zipfile.ZIP_DEFLATED
                            ),
                        )
        except Exception:
            export_path.unlink(missing_ok=True)
            raise
        return export_path, f"liberox_{slug}.zip"
