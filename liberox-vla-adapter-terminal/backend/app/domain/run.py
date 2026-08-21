"""Run and draft state owned by the application domain.

These objects intentionally contain no MuJoCo, FastAPI, database, or filesystem
operations. Adapters add artifact URLs and persistence around them.
"""

from __future__ import annotations

import threading
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Any


TERMINAL_STATES = frozenset({"COMPLETED", "ERROR"})
ACTIVE_STATES = frozenset({"LOADING", "READY", "RUNNING", "STOPPING", "POSTPROCESSING"})


def utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


@dataclass
class SimulationSession:
    """One original or branch run and its observable state."""

    id: str
    kind: str
    output_dir: Path
    max_steps: int
    open_loop_steps: int
    seed: int = 0
    disabled_policy_cameras: tuple[str, ...] = ()
    policy_id: str = "base"
    policy_label: str | None = None
    policy_base_checkpoint: str | None = None
    policy_overlay: str | None = None
    policy_compatibility_sha256: str | None = None
    task_id: str = ""
    task_level: str | None = None
    task_name: str | None = None
    task_prompt: str | None = None
    parent_session_id: str | None = None
    root_session_id: str | None = None
    source_trajectory: str | None = None
    resume_step: int | None = None
    control_mode: str = "policy"
    manual_source: str | None = None
    manual_translation_gain: float | None = None
    manual_rotation_gain: float | None = None
    status: str = "IDLE"
    created_at: str = field(default_factory=utc_now)
    completed_at: str | None = None
    current_step: int = 0
    state_count: int = 0
    action_count: int = 0
    policy_queries: int = 0
    success: bool = False
    error: str | None = None
    stopped_reason: str | None = None
    measured_control_hz: float | None = None
    simulated_duration_seconds: float = 0.0
    trajectory: str | None = None
    branchable: bool = False
    managed: bool = True
    preparation_phase: str | None = "queued"
    preparation_message: str | None = "等待后台任务"
    countdown_remaining: int | None = None
    preview_ready: bool = False
    preparation_timing: dict[str, float] = field(default_factory=dict)
    stop_event: threading.Event = field(default_factory=threading.Event, repr=False)
    latest_jpeg: bytes | None = field(default=None, repr=False)
    latest_frame_version: int = field(default=0, repr=False)
    thread: threading.Thread | None = field(default=None, repr=False)
    manual_connected: bool = field(default=False, repr=False)
    preview_event: threading.Event = field(default_factory=threading.Event, repr=False)
    preview_error: str | None = field(default=None, repr=False)
    spacemouse_status: str | None = None
    spacemouse_connected: bool | None = None
    spacemouse_stale: bool | None = None
    spacemouse_latency_ms: float | None = None
    spacemouse_deadman_ms: int | None = None
    spacemouse_calibration: dict[str, Any] | None = field(default=None, repr=False)
    spacemouse_diagnostics: dict[str, Any] | None = field(default=None, repr=False)
    spacemouse_samples: list[dict[str, Any]] = field(default_factory=list, repr=False)

    @property
    def episode_dir(self) -> Path:
        return self.output_dir / "episodes" / "episode_000"

    def public(self, artifacts: dict[str, str] | None = None) -> dict[str, Any]:
        return {
            "id": self.id,
            "kind": self.kind,
            "task_id": self.task_id or None,
            "level": self.task_level,
            "task_name": self.task_name,
            "task": self.task_prompt,
            "parent_session_id": self.parent_session_id,
            "root_session_id": self.root_session_id,
            "source_trajectory": self.source_trajectory,
            "resume_step": self.resume_step,
            "control_mode": self.control_mode,
            "policy_id": self.policy_id,
            "policy_label": self.policy_label,
            "policy_base_checkpoint": self.policy_base_checkpoint,
            "policy_overlay": self.policy_overlay,
            "policy_compatibility_sha256": self.policy_compatibility_sha256,
            "manual_source": self.manual_source,
            "manual_translation_gain": self.manual_translation_gain,
            "manual_rotation_gain": self.manual_rotation_gain,
            "spacemouse_status": self.spacemouse_status,
            "spacemouse_connected": self.spacemouse_connected,
            "spacemouse_stale": self.spacemouse_stale,
            "spacemouse_latency_ms": self.spacemouse_latency_ms,
            "spacemouse_deadman_ms": self.spacemouse_deadman_ms,
            "status": self.status,
            "created_at": self.created_at,
            "completed_at": self.completed_at,
            "max_steps": self.max_steps,
            "open_loop_steps": self.open_loop_steps,
            "seed": self.seed,
            "disabled_policy_cameras": list(self.disabled_policy_cameras),
            "current_step": self.current_step,
            "state_count": self.state_count,
            "action_count": self.action_count,
            "policy_queries": self.policy_queries,
            "success": self.success,
            "error": self.error,
            "stopped_reason": self.stopped_reason,
            "measured_control_hz": self.measured_control_hz,
            "simulated_duration_seconds": self.simulated_duration_seconds,
            "output_dir": str(self.output_dir),
            "trajectory": self.trajectory,
            "branchable": self.branchable and self.status in TERMINAL_STATES,
            "legacy": False,
            "managed": self.managed,
            "preparation_phase": self.preparation_phase,
            "preparation_message": self.preparation_message,
            "countdown_remaining": self.countdown_remaining,
            "preview_ready": self.preview_ready,
            "preparation_timing": dict(self.preparation_timing),
            "artifacts": artifacts or {},
        }


@dataclass
class SimulationDraft:
    """One non-persistent, editable original-run draft."""

    id: str
    task_id: str
    max_steps: int
    open_loop_steps: int
    seed: int = 0
    disabled_policy_cameras: tuple[str, ...] = ()
    policy_id: str = "base"
    policy_label: str | None = None
    preview_status: str = "PREPARING"
    preview_revision: int = 1
    error: str | None = None
    latest_jpeg: bytes | None = field(default=None, repr=False)

    def public(self, task: dict[str, Any]) -> dict[str, Any]:
        return {
            "id": self.id,
            "task_id": self.task_id,
            "max_steps": self.max_steps,
            "open_loop_steps": self.open_loop_steps,
            "seed": self.seed,
            "disabled_policy_cameras": list(self.disabled_policy_cameras),
            "policy_id": self.policy_id,
            "policy_label": self.policy_label,
            "preview_status": self.preview_status,
            "preview_revision": self.preview_revision,
            "preview_ready": self.preview_status == "READY",
            "preview_available": self.latest_jpeg is not None,
            "error": self.error,
            "task": task,
        }
