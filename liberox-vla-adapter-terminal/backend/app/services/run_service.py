"""Application façade for run, draft, controller, and dataset use cases.

FastAPI depends on this service only. The service owns no MuJoCo object; all
long-running work is delegated to the simulation worker.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any


class RunService:
    def __init__(self, worker: Any):
        self.worker = worker

    def close(self): return self.worker.close()
    def bootstrap(self): return self.worker.bootstrap()
    def list_runs(self): return self.worker.list_sessions()
    def dataset_summary(self):
        method = getattr(self.worker, "dataset_summary", None)
        if method is None:
            runs = self.list_runs()
            successes = sum(bool(item.get("success")) for item in runs)
            return {"runs": len(runs), "successes": successes, "success_rate": successes / len(runs) if runs else 0.0, "tasks": []}
        return method()
    def get_run(self, run_id): return self.worker.get_public(run_id)
    def stop(self, run_id): return self.worker.stop(run_id)
    def create_branch(self, run_id, *args, **kwargs): return self.worker.create_branch(run_id, *args, **kwargs)
    def delete(self, run_id, confirmation): return self.worker.delete_session(run_id, confirmation)
    def frame(self, run_id, step): return self.worker.render_frame(run_id, step)
    def frame_state(self, run_id, step): return self.worker.frame_state(run_id, step)
    def artifact(self, run_id, name) -> Path: return self.worker.artifact_path(run_id, name)
    def get_draft(self): return self.worker.get_draft()
    def create_draft(self, *args): return self.worker.create_draft(*args)
    def update_draft(self, **kwargs): return self.worker.update_draft(**kwargs)
    def discard_draft(self): return self.worker.discard_draft()
    def draft_preview(self): return self.worker.draft_preview()
    def start_draft(self): return self.worker.start_draft()
    def controller_status(self): return self.worker.controller_status()
    def calibrate_controller(self): return self.worker.calibrate_controller()
    def manual_connect(self, run_id): return self.worker.manual_connect(run_id)
    def manual_disconnect(self, run_id): return self.worker.manual_disconnect(run_id)
    def manual_settings(self, run_id, translation, rotation): return self.worker.manual_settings(run_id, translation, rotation)

    def latest_frame(self, run_id: str) -> tuple[bytes | None, int]:
        lock = getattr(self.worker, "lock", None)
        sessions = getattr(self.worker, "sessions", {})
        if lock is None:
            return None, -1
        with lock:
            record = sessions.get(run_id)
            return (
                None if record is None else record.latest_jpeg,
                -1 if record is None else record.latest_frame_version,
            )
