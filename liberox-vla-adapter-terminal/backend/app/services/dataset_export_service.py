"""Bounded, CPU-only background export with lightweight progress polling."""
from __future__ import annotations

import threading
import uuid
import logging
from concurrent.futures import ThreadPoolExecutor
from contextlib import contextmanager
from datetime import datetime, timezone

from ..storage.files import atomic_write_json
from .dataset_export_rewards import read_json
from .dataset_transfer import export_dataset


class DatasetExportService:
    def __init__(self, datasets, config):
        self.datasets, self.config = datasets, config
        self.root = config.offline_rl_root.parent / "dataset-exports"
        self.lock = threading.RLock()
        self.worker = ThreadPoolExecutor(max_workers=1, thread_name_prefix="dataset-export")
        self.active = False
        self.closed = False
        self.latest: dict[str, dict] = {}
        for path in sorted((self.root / ".jobs").glob("*.json")):
            try:
                state = read_json(path)
                if state["status"] in {"QUEUED", "RUNNING"}:
                    state.update(status="FAILED", error="服务已重启；上次导出未完成，请重新导出")
                    atomic_write_json(path, state)
                previous = self.latest.get(state["dataset_id"])
                if previous is None or previous["created_at"] < state["created_at"]:
                    self.latest[state["dataset_id"]] = state
            except (OSError, ValueError, KeyError):
                continue

    @contextmanager
    def mutation_guard(self):
        """Dataset deletion must not remove active export inputs or legacy results."""
        with self.lock:
            if self.active:
                raise RuntimeError("数据集正在导出，请完成后再删除或修复存储")
            yield

    def get(self, dataset_id: str):
        with self.lock:
            return dict(self.latest[dataset_id]) if dataset_id in self.latest else None

    def start(self, dataset_id: str):
        with self.mutation_guard():
            if self.closed:
                raise RuntimeError("Export service is closing")
            # Heavy validation/copying starts only after the job is accepted.
            dataset = self.datasets.get(dataset_id, quick_verify=False)
            now = datetime.now(timezone.utc)
            identifier = f"{dataset_id}_{now:%Y%m%d_%H%M%S_%f}_{uuid.uuid4().hex[:8]}"
            state = {"id": identifier, "dataset_id": dataset_id, "status": "QUEUED",
                "created_at": now.isoformat(), "stage": "等待导出", "completed_runs": 0,
                "total_runs": dataset["member_count"], "current_run": "", "current_file": "",
                "output_path": None, "error": None}
            atomic_write_json(self.root / ".jobs" / f"{identifier}.json", state)
            self.latest[dataset_id] = state
            self.active = True
            try:
                self.worker.submit(self._run, state)
            except Exception as exc:
                self.active = False
                state.update(status="FAILED", error=str(exc))
                atomic_write_json(self.root / ".jobs" / f"{identifier}.json", state)
                raise
            return dict(state)

    def _run(self, state):
        def update(**values):
            with self.lock:
                state.update(values)
                snapshot = dict(state)
            atomic_write_json(self.root / ".jobs" / f"{state['id']}.json", snapshot)
        try:
            update(status="RUNNING")
            destination = export_dataset(self.config.project_root, state["dataset_id"],
                self.root / state["id"], self.config.offline_rl_root / "configs/training/iql.yaml",
                self.config.offline_rl_root, progress=update)
            update(status="COMPLETED", stage="导出完成", output_path=str(destination / "runs"),
                   current_file="", completed_at=datetime.now(timezone.utc).isoformat())
        except Exception as exc:
            try:
                update(status="FAILED", error=f"{type(exc).__name__}: {exc}")
            except OSError:
                logging.getLogger(__name__).exception("Cannot persist failed dataset export")
        finally:
            with self.lock:
                self.active = False

    def close(self):
        with self.lock:
            self.closed = True
        self.worker.shutdown(wait=True)
