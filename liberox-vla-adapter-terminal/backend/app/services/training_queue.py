"""Shared durable FIFO dispatch for UI training and policy test jobs."""
from __future__ import annotations

import fcntl
import logging
import threading
from datetime import datetime, timezone

from ..core.exceptions import ConflictError
from ..storage.files import atomic_write_json


class TrainingQueue:
    def start_training_queue(self) -> None:
        self._queue_stop = threading.Event()
        self._queue_wake = threading.Event()
        self._queue_wait_reason = None
        self._queue_thread = threading.Thread(target=self._queue_loop, name="offline-queue", daemon=True)
        self._queue_thread.start()

    def _wake_training_queue(self) -> None:
        event = getattr(self, "_queue_wake", None)
        if event is not None:
            event.set()

    def close_training_queue(self) -> None:
        event = getattr(self, "_queue_stop", None)
        if event is not None:
            event.set()
            self._wake_training_queue()
            self._queue_thread.join()

    def enqueue_training(self, dataset_id: str, parameters: dict) -> dict:
        with self.lock:
            self._validate_training_parameters(parameters, self.training_models)
            if parameters.get("resume_checkpoint"):
                raise ValueError("Queued training starts independently; checkpoint resume is not supported")
            dataset = self.datasets.require_ready_for_training(dataset_id)
            version, normalized = self.training_inputs(dataset, parameters)
            # Reuse the normal config builder, but reserve no GPU and start no process.
            job = self._launch_training(dataset_id, dataset, {**normalized, "resume_checkpoint": None},
                                        reward_version=version, queued=True)
            self._wake_training_queue()
            return job

    def training_queue(self) -> dict:
        return self._job_queue("training")

    def evaluation_queue(self) -> dict:
        return self._job_queue("evaluation")

    def enqueue_evaluation(self, request: dict) -> dict:
        with self.lock:
            preview = self.preview_evaluation(request)
            if request.get("schedule_sha256") != preview["schedule_sha256"]:
                raise ValueError("测试调度已变化，请重新预览后注册")
            job = self._launch_evaluation(preview, queued=True)
            self._wake_training_queue()
            return job

    def _job_queue(self, kind: str) -> dict:
        # Only small job manifests: no model hashes, reward arrays, metrics or log reads.
        jobs = []
        for identifier in self.repository.queue_ids(kind=kind):
            try:
                _, job = self._load_job(identifier)
            except (OSError, ValueError, KeyError, TypeError):
                job = {**self.repository.indexed_job(identifier), "parameters": {}, "stage": "failed",
                       "stage_label": "任务记录无法读取", "error": "任务清单缺失或损坏，请检查后台日志。"}
            jobs.append({key: job.get(key) for key in (
                "id", "kind", "status", "dataset_id", "created_at", "started_at", "completed_at",
                "stage", "stage_label", "error", "parameters",
            )})
        return {"jobs": jobs, "waiting_reason": getattr(self, "_queue_wait_reason", None)}

    def _queue_loop(self) -> None:
        while not self._queue_stop.is_set():
            try:
                self._dispatch_training_queue()
            except Exception:
                logging.getLogger(__name__).exception("Offline queue dispatch failed; retrying")
            self._queue_wake.wait(2.0)
            self._queue_wake.clear()

    def _dispatch_training_queue(self) -> None:
        # Protect selection + STARTING publication across backend processes too.
        with self.lock, (self.project_root / ".training-queue.lock").open("a+") as dispatch_lock:
            try:
                fcntl.flock(dispatch_lock.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)
            except BlockingIOError:
                return
            if getattr(self, "_queue_stop", None) is not None and self._queue_stop.is_set():
                return
            queued = []
            for identifier in self.repository.queue_ids(pending_only=True):
                try:
                    _, job = self._load_job(identifier)
                    if job["status"] != "QUEUED":
                        self._reconcile(identifier)
                    else:
                        queued.append(job)
                except (OSError, ValueError, KeyError, TypeError):
                    logging.getLogger(__name__).exception("Cannot read queued job %s; skipping", identifier)
                    self.repository.fail_unreadable_job(identifier)
            if not queued:
                self._queue_wait_reason = None
                return
            selected = queued[0]
            try:
                self._prepare_launch()
            except ConflictError as exc:
                self._queue_wait_reason = str(exc)
                return
            except Exception as exc:
                self._queue_wait_reason = f"等待 GPU 任务资源：{exc}"
                return
            self._queue_wait_reason = None
            spawning = False
            try:
                selected.update(status="STARTING", stage="starting", stage_label="准备后台任务",
                                dispatched_at=datetime.now(timezone.utc).isoformat())
                path = self._job_path(selected["id"])
                atomic_write_json(path, selected)
                self.repository.upsert(selected, path)
                if selected["kind"] == "evaluation":
                    result_path = self._evaluation_manifest(selected["id"])
                    result = self._load_evaluation_manifest(result_path)
                    result["status"] = "STARTING"
                    atomic_write_json(result_path, result)
                    self.evaluation_repository.upsert(result, result_path)
                spawning = True
                self._spawn_job(selected)
            except Exception as exc:
                # _spawn_job owns launch failures; once Popen succeeds, only the
                # runner may update its manifest. Never overwrite a live child.
                if not spawning:
                    selected.update(status="FAILED", error=f"Cannot prepare queued job: {type(exc).__name__}: {exc}",
                                    completed_at=datetime.now(timezone.utc).isoformat())
                    atomic_write_json(self._job_path(selected["id"]), selected)
                    self.repository.upsert(selected, self._job_path(selected["id"]))
                logging.getLogger(__name__).exception("Cannot launch queued job %s", selected["id"])
                raise
            finally:
                self.launch_reserved = False
