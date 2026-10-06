"""Shared FIFO for inference, data processing, evaluation and training."""
from __future__ import annotations

import fcntl
import logging
import threading
import uuid
from concurrent.futures import Future
from datetime import datetime, timezone

from ..core.exceptions import ConflictError
from ..storage.files import atomic_write_json
from .work_resources import ResourceLease


class WorkQueue:
    def enqueue_dataset(self, parameters, *, parent_id=None):
        parameters = dict(parameters)
        if parent_id is not None:
            parent = self.datasets.get(parent_id, quick_verify=False)
            parameters.update(task_id=parent["task_id"], parent_dataset_id=parent_id)
            parameters.setdefault("include_post_success", parent.get("include_post_success", True))
        resolved = self.datasets.preview(parameters["task_id"], parameters["selection"])
        parameters["resolved_run_ids"] = resolved["run_ids"]
        job, _ = self.submit_local("dataset", lambda: self.datasets.create(**parameters),
            dataset_id=parent_id, parameters={"run_ids": resolved["run_ids"], **parameters})
        return job

    def submit_local(self, kind, callback, *, dataset_id=None, parameters=None,
                     cancel=None, created_at=None):
        """Local executors share the same index/lease as detached model jobs."""
        with self.lock:
            identifier = f"{kind}_{uuid.uuid4().hex[:12]}"
            directory = self.jobs_root / identifier
            directory.mkdir(parents=True)
            future = Future()
            self._local_tasks[identifier] = (callback, future, cancel)
            payload = {"schema_version": 1, "id": identifier, "kind": kind,
                       "executor": "local", "status": "QUEUED", "dataset_id": dataset_id,
                       "created_at": created_at or datetime.now(timezone.utc).isoformat(),
                       "parameters": parameters or {}, "stage": "queued", "stage_label": "等待前序任务完成",
                       "pid": None, "error": None}
            try:
                self._save_work(payload)
            except Exception:
                self._local_tasks.pop(identifier, None)
                future.cancel()
                raise
            self._wake_training_queue()
            return payload, future

    def _save_work(self, payload):
        path = self.jobs_root / payload["id"] / "job.json"
        atomic_write_json(path, payload)
        self.repository.upsert(payload, path)

    def immediate_lease(self):
        with self.lock:
            if self.repository.queue_ids(pending_only=True):
                raise ConflictError("工作队列尚未空闲；人工接管不会入队，请完成后重试。",
                                    code="WORK_QUEUE_BUSY", context={"severity": "warning"})
            return ResourceLease(self.gpu_lock_path)

    def _run_local(self, payload, lease):
        callback, future, _ = self._local_tasks[payload["id"]]
        result = error = None
        try:
            if not future.set_running_or_notify_cancel():
                return
            result = callback()
            payload.update(status="COMPLETED", stage="completed")
        except BaseException as exc:
            error = exc
            payload.update(status="FAILED", stage="failed", error=f"{type(exc).__name__}: {exc}")
        finally:
            try:
                with self.lock:
                    _, current = self._load_job(payload["id"])
                    if current["status"] in {"STOPPING", "CANCELED"}:
                        payload.update(status="CANCELED", stage="canceled")
                    payload["completed_at"] = datetime.now(timezone.utc).isoformat()
                    self._save_work(payload)
            except BaseException as exc:
                error = error or exc
            finally:
                lease.close()
                with self.lock:
                    self._local_tasks.pop(payload["id"], None)
                if not future.done():
                    if error is not None:
                        future.set_exception(error)
                    else:
                        future.set_result(result)
                self._wake_training_queue()

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
            if request.get("policy_content_sha256") != preview["policy_content_sha256"]:
                raise ValueError("所选模型已变化或未验证，请重新预览后注册")
            job = self._launch_evaluation(preview, queued=True)
            self._wake_training_queue()
            return job

    def _job_queue(self, kind: str | None = None) -> dict:
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
            for identifier in self.repository.queue_ids(pending_only=True):
                try:
                    _, job = self._load_job(identifier)
                    if job["status"] != "QUEUED":
                        self._reconcile(identifier)
                except (OSError, ValueError, KeyError, TypeError):
                    logging.getLogger(__name__).exception("Cannot read queued job %s; skipping", identifier)
                    self.repository.fail_unreadable_job(identifier)
            # Reconciliation can enqueue result publication ahead of later jobs.
            queued = []
            for identifier in self.repository.queue_ids(pending_only=True):
                _, job = self._load_job(identifier)
                if job["status"] == "QUEUED":
                    queued.append(job)
            if not queued:
                self._queue_wait_reason = None
                return
            selected = queued[0]
            try:
                self._prepare_launch()
                lease = ResourceLease(self.gpu_lock_path)
            except ConflictError as exc:
                self.launch_reserved = False
                self._queue_wait_reason = str(exc)
                return
            except Exception as exc:
                self.launch_reserved = False
                self._queue_wait_reason = f"等待 GPU 任务资源：{exc}"
                return
            self._queue_wait_reason = None
            spawning = False
            try:
                provider = getattr(getattr(self, "manager", None), "provider", None)
                if selected.get("requires_gpu", False) and provider is not None:
                    provider.unload()
                selected.update(status="STARTING", stage="starting", stage_label="准备后台任务",
                                dispatched_at=datetime.now(timezone.utc).isoformat())
                path = self._job_path(selected["id"])
                atomic_write_json(path, selected)
                self.repository.upsert(selected, path)
                if selected.get("executor") == "local":
                    if selected["id"] not in self._local_tasks:
                        raise RuntimeError("Backend restarted; please register this local task again")
                    selected.update(status="RUNNING", started_at=datetime.now(timezone.utc).isoformat())
                    self._save_work(selected)
                    threading.Thread(target=self._run_local, args=(selected, lease),
                                     name=f"work-{selected['id']}", daemon=True).start()
                    lease = None
                    return
                if selected["kind"] == "evaluation":
                    result_path = self._evaluation_manifest(selected["id"])
                    result = self._load_evaluation_manifest(result_path)
                    result["status"] = "STARTING"
                    atomic_write_json(result_path, result)
                    self.evaluation_repository.upsert(result, result_path)
                spawning = True
                self._spawn_job(selected, lease=lease)
            except Exception as exc:
                # _spawn_job owns launch failures; once Popen succeeds, only the
                # runner may update its manifest. Never overwrite a live child.
                if not spawning:
                    selected.update(status="FAILED", error=f"Cannot prepare queued job: {type(exc).__name__}: {exc}",
                                    completed_at=datetime.now(timezone.utc).isoformat())
                    atomic_write_json(self._job_path(selected["id"]), selected)
                    self.repository.upsert(selected, self._job_path(selected["id"]))
                    task = self._local_tasks.pop(selected["id"], None)
                    if task is not None and not task[1].done():
                        task[1].set_exception(exc)
                logging.getLogger(__name__).exception("Cannot launch queued job %s", selected["id"])
                raise
            finally:
                if lease is not None:
                    lease.close()
                self.launch_reserved = False
