"""Persistent two-environment annotation/training orchestration."""

from __future__ import annotations

import copy
import fcntl
import hashlib
import json
import math
import os
import re
import signal
import shutil
import socket
import subprocess
import sys
import threading
import urllib.error
import urllib.request
import uuid
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import yaml

from ..core.exceptions import ConflictError
from ..storage.files import atomic_write_json, atomic_write_yaml
from ..storage.repositories import EvaluationRepository, OfflineJobRepository
from .dataset_reward_versions import DatasetRewardVersions


ACTIVE_JOB_STATES = frozenset({"STARTING", "RUNNING", "STOPPING"})
TERMINAL_JOB_STATES = frozenset({"COMPLETED", "FAILED", "CANCELED"})


def _utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def _stable_hash(value: Any) -> str:
    return hashlib.sha256(
        json.dumps(value, sort_keys=True, default=str, separators=(",", ":")).encode()
    ).hexdigest()


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _training_replay_counts(prepared: dict[str, Any]) -> dict[str, int]:
    """Count the train split's full replay without rewriting the job's pin.

    Older schema-4 manifests keep post-confirmation actions in evaluation_chunks.
    Count those complete boundaries, including their actual controller breaks,
    and count physically copied branch prefixes once, as the replay loader does.
    """
    episodes = []
    for episode in prepared["episodes"]:
        chunks = episode["chunks"]
        recorded_end = int(episode["recorded_action_count"])
        if not chunks or int(chunks[-1]["end"]) != recorded_end:
            chunks = episode.get("evaluation_chunks", [])
        if not chunks or int(chunks[-1]["end"]) != recorded_end:
            raise ValueError(f"Run {episode['run_id']} lacks full-recording replay chunks")
        episodes.append((episode, chunks))

    def key(episode: dict[str, Any], chunk: dict[str, Any]) -> tuple[str, int, int, str]:
        return (
            str(episode["root_run_id"]), int(chunk["start"]),
            int(chunk["end"]), str(chunk["action_source"]),
        )

    copied = {
        key(episode, chunk) for episode, chunks in episodes for chunk in chunks
        if chunk.get("copied_prefix", False)
    }
    seen: set[tuple[str, int, int, str]] = set()
    action_count = chunk_count = 0
    for episode, chunks in sorted(
        episodes, key=lambda item: (item[0].get("kind") == "branch", str(item[0]["run_id"])),
    ):
        if episode["split"] != "train":
            continue
        for chunk in chunks:
            current = key(episode, chunk)
            if current in copied:
                if current in seen:
                    continue
                seen.add(current)
            action_count += int(chunk["length"])
            chunk_count += 1
    return {"action_count": action_count, "chunk_count": chunk_count}


class OfflineJobService(DatasetRewardVersions):
    def __init__(
        self, ui_config: Any, manager: Any, datasets: Any,
        trajectory_evaluations: Any | None = None,
        robometer_evaluations: Any | None = None,
        stage_annotations: Any | None = None,
    ):
        self.ui_config = ui_config
        self.manager = manager
        self.datasets = datasets
        self.trajectory_evaluations = trajectory_evaluations
        self.robometer_evaluations = robometer_evaluations
        self.stage_annotations = stage_annotations
        self.project_root = ui_config.project_root
        self.jobs_root = self.project_root / "jobs"
        self.training_root = self.project_root / "training"
        self.cache_root = self.project_root / "annotation-cache"
        self.evaluations_root = self.project_root / "evaluations"
        self.gpu_lock_path = self.project_root / ".gpu-task.lock"
        for path in (
            self.jobs_root, self.training_root, self.cache_root,
            self.evaluations_root,
        ):
            path.mkdir(parents=True, exist_ok=True)
        self.repository = OfflineJobRepository(
            ui_config.catalog_path, ui_config.project_id
        )
        self.evaluation_repository = EvaluationRepository(
            ui_config.catalog_path, ui_config.project_id
        )
        self.lock = threading.RLock()
        self.launch_reserved = False
        self.tensorboard_process: subprocess.Popen[Any] | None = None
        self.tensorboard_log = self.project_root / "tensorboard.log"
        self._reconcile_all()
        self._reconcile_all_evaluations()

    @property
    def base_config_path(self) -> Path:
        return self.ui_config.offline_rl_root / "configs" / "liberox_iql.yaml"

    def evaluator_capabilities(self) -> dict[str, dict[str, Any]]:
        checkout = self.ui_config.robometer_root.parent / "Robometer"
        checkout_ready = (checkout / "robometer" / "__init__.py").is_file()
        environment_ready = False
        try:
            result = subprocess.run(
                ["conda", "env", "list", "--json"], check=True, capture_output=True,
                text=True, timeout=5,
            )
            environment_ready = any(
                Path(value).name == self.ui_config.robometer_environment
                for value in json.loads(result.stdout).get("envs", [])
            )
        except Exception:
            environment_ready = False
        configured = checkout_ready and environment_ready
        reason = None
        if not checkout_ready:
            reason = f"Official Robometer checkout not found: {checkout}"
        elif not environment_ready:
            reason = f"Conda environment not found: {self.ui_config.robometer_environment}"
        return {
            "rynnvalue": {"available": True, "reason": None},
            "robometer": {"available": configured, "reason": reason},
        }

    def _load_base_config(self) -> dict[str, Any]:
        if not self.base_config_path.is_file():
            raise FileNotFoundError(f"Offline RL config not found: {self.base_config_path}")
        raw = yaml.safe_load(self.base_config_path.read_text(encoding="utf-8"))
        if not isinstance(raw, dict):
            raise ValueError("Offline RL config root must be an object")
        base = self.base_config_path.parent
        paths = raw["paths"]
        for name, value in list(paths.items()):
            if name == "dataset_sources":
                paths[name] = [
                    str((Path(item) if Path(item).is_absolute() else base / item).resolve())
                    for item in value
                ]
            elif isinstance(value, str):
                path = Path(value).expanduser()
                paths[name] = str((path if path.is_absolute() else base / path).resolve())
        return raw

    def defaults(self, dataset_id: str | None = None, reward_source: str | None = None) -> dict[str, Any]:
        raw = self._load_base_config()
        version = None
        availability = None
        if reward_source is not None:
            self._reward_source(raw, {"reward_source": reward_source})
        if dataset_id:
            dataset = self.datasets.get(dataset_id, quick_verify=False)
            identifier = (dataset.get("evaluation_version_ids", {}).get(reward_source)
                          if reward_source else dataset.get("reward_version_id"))
            if identifier:
                version = self.datasets.get_version(dataset_id, identifier)
                sealed = yaml.safe_load(Path(version["config_path"]).read_text(encoding="utf-8"))
                raw["reward"] = sealed["reward"]
                availability = {"ready": bool(version.get("complete") and version.get("status") == "READY"),
                                "origin": "dataset", "missing_run_ids": []}
            else:
                from .global_reward_binding import global_members
                source = reward_source or self._reward_source(raw, {})
                records, missing = global_members(self, dataset, source)
                availability = {"ready": bool(records) and not missing, "origin": "global",
                                "missing_run_ids": [item["run_id"] for item in missing], "errors": missing}
                from .trajectory_reward_snapshot import needs_snapshot_validation
                availability["pending"] = False
                for item in missing:
                    run = self.datasets.run_service.get_run(item["run_id"])
                    if needs_snapshot_validation(run, source):
                        availability["pending"] = True
                        self.schedule_first_reward_snapshot(run)
                if records:
                    raw["reward"].update(records[0][1]["reward_config"])
        if reward_source is not None:
            raw["reward"].update(source=reward_source, rynnvalue=reward_source == "rynnvalue")
        reward_source = self._reward_source(raw, {})
        return {
            "reward_version": version,
            "reward_availability": availability,
            "reward_parameters_locked": False,
            "reward_locked_parameters": [
                "reward_stage_exponent", "reward_shaping_weight",
            ] if dataset_id else [],
            "reward_editable_parameters": ["reward_gamma", "reward_accumulate_primitive_steps"],
            "basic": {
                name: raw["iql"][name] for name in (
                    "train_steps", "critic_warmup_steps",
                    "micro_batch_size", "gradient_accumulation_steps",
                    "checkpoint_interval", "seed",
                )
            },
            "advanced": {
                name: raw["iql"][name] for name in (
                    "critic_optimizer", "critic_lr", "critic_weight_decay",
                    "critic_max_grad_norm", "value_optimizer", "value_lr",
                    "value_weight_decay", "value_max_grad_norm",
                    "policy_peak_lr", "policy_final_lr",
                    "expectile", "beta", "max_advantage_weight", "target_tau",
                )
            } | {
                "reward_source": reward_source,
                "reward_stage_exponent": raw["reward"].get("stage_exponent", 2.0),
                "reward_rynnvalue": reward_source == "rynnvalue",
                "reward_gamma": raw["reward"]["gamma"],
                "reward_shaping_weight": raw["reward"]["shaping_weight"],
                "reward_accumulate_primitive_steps": raw["reward"][
                    "accumulate_primitive_steps"
                ],
            } | {
                "console_interval_steps": raw["logging"]["console_interval_steps"],
                "flush_seconds": raw["logging"]["flush_seconds"],
            },
            "monitoring": {
                "tensorboard": raw["logging"]["tensorboard"],
                "wandb_enabled": raw["logging"]["wandb"]["enabled"],
                "wandb_mode": raw["logging"]["wandb"]["mode"],
                "wandb_project": raw["logging"]["wandb"]["project"],
                "wandb_entity": raw["logging"]["wandb"]["entity"],
                "wandb_run_name": raw["logging"]["wandb"]["run_name"],
                "wandb_group": raw["logging"]["wandb"]["group"],
                "wandb_tags": ", ".join(raw["logging"]["wandb"]["tags"]),
                "wandb_log_interval_steps": raw["logging"]["wandb"]["log_interval_steps"],
            },
            "fixed": {
                "dtype": raw["iql"]["dtype"],
                "critic_image_size": raw["iql"]["critic_image_size"],
                "action_horizon": raw["data"]["action_horizon"],
                "action_dim": raw["data"]["action_dim"],
                "proprio_dim": raw["data"]["proprio_dim"],
                "base_checkpoint": raw["vla"]["base_checkpoint"],
                "stats_key": raw["vla"]["stats_key"],
                "freeze_backbone": raw["vla"]["freeze_backbone"],
            },
            "environments": {
                "prepare": self.ui_config.train_environment,
                "annotation": self.ui_config.reward_environment,
                "training": self.ui_config.train_environment,
            },
            "checkpoints": self.available_checkpoints(dataset_id),
        }

    def _job_path(self, job_id: str) -> Path:
        if not job_id or Path(job_id).name != job_id or ".." in job_id:
            raise ValueError("Invalid job id")
        path = self.jobs_root / job_id / "job.json"
        if not path.is_file() or path.is_symlink():
            raise KeyError(job_id)
        return path

    def _load_job(self, job_id: str) -> tuple[Path, dict[str, Any]]:
        path = self._job_path(job_id)
        payload = json.loads(path.read_text(encoding="utf-8"))
        if payload.get("id") != job_id or payload.get("schema_version") != 1:
            raise ValueError(f"Invalid job manifest: {path}")
        return path, payload

    @staticmethod
    def _pid_alive(pid: int | None) -> bool:
        if not pid:
            return False
        try:
            os.kill(int(pid), 0)
        except (ProcessLookupError, PermissionError):
            return False
        return True

    def _reconcile(self, job_id: str) -> dict[str, Any]:
        path, job = self._load_job(job_id)
        process_id = job.get("pid") or job.get("launcher_pid")
        created = datetime.fromisoformat(job["created_at"])
        missing_process = (
            job["status"] == "STARTING"
            and not process_id
            and (datetime.now(timezone.utc) - created).total_seconds() > 5
        )
        dead_process = bool(process_id) and not self._pid_alive(process_id)
        if job["status"] in ACTIVE_JOB_STATES and (missing_process or dead_process):
            job["status"] = "FAILED"
            job["completed_at"] = _utc_now()
            job["error"] = "Detached job process exited without a terminal status"
            atomic_write_json(path, job)
        self.repository.upsert(job, path)
        if job["kind"] == "annotation" and (job.get("parameters") or {}).get("reward_version_id"):
            version = self.datasets.get_version(job["dataset_id"], job["id"])
            if job["status"] == "COMPLETED" and not version.get("complete"):
                job.update(status="FAILED", error="Evaluation version was not sealed")
                atomic_write_json(path, job)
                self.repository.upsert(job, path)
            status = {"COMPLETED": "READY", "FAILED": "ERROR", "CANCELED": "CANCELED"}.get(job["status"], "RUNNING")
            if version.get("status") != "READY":
                version.update(status=status, error=job.get("error"), completed_at=job.get("completed_at"))
                if status in {"ERROR", "CANCELED"} and not version.get("complete"):
                    version_path = self.datasets.root / job["dataset_id"] / "annotations" / job["id"] / "version.json"
                    atomic_write_json(version_path, version)
            self.datasets.update_version(job["dataset_id"], version)
        elif job["kind"] == "annotation":
            desired = {
                "COMPLETED": "READY",
                "FAILED": "ERROR",
                "CANCELED": "CANCELED",
            }.get(job["status"], "RUNNING")
            try:
                current = self.datasets.get(job["dataset_id"], quick_verify=False)
                synchronized = (
                    current.get("pending_annotation_id") == job["id"]
                    if desired == "RUNNING"
                    else current.get("last_annotation_id") == job["id"]
                    and current.get("last_annotation_status") == desired
                )
                if not synchronized:
                    parameters = job.get("parameters") or {}
                    annotation_config = {"max_frames": parameters.get("max_frames")}
                    if "accumulate_primitive_steps" in parameters:
                        annotation_config["accumulate_primitive_steps"] = parameters[
                            "accumulate_primitive_steps"
                        ]
                    self.datasets.update_annotation(
                        job["dataset_id"], desired, annotation_id=job["id"],
                        annotation_config=annotation_config,
                    )
            except Exception:
                pass
        if job["kind"] in {"annotation", "trajectory_evaluation"} and (
            job["kind"] == "trajectory_evaluation"
            or (job.get("parameters") or {}).get("reward_version_id")
        ):
            self._schedule_result_binding(job)
        elif job["kind"] == "evaluation":
            manifest = Path(str(job.get("output_path") or "")) / "evaluation.json"
            try:
                self._index_evaluation_manifest(manifest)
            except Exception:
                pass
        return self._public_job(job)

    def _reconcile_all(self) -> None:
        for path in self.jobs_root.glob("*/job.json"):
            try:
                self._reconcile(path.parent.name)
            except Exception:
                continue

    def list(self) -> list[dict[str, Any]]:
        result = []
        for path in self.jobs_root.glob("*/job.json"):
            try:
                result.append(self._reconcile(path.parent.name))
            except Exception:
                continue
        return sorted(result, key=lambda item: item["created_at"], reverse=True)

    def get(self, job_id: str) -> dict[str, Any]:
        return self._reconcile(job_id)

    def delete_dataset(
        self, dataset_id: str, confirm_dataset_id: str, *, force: bool = False
    ) -> dict[str, Any]:
        """Delete a dataset while retaining, and optionally detaching, job history."""
        with self.lock:
            live_references = [
                job for job in self.list() if job.get("dataset_id") == dataset_id
            ]
            references_by_id = {
                job["id"]: job
                for job in self.repository.references_for_dataset(dataset_id)
            }
            references_by_id.update({job["id"]: job for job in live_references})
            references = list(references_by_id.values())
            active = [
                job["id"] for job in references
                if job.get("status") in ACTIVE_JOB_STATES
            ]
            if active:
                raise ConflictError(
                    "Dataset has an active background job",
                    code="DATASET_JOB_ACTIVE",
                    context={"dataset_id": dataset_id, "jobs": active},
                )
            training = [
                job["id"] for job in references if job.get("kind") == "training"
            ]
            if training and not force:
                raise ConflictError(
                    "Dataset is referenced by training history",
                    code="DATASET_HAS_TRAINING",
                    context={"dataset_id": dataset_id, "training_jobs": training},
                )
            result = self.datasets.delete_dataset(dataset_id, confirm_dataset_id)
            deleted_at = _utc_now()
            for job_id in training:
                try:
                    path, job = self._load_job(job_id)
                    job["source_dataset_deleted"] = True
                    job["source_dataset_deleted_at"] = deleted_at
                    atomic_write_json(path, job)
                    self.repository.upsert(job, path)
                except (KeyError, OSError, ValueError, json.JSONDecodeError):
                    # The SQLite reference may outlive a manually removed job manifest.
                    continue
            return {
                **result,
                "retained_training_jobs": training,
            }

    def _public_job(self, job: dict[str, Any]) -> dict[str, Any]:
        result = {
            key: value for key, value in job.items()
            if key not in {"stages", "gpu_lock_path"}
        }
        if job.get("trajectory_binding_status") == "RUNNING":
            result.update(status="RUNNING", stage="bind", stage_label="保存轨迹评价")
        elif job.get("trajectory_binding_status") == "ERROR":
            # Dataset outputs are already sealed and valid. Global publication
            # is a separate convenience copy, not a failed reward computation.
            result.update(stage="bind", stage_label="评价成功，全局结果同步失败",
                          warning=job.get("trajectory_binding_error"))
        job_dir = self.jobs_root / job["id"]
        result["log_size"] = (job_dir / "job.log").stat().st_size if (job_dir / "job.log").is_file() else 0
        if job["kind"] == "training":
            metrics = self._latest_metrics(job)
            if metrics is not None:
                result["metrics"] = metrics
            summary = self._training_summary(job)
            if summary is not None:
                result["training_summary"] = summary
        elif job["kind"] == "evaluation":
            manifest = Path(str(job.get("output_path") or "")) / "evaluation.json"
            try:
                evaluation = self._load_evaluation_manifest(manifest)
                public = self._public_evaluation(
                    evaluation, manifest, detail=False
                )
                aggregate = public["aggregate"]
                attempted = int(aggregate.get("attempted_trials", 0) or 0)
                total = int(aggregate.get("total_trials", 0) or 0)
                elapsed = float(public.get("wall_time_seconds") or 0.0)
                trial_mean = aggregate.get("elapsed_seconds_mean")
                remaining = (
                    float(trial_mean) * max(0, total - attempted)
                    if trial_mean is not None else None
                )
                completed_trials = evaluation.get("trials") or evaluation.get("episodes") or []
                latest = completed_trials[-1] if completed_trials else {}
                schedule = evaluation.get("schedule") or []
                active = (
                    evaluation.get("status") in ACTIVE_JOB_STATES
                    and attempted < total
                    and attempted < len(schedule)
                )
                scheduled = schedule[attempted] if active else latest
                result["evaluation_summary"] = {
                    **aggregate,
                    "evaluation_id": evaluation["id"],
                    "current_trial": attempted + 1 if active else attempted,
                    "progress_percent": attempted / total * 100.0 if total else 0.0,
                    "elapsed_seconds": elapsed,
                    "estimated_remaining_seconds": remaining,
                    "init_state_index": scheduled.get("init_state_index"),
                    "seed": scheduled.get("seed"),
                    "measured_control_hz": latest.get("measured_control_hz"),
                }
            except Exception:
                pass
        return result

    def logs(self, job_id: str, offset: int = 0, limit: int = 256_000) -> dict[str, Any]:
        self.get(job_id)
        if offset < 0 or limit < 1 or limit > 1_000_000:
            raise ValueError("Invalid log range")
        path = self.jobs_root / job_id / "job.log"
        if not path.is_file():
            return {"offset": 0, "next_offset": 0, "text": ""}
        size = path.stat().st_size
        start = min(offset, size)
        with path.open("rb") as stream:
            stream.seek(start)
            data = stream.read(limit)
        return {
            "offset": start,
            "next_offset": start + len(data),
            "text": data.decode("utf-8", errors="replace"),
        }

    def has_active_job(self) -> bool:
        return any(job["status"] in ACTIVE_JOB_STATES for job in self.list())

    def has_active_gpu_job(self) -> bool:
        return any(job["status"] in ACTIVE_JOB_STATES
                   and job.get("parameters", {}).get("requires_gpu", True)
                   for job in self.list())

    def _external_gpu_lock(self) -> bool:
        self.gpu_lock_path.parent.mkdir(parents=True, exist_ok=True)
        with self.gpu_lock_path.open("a+") as stream:
            try:
                fcntl.flock(stream.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)
            except BlockingIOError:
                return True
            fcntl.flock(stream.fileno(), fcntl.LOCK_UN)
        return False

    def assert_simulation_allowed(self) -> None:
        if self.launch_reserved or self.has_active_gpu_job() or self._external_gpu_lock():
            raise ConflictError(
                "An annotation, training, or evaluation job is using the GPU",
                code="GPU_TASK_ACTIVE",
            )

    def _prepare_launch(self) -> None:
        if self.has_active_gpu_job() or self._external_gpu_lock():
            raise ConflictError("Another GPU task is active", code="GPU_TASK_ACTIVE")
        active = getattr(self.manager, "active_session_id", None)
        if active is not None:
            raise ConflictError("A simulation is active", code="SIMULATION_ACTIVE")
        if getattr(self.manager, "draft", None) is not None:
            raise ConflictError("Cancel the simulation draft first", code="SIMULATION_DRAFT_ACTIVE")
        controller = self.manager.controller_status()
        if controller.get("state") in {"CALIBRATING", "ARMED"}:
            raise ConflictError("The controller is calibrating or armed", code="CONTROLLER_BUSY")
        # Close the in-process race between the last conflict check and
        # publication of the STARTING job manifest. SimulationManager's guard
        # observes this reservation immediately.
        self.launch_reserved = True
        provider = getattr(self.manager, "provider", None)
        try:
            if provider is not None:
                provider.unload()
        except Exception:
            self.launch_reserved = False
            raise

    def _new_job(
        self,
        *,
        kind: str,
        dataset_id: str | None,
        stages: list[dict[str, Any]],
        config_path: Path,
        output_path: Path,
        parameters: dict[str, Any],
    ) -> dict[str, Any]:
        now = _utc_now()
        job_id = config_path.parent.name
        payload = {
            "schema_version": 1,
            "id": job_id,
            "kind": kind,
            "status": "STARTING",
            "dataset_id": dataset_id,
            "created_at": now,
            "started_at": None,
            "completed_at": None,
            "heartbeat_at": None,
            "pid": None,
            "process_group_id": None,
            "stage": "starting",
            "stage_label": "准备后台任务",
            "error": None,
            "config_path": str(config_path.resolve()),
            "output_path": str(output_path.resolve()),
            "parameters": parameters,
            "requires_gpu": parameters.get("requires_gpu", True),
            "gpu_lock_path": str(self.gpu_lock_path.resolve()),
            "stages": stages,
        }
        job_dir = config_path.parent
        atomic_write_json(job_dir / "job.json", payload)
        self.repository.upsert(payload, job_dir / "job.json")
        runner = Path(__file__).resolve().parents[1] / "workers" / "offline_job_runner.py"
        try:
            process = subprocess.Popen(
                [sys.executable, str(runner), "--job-dir", str(job_dir)],
                cwd=str(self.ui_config.offline_rl_root),
                stdin=subprocess.DEVNULL,
                stdout=subprocess.DEVNULL,
                stderr=subprocess.DEVNULL,
                start_new_session=True,
            )
        except Exception as exc:
            payload.update(
                status="FAILED", completed_at=_utc_now(),
                error=f"Cannot launch detached runner: {type(exc).__name__}: {exc}",
            )
            atomic_write_json(job_dir / "job.json", payload)
            self.repository.upsert(payload, job_dir / "job.json")
            raise
        payload["launcher_pid"] = process.pid
        atomic_write_json(job_dir / "job.json", payload)
        self.repository.upsert(payload, job_dir / "job.json")
        return self._public_job(payload)

    def _effective_config(self, dataset: dict[str, Any]) -> dict[str, Any]:
        raw = copy.deepcopy(self._load_base_config())
        raw["data"]["task_ids"] = [dataset["task_id"]]
        raw["data"]["selection_manifest"] = str(
            (self.datasets.root / dataset["id"] / "dataset.json").resolve()
        )
        raw["data"]["validation_fraction"] = dataset["validation_fraction"]
        raw["data"]["split_seed"] = dataset["split_seed"]
        raw["data"]["success_consecutive_steps"] = dataset["success_consecutive_steps"]
        raw["paths"]["annotation_cache"] = str(self.cache_root.resolve())
        return raw

    def start_annotation(
        self, dataset_id: str, *, max_frames: int | None = None,
        accumulate_primitive_steps: bool | None = None, source: str = "rynnvalue", **options: Any,
    ) -> dict[str, Any]:
        return self.start_reward_version(dataset_id, source=source, max_frames=max_frames,
            accumulate_primitive_steps=accumulate_primitive_steps, **options)

    def start_trajectory_evaluation(
        self,
        *,
        task_id: str,
        run_ids: list[str] | None,
        overwrite: bool,
        evaluators: list[str] | None = None,
    ) -> dict[str, Any]:
        """Evaluate selected trajectories independently from frozen datasets."""
        with self.lock:
            runs = self.datasets.list_runs(task_id=task_id, eligible=True)
            available = {item["id"]: item for item in runs}
            # A task-wide batch is a data-maintenance operation and excludes
            # held-out tests by default. Explicitly selected trajectories may
            # still be evaluated for inspection.
            requested = (
                [run_id for run_id, run in available.items() if not run.get("is_test")]
                if run_ids is None else list(run_ids)
            )
            if not requested:
                raise ValueError("No eligible non-test trajectories were selected")
            if len(requested) != len(set(requested)):
                raise ValueError("run_ids must not contain duplicates")
            missing = [run_id for run_id in requested if run_id not in available]
            if missing:
                raise ValueError(f"Unavailable trajectories: {missing}")
            evaluators = list(evaluators or ["rynnvalue"])
            if not evaluators or len(evaluators) != len(set(evaluators)):
                raise ValueError("evaluators must be a non-empty unique list")
            if any(value not in {"rynnvalue", "robometer"} for value in evaluators):
                raise ValueError(f"Unsupported evaluator list: {evaluators}")
            services = {
                "rynnvalue": self.trajectory_evaluations,
                "robometer": self.robometer_evaluations,
            }
            for evaluator in evaluators:
                if services[evaluator] is None:
                    raise RuntimeError(f"{evaluator} evaluation service is unavailable")
            selected_by_evaluator: dict[str, list[str]] = {}
            skipped_by_evaluator: dict[str, list[str]] = {}
            for evaluator in evaluators:
                service = services[evaluator]
                selected_by_evaluator[evaluator] = [
                    run_id for run_id in requested
                    if overwrite or not service.exists(available[run_id])
                ]
                skipped_by_evaluator[evaluator] = [
                    run_id for run_id in requested
                    if run_id not in selected_by_evaluator[evaluator]
                ]
            selected_union = list(dict.fromkeys(
                run_id for evaluator in evaluators
                for run_id in selected_by_evaluator[evaluator]
            ))
            if not selected_union:
                return {
                    "kind": "trajectory_evaluation",
                    "status": "COMPLETED",
                    "job": None,
                    "evaluators": evaluators,
                    "selected_count": 0,
                    "skipped_count": len(requested) * len(evaluators),
                    "selected_by_evaluator": selected_by_evaluator,
                    "skipped_by_evaluator": skipped_by_evaluator,
                    "message": "All selected trajectories already have reusable evaluations",
                }
            self._prepare_launch()
            try:
                job = self._launch_trajectory_evaluation(
                    task_id=task_id, requested_run_ids=requested,
                    selected_by_evaluator=selected_by_evaluator,
                    skipped_by_evaluator=skipped_by_evaluator,
                    evaluators=evaluators, overwrite=overwrite,
                )
            finally:
                self.launch_reserved = False
            return {
                "kind": "trajectory_evaluation",
                "status": job["status"],
                "job": job,
                "evaluators": evaluators,
                "selected_count": len(selected_union),
                "skipped_count": sum(map(len, skipped_by_evaluator.values())),
                "selected_by_evaluator": selected_by_evaluator,
                "skipped_by_evaluator": skipped_by_evaluator,
            }

    def _launch_trajectory_evaluation(
        self,
        *,
        task_id: str,
        requested_run_ids: list[str],
        selected_by_evaluator: dict[str, list[str]],
        skipped_by_evaluator: dict[str, list[str]],
        evaluators: list[str],
        overwrite: bool,
    ) -> dict[str, Any]:
        job_id = f"eval_{datetime.now().strftime('%Y%m%d_%H%M%S')}_{uuid.uuid4().hex[:8]}"
        job_dir = self.jobs_root / job_id
        job_dir.mkdir(parents=True, exist_ok=False)
        output_root = self.project_root / "trajectory-evaluations" / job_id
        work_dir = output_root / "work"
        work_dir.mkdir(parents=True, exist_ok=False)
        stages: list[dict[str, Any]] = []
        config_paths: dict[str, str] = {}
        selections: dict[str, Any] = {}
        if selected_by_evaluator.get("rynnvalue"):
            raw = copy.deepcopy(self._load_base_config())
            raw["reward"].update(source="rynnvalue", rynnvalue=True)
            raw["data"].pop("stage_annotations_manifest", None)
            rynn_work = work_dir / "rynnvalue"
            rynn_work.mkdir(parents=True)
            raw["data"]["task_ids"] = [task_id]
            raw["paths"]["annotation_cache"] = str(self.cache_root.resolve())
            raw["paths"]["work_dir"] = str(rynn_work.resolve())
            selection_path = job_dir / "rynnvalue_selection.json"
            selections["rynnvalue"] = self.datasets.write_evaluation_selection(
                selection_path, task_id=task_id,
                run_ids=selected_by_evaluator["rynnvalue"],
                split_seed=int(raw["data"]["split_seed"]),
                validation_fraction=float(raw["data"]["validation_fraction"]),
                success_consecutive_steps=int(raw["data"]["success_consecutive_steps"]),
            )
            raw["data"]["selection_manifest"] = str(selection_path.resolve())
            config_path = job_dir / "rynnvalue_config.yaml"
            atomic_write_yaml(config_path, raw)
            config_paths["rynnvalue"] = str(config_path)
            scripts = self.ui_config.offline_rl_root / "scripts"
            annotate_argv = ["python", str(scripts / "annotate_rewards.py"), "--config", str(config_path)]
            if overwrite:
                annotate_argv.append("--overwrite")
            stages.extend([
                {"id": "rynn_prepare", "label": "准备 RynnValue 轨迹输入",
                 "environment": self.ui_config.train_environment,
                 "argv": ["python", str(scripts / "prepare_dataset.py"), "--config", str(config_path)],
                 "cwd": str(self.ui_config.offline_rl_root)},
                {"id": "rynn_annotate", "label": "RynnValue-4B 轨迹评价",
                 "environment": self.ui_config.reward_environment, "argv": annotate_argv,
                 "cwd": str(self.ui_config.offline_rl_root)},
                {"id": "rynn_rewards", "label": "生成默认 IQL 奖励缓存",
                 "environment": self.ui_config.train_environment,
                 "argv": ["python", str(scripts / "materialize_rewards.py"),
                          "--config", str(config_path)],
                 "cwd": str(self.ui_config.offline_rl_root)},
            ])
        if selected_by_evaluator.get("robometer"):
            base_path = self.ui_config.robometer_root / "configs" / "robometer_evaluation.yaml"
            if not base_path.is_file():
                raise FileNotFoundError(f"Robometer config not found: {base_path}")
            robo = yaml.safe_load(base_path.read_text(encoding="utf-8"))
            selection_path = job_dir / "robometer_selection.json"
            base_rynn = self._load_base_config()
            selections["robometer"] = self.datasets.write_evaluation_selection(
                selection_path, task_id=task_id,
                run_ids=selected_by_evaluator["robometer"],
                split_seed=int(base_rynn["data"]["split_seed"]),
                validation_fraction=float(base_rynn["data"]["validation_fraction"]),
                success_consecutive_steps=int(base_rynn["data"]["success_consecutive_steps"]),
            )
            robo["paths"]["selection_manifest"] = str(selection_path.resolve())
            robo["paths"]["output_dir"] = str((work_dir / "robometer").resolve())
            raw_root = Path(str(robo["paths"]["robometer_root"]))
            if not raw_root.is_absolute():
                robo["paths"]["robometer_root"] = str((base_path.parent / raw_root).resolve())
            checkout = Path(robo["paths"]["robometer_root"])
            if not (checkout / "robometer" / "__init__.py").is_file():
                raise FileNotFoundError(
                    "Official Robometer checkout is unavailable; install it as documented in "
                    f"{self.ui_config.robometer_root / 'README.md'}"
                )
            config_path = job_dir / "robometer_config.yaml"
            atomic_write_yaml(config_path, robo)
            config_paths["robometer"] = str(config_path)
            argv = ["python", str(self.ui_config.robometer_root / "scripts" / "evaluate_trajectories.py"),
                    "--config", str(config_path)]
            if overwrite:
                argv.append("--overwrite")
            stages.append({
                "id": "robometer_annotate", "label": "Robometer-4B 轨迹评价",
                "environment": self.ui_config.robometer_environment, "argv": argv,
                "cwd": str(self.ui_config.robometer_root),
            })
        if not stages:
            raise ValueError("No evaluator has trajectories requiring evaluation")
        config_path = Path(next(iter(config_paths.values())))
        try:
            return self._new_job(
                kind="trajectory_evaluation", dataset_id=None, stages=stages,
                config_path=config_path, output_path=output_root,
                parameters={
                    "task_id": task_id,
                    "run_ids": requested_run_ids,
                    "member_count": len(requested_run_ids),
                    "evaluators": evaluators,
                    "selected_by_evaluator": selected_by_evaluator,
                    "skipped_by_evaluator": skipped_by_evaluator,
                    "overwrite": overwrite,
                    "selection_sha256": {
                        key: value["dataset_sha256"] for key, value in selections.items()
                    },
                    "config_paths": config_paths,
                },
            )
        except Exception:
            shutil.rmtree(output_root, ignore_errors=True)
            raise


    @staticmethod
    def _validate_training_parameters(parameters: dict[str, Any]) -> None:
        allowed = {
            "train_steps", "critic_warmup_steps", "gradient_accumulation_steps",
            "micro_batch_size", "checkpoint_interval", "seed",
            "critic_optimizer", "critic_lr", "critic_weight_decay",
            "critic_max_grad_norm", "value_optimizer", "value_lr",
            "value_weight_decay", "value_max_grad_norm",
            "policy_peak_lr", "policy_final_lr", "expectile", "beta",
            "max_advantage_weight", "target_tau", "console_interval_steps",
            "flush_seconds", "resume_checkpoint", "tensorboard", "wandb_enabled",
            "wandb_mode", "wandb_project", "wandb_entity", "wandb_run_name",
            "wandb_group", "wandb_tags", "wandb_log_interval_steps",
            "reward_rynnvalue", "reward_gamma", "reward_shaping_weight",
            "reward_accumulate_primitive_steps", "reward_source", "reward_stage_exponent",
            "reward_version_id",
        }
        unknown = sorted(set(parameters) - allowed)
        if unknown:
            raise ValueError(f"Unknown training parameters: {unknown}")
        integer_positive = (
            "train_steps", "micro_batch_size", "gradient_accumulation_steps",
            "checkpoint_interval", "console_interval_steps", "wandb_log_interval_steps",
        )
        for name in integer_positive:
            value = parameters.get(name)
            if value is not None and (type(value) is not int or value < 1):
                raise ValueError(f"{name} must be a positive integer")
        for name in ("critic_warmup_steps", "seed"):
            value = parameters.get(name)
            if value is not None and (type(value) is not int or value < 0):
                raise ValueError(f"{name} must be a non-negative integer")
        for name in (
            "critic_lr", "value_lr", "policy_peak_lr", "policy_final_lr", "beta",
            "max_advantage_weight", "target_tau", "flush_seconds",
            "critic_weight_decay", "value_weight_decay",
            "reward_gamma", "reward_shaping_weight",
        ):
            value = parameters.get(name)
            if value is not None and (
                isinstance(value, bool) or not isinstance(value, (int, float)) or value < 0
            ):
                raise ValueError(f"{name} must be a non-negative number")
        for name in ("critic_max_grad_norm", "value_max_grad_norm"):
            value = parameters.get(name)
            if value is not None and (
                isinstance(value, bool) or not isinstance(value, (int, float)) or value <= 0
            ):
                raise ValueError(f"{name} must be a positive number")
        expectile = parameters.get("expectile")
        if expectile is not None and (
            isinstance(expectile, bool) or not isinstance(expectile, (int, float))
            or not 0 <= expectile <= 1
        ):
            raise ValueError("expectile must be in [0, 1]")
        target_tau = parameters.get("target_tau")
        if target_tau is not None and target_tau > 1:
            raise ValueError("target_tau must be in [0, 1]")
        reward_gamma = parameters.get("reward_gamma")
        if "reward_gamma" in parameters and (
            type(reward_gamma) not in (int, float) or not math.isfinite(reward_gamma)
            or not 0 <= reward_gamma <= 1
        ):
            raise ValueError("reward_gamma must be in [0, 1]")
        if "reward_accumulate_primitive_steps" in parameters and type(parameters["reward_accumulate_primitive_steps"]) is not bool:
            raise ValueError("reward_accumulate_primitive_steps must be boolean")
        if "reward_source" in parameters and parameters["reward_source"] not in (
            "sparse", "rynnvalue", "stage",
        ):
            raise ValueError("reward_source must be sparse, rynnvalue, or stage")
        if "reward_stage_exponent" in parameters:
            exponent = parameters["reward_stage_exponent"]
            if (
                isinstance(exponent, bool) or not isinstance(exponent, (int, float))
                or not math.isfinite(exponent) or exponent < 1
            ):
                raise ValueError("reward_stage_exponent must be a finite number >= 1")
        for name in ("critic_optimizer", "value_optimizer"):
            value = parameters.get(name)
            if value is not None and value not in {"adam", "adamw"}:
                raise ValueError(f"{name} must be adam or adamw")
        if parameters.get("wandb_mode") not in (None, "online", "offline", "disabled"):
            raise ValueError("wandb_mode must be online, offline, or disabled")
        for name in (
            "tensorboard", "wandb_enabled", "reward_rynnvalue",
            "reward_accumulate_primitive_steps",
        ):
            value = parameters.get(name)
            if value is not None and type(value) is not bool:
                raise ValueError(f"{name} must be boolean")
        for name in ("wandb_project", "wandb_entity", "wandb_run_name", "wandb_group", "wandb_tags"):
            value = parameters.get(name)
            if value is not None and not isinstance(value, str):
                raise ValueError(f"{name} must be a string or null")
        project = parameters.get("wandb_project")
        if project is not None and not project.strip():
            raise ValueError("wandb_project must not be empty")

    def _resolve_resume(self, value: Any) -> str | None:
        if value in (None, ""):
            return None
        if not isinstance(value, str):
            raise ValueError("resume_checkpoint must be a platform checkpoint path")
        path = Path(value).resolve()
        try:
            path.relative_to(self.training_root.resolve())
        except ValueError:
            raise ValueError("resume_checkpoint must be inside the managed training root") from None
        if not path.is_dir() or path.is_symlink() or not (path / "checkpoint.json").is_file():
            raise FileNotFoundError(path)
        return str(path)

    def _validated_annotation_work(
        self, dataset: dict[str, Any]
    ) -> tuple[Path, dict[str, Any], dict[str, Any]]:
        annotation_id = dataset["annotation_id"]
        work = self.datasets.root / dataset["id"] / "annotations" / annotation_id / "work"
        prepared_path = work / "dataset_manifest.json"
        annotation_path = work / "annotations" / "annotation_manifest.json"
        legacy_reward_path = work / "rewards" / "reward_manifest.json"
        manifest_path = (
            annotation_path if annotation_path.is_file() else legacy_reward_path
        )
        if (
            prepared_path.is_symlink() or manifest_path.is_symlink()
            or not prepared_path.is_file() or not manifest_path.is_file()
        ):
            raise FileNotFoundError("Prepared dataset or RynnValue annotation manifest is missing")
        prepared = json.loads(prepared_path.read_text(encoding="utf-8"))
        annotation_manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
        if (
            prepared.get("source_dataset_id") != dataset["id"]
            or prepared.get("source_dataset_sha256") != dataset["dataset_sha256"]
        ):
            raise ConflictError(
                "Prepared data does not match the immutable dataset",
                code="ANNOTATION_DATASET_MISMATCH",
            )
        separated = manifest_path == annotation_path
        if separated and annotation_manifest.get("kind") != "rynnvalue_annotation":
            raise ConflictError(
                "RynnValue annotation manifest has an invalid artifact kind",
                code="ANNOTATION_INCOMPLETE",
            )
        if (
            annotation_manifest.get("complete") is not True
            or annotation_manifest.get("dataset_sha256") != prepared.get("dataset_sha256")
        ):
            raise ConflictError(
                "RynnValue annotation is incomplete or belongs to different prepared data",
                code="ANNOTATION_INCOMPLETE",
            )
        expected_runs = {episode["run_id"] for episode in prepared.get("episodes", [])}
        annotation_runs = {
            episode.get("run_id") for episode in annotation_manifest.get("episodes", [])
        }
        if annotation_runs != expected_runs:
            raise ConflictError(
                "Reward annotation membership does not match prepared data",
                code="ANNOTATION_MEMBERSHIP_MISMATCH",
            )
        dataset_annotation_config = dataset.get("annotation_config") or {}
        manifest_annotation_config = (
            annotation_manifest.get("annotation_config")
            or annotation_manifest.get("reward_config")
            or {}
        )
        expected_max_frames = dataset_annotation_config.get("max_frames")
        if (
            expected_max_frames is not None
            and manifest_annotation_config.get("max_frames") != expected_max_frames
        ):
            raise ConflictError(
                "Active RynnValue evaluation does not match dataset max_frames",
                code="ANNOTATION_CONFIG_MISMATCH",
            )
        cache_root = self.cache_root.resolve()
        for episode in annotation_manifest.get("episodes", []):
            raw_annotation = Path(str(episode.get("annotation_path") or ""))
            if raw_annotation.is_symlink():
                raise ValueError("Symlink reward annotations are not allowed")
            annotation = raw_annotation.resolve()
            try:
                annotation.relative_to(cache_root)
            except ValueError:
                raise ValueError("Reward annotation path is outside the managed cache") from None
            if not annotation.is_file():
                raise FileNotFoundError(annotation)
            digest = _sha256(annotation)
            if digest != episode.get("annotation_sha256"):
                raise ConflictError(
                    f"Reward annotation hash changed for {episode.get('run_id')}",
                    code="ANNOTATION_HASH_MISMATCH",
                )
        return work, prepared, annotation_manifest

    @staticmethod
    def _reward_source(raw: dict[str, Any], parameters: dict[str, Any]) -> str:
        if "reward_source" in parameters:
            source = parameters["reward_source"]
        elif "reward_rynnvalue" in parameters:
            # Old clients used a boolean checkbox; false meant sparse-only.
            source = "rynnvalue" if parameters["reward_rynnvalue"] else "sparse"
        else:
            source = raw["reward"].get("source") or (
                "rynnvalue" if raw["reward"].get("rynnvalue", True) else "sparse"
            )
        if source not in ("sparse", "rynnvalue", "stage"):
            raise ValueError("reward_source must be sparse, rynnvalue, or stage")
        return source

    def start_training(self, dataset_id: str, parameters: dict[str, Any]) -> dict[str, Any]:
        with self.lock:
            self._validate_training_parameters(parameters)
            dataset = self.datasets.require_ready_for_training(dataset_id)
            version, normalized = self.pinned_reward(dataset, parameters)
            self._prepare_launch()
            try:
                return self._launch_training(
                    dataset_id, dataset, normalized, reward_version=version,
                )
            finally:
                self.launch_reserved = False

    def _launch_training(
        self, dataset_id: str, dataset: dict[str, Any], parameters: dict[str, Any],
        *, reward_version: dict[str, Any],
    ) -> dict[str, Any]:
        # Start from the sealed evaluation configuration: input sampling, reward
        # parameters and prepared-data identity must not drift with base YAML.
        sealed = yaml.safe_load(Path(reward_version["config_path"]).read_text(encoding="utf-8"))
        raw = self._effective_config(dataset)
        for section in ("data", "reward", "paths", "vla"):
            raw[section] = copy.deepcopy(sealed[section])
        source = self._reward_source(raw, parameters)
        raw["reward"].update(source=source, rynnvalue=source == "rynnvalue")
        annotation_id = reward_version["id"]
        work_dir = Path(reward_version["work_dir"])
        prepared = json.loads(
            Path(reward_version["prepared_manifest_path"]).read_text(encoding="utf-8")
        )
        replay_counts = _training_replay_counts(prepared)
        raw["reward"].update(manifest_path=reward_version["reward_manifest_path"],
            manifest_sha256=reward_version["reward_manifest_sha256"], version_id=annotation_id)
        for key, value in parameters.items():
            if key in raw["iql"]:
                raw["iql"][key] = value
            elif key in raw["logging"]:
                raw["logging"][key] = value
        for parameter_name, config_name in {
            "reward_gamma": "gamma",
            "reward_shaping_weight": "shaping_weight",
            "reward_accumulate_primitive_steps": "accumulate_primitive_steps",
            "reward_stage_exponent": "stage_exponent",
        }.items():
            if parameter_name in parameters:
                raw["reward"][config_name] = parameters[parameter_name]
        wandb = raw["logging"]["wandb"]
        for parameter_name, config_name in {
            "wandb_enabled": "enabled", "wandb_mode": "mode",
            "wandb_project": "project", "wandb_entity": "entity",
            "wandb_run_name": "run_name", "wandb_group": "group",
            "wandb_log_interval_steps": "log_interval_steps",
        }.items():
            if parameter_name in parameters:
                wandb[config_name] = parameters[parameter_name]
        if "wandb_tags" in parameters:
            wandb["tags"] = [
                tag.strip() for tag in str(parameters["wandb_tags"] or "").split(",")
                if tag.strip()
            ]
        raw["iql"]["resume_checkpoint"] = self._resolve_resume(
            parameters.get("resume_checkpoint")
        )
        if raw["iql"]["checkpoint_interval"] % raw["iql"]["gradient_accumulation_steps"]:
            raise ValueError(
                "checkpoint_interval must be divisible by gradient_accumulation_steps"
            )
        job_id = f"train_{datetime.now().strftime('%Y%m%d_%H%M%S')}_{uuid.uuid4().hex[:8]}"
        job_dir = self.jobs_root / job_id
        job_dir.mkdir(parents=True, exist_ok=False)
        raw["paths"]["work_dir"] = str(work_dir.resolve())
        output_root = self.training_root / job_id
        output_root.mkdir(parents=True, exist_ok=False)
        raw["paths"]["output_dir"] = str(output_root.resolve())
        config_path = job_dir / "effective_config.yaml"
        atomic_write_yaml(config_path, raw)
        scripts = self.ui_config.offline_rl_root / "scripts"
        stages = [
            {
                "id": "train", "label": "VLA-Adapter Pixel-IQL 后训练",
                "environment": self.ui_config.train_environment,
                "argv": ["python", str(scripts / "train_iql.py"),
                         "--config", str(config_path)],
                "cwd": str(self.ui_config.offline_rl_root),
            },
        ]
        return self._new_job(
            kind="training", dataset_id=dataset_id, stages=stages,
            config_path=config_path, output_path=output_root,
            parameters={
                "task_id": dataset["task_id"], "member_count": dataset["member_count"],
                **replay_counts,
                "replay_policy": "full_recording_v1",
                "annotation_id": annotation_id,
                "reward_version_id": annotation_id,
                "reward_manifest_sha256": reward_version["reward_manifest_sha256"],
                "source_dataset_sha256": dataset.get("dataset_sha256"),
                "reward": {
                    "source": source,
                    "stage_exponent": raw["reward"].get("stage_exponent", 2.0),
                    "rynnvalue": raw["reward"]["rynnvalue"],
                    "gamma": raw["reward"]["gamma"],
                    "shaping_weight": raw["reward"]["shaping_weight"],
                    "accumulate_primitive_steps": raw["reward"][
                        "accumulate_primitive_steps"
                    ],
                },
                **{name: raw["iql"][name] for name in (
                    "train_steps", "critic_warmup_steps", "micro_batch_size",
                    "gradient_accumulation_steps", "checkpoint_interval", "seed",
                    "critic_optimizer", "critic_lr", "critic_weight_decay",
                    "critic_max_grad_norm", "value_optimizer", "value_lr",
                    "value_weight_decay", "value_max_grad_norm",
                    "policy_peak_lr", "policy_final_lr", "expectile", "beta",
                    "max_advantage_weight", "target_tau",
                )},
                "tensorboard": raw["logging"]["tensorboard"],
                "wandb": raw["logging"]["wandb"],
            },
        )

    @staticmethod
    def _evaluation_id(value: Any) -> str:
        evaluation_id = str(value or "")
        if (
            not evaluation_id
            or Path(evaluation_id).name != evaluation_id
            or ".." in evaluation_id
            or re.fullmatch(r"[A-Za-z0-9_-]+", evaluation_id) is None
        ):
            raise ValueError("Invalid evaluation id")
        return evaluation_id

    @staticmethod
    def _task_slug(task: dict[str, Any]) -> str:
        source = str(task.get("task_name") or task.get("task_id") or "task")
        slug = re.sub(r"[^a-z0-9]+", "_", source.lower()).strip("_")
        if not slug:
            raise ValueError("Task cannot be converted to a safe evaluation slug")
        return slug

    @staticmethod
    def _validate_date_filter(value: str | None, name: str) -> str | None:
        if value is None:
            return None
        try:
            datetime.strptime(value, "%Y-%m-%d")
        except ValueError:
            raise ValueError(f"{name} must use YYYY-MM-DD") from None
        return value

    @staticmethod
    def _evaluation_config(
        request: dict[str, Any], *, state_indices: list[int], seed_count: int,
        control_hz: float,
    ) -> dict[str, Any]:
        return {
            "trials": int(request["trials"]),
            "max_steps": int(request["max_steps"]),
            "open_loop_steps": int(request["open_loop_steps"]),
            "realtime": bool(request["realtime"]),
            "init_state_indices": state_indices,
            "base_seed": int(request["base_seed"]),
            "seed_count": seed_count,
            "schedule_seed": int(request["schedule_seed"]),
            "control_hz": int(control_hz),
            "success_streak": 5,
            "consecutive_error_limit": 3,
            "disabled_policy_cameras": [],
        }

    def _evaluation_snapshots(
        self, task_id: str, policy_id: str
    ) -> tuple[dict[str, Any], dict[str, Any]]:
        task = self.manager.catalog.metadata(task_id)
        bddl_path, init_path = self.manager.catalog.paths(task_id)
        task_snapshot = {
            "task_id": task["task_id"],
            "level": task["level"],
            "task_name": task["task_name"],
            "prompt": task["prompt"],
            "bddl_path": str(bddl_path.resolve()),
            "init_path": str(init_path.resolve()),
        }
        self.manager.policy_catalog.refresh()
        policy = self.manager.policy_catalog.entry(policy_id)
        policy_snapshot = {
            "policy_id": policy.policy_id,
            "label": policy.label,
            "base_checkpoint": policy.base_checkpoint,
            "stats_key": policy.stats_key,
            "manifest": None if policy.manifest is None else str(policy.manifest),
            "action_head": None if policy.action_head is None else str(policy.action_head),
            "proprio_projector": None
            if policy.proprio_projector is None else str(policy.proprio_projector),
            "training_step": policy.training_step,
            "compatibility_sha256": policy.compatibility_sha256,
        }
        return task_snapshot, policy_snapshot

    def preview_evaluation(self, request: dict[str, Any]) -> dict[str, Any]:
        """Validate an evaluation and return its deterministic frozen schedule."""
        from ..evaluation.batch import build_evaluation_preview

        task_id = str(request["task_id"])
        task_snapshot, policy_snapshot = self._evaluation_snapshots(
            task_id, str(request["policy_id"])
        )
        state_count = int(self.manager.catalog.initial_state_count(task_id))
        requested_states = request.get("init_state_indices")
        state_indices = (
            list(range(state_count))
            if requested_states is None else [int(value) for value in requested_states]
        )
        if not state_indices:
            raise ValueError("At least one init state must be selected")
        if len(state_indices) != len(set(state_indices)):
            raise ValueError("init_state_indices must not contain duplicates")
        invalid = [value for value in state_indices if not 0 <= value < state_count]
        if invalid:
            raise ValueError(
                f"init_state_indices for {task_id} must be in [0, {state_count - 1}]; "
                f"invalid values: {invalid}"
            )
        trials = int(request["trials"])
        seed_count = request.get("seed_count")
        effective_seed_count = (
            math.ceil(trials / len(state_indices))
            if seed_count is None else int(seed_count)
        )
        base_seed = int(request["base_seed"])
        if base_seed + effective_seed_count - 1 > 2147483647:
            raise ValueError("base_seed + seed_count exceeds 2147483647")
        control_hz = int(getattr(self.manager.eval_config, "control_hz", 20))
        preview = build_evaluation_preview(
            trials=trials,
            init_state_indices=state_indices,
            base_seed=base_seed,
            seed_count=effective_seed_count,
            schedule_seed=int(request["schedule_seed"]),
            control_hz=control_hz,
            max_steps=int(request["max_steps"]),
        )
        config = self._evaluation_config(
            request,
            state_indices=state_indices,
            seed_count=effective_seed_count,
            control_hz=control_hz,
        )
        distribution = preview.get("distribution") or {}
        public_schedule = [
            {
                "trial_index": int(item.get("trial_index", item.get("episode_index", 0))),
                "init_state_index": int(item["init_state_index"]),
                "seed": int(item["seed"]),
            }
            for item in preview["schedule"]
        ]
        return {
            **preview,
            "config": {
                "task_id": task_snapshot["task_id"],
                "policy_id": policy_snapshot["policy_id"],
                **config,
            },
            "schedule": public_schedule,
            "task_snapshot": task_snapshot,
            "policy_snapshot": policy_snapshot,
            "init_state_count": state_count,
            "init_state_counts": preview.get("init_state_counts")
            or distribution.get("init_state_counts", {}),
            "seed_counts": preview.get("seed_counts")
            or distribution.get("seed_counts", {}),
            "combination_counts": preview.get("combination_counts")
            or distribution.get("combination_counts", {}),
            "estimated_duration_seconds": preview.get(
                "estimated_duration_seconds",
                preview.get("estimated_simulation_seconds"),
            ),
        }

    def start_evaluation(self, request: dict[str, Any]) -> dict[str, Any]:
        with self.lock:
            preview = self.preview_evaluation(request)
            self._prepare_launch()
            try:
                return self._launch_evaluation(preview)
            finally:
                self.launch_reserved = False

    def _launch_evaluation(self, preview: dict[str, Any]) -> dict[str, Any]:
        evaluation_id = (
            f"eval_{datetime.now().strftime('%Y%m%d_%H%M%S')}_"
            f"{uuid.uuid4().hex[:8]}"
        )
        now = _utc_now()
        task = preview["task_snapshot"]
        policy = preview["policy_snapshot"]
        date = datetime.now().strftime("%Y-%m-%d")
        directory_name = (
            f"{datetime.now().strftime('%Y-%m-%d_%H%M%S')}__{evaluation_id}"
        )
        result_dir = self.evaluations_root / self._task_slug(task) / date / directory_name
        job_dir = self.jobs_root / evaluation_id
        result_dir.mkdir(parents=True, exist_ok=False)
        try:
            job_dir.mkdir(parents=True, exist_ok=False)
        except Exception:
            shutil.rmtree(result_dir)
            raise
        result_path = result_dir / "evaluation.json"
        config_path = job_dir / "effective_config.yaml"
        config_keys = {
            "trials", "max_steps", "open_loop_steps", "realtime",
            "init_state_indices", "base_seed", "seed_count", "schedule_seed",
            "control_hz", "success_streak", "consecutive_error_limit",
            "disabled_policy_cameras",
        }
        runtime_config = {
            key: value for key, value in preview["config"].items() if key in config_keys
        }
        raw_schedule = [
            {
                "trial_index": int(item.get("trial_index", index)),
                "init_state_index": int(item["init_state_index"]),
                "seed": int(item["seed"]),
            }
            for index, item in enumerate(preview["schedule"])
        ]
        effective = {
            "schema_version": 1,
            "evaluation_id": evaluation_id,
            "result_path": str(result_path.resolve()),
            "task_snapshot": task,
            "policy_snapshot": policy,
            "config": runtime_config,
            "schedule": raw_schedule,
            "schedule_sha256": preview["schedule_sha256"],
        }
        manifest = {
            "schema_version": 1,
            "id": evaluation_id,
            "job_id": evaluation_id,
            "status": "STARTING",
            "created_at": now,
            "started_at": None,
            "completed_at": None,
            "error": None,
            "task_snapshot": task,
            "policy_snapshot": policy,
            "config": runtime_config,
            "schedule": raw_schedule,
            "schedule_sha256": preview["schedule_sha256"],
            "success_rule": {
                "done_consecutive_steps": runtime_config["success_streak"],
                "success_latched": True,
                "run_full_horizon": True,
                "errors_in_denominator": True,
            },
            "aggregate": {
                "attempted": 0,
                "successes": 0,
                "failures": 0,
                "errors": 0,
                "completion_rate": 0.0,
                "success_rate": 0.0,
            },
            "trials": [],
        }
        atomic_write_yaml(config_path, effective)
        atomic_write_json(result_path, manifest)
        self.evaluation_repository.upsert(manifest, result_path)
        terminal_root = Path(__file__).resolve().parents[3]
        script = terminal_root / "scripts" / "run_policy_evaluation.py"
        stages = [{
            "id": "evaluate",
            "label": "LIBERO-X 策略批量测试",
            "environment": self.ui_config.train_environment,
            "argv": ["python", str(script), "--config", str(config_path)],
            "cwd": str(terminal_root),
        }]
        try:
            job = self._new_job(
                kind="evaluation",
                dataset_id=None,
                stages=stages,
                config_path=config_path,
                output_path=result_dir,
                parameters={
                    "evaluation_id": evaluation_id,
                    "task_id": task["task_id"],
                    "policy_id": policy["policy_id"],
                    **runtime_config,
                    "schedule_sha256": preview["schedule_sha256"],
                },
            )
        except Exception as exc:
            manifest.update(
                status="FAILED", completed_at=_utc_now(),
                error=f"Cannot launch evaluation: {type(exc).__name__}: {exc}",
            )
            atomic_write_json(result_path, manifest)
            self.evaluation_repository.upsert(manifest, result_path)
            raise
        return job

    def _ensure_evaluation_path(self, path: Path) -> Path:
        root = self.evaluations_root.resolve()
        if path.name != "evaluation.json" or path.is_symlink() or not path.is_file():
            raise FileNotFoundError(path)
        absolute = path.absolute()
        try:
            lexical = absolute.relative_to(root)
        except ValueError:
            raise ValueError("Evaluation manifest is outside the managed root") from None
        current = root
        for part in lexical.parts[:-1]:
            current = current / part
            if current.is_symlink():
                raise ValueError("Symlink evaluation paths are not allowed")
        resolved = path.resolve()
        try:
            relative = resolved.relative_to(root)
        except ValueError:
            raise ValueError("Evaluation manifest is outside the managed root") from None
        return resolved

    def _load_evaluation_manifest(self, path: Path) -> dict[str, Any]:
        path = self._ensure_evaluation_path(path)
        payload = json.loads(path.read_text(encoding="utf-8"))
        evaluation_id = self._evaluation_id(payload.get("id"))
        if payload.get("schema_version") != 1:
            raise ValueError(f"Unsupported evaluation manifest: {path}")
        if payload.get("job_id", evaluation_id) != evaluation_id:
            raise ValueError(f"Evaluation/job identity mismatch: {path}")
        return payload

    def _index_evaluation_manifest(self, path: Path) -> dict[str, Any]:
        payload = self._load_evaluation_manifest(path)
        self.evaluation_repository.upsert(payload, path)
        return payload

    def _reconcile_all_evaluations(self) -> None:
        for path in self.evaluations_root.glob("*/*/*/evaluation.json"):
            try:
                self._index_evaluation_manifest(path)
            except Exception:
                continue

    def _evaluation_manifest(self, evaluation_id: str) -> Path:
        evaluation_id = self._evaluation_id(evaluation_id)
        indexed = self.evaluation_repository.manifest_path(evaluation_id)
        candidates = [indexed] if indexed is not None else []
        candidates.extend(self.evaluations_root.glob("*/*/*/evaluation.json"))
        seen: set[Path] = set()
        for candidate in candidates:
            if candidate is None or candidate in seen:
                continue
            seen.add(candidate)
            try:
                payload = self._load_evaluation_manifest(candidate)
            except Exception:
                continue
            if payload["id"] == evaluation_id:
                self.evaluation_repository.upsert(payload, candidate)
                return candidate.resolve()
        raise KeyError(evaluation_id)

    def _synchronize_evaluation(
        self, path: Path, payload: dict[str, Any]
    ) -> dict[str, Any]:
        if payload.get("status") not in ACTIVE_JOB_STATES:
            self.evaluation_repository.upsert(payload, path)
            return payload
        try:
            self._load_job(payload["id"])
            public_job = self._reconcile(payload["id"])
        except Exception:
            self.evaluation_repository.upsert(payload, path)
            return payload
        if public_job["status"] in TERMINAL_JOB_STATES:
            # Normally the child writes the terminal result first. This fallback
            # covers launch failures or a child killed before it could persist.
            latest = self._load_evaluation_manifest(path)
            if latest.get("status") in ACTIVE_JOB_STATES:
                latest["status"] = public_job["status"]
                latest["completed_at"] = public_job.get("completed_at") or _utc_now()
                latest["error"] = public_job.get("error")
                atomic_write_json(path, latest)
            payload = latest
        self.evaluation_repository.upsert(payload, path)
        return payload

    @staticmethod
    def _public_evaluation(
        payload: dict[str, Any], path: Path, *, detail: bool
    ) -> dict[str, Any]:
        value = copy.deepcopy(payload)
        task = value.get("task_snapshot") or value.get("task") or {}
        policy = value.get("policy_snapshot") or value.get("policy") or {}
        value.update({
            "task_id": task.get("task_id"),
            "task_name": task.get("task_name"),
            "task_prompt": task.get("prompt"),
            "policy_id": policy.get("policy_id"),
            "policy_label": policy.get("label"),
            "base_checkpoint": policy.get("base_checkpoint"),
            "overlay_id": None
            if policy.get("policy_id") == "base" else policy.get("policy_id"),
            "training_step": policy.get("training_step"),
            "compatibility_sha256": policy.get("compatibility_sha256"),
            "output_path": str(path.parent.resolve()),
        })
        config = value.get("config") or {}
        value["config"] = {
            "task_id": task.get("task_id"),
            "policy_id": policy.get("policy_id"),
            **config,
        }
        schedule = value.get("schedule") or []
        value["schedule"] = [
            {
                **item,
                "trial_index": int(item.get("trial_index", item.get("episode_index", index))),
            }
            for index, item in enumerate(schedule)
        ]
        trials = value.get("trials", value.get("episodes", [])) or []
        value["trials"] = []
        for index, item in enumerate(trials):
            value["trials"].append({
                **item,
                "trial_index": int(item.get("trial_index", item.get("episode_index", index))),
                "steps": int(item.get("steps", item.get("executed_steps", 0)) or 0),
                "inference_latency_ms": item.get(
                    "inference_latency_ms", item.get("inference_latency_mean_ms")
                ),
                "elapsed_seconds": float(
                    item.get("elapsed_seconds", item.get("wall_seconds", 0.0)) or 0.0
                ),
            })
        aggregate = value.get("aggregate") or value.get("summary") or {}
        total = int(
            aggregate.get(
                "total_trials", aggregate.get("scheduled_trials", config.get("trials", 0))
            ) or 0
        )
        attempted = int(
            aggregate.get("attempted_trials", aggregate.get("attempted", len(trials))) or 0
        )
        successes = int(
            aggregate.get("successes", aggregate.get("success_count", 0)) or 0
        )
        errors = int(aggregate.get("errors", aggregate.get("error_count", 0)) or 0)
        failures = int(
            aggregate.get("failures", max(0, attempted - successes - errors)) or 0
        )
        wilson = aggregate.get("wilson_95") or [0.0, 0.0]

        def mean(name: str) -> Any:
            metric = aggregate.get(name)
            return metric.get("mean") if isinstance(metric, dict) else metric

        def breakdown(name: str) -> dict[str, Any]:
            groups = aggregate.get(name) or {}
            return {
                str(key): {
                    **group,
                    "trials": int(
                        group.get("trials", group.get("attempted", 0)) or 0
                    ),
                }
                for key, group in groups.items()
            }

        value["aggregate"] = {
            "total_trials": total,
            "attempted_trials": attempted,
            "completed_trials": int(
                aggregate.get("completed_trials", attempted) or 0
            ),
            "successes": successes,
            "failures": failures,
            "errors": errors,
            "success_rate": float(
                aggregate.get("success_rate", successes / attempted if attempted else 0.0)
                or 0.0
            ),
            "wilson_lower": float(
                aggregate.get("wilson_lower", wilson[0] if len(wilson) > 0 else 0.0)
                or 0.0
            ),
            "wilson_upper": float(
                aggregate.get("wilson_upper", wilson[1] if len(wilson) > 1 else 0.0)
                or 0.0
            ),
            "completion_rate": float(
                aggregate.get(
                    "completion_rate",
                    aggregate.get(
                        "completion_coverage", attempted / total if total else 0.0
                    ),
                )
                or 0.0
            ),
            "by_init_state": breakdown("by_init_state"),
            "by_seed": breakdown("by_seed"),
            "by_combination": breakdown("by_combination"),
            "first_success_step_mean": aggregate.get(
                "first_success_step_mean", mean("first_success_step")
            ),
            "policy_queries_mean": aggregate.get(
                "policy_queries_mean", mean("policy_queries")
            ),
            "inference_latency_ms_mean": aggregate.get(
                "inference_latency_ms_mean", mean("inference_latency_ms")
            ),
            "measured_control_hz_mean": aggregate.get(
                "measured_control_hz_mean", mean("control_hz")
            ),
            "elapsed_seconds_mean": aggregate.get(
                "elapsed_seconds_mean", mean("wall_seconds")
            ),
        }
        value["model_load_seconds"] = value.get(
            "model_load_seconds", (value.get("timing") or {}).get("model_load_seconds")
        )
        value["wall_time_seconds"] = value.get(
            "wall_time_seconds",
            (value.get("timing") or {}).get(
                "wall_time_seconds", (value.get("timing") or {}).get("wall_seconds")
            ),
        )
        value["simulated_time_seconds"] = value.get(
            "simulated_time_seconds",
            (value.get("timing") or {}).get(
                "simulated_time_seconds",
                (value.get("timing") or {}).get("simulated_seconds"),
            ),
        )
        if not detail:
            value["schedule_count"] = len(value.pop("schedule", []))
            value["attempted_count"] = len(value.pop("trials", []))
        return value

    def get_evaluation(self, evaluation_id: str) -> dict[str, Any]:
        path = self._evaluation_manifest(evaluation_id)
        payload = self._synchronize_evaluation(
            path, self._load_evaluation_manifest(path)
        )
        return self._public_evaluation(payload, path, detail=True)

    def list_evaluations(
        self,
        *,
        task_id: str | None = None,
        policy_id: str | None = None,
        status: str | None = None,
        date_from: str | None = None,
        date_to: str | None = None,
    ) -> list[dict[str, Any]]:
        date_from = self._validate_date_filter(date_from, "date_from")
        date_to = self._validate_date_filter(date_to, "date_to")
        if date_from and date_to and date_from > date_to:
            raise ValueError("date_from must not be later than date_to")
        result: list[dict[str, Any]] = []
        for path in self.evaluations_root.glob("*/*/*/evaluation.json"):
            try:
                payload = self._synchronize_evaluation(
                    path, self._load_evaluation_manifest(path)
                )
            except Exception:
                continue
            task = (
                payload.get("task_snapshot") or payload.get("task") or {}
            ).get("task_id")
            policy = (
                payload.get("policy_snapshot") or payload.get("policy") or {}
            ).get("policy_id")
            created_date = str(payload.get("created_at") or "")[:10]
            if task_id is not None and task != task_id:
                continue
            if policy_id is not None and policy != policy_id:
                continue
            if status is not None and payload.get("status") != status:
                continue
            if date_from is not None and created_date < date_from:
                continue
            if date_to is not None and created_date > date_to:
                continue
            result.append(self._public_evaluation(payload, path, detail=False))
        return sorted(result, key=lambda item: item.get("created_at", ""), reverse=True)

    def stop_evaluation(self, evaluation_id: str) -> dict[str, Any]:
        path = self._evaluation_manifest(evaluation_id)
        current_job = self.get(evaluation_id)
        if current_job["kind"] != "evaluation":
            raise ValueError("Evaluation/job identity mismatch")
        if current_job["status"] not in ACTIVE_JOB_STATES:
            return current_job
        job = self.stop(evaluation_id)
        latest = self._load_evaluation_manifest(path)
        if latest.get("status") in ACTIVE_JOB_STATES:
            latest["status"] = (
                job["status"] if job["status"] in TERMINAL_JOB_STATES else "STOPPING"
            )
            if job["status"] in TERMINAL_JOB_STATES:
                latest["completed_at"] = job.get("completed_at") or _utc_now()
                latest["error"] = job.get("error")
            atomic_write_json(path, latest)
            self.evaluation_repository.upsert(latest, path)
        return job

    def delete_evaluation(
        self, evaluation_id: str, confirm_evaluation_id: str
    ) -> dict[str, Any]:
        evaluation_id = self._evaluation_id(evaluation_id)
        if confirm_evaluation_id != evaluation_id:
            raise ValueError(
                "confirm_evaluation_id must exactly match evaluation_id"
            )
        with self.lock:
            path = self._evaluation_manifest(evaluation_id)
            payload = self._synchronize_evaluation(
                path, self._load_evaluation_manifest(path)
            )
            if payload.get("status") in ACTIVE_JOB_STATES:
                raise ConflictError(
                    "Active evaluation must be stopped before deletion",
                    code="EVALUATION_ACTIVE",
                    context={"evaluation_id": evaluation_id},
                )
            result_dir = path.parent
            if result_dir.is_symlink():
                raise ValueError("Symlink evaluation directories are not allowed")
            job_dir = self.jobs_root / evaluation_id
            if job_dir.exists():
                if job_dir.is_symlink() or job_dir.parent.resolve() != self.jobs_root.resolve():
                    raise ValueError("Unsafe evaluation job directory")
                job_path = job_dir / "job.json"
                if job_path.is_file():
                    job = json.loads(job_path.read_text(encoding="utf-8"))
                    if job.get("id") != evaluation_id or job.get("kind") != "evaluation":
                        raise ValueError("Evaluation job identity mismatch")
                    if job.get("status") in ACTIVE_JOB_STATES:
                        raise ConflictError(
                            "Active evaluation must be stopped before deletion",
                            code="EVALUATION_ACTIVE",
                            context={"evaluation_id": evaluation_id},
                        )
            shutil.rmtree(result_dir)
            if job_dir.is_dir():
                shutil.rmtree(job_dir)
            self.evaluation_repository.delete(evaluation_id)
            self.repository.delete(evaluation_id)
            for directory in (result_dir.parent, result_dir.parent.parent):
                try:
                    directory.rmdir()
                except OSError:
                    pass
            return {
                "deleted": evaluation_id,
                "policy_deleted": False,
                "dataset_deleted": False,
            }

    def stop(self, job_id: str) -> dict[str, Any]:
        path, job = self._load_job(job_id)
        if job["status"] not in ACTIVE_JOB_STATES:
            return self._public_job(job)
        job["status"] = "STOPPING"
        job["stage_label"] = "正在安全停止"
        atomic_write_json(path, job)
        process_group = job.get("process_group_id")
        pid = job.get("pid") or job.get("launcher_pid")
        try:
            if process_group:
                os.killpg(int(process_group), signal.SIGTERM)
            elif pid:
                os.kill(int(pid), signal.SIGTERM)
        except ProcessLookupError:
            pass
        self.repository.upsert(job, path)
        return self._public_job(job)

    def _latest_metrics(self, job: dict[str, Any]) -> dict[str, Any] | None:
        output = Path(job["output_path"])
        candidates = sorted(output.glob("*/metrics.jsonl"), key=lambda path: path.stat().st_mtime)
        if not candidates:
            return None
        try:
            with candidates[-1].open("rb") as stream:
                stream.seek(0, os.SEEK_END)
                end = stream.tell()
                stream.seek(max(0, end - 65536))
                lines = stream.read().decode("utf-8", errors="replace").splitlines()
            for line in reversed(lines):
                if line.strip().startswith("{"):
                    return json.loads(line)
        except Exception:
            return None
        return None

    @staticmethod
    def _training_summary(job: dict[str, Any]) -> dict[str, Any] | None:
        output = Path(job["output_path"])
        summaries = sorted(output.glob("*/summary.json"))
        if not summaries:
            return None
        try:
            return json.loads(summaries[-1].read_text(encoding="utf-8"))
        except Exception:
            return None

    def available_checkpoints(self, dataset_id: str | None = None) -> list[dict[str, Any]]:
        expected_dataset_hash = None
        dataset = None
        if dataset_id:
            dataset = self.datasets.get(dataset_id)
            annotation_id = dataset.get("annotation_id")
            if dataset.get("annotation_status") == "READY" and annotation_id:
                work = self.datasets.root / dataset_id / "annotations" / annotation_id / "work"
                prepared = work / "dataset_manifest.json"
                if prepared.is_file():
                    prepared_payload = json.loads(prepared.read_text(encoding="utf-8"))
                    expected_dataset_hash = prepared_payload.get("dataset_sha256")
        result = []
        jobs: dict[str, dict[str, Any] | None] = {}
        jobs_root = getattr(self, "jobs_root", self.training_root.parent / "jobs")
        for path in sorted(
            self.training_root.glob("*/*/checkpoints/step_*"), reverse=True
        ):
            metadata_path = path / "checkpoint.json"
            if not path.is_dir() or path.is_symlink() or not metadata_path.is_file():
                continue
            try:
                metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
            except Exception:
                continue
            job_id = path.parents[2].name
            if job_id not in jobs:
                job_path = jobs_root / job_id / "job.json"
                try:
                    job = json.loads(job_path.read_text(encoding="utf-8"))
                    jobs[job_id] = job if (
                        not job_path.is_symlink() and job.get("id") == job_id
                        and job.get("kind") == "training"
                    ) else None
                except (OSError, ValueError):
                    jobs[job_id] = None
            job = jobs[job_id]
            if dataset_id:
                if job is not None:
                    if job.get("dataset_id") != dataset_id:
                        continue
                    frozen_hash = job.get("parameters", {}).get("source_dataset_sha256")
                    if frozen_hash and frozen_hash != dataset.get("dataset_sha256"):
                        continue
                elif (
                    expected_dataset_hash is None
                    or metadata.get("dataset_sha256") != expected_dataset_hash
                ):
                    continue
            # List this dataset's checkpoints across reward sources. Exact
            # reward/config compatibility remains enforced by checkpoint restore.
            source = (job or {}).get("parameters", {}).get("reward", {}).get("source")
            label = f"{job_id}/{path.parent.parent.name}/{path.name}"
            result.append({
                "path": str(path.resolve()),
                "label": f"{label} · {source}" if source else label,
            })
        return result

    def tensorboard_status(self) -> dict[str, Any]:
        url = f"http://{self.ui_config.tensorboard_host}:{self.ui_config.tensorboard_port}/"
        healthy = False
        try:
            with urllib.request.urlopen(url + "data/environment", timeout=0.5) as response:
                environment = json.loads(response.read().decode("utf-8"))
                healthy = response.status == 200 and isinstance(environment, dict)
        except (urllib.error.URLError, TimeoutError, json.JSONDecodeError):
            healthy = False
        process = self.tensorboard_process
        return {
            "url": url,
            "running": healthy,
            "managed": process is not None and process.poll() is None,
            "pid": None if process is None or process.poll() is not None else process.pid,
            "logdir": str(self.training_root.resolve()),
        }

    def start_tensorboard(self) -> dict[str, Any]:
        with self.lock:
            status = self.tensorboard_status()
            if status["running"]:
                return status
            if self.tensorboard_process is not None and self.tensorboard_process.poll() is None:
                return {**status, "starting": True, "managed": True, "pid": self.tensorboard_process.pid}
            with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as probe:
                if probe.connect_ex((self.ui_config.tensorboard_host, self.ui_config.tensorboard_port)) == 0:
                    raise ConflictError(
                        "TensorBoard port is occupied by another service",
                        code="TENSORBOARD_PORT_OCCUPIED",
                    )
            output = self.tensorboard_log.open("ab")
            self.tensorboard_process = subprocess.Popen(
                [
                    "conda", "run", "--no-capture-output", "-n",
                    self.ui_config.train_environment, "tensorboard",
                    "--logdir", str(self.training_root.resolve()),
                    "--host", self.ui_config.tensorboard_host,
                    "--port", str(self.ui_config.tensorboard_port),
                ],
                cwd=str(self.ui_config.offline_rl_root),
                stdin=subprocess.DEVNULL,
                stdout=output,
                stderr=subprocess.STDOUT,
                start_new_session=True,
            )
            return {
                **self.tensorboard_status(),
                "starting": True,
                "pid": self.tensorboard_process.pid,
            }

    def close(self) -> None:
        # Offline jobs and TensorBoard intentionally survive a UI backend restart.
        executor = getattr(self, "_binding_executor", None)
        if executor is not None:
            # Cancel queued publishers; unfinished jobs will be rediscovered on
            # restart. Let an in-flight atomic publication finish before exit.
            executor.shutdown(wait=True, cancel_futures=True)
