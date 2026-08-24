"""Persistent two-environment annotation/training orchestration."""

from __future__ import annotations

import copy
import fcntl
import hashlib
import json
import os
import signal
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
from ..storage.repositories import OfflineJobRepository


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


class OfflineJobService:
    def __init__(self, ui_config: Any, manager: Any, datasets: Any):
        self.ui_config = ui_config
        self.manager = manager
        self.datasets = datasets
        self.project_root = ui_config.project_root
        self.jobs_root = self.project_root / "jobs"
        self.training_root = self.project_root / "training"
        self.cache_root = self.project_root / "annotation-cache"
        self.gpu_lock_path = self.project_root / ".gpu-task.lock"
        for path in (self.jobs_root, self.training_root, self.cache_root):
            path.mkdir(parents=True, exist_ok=True)
        self.repository = OfflineJobRepository(
            ui_config.catalog_path, ui_config.project_id
        )
        self.lock = threading.RLock()
        self.launch_reserved = False
        self.tensorboard_process: subprocess.Popen[Any] | None = None
        self.tensorboard_log = self.project_root / "tensorboard.log"
        self._reconcile_all()

    @property
    def base_config_path(self) -> Path:
        return self.ui_config.offline_rl_root / "configs" / "liberox_iql.yaml"

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

    def defaults(self, dataset_id: str | None = None) -> dict[str, Any]:
        raw = self._load_base_config()
        return {
            "basic": {
                name: raw["iql"][name] for name in (
                    "train_steps", "critic_warmup_steps",
                    "gradient_accumulation_steps", "checkpoint_interval", "seed",
                )
            },
            "advanced": {
                name: raw["iql"][name] for name in (
                    "critic_lr", "value_lr", "policy_peak_lr", "policy_final_lr",
                    "expectile", "beta", "max_advantage_weight", "target_tau",
                )
            } | {
                "console_interval_steps": raw["logging"]["console_interval_steps"],
                "flush_seconds": raw["logging"]["flush_seconds"],
            },
            "fixed": {
                "dtype": raw["iql"]["dtype"],
                "micro_batch_size": raw["iql"]["micro_batch_size"],
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
        if job["kind"] == "annotation":
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
                    self.datasets.update_annotation(
                        job["dataset_id"], desired, annotation_id=job["id"]
                    )
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
        self, dataset_id: str, confirm_dataset_id: str
    ) -> dict[str, Any]:
        """Delete a dataset only when no active or training job references it."""
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
            if training:
                raise ConflictError(
                    "Dataset is referenced by training history",
                    code="DATASET_HAS_TRAINING",
                    context={"dataset_id": dataset_id, "training_jobs": training},
                )
            return self.datasets.delete_dataset(dataset_id, confirm_dataset_id)

    def _public_job(self, job: dict[str, Any]) -> dict[str, Any]:
        result = {
            key: value for key, value in job.items()
            if key not in {"stages", "gpu_lock_path"}
        }
        job_dir = self.jobs_root / job["id"]
        result["log_size"] = (job_dir / "job.log").stat().st_size if (job_dir / "job.log").is_file() else 0
        if job["kind"] == "training":
            metrics = self._latest_metrics(job)
            if metrics is not None:
                result["metrics"] = metrics
            summary = self._training_summary(job)
            if summary is not None:
                result["training_summary"] = summary
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
        if self.launch_reserved or self.has_active_job() or self._external_gpu_lock():
            raise ConflictError(
                "An annotation or training job is using the GPU",
                code="GPU_TASK_ACTIVE",
            )

    def _prepare_launch(self) -> None:
        if self.has_active_job() or self._external_gpu_lock():
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
        dataset_id: str,
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

    def start_annotation(self, dataset_id: str) -> dict[str, Any]:
        with self.lock:
            dataset = self.datasets.require_ready_for_annotation(dataset_id)
            self._prepare_launch()
            try:
                return self._launch_annotation(dataset_id, dataset)
            finally:
                self.launch_reserved = False

    def _launch_annotation(
        self, dataset_id: str, dataset: dict[str, Any]
    ) -> dict[str, Any]:
        job_id = f"ann_{datetime.now().strftime('%Y%m%d_%H%M%S')}_{uuid.uuid4().hex[:8]}"
        job_dir = self.jobs_root / job_id
        job_dir.mkdir(parents=True, exist_ok=False)
        annotation_root = self.datasets.root / dataset_id / "annotations" / job_id
        work_dir = annotation_root / "work"
        work_dir.mkdir(parents=True, exist_ok=False)
        raw = self._effective_config(dataset)
        raw["paths"]["work_dir"] = str(work_dir.resolve())
        config_path = job_dir / "effective_config.yaml"
        atomic_write_yaml(config_path, raw)
        scripts = self.ui_config.offline_rl_root / "scripts"
        stages = [
            {
                "id": "prepare", "label": "准备训练数据集",
                "environment": self.ui_config.train_environment,
                "argv": ["python", str(scripts / "prepare_dataset.py"), "--config", str(config_path)],
                "cwd": str(self.ui_config.offline_rl_root),
            },
            {
                "id": "annotate", "label": "RynnValue-4B 奖励标注",
                "environment": self.ui_config.reward_environment,
                "argv": ["python", str(scripts / "annotate_rewards.py"), "--config", str(config_path)],
                "cwd": str(self.ui_config.offline_rl_root),
            },
        ]
        self.datasets.update_annotation(dataset_id, "RUNNING", annotation_id=job_id)
        try:
            return self._new_job(
                kind="annotation", dataset_id=dataset_id, stages=stages,
                config_path=config_path, output_path=annotation_root,
                parameters={
                    "task_id": dataset["task_id"], "member_count": dataset["member_count"],
                    "action_count": dataset["action_count"], "chunk_count": dataset["chunk_count"],
                    "model": raw["reward"]["model"], "revision": raw["reward"]["revision"],
                },
            )
        except Exception:
            self.datasets.update_annotation(dataset_id, "ERROR", annotation_id=job_id)
            raise

    @staticmethod
    def _validate_training_parameters(parameters: dict[str, Any]) -> None:
        allowed = {
            "train_steps", "critic_warmup_steps", "gradient_accumulation_steps",
            "checkpoint_interval", "seed", "critic_lr", "value_lr",
            "policy_peak_lr", "policy_final_lr", "expectile", "beta",
            "max_advantage_weight", "target_tau", "console_interval_steps",
            "flush_seconds", "resume_checkpoint",
        }
        unknown = sorted(set(parameters) - allowed)
        if unknown:
            raise ValueError(f"Unknown training parameters: {unknown}")
        integer_positive = (
            "train_steps", "gradient_accumulation_steps", "checkpoint_interval",
            "console_interval_steps",
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
        ):
            value = parameters.get(name)
            if value is not None and (
                isinstance(value, bool) or not isinstance(value, (int, float)) or value < 0
            ):
                raise ValueError(f"{name} must be a non-negative number")
        expectile = parameters.get("expectile")
        if expectile is not None and (
            isinstance(expectile, bool) or not isinstance(expectile, (int, float))
            or not 0 <= expectile <= 1
        ):
            raise ValueError("expectile must be in [0, 1]")

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
        reward_path = work / "rewards" / "reward_manifest.json"
        if (
            prepared_path.is_symlink() or reward_path.is_symlink()
            or not prepared_path.is_file() or not reward_path.is_file()
        ):
            raise FileNotFoundError("Prepared dataset or reward manifest is missing")
        prepared = json.loads(prepared_path.read_text(encoding="utf-8"))
        reward = json.loads(reward_path.read_text(encoding="utf-8"))
        if (
            prepared.get("source_dataset_id") != dataset["id"]
            or prepared.get("source_dataset_sha256") != dataset["dataset_sha256"]
        ):
            raise ConflictError(
                "Prepared data does not match the immutable dataset",
                code="ANNOTATION_DATASET_MISMATCH",
            )
        if reward.get("complete") is not True or reward.get("dataset_sha256") != prepared.get("dataset_sha256"):
            raise ConflictError(
                "Reward annotation is incomplete or belongs to different prepared data",
                code="ANNOTATION_INCOMPLETE",
            )
        expected_runs = {episode["run_id"] for episode in prepared.get("episodes", [])}
        reward_runs = {episode.get("run_id") for episode in reward.get("episodes", [])}
        if reward_runs != expected_runs:
            raise ConflictError(
                "Reward annotation membership does not match prepared data",
                code="ANNOTATION_MEMBERSHIP_MISMATCH",
            )
        cache_root = self.cache_root.resolve()
        for episode in reward.get("episodes", []):
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
        return work, prepared, reward

    def start_training(self, dataset_id: str, parameters: dict[str, Any]) -> dict[str, Any]:
        with self.lock:
            dataset = self.datasets.require_ready_for_training(dataset_id)
            self._validate_training_parameters(parameters)
            self._prepare_launch()
            try:
                return self._launch_training(dataset_id, dataset, parameters)
            finally:
                self.launch_reserved = False

    def _launch_training(
        self, dataset_id: str, dataset: dict[str, Any], parameters: dict[str, Any]
    ) -> dict[str, Any]:
        raw = self._effective_config(dataset)
        annotation_id = dataset["annotation_id"]
        work_dir, _, _ = self._validated_annotation_work(dataset)
        raw["paths"]["work_dir"] = str(work_dir.resolve())
        for key, value in parameters.items():
            if key in raw["iql"]:
                raw["iql"][key] = value
            elif key in raw["logging"]:
                raw["logging"][key] = value
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
        output_root = self.training_root / job_id
        output_root.mkdir(parents=True, exist_ok=False)
        raw["paths"]["output_dir"] = str(output_root.resolve())
        config_path = job_dir / "effective_config.yaml"
        atomic_write_yaml(config_path, raw)
        script = self.ui_config.offline_rl_root / "scripts" / "train_iql.py"
        stages = [{
            "id": "train", "label": "VLA-Adapter Pixel-IQL 后训练",
            "environment": self.ui_config.train_environment,
            "argv": ["python", str(script), "--config", str(config_path)],
            "cwd": str(self.ui_config.offline_rl_root),
        }]
        return self._new_job(
            kind="training", dataset_id=dataset_id, stages=stages,
            config_path=config_path, output_path=output_root,
            parameters={
                "task_id": dataset["task_id"], "member_count": dataset["member_count"],
                "action_count": dataset["action_count"], "chunk_count": dataset["chunk_count"],
                "annotation_id": annotation_id,
                **{name: raw["iql"][name] for name in (
                    "train_steps", "critic_warmup_steps", "gradient_accumulation_steps",
                    "checkpoint_interval", "seed", "critic_lr", "value_lr",
                    "policy_peak_lr", "policy_final_lr", "expectile", "beta",
                    "max_advantage_weight", "target_tau",
                )},
            },
        )

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
        expected: tuple[str, str] | None = None
        if dataset_id:
            dataset = self.datasets.get(dataset_id)
            annotation_id = dataset.get("annotation_id")
            if dataset.get("annotation_status") == "READY" and annotation_id:
                work = self.datasets.root / dataset_id / "annotations" / annotation_id / "work"
                prepared = work / "dataset_manifest.json"
                rewards = work / "rewards" / "reward_manifest.json"
                if prepared.is_file() and rewards.is_file():
                    prepared_payload = json.loads(prepared.read_text(encoding="utf-8"))
                    reward_payload = json.loads(rewards.read_text(encoding="utf-8"))
                    expected = (
                        str(prepared_payload.get("dataset_sha256")),
                        _stable_hash(reward_payload),
                    )
        result = []
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
            if expected is not None and (
                metadata.get("dataset_sha256"), metadata.get("reward_sha256")
            ) != expected:
                continue
            result.append({
                "path": str(path.resolve()),
                "label": f"{path.parents[2].name}/{path.parent.parent.name}/{path.name}",
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
        pass
