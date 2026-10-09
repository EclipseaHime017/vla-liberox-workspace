"""Read-only platform adapter for the independent keyframe experiment worker."""
from __future__ import annotations

from dataclasses import asdict
from contextlib import contextmanager
from datetime import datetime, timezone
import fcntl
import importlib
import json
import os
from pathlib import Path
import re
import shutil
import sys
import uuid

import yaml

from ..storage.files import atomic_write_json
from ..storage.paths import storage_path


class AnnotationLabService:
    def __init__(self, runs, project_root: Path, coordinator, package_root: Path | None = None):
        self.runs, self.coordinator = runs, coordinator
        self.root = project_root / "annotation-lab"
        self.package = package_root or Path(__file__).resolve().parents[4] / "keyframe-annotation"
        source = str(self.package / "src")
        if source not in sys.path:
            sys.path.insert(0, source)
        self._reconciled = False

    @contextmanager
    def storage_lock(self):
        self.root.mkdir(parents=True, exist_ok=True)
        with (self.root / ".lock").open("a") as lock:
            fcntl.flock(lock, fcntl.LOCK_EX)
            yield

    def current(self):
        path = self.root / "current.json"
        return json.loads(path.read_text()) if path.is_file() else {}

    def recover(self):
        if self._reconciled:
            return
        with self.coordinator.lock, self.storage_lock():
            self._reconciled = True  # prune/get can reenter during reconciliation
            try:
                self.reconcile()
            except Exception:
                self._reconciled = False
                raise

    def reconcile(self):
        """Recover publication/pruning after an interrupted registration."""
        requests = []
        for path in self.root.glob("lab_*/request.json"):
            identifier = path.parent.name
            directory = self.directory(identifier)
            job_path = self.coordinator.jobs_root / identifier / "job.json"
            if not job_path.is_file():
                continue  # A request alone is not a registered job.
            payload = json.loads(job_path.read_text())
            if payload.get("kind") != "annotation_lab" or payload.get("id") != identifier:
                raise ValueError("Invalid annotation job identity")
            request = json.loads((directory / "request.json").read_text())
            if request.get("id") != identifier:
                raise ValueError("Invalid annotation request identity")
            if any(not re.fullmatch(r"[A-Za-z0-9_-]{1,100}", run_id) for run_id in request["run_ids"]):
                raise ValueError("Invalid annotation member identity")
            self.coordinator.repository.upsert(payload, job_path)
            requests.append(request)
        current = {}
        for request in sorted(requests, key=lambda item: (item["created_at"], item["id"])):
            current.update({run_id: request["id"] for run_id in request["run_ids"]})
        atomic_write_json(self.root / "current.json", current)
        for request in requests:
            retained = [run_id for run_id in request["run_ids"] if current[run_id] == request["id"]]
            if retained != request["run_ids"]:
                self.prune(request["id"], retained)
        # Finish a crash between removing an output directory and removing its job.
        for identifier in self.coordinator.repository.queue_ids(kind="annotation_lab"):
            if identifier in current.values() or (self.root / identifier).exists():
                continue
            job_dir = self.coordinator.jobs_root / identifier
            if not re.fullmatch(r"lab_[A-Za-z0-9_]+", identifier) or job_dir.is_symlink():
                raise ValueError("Invalid orphan lab job path")
            if (job_dir / "job.json").exists():
                payload = json.loads((job_dir / "job.json").read_text())
                if payload["kind"] != "annotation_lab" or payload["status"] not in {"COMPLETED", "FAILED", "CANCELED"}:
                    continue
                shutil.rmtree(job_dir)
            self.coordinator.repository.delete(identifier)

    def prune(self, identifier, retained):
        """Remove only superseded lab members, never source data or reward files."""
        directory = self.directory(identifier)
        request = json.loads((directory / "request.json").read_text())
        job = self.coordinator.get(identifier)
        if job["status"] not in {"COMPLETED", "FAILED", "CANCELED"}:
            raise ValueError("Cannot replace an active annotation experiment")
        for run_id in set(request["run_ids"]) - set(retained):
            member = directory / run_id
            if member.is_symlink() or member.resolve().parent != directory.resolve():
                raise ValueError("Invalid result directory")
            if member.exists():
                shutil.rmtree(member)
        if retained:
            request.update(run_ids=retained, sources=[s for s in request["sources"] if s["run_id"] in retained])
            atomic_write_json(directory / "request.json", request)
        else:
            job_dir = self.coordinator.jobs_root / identifier
            if job_dir.is_symlink() or job_dir.resolve().parent != self.coordinator.jobs_root.resolve():
                raise ValueError("Invalid lab job directory")
            payload = json.loads((job_dir / "job.json").read_text())
            if payload["kind"] != "annotation_lab":
                raise ValueError("Refusing to remove a non-lab job")
            shutil.rmtree(directory)
            shutil.rmtree(job_dir)
            self.coordinator.repository.delete(identifier)

    def config(self):
        return importlib.import_module("keyframe_annotation.config").load_config(self.package / "configs/qwen3_vl.yaml")

    def defaults(self):
        config = self.config()
        snapshot = Path(config.cache_dir) / ("models--" + config.model_id.replace("/", "--")) / "snapshots" / config.revision
        index = snapshot / "model.safetensors.index.json"
        try:
            weights = set(json.loads(index.read_text())["weight_map"].values())
            available = bool(weights) and all((snapshot / name).is_file() for name in weights)
        except (OSError, ValueError, KeyError):
            available = False
        prefixes = [Path(sys.prefix).parent / config.environment, Path(sys.prefix) / "envs" / config.environment]
        if os.environ.get("CONDA_EXE"):
            prefixes.append(Path(os.environ["CONDA_EXE"]).parent.parent / "envs" / config.environment)
        registry = Path.home() / ".conda/environments.txt"
        if registry.is_file():
            prefixes.extend(Path(line) for line in registry.read_text().splitlines() if Path(line).name == config.environment)
        environment_ready = any((prefix / "bin/python").is_file() for prefix in prefixes)
        return {"config": asdict(config), "model_available": available,
            "environment_available": environment_ready, "proposal_only": True,
            "message": None if available and environment_ready else "请先安装 keyframe-vlm 环境并下载锁定的 Qwen 权重；原平台功能不受影响。"}

    def source(self, run_id):
        if not isinstance(run_id, str) or not re.fullmatch(r"[A-Za-z0-9_-]{1,100}", run_id):
            raise ValueError("Invalid run ID")
        run = self.runs.get_run(run_id)
        if run.get("status") not in {"COMPLETED", "ERROR"} or not run.get("action_count"):
            raise ValueError(f"轨迹尚未完成或没有动作：{run_id}")
        trajectory = storage_path(str(run.get("trajectory") or ""))
        if not trajectory.is_file() or trajectory.is_symlink() or trajectory.name != "trajectory.npz":
            raise ValueError(f"轨迹不可用：{run_id}")
        trajectory = trajectory.resolve()
        observations = trajectory.with_name("trajectory_observations.npz")
        if not observations.is_file() or observations.is_symlink():
            raise ValueError(f"缺少原始 observation：{run_id}")
        manifest = trajectory.parent.parent.parent / "run.json"
        metadata = json.loads(manifest.read_text())
        if metadata.get("id") != run_id or metadata.get("task_id") != run.get("task_id"):
            raise ValueError("轨迹清单与当前记录身份不一致")
        prompt = run.get("task") or metadata.get("task")
        if not isinstance(prompt, str) or not prompt.strip():
            raise ValueError("缺少任务提示词")
        return {"run_id": run_id, "task_id": run["task_id"], "prompt": prompt,
            "trajectory_path": str(trajectory), "observations_path": str(observations),
            "manifest_path": str(manifest), "control_hz": metadata.get("control_hz"),
            "orientation": metadata.get("observation_orientation", "libero_raw")}

    def start(self, run_ids, overrides):
        if not 1 <= len(run_ids) <= 100 or len(set(run_ids)) != len(run_ids):
            raise ValueError("请选择 1–100 条不重复轨迹")
        allowed = {"mode", "coarse_fps", "window_seconds", "cameras"}
        if set(overrides) - allowed:
            raise ValueError("Unknown lab options")
        config = type(self.config()).from_dict({**asdict(self.config()), **overrides}, self.package)
        ready = self.defaults()
        if not ready["model_available"] or not ready["environment_available"]:
            raise ValueError(ready["message"])
        self.recover()
        # Reuse only scheduling/storage primitives, never the dataset reward services.
        with self.coordinator.lock, self.storage_lock():
            self.reconcile()
            sources = [self.source(run_id) for run_id in run_ids]
            current = self.current()
            replaced = {current[run_id] for run_id in run_ids if run_id in current}
            for previous in replaced:
                if self.coordinator.get(previous)["status"] not in {"COMPLETED", "FAILED", "CANCELED"}:
                    raise ValueError("所选轨迹已有未结束的标注任务，请先完成或停止该任务")
            identifier = f"lab_{datetime.now(timezone.utc):%Y%m%d_%H%M%S}_{uuid.uuid4().hex[:8]}"
            job_dir = self.coordinator.jobs_root / identifier
            job_dir.mkdir(parents=True)
            output = self.root / identifier
            output.mkdir(parents=True)
            config_path = job_dir / "config.yaml"
            config_path.write_text(yaml.safe_dump(asdict(config), allow_unicode=True), encoding="utf-8")
            atomic_write_json(job_dir / "inputs.json", sources)
            atomic_write_json(output / "request.json", {"id": identifier, "run_ids": run_ids,
                "created_at": datetime.now(timezone.utc).isoformat(), "config": asdict(config), "sources": sources})
            try:
                self.coordinator.enqueue_external(kind="annotation_lab", config_path=config_path,
                    output_path=output, parameters={"run_ids": run_ids, "requires_gpu": True},
                    stages=[{"id": "annotation_lab", "label": "Qwen 目标区间实验", "environment": config.environment,
                        "argv": ["python", str(self.package / "scripts/annotate.py"), "--config", str(config_path),
                                 "--inputs", str(job_dir / "inputs.json"), "--output", str(output)], "cwd": str(self.package)}])
            except Exception:
                self._reconciled = False
                # A rejected submission must not erase the last available result.
                if not (job_dir / "job.json").exists():
                    shutil.rmtree(output)
                    shutil.rmtree(job_dir)
                raise
            try:
                current.update({run_id: identifier for run_id in run_ids})
                atomic_write_json(self.root / "current.json", current)
                for previous in replaced:
                    self.prune(previous, [run_id for run_id, owner in current.items() if owner == previous])
            except Exception:
                self._reconciled = False
                raise
            return self.get(identifier)

    def directory(self, identifier):
        if not re.fullmatch(r"lab_[A-Za-z0-9_]+", identifier):
            raise ValueError("Invalid experiment ID")
        path = self.root / identifier
        if not (path / "request.json").is_file() or path.is_symlink():
            raise FileNotFoundError("实验不存在")
        return path

    def get(self, identifier):
        self.recover()
        directory = self.directory(identifier)
        request = json.loads((directory / "request.json").read_text())
        job = self.coordinator.get(identifier)
        progress = json.loads((directory / "progress.json").read_text()) if (directory / "progress.json").exists() else None
        status, error = job["status"], (progress or {}).get("error") or job.get("error")
        if progress and len(request["run_ids"]) != progress["total_runs"]:
            members = [item for item in progress.get("runs", []) if item["run_id"] in request["run_ids"]]
            failed = sum(item["status"] == "FAILED" for item in members)
            completed = sum(item["status"] == "COMPLETED" for item in members)
            error = "; ".join(item["error"] for item in members if item.get("error")) or None
            status = "FAILED" if failed else "COMPLETED" if completed == len(request["run_ids"]) else status
            progress = {**progress, "runs": members, "completed_runs": completed, "failed_runs": failed,
                        "total_runs": len(request["run_ids"]), "error": error, "status": status}
        return {"id": identifier, "status": status, "error": error,
            "created_at": request["created_at"], "run_ids": request["run_ids"], "config": request["config"],
            "progress": progress, "proposal_only": True}

    def list(self):
        self.recover()
        result = []
        for identifier in sorted(set(self.current().values()), reverse=True):
            try:
                result.append(self.get(identifier))
            except (OSError, ValueError, KeyError):
                continue
        return result

    def result(self, identifier, run_id):
        directory = self.member_directory(identifier, run_id)
        path = directory / "result.json"
        result = json.loads(path.read_text()) if path.exists() else None
        if result is not None:
            result["calls"] = [json.loads(p.read_text()) for p in sorted((directory / "calls").glob("*.json"))]
        return result

    def member_directory(self, identifier, run_id):
        directory = self.directory(identifier)
        request = json.loads((directory / "request.json").read_text())
        if not re.fullmatch(r"[A-Za-z0-9_-]{1,100}", run_id) or run_id not in request["run_ids"]:
            raise ValueError("轨迹不属于这个实验")
        if self.current().get(run_id) != identifier:
            raise ValueError("该轨迹的旧实验结果已被替换")
        member = directory / run_id
        if member.resolve().parent != directory.resolve():
            raise ValueError("Invalid experiment member path")
        return member

    def evidence(self, identifier, run_id, camera, step):
        directory = self.member_directory(identifier, run_id)
        if camera not in {"agentview_image", "wrist_image"} or step < 0:
            raise ValueError("Invalid evidence reference")
        path = directory / "evidence" / f"{camera}_{step}.jpg"
        if not path.is_file() or path.is_symlink() or directory.resolve() not in path.resolve().parents:
            raise FileNotFoundError("该帧尚未作为模型证据采样")
        return path

    def stop(self, identifier):
        self.directory(identifier)
        self.coordinator.stop(identifier)
        return self.get(identifier)
