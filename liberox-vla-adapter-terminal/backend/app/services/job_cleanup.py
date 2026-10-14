"""Delete completed job artifacts within their owning platform directories."""
from __future__ import annotations

import json
import fcntl
import os
import shutil
import uuid
from pathlib import Path

from ..core.exceptions import ConflictError
from .policy_management_service import PolicyManagementService


def _owned_directory(path: Path, root: Path, identifier: str) -> Path:
    if path.is_symlink() or path.name != identifier or path.resolve().parent != root.resolve():
        raise ValueError(f"Job artifact is outside its managed root: {path}")
    return path


def _remove_exports(service, output: Path) -> list[str]:
    registry = service.manager.policy_catalog.registry
    with (registry / ".registry.lock").open("a+") as registry_lock:
        fcntl.flock(registry_lock.fileno(), fcntl.LOCK_EX)
        return _remove_exports_locked(service, output, registry)


def _remove_exports_locked(service, output: Path, registry: Path) -> list[str]:
    policies = PolicyManagementService(service.manager, service)
    deleted = []
    runs = [run for run in output.iterdir() if run.is_dir() and not run.is_symlink()] if output.is_dir() else []
    def exported_at(run):
        for name in ("export.json", "summary.json"):
            path = run / name
            if path.is_file() and not path.is_symlink():
                return path.stat().st_mtime_ns
        return run.stat().st_mtime_ns
    for run in sorted(runs, key=exported_at, reverse=True):
        marker = run / "export.json"
        summary = run / "summary.json"
        if marker.is_file() and not marker.is_symlink():
            owner = json.loads(marker.read_text())
            if Path(owner["registry"]).resolve() != registry.resolve():
                raise ValueError("Export registry does not match the platform registry")
            identifier = owner["policy_id"]
            previous = owner.get("previous_latest")
        elif summary.is_file() and not summary.is_symlink():
            exported = json.loads(summary.read_text()).get("policy_overlay")
            if not exported:
                continue
            manifest = Path(exported)
            if manifest.name != "policy.yaml" or manifest.parent.parent.resolve() != registry.resolve():
                raise ValueError("Exported model is outside the platform registry")
            identifier, previous = manifest.parent.name, None
        else:
            continue
        if not isinstance(identifier, str) or Path(identifier).name != identifier or identifier in {".", "..", "latest"}:
            raise ValueError("Invalid exported model id")
        if identifier in service.manager.policy_catalog.base_models:
            raise ValueError("Cannot delete a registered base model")
        directory = _owned_directory(registry / identifier, registry, identifier)
        if not directory.exists():
            continue
        policies._assert_idle(identifier)
        provider = getattr(service.manager, "provider", None)
        if provider is not None and getattr(provider, "current_policy_id", None) == identifier:
            provider.unload()
        latest = registry / "latest"
        restore_latest = latest.is_symlink() and latest.resolve(strict=False) == directory.resolve()
        shutil.rmtree(directory)
        if restore_latest:
            latest.unlink()
            if isinstance(previous, str) and Path(previous).name == previous and previous not in {".", "..", "latest"}:
                restored = registry / previous
                if restored.is_dir() and not restored.is_symlink():
                    temporary = registry / f".latest-{uuid.uuid4().hex}"
                    temporary.symlink_to(previous, target_is_directory=True)
                    os.replace(temporary, latest)
        deleted.append(identifier)
    service.manager.policy_catalog.refresh()
    return deleted


def delete_job(service, job_id: str, confirmation: str, *, delete_exported_models: bool = False) -> dict:
    if confirmation != job_id:
        raise ValueError("confirm_job_id must exactly match job_id")
    with service.lock:
        path, job = service._load_job(job_id)
        if (job.get("status") not in {"COMPLETED", "FAILED", "CANCELED"}
                or job_id in service._local_tasks
                or any(service._pid_alive(job.get(key)) for key in ("pid", "launcher_pid"))):
            raise ConflictError("Job must finish releasing its resources before deletion", code="JOB_ACTIVE")
        if job["kind"] == "evaluation":
            return service.delete_evaluation(job_id, confirmation)
        if job["kind"] != "training" and job.get("executor") != "local":
            raise ValueError("Only training and local task records can be deleted here")
        if delete_exported_models and job["kind"] != "training":
            raise ValueError("Exported models can only be deleted with their training job")
        directory = _owned_directory(path.parent, service.jobs_root, job_id)
        exports = []
        if job["kind"] == "training":
            output = _owned_directory(Path(job["output_path"]), service.training_root, job_id)
            for other in service.list():
                resume = (other.get("parameters") or {}).get("resume_checkpoint")
                if other["id"] != job_id and other["status"] in {"QUEUED", "STARTING", "RUNNING", "STOPPING"} and resume:
                    if Path(resume).resolve().is_relative_to(output.resolve()):
                        raise ConflictError("Checkpoint is used by another active job", code="JOB_CHECKPOINT_REFERENCED")
            if delete_exported_models:
                exports = _remove_exports(service, output)
            if output.is_dir():
                shutil.rmtree(output)
        shutil.rmtree(directory)
        service.repository.delete(job_id)
        return {"deleted": job_id, "models_deleted": exports}
