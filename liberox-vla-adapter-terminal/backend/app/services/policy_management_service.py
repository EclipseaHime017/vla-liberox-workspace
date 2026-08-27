"""Safe local management of VLA policy overlays."""

from __future__ import annotations

import re
import shutil
import tempfile
from pathlib import Path
from typing import Any

import yaml

from ..core.exceptions import ConflictError
from ..storage.files import atomic_write_yaml


class PolicyManagementService:
    def __init__(self, manager: Any, jobs: Any | None = None):
        self.manager = manager
        self.catalog = manager.policy_catalog
        self.jobs = jobs

    def list(self) -> list[dict[str, Any]]:
        return self.catalog.list_policies()

    def _entry(self, policy_id: str):
        self.catalog.refresh()
        return self.catalog.entry(policy_id)

    @staticmethod
    def _load_manifest(path: Path) -> dict[str, Any]:
        payload = yaml.safe_load(path.read_text(encoding="utf-8"))
        if not isinstance(payload, dict):
            raise ValueError("Policy manifest is invalid")
        return payload

    def detail(self, policy_id: str) -> dict[str, Any]:
        entry = self._entry(policy_id)
        result = entry.public()
        if entry.is_base:
            result.update(manifest=None, components=[], training_records=[])
            return result
        assert entry.manifest is not None
        manifest = self._load_manifest(entry.manifest)
        components = [
            {
                "name": name,
                "filename": path.name,
                "size_bytes": path.stat().st_size,
                "sha256": manifest["component_sha256"][name],
            }
            for name, path in (
                ("action_head", entry.action_head),
                ("proprio_projector", entry.proprio_projector),
            )
            if path is not None
        ]
        records: list[dict[str, Any]] = []
        if self.jobs is not None:
            for job in self.jobs.list():
                if job.get("kind") != "training":
                    continue
                summary = job.get("training_summary") or {}
                overlay = str(
                    summary.get("policy_overlay") or summary.get("overlay_path")
                    or summary.get("policy_manifest") or ""
                )
                if (
                    policy_id == job.get("parameters", {}).get("policy_id")
                    or (overlay and Path(overlay).parent.name == policy_id)
                    or job.get("parameters", {}).get("dataset_sha256") == manifest.get("dataset_sha256")
                ):
                    records.append(job)
        result.update(
            manifest={
                key: manifest.get(key) for key in (
                    "dataset_sha256", "reward_sha256", "training_step",
                    "action_horizon", "action_dim", "proprio_dim",
                    "compatibility_sha256",
                )
            },
            components=components,
            training_records=records,
        )
        return result

    def _assert_mutable(self, policy_id: str) -> tuple[Any, Path]:
        if policy_id == "base":
            raise ConflictError("The base model is read-only", code="BASE_POLICY_READ_ONLY")
        entry = self._entry(policy_id)
        assert entry.manifest is not None
        directory = entry.manifest.parent
        root = self.catalog.registry.resolve()
        if directory.is_symlink() or directory.resolve().parent != root:
            raise ValueError("Policy overlay is outside the managed registry")
        return entry, directory

    def _assert_idle(self, policy_id: str) -> None:
        active_id = getattr(self.manager, "active_session_id", None)
        if active_id:
            active = self.manager.get_public(active_id)
            if active.get("policy_id") == policy_id:
                raise ConflictError("Policy is used by the active simulation", code="POLICY_ACTIVE")
        draft = getattr(self.manager, "draft", None)
        if draft is not None and getattr(draft, "policy_id", None) == policy_id:
            raise ConflictError("Policy is selected by the simulation draft", code="POLICY_DRAFT_ACTIVE")
        if self.jobs is not None:
            for job in self.jobs.list():
                if job.get("status") not in {"STARTING", "RUNNING", "STOPPING"}:
                    continue
                if job.get("parameters", {}).get("policy_id") == policy_id:
                    raise ConflictError("Policy is used by an active job", code="POLICY_JOB_ACTIVE")

    def rename(self, policy_id: str, label: str) -> dict[str, Any]:
        clean = label.strip()
        if not clean or len(clean) > 100:
            raise ValueError("Model name must contain 1..100 characters")
        entry, _ = self._assert_mutable(policy_id)
        self._assert_idle(policy_id)
        assert entry.manifest is not None
        payload = self._load_manifest(entry.manifest)
        payload["label"] = clean
        atomic_write_yaml(entry.manifest, payload)
        self.catalog.refresh()
        return self.detail(policy_id)

    @staticmethod
    def _slug(label: str) -> str:
        slug = re.sub(r"[^a-z0-9]+", "-", label.lower()).strip("-")
        return slug[:40] or "iql-model"

    def copy(self, policy_id: str, label: str) -> dict[str, Any]:
        clean = label.strip()
        if not clean or len(clean) > 100:
            raise ValueError("Model name must contain 1..100 characters")
        entry, source = self._assert_mutable(policy_id)
        root = self.catalog.registry.resolve()
        root.mkdir(parents=True, exist_ok=True)
        stem = self._slug(clean)
        target_id = stem
        suffix = 2
        while (root / target_id).exists():
            target_id = f"{stem}-{suffix}"
            suffix += 1
        temporary = Path(tempfile.mkdtemp(prefix=".policy-copy-", dir=root))
        try:
            for component in (entry.action_head, entry.proprio_projector):
                assert component is not None
                shutil.copy2(component, temporary / component.name)
            assert entry.manifest is not None
            payload = self._load_manifest(entry.manifest)
            payload["policy_id"] = target_id
            payload["label"] = clean
            atomic_write_yaml(temporary / "policy.yaml", payload)
            temporary.replace(root / target_id)
        except BaseException:
            shutil.rmtree(temporary, ignore_errors=True)
            raise
        self.catalog.refresh()
        return self.detail(target_id)

    def delete(self, policy_id: str, confirmation: str) -> dict[str, Any]:
        if confirmation != policy_id:
            raise ValueError("confirm_policy_id must exactly match policy_id")
        _, directory = self._assert_mutable(policy_id)
        self._assert_idle(policy_id)
        latest = self.catalog.registry.resolve() / "latest"
        if latest.is_symlink() and latest.resolve(strict=False) == directory.resolve():
            latest.unlink()
        shutil.rmtree(directory)
        self.catalog.refresh()
        return {"deleted": policy_id}
