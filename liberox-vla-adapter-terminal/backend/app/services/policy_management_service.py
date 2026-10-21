"""Safe local management of registered policy overlays."""

from __future__ import annotations

import hashlib
import json
import re
import shutil
import tempfile
from contextlib import nullcontext
from pathlib import Path
from typing import Any

import yaml

from ..core.exceptions import ConflictError
from ..storage.files import atomic_write_json, atomic_write_yaml


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
            result.update(manifest=None, components=[], training_records=self._training_records(entry, {}))
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
            for name, path in entry.component_paths.items()
            if path is not None
        ]
        result.update(
            manifest={
                key: manifest.get(key) for key in (
                    "dataset_sha256", "reward_sha256", "training_step", "algorithm",
                    "action_horizon", "action_dim", "proprio_dim",
                    "compatibility_sha256",
                )
            },
            components=components,
            training_records=self._training_records(entry, manifest),
        )
        return result

    @staticmethod
    def _training_source(entry) -> dict:
        manifest = yaml.safe_load(entry.manifest.read_text(encoding="utf-8")) if entry.manifest else {}
        fingerprint = hashlib.sha256(json.dumps({key: value for key, value in manifest.items()
            if key not in {"policy_id", "label"}}, sort_keys=True, separators=(",", ":")).encode()).hexdigest()
        result = {"source_policy_id": entry.policy_id,
                  "source_manifest": str(entry.manifest.resolve()) if entry.manifest else None,
                  "manifest_sha256": fingerprint}
        path = entry.manifest.with_name("provenance.json") if entry.manifest else None
        if path is not None and path.is_file():
            source = json.loads(path.read_text(encoding="utf-8"))
            if (not isinstance(source, dict) or not isinstance(source.get("source_policy_id"), str)
                    or not isinstance(source.get("source_manifest"), str)
                    or source.get("manifest_sha256") != fingerprint):
                raise ValueError("Copied-model training provenance does not match its weights")
            result = source
        return result

    def _training_records(self, entry, manifest: dict) -> list[dict]:
        if self.jobs is None:
            return []
        source = self._training_source(entry)
        records = []
        for job in self.jobs.list():
            if job.get("kind") != "training":
                continue
            settings = job.get("parameters", {}).get("model") or {}
            if entry.is_base:
                matches = (settings.get("base_id") == entry.policy_id and settings.get("family") == entry.family
                           and settings.get("base_checkpoint") == entry.base_checkpoint)
                if entry.base_revision and settings.get("base_revision") != entry.base_revision:
                    matches = False
            else:
                summary = job.get("training_summary") or {}
                exported = summary.get("policy_overlay") or summary.get("overlay_path") or summary.get("policy_manifest")
                matches = bool(exported and Path(exported).resolve() == Path(source["source_manifest"]).resolve())
                if not matches and job.get("output_path"):
                    for marker in Path(job["output_path"]).glob("*/export.json"):
                        try:
                            owner = json.loads(marker.read_text())
                            if (owner.get("policy_id") == source["source_policy_id"]
                                    and Path(str(owner.get("registry", ""))).resolve() / owner["policy_id"] / "policy.yaml"
                                    == Path(source["source_manifest"]).resolve()):
                                matches = True
                                break
                        except (OSError, ValueError, AttributeError):
                            continue
                for key, summary_key in (("dataset_sha256", "dataset_sha256"), ("training_step", "steps"), ("algorithm", "algorithm")):
                    expected = manifest.get(key, "iql" if key == "algorithm" else None)
                    if summary.get(summary_key) is not None and summary[summary_key] != expected:
                        matches = False
            if matches:
                records.append({**job, "model_relation": "source" if entry.is_base else "output"})
        return records

    def configuration(self, policy_id: str, job_id: str) -> dict[str, str]:
        detail = self.detail(policy_id)
        if not any(job["id"] == job_id for job in detail["training_records"]):
            raise KeyError("该训练记录不属于所选模型")
        return self.jobs.configuration(job_id)

    def _assert_mutable(self, policy_id: str) -> tuple[Any, Path]:
        entry = self._entry(policy_id)
        if entry.is_base:
            raise ConflictError("The base model is read-only", code="BASE_POLICY_READ_ONLY")
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
                if job.get("status") not in {"QUEUED", "STARTING", "RUNNING", "STOPPING"}:
                    continue
                if job.get("parameters", {}).get("policy_id") == policy_id:
                    raise ConflictError("Policy is used by an active job", code="POLICY_JOB_ACTIVE")

    def rename(self, policy_id: str, label: str) -> dict[str, Any]:
        # Registration and model mutation must not race between snapshot and
        # publication of the queued job that protects this model.
        with getattr(self.jobs, "lock", nullcontext()):
            return self._rename(policy_id, label)

    def _rename(self, policy_id: str, label: str) -> dict[str, Any]:
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
            for component in entry.component_paths.values():
                if component is None:
                    continue
                relative = component.resolve().relative_to(source.resolve())
                destination = temporary / relative
                destination.parent.mkdir(parents=True, exist_ok=True)
                shutil.copy2(component, destination)
            assert entry.manifest is not None
            payload = self._load_manifest(entry.manifest)
            atomic_write_json(temporary / "provenance.json", self._training_source(entry))
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
        with getattr(self.jobs, "lock", nullcontext()):
            return self._delete(policy_id, confirmation)

    def _delete(self, policy_id: str, confirmation: str) -> dict[str, Any]:
        if confirmation != policy_id:
            raise ValueError("confirm_policy_id must exactly match policy_id")
        _, directory = self._assert_mutable(policy_id)
        self._assert_idle(policy_id)
        provider = getattr(self.manager, "provider", None)
        if provider is not None and getattr(provider, "current_policy_id", None) == policy_id:
            provider.unload()
        latest = self.catalog.registry.resolve() / "latest"
        if latest.is_symlink() and latest.resolve(strict=False) == directory.resolve():
            latest.unlink()
        shutil.rmtree(directory)
        self.catalog.refresh()
        return {"deleted": policy_id}
