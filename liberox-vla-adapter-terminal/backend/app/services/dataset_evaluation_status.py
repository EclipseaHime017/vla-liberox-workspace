"""Lightweight member-list status; strict array validation stays off list requests."""

from __future__ import annotations

import json
from pathlib import Path

from .trajectory_reward_snapshot import snapshot_path


def _json(path: Path) -> dict:
    if path.is_symlink() or not path.is_file():
        raise ValueError(f"Evaluation metadata is unavailable: {path.name}")
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise ValueError(f"Invalid evaluation metadata: {path.name}")
    return value


def _values(path: str | Path) -> None:
    candidate = Path(path)
    if candidate.is_symlink() or not candidate.is_file():
        raise ValueError("Evaluation values are unavailable")


class MemberEvaluationStatus:
    """Resolve each evaluator independently, sharing small manifests within a page."""

    def __init__(self, datasets, dataset: dict):
        self.datasets = datasets
        self.dataset = dataset
        self.current = datasets.current_evaluation_ids(dataset)
        self.versions = {}
        self.legacy_candidates = None

    def _version(self, owner: str, identifier: str, source: str, run_id: str) -> dict:
        key = (owner, identifier, source)
        if key not in self.versions:
            version = self.datasets.get_version(owner, identifier)
            if version.get("status") != "READY" or version.get("evaluator") != source:
                raise ValueError("Dataset evaluation is not ready")
            manifest_key = "robometer_manifest_path" if source == "robometer" else "reward_manifest_path"
            manifest = _json(Path(version[manifest_key]))
            if manifest.get("complete") is not True:
                raise ValueError("Dataset evaluation is incomplete")
            entries = {}
            for entry in manifest.get("episodes", []):
                member_id = entry["run_id"]
                if member_id in entries:
                    raise ValueError("Duplicate evaluation member")
                entries[member_id] = entry
            self.versions[key] = version, entries
        version, entries = self.versions[key]
        entry = entries[run_id]
        _values(entry.get("reward_path") or entry["annotation_path"])
        return {"status": "READY", "version_id": identifier,
                "evaluated_at": version.get("completed_at")}

    def _global(self, source: str, run: dict) -> dict:
        # Dataset-generated Rynn rewards may be stored separately from the
        # native evaluator sidecar. Match training/detail precedence without
        # hashing or opening trajectory/observation/reward arrays here.
        if source == "rynnvalue":
            path = snapshot_path(run, source)
            if path is not None and path.exists():
                payload = _json(path)
                if (payload.get("schema_version") != 1 or payload.get("run_id") != run["id"]
                        or payload.get("source") != source):
                    raise ValueError("Global evaluation identity mismatch")
                name = payload["values_file"]
                if not isinstance(name, str) or Path(name).name != name:
                    raise ValueError("Invalid global evaluation values path")
                _values(path.with_name(name))
                return {"status": "READY", "evaluated_at": payload.get("evaluated_at")}
        service = (self.datasets.evaluations if source == "rynnvalue"
                   else self.datasets.robometer_evaluations)
        status = service.status(run) if service is not None else {"status": "NOT_EVALUATED"}
        if status["status"] == "READY":
            return status
        if Path(run["trajectory"]).with_name(f"{source}_evaluation.json").exists():
            return status

        # Compatibility with results created before global sidecars existed:
        # the first saved evaluation is the global fallback, as in detail views.
        if self.legacy_candidates is None:
            self.legacy_candidates = sorted(
                (version.get("completed_at") or version.get("created_at") or "",
                 dataset["id"], version["id"], version["evaluator"],
                 frozenset(member["run_id"] for member in dataset.get("members", [])))
                for dataset in self.datasets.list(self.dataset.get("task_id"))
                for version in dataset.get("evaluation_versions", [])
                if version.get("status") == "READY"
                and version.get("evaluator") in {"rynnvalue", "robometer"}
            )
        for _, owner, identifier, evaluator, members in self.legacy_candidates:
            if evaluator == source and run["id"] in members:
                try:
                    return self._version(owner, identifier, source, run["id"])
                except (OSError, KeyError, TypeError, ValueError):
                    continue
        return status

    def status(self, source: str, run: dict) -> dict:
        identifier = self.current.get(source)
        origin = "dataset" if identifier else "global"
        try:
            result = (self._version(self.dataset["id"], identifier, source, run["id"])
                      if identifier else self._global(source, run))
            return {**result, "origin": origin}
        except (OSError, KeyError, TypeError, ValueError) as exc:
            # A broken local override must not be disguised by a valid global.
            return {"status": "ERROR", "origin": origin, "error": str(exc)}
