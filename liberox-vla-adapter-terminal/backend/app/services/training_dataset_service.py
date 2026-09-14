"""Immutable, single-task training-dataset selection and integrity management."""

from __future__ import annotations

import hashlib
import json
import math
import random
import shutil
import threading
import uuid
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Iterable

from ..core.exceptions import ConflictError
from ..storage.files import atomic_write_json
from ..storage.repositories import RunLabelRepository, TrainingDatasetRepository


SOURCE_TYPES = frozenset({"inference", "manual", "policy_requery"})
OUTCOMES = frozenset({"success", "failure"})


def _utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _stable_hash(value: Any) -> str:
    payload = json.dumps(value, sort_keys=True, separators=(",", ":")).encode("utf-8")
    return hashlib.sha256(payload).hexdigest()


def _split(root_id: str, seed: int, fraction: float) -> str:
    digest = hashlib.sha256(f"{seed}:{root_id}".encode()).digest()
    value = int.from_bytes(digest[:8], "big") / float(2**64)
    return "validation" if value < fraction else "train"


class TrainingDatasetService:
    """Owns immutable manifests while run files remain read-only source data."""

    def __init__(
        self, run_service: Any, ui_config: Any, evaluations: Any | None = None,
        robometer_evaluations: Any | None = None,
    ):
        self.run_service = run_service
        self.ui_config = ui_config
        self.root = ui_config.project_root / "datasets"
        self.root.mkdir(parents=True, exist_ok=True)
        self.repository = TrainingDatasetRepository(
            ui_config.catalog_path, ui_config.project_id
        )
        self.labels = RunLabelRepository(ui_config.catalog_path, ui_config.project_id)
        self.evaluations = evaluations
        self.robometer_evaluations = robometer_evaluations
        self.lock = threading.RLock()
        self._index_existing()

    @staticmethod
    def classify(run: dict[str, Any]) -> tuple[str, str, bool, str | None]:
        status = str(run.get("status") or "")
        if status != "COMPLETED" or run.get("error"):
            return "incomplete", "failure", False, "运行未正常完成"
        if int(run.get("action_count") or 0) <= 0:
            return "incomplete", "failure", False, "轨迹没有动作"
        if run.get("kind") == "branch":
            source = "manual" if run.get("control_mode") == "manual" else "policy_requery"
        else:
            source = "inference"
        outcome = "success" if bool(run.get("success")) else "failure"
        trajectory = Path(str(run.get("trajectory") or ""))
        observations = trajectory.with_name("trajectory_observations.npz")
        output = Path(str(run.get("output_dir") or trajectory.parent.parent.parent))
        manifest = output / "run.json"
        if not manifest.is_file():
            manifest = output / "session.json"
        missing = [
            label for label, path in (
                ("run manifest", manifest), ("trajectory", trajectory),
                ("observations", observations),
            ) if not path.is_file()
        ]
        if missing:
            return source, outcome, False, "缺少 " + ", ".join(missing)
        return source, outcome, True, None

    def list_runs(
        self,
        task_id: str | None = None,
        source_type: str | None = None,
        outcome: str | None = None,
        eligible: bool | None = None,
    ) -> list[dict[str, Any]]:
        if source_type is not None and source_type not in SOURCE_TYPES:
            raise ValueError(f"Unknown source_type: {source_type}")
        if outcome is not None and outcome not in OUTCOMES:
            raise ValueError(f"Unknown outcome: {outcome}")
        result = []
        test_ids = self.labels.test_ids()
        for run in self.run_service.list_runs():
            if task_id and run.get("task_id") != task_id:
                continue
            current_source, current_outcome, trainable, reason = self.classify(run)
            if source_type and current_source != source_type:
                continue
            if outcome and current_outcome != outcome:
                continue
            if eligible is not None and trainable is not eligible:
                continue
            resume = int(run.get("resume_step") or 0) if run.get("kind") == "branch" else 0
            actions = max(0, int(run.get("action_count") or 0))
            chunk_count = (
                math.ceil(resume / 8) + math.ceil(max(0, actions - resume) / 8)
                if run.get("kind") == "branch" and actions
                else math.ceil(actions / 8) if actions else 0
            )
            public = {
                **run,
                "is_test": run["id"] in test_ids,
                "source_type": current_source,
                "outcome": current_outcome,
                "training_eligible": trainable,
                "ineligible_reason": reason,
                "training_start_step": 0,
                "training_action_count": actions,
                "training_chunk_count": chunk_count,
            }
            result.append(public)
        return sorted(result, key=lambda item: (item.get("created_at") or "", item["id"]), reverse=True)

    def list_runs_page(
        self,
        task_id: str | None = None,
        source_type: str | None = None,
        outcome: str | None = None,
        eligible: bool | None = None,
        *,
        page: int = 1,
        page_size: int = 5,
    ) -> dict[str, Any]:
        if type(page) is not int or page < 1:
            raise ValueError("page must be a positive integer")
        if type(page_size) is not int or not 1 <= page_size <= 50:
            raise ValueError("page_size must be in [1, 50]")
        values = self.list_runs(task_id, source_type, outcome, eligible)
        total = len(values)
        pages = max(1, math.ceil(total / page_size))
        if page > pages and total:
            page = pages
        start = (page - 1) * page_size
        eligible_count = sum(
            bool(item.get("training_eligible")) and not bool(item.get("is_test"))
            for item in values
        )
        test_count = sum(bool(item.get("is_test")) for item in values)
        items = values[start:start + page_size]
        if self.evaluations is not None:
            for item in items:
                item["rynn_evaluation"] = self.evaluations.status(item)
            evaluated_count = sum(
                self.evaluations.status(item).get("status") == "READY" for item in values
            )
        else:
            evaluated_count = 0
        robometer_evaluated_count = 0
        both_evaluated_count = 0
        if self.robometer_evaluations is not None:
            for item in items:
                item["robometer_evaluation"] = self.robometer_evaluations.status(item)
            robometer_evaluated_count = sum(
                self.robometer_evaluations.status(item).get("status") == "READY"
                for item in values
            )
            if self.evaluations is not None:
                both_evaluated_count = sum(
                    self.evaluations.status(item).get("status") == "READY"
                    and self.robometer_evaluations.status(item).get("status") == "READY"
                    for item in values
                )
        return {
            "items": items,
            "total": total,
            "eligible_count": eligible_count,
            "evaluated_count": evaluated_count,
            "rynn_evaluated_count": evaluated_count,
            "robometer_evaluated_count": robometer_evaluated_count,
            "both_evaluated_count": both_evaluated_count,
            "test_count": test_count,
            "page": page,
            "page_size": page_size,
            "pages": pages,
        }

    @staticmethod
    def _validate_selection(selection: dict[str, Any]) -> None:
        mode = selection.get("mode")
        if mode not in {"random", "sequential", "rule", "manual"}:
            raise ValueError("selection.mode must be random, sequential, rule, or manual")
        source_types = selection.get("source_types") or sorted(SOURCE_TYPES)
        outcomes = selection.get("outcomes") or sorted(OUTCOMES)
        if not source_types or any(value not in SOURCE_TYPES for value in source_types):
            raise ValueError("selection.source_types contains an unsupported value")
        if not outcomes or any(value not in OUTCOMES for value in outcomes):
            raise ValueError("selection.outcomes contains an unsupported value")
        if mode in {"random", "sequential"}:
            size = selection.get("size")
            if type(size) is not int or size < 1:
                raise ValueError("selection.size must be a positive integer")
        if mode == "random" and type(selection.get("seed", 0)) is not int:
            raise ValueError("selection.seed must be an integer")
        if mode == "sequential" and selection.get("order", "newest") not in {"oldest", "newest"}:
            raise ValueError("selection.order must be oldest or newest")
        if mode == "manual":
            ids = selection.get("run_ids")
            if not isinstance(ids, list) or not ids or any(not isinstance(value, str) for value in ids):
                raise ValueError("selection.run_ids must be a non-empty list")
            if len(ids) != len(set(ids)):
                raise ValueError("selection.run_ids must not contain duplicates")
        if mode == "rule":
            quotas = selection.get("quotas")
            if not isinstance(quotas, list) or not quotas:
                raise ValueError("selection.quotas must be a non-empty list")
            seen: set[tuple[str, str]] = set()
            for quota in quotas:
                if not isinstance(quota, dict):
                    raise ValueError("Every quota must be an object")
                source, outcome = quota.get("source_type"), quota.get("outcome")
                key = (source, outcome)
                if source not in SOURCE_TYPES or outcome not in OUTCOMES or key in seen:
                    raise ValueError("Rule quotas must use unique supported source/outcome pairs")
                seen.add(key)
                if type(quota.get("count")) is not int or quota["count"] < 0:
                    raise ValueError("quota.count must be a non-negative integer")
                if quota.get("order", "random") not in {"random", "oldest", "newest"}:
                    raise ValueError("quota.order must be random, oldest, or newest")

    @staticmethod
    def _ordered(items: Iterable[dict[str, Any]], order: str) -> list[dict[str, Any]]:
        return sorted(
            items,
            key=lambda item: (item.get("created_at") or "", item["id"]),
            reverse=order == "newest",
        )

    def preview(self, task_id: str, selection: dict[str, Any]) -> dict[str, Any]:
        if not isinstance(task_id, str) or not task_id.strip():
            raise ValueError("task_id is required")
        self._validate_selection(selection)
        # Test-labelled trajectories are deliberately held out from every new
        # frozen training dataset, including explicit/manual selections.
        all_runs = [
            run for run in self.list_runs(task_id=task_id, eligible=True)
            if not run["is_test"]
        ]
        source_types = set(selection.get("source_types") or SOURCE_TYPES)
        outcomes = set(selection.get("outcomes") or OUTCOMES)
        candidates = [
            run for run in all_runs
            if run["source_type"] in source_types and run["outcome"] in outcomes
        ]
        mode = selection["mode"]
        if mode == "manual":
            by_id = {run["id"]: run for run in all_runs}
            missing = [run_id for run_id in selection["run_ids"] if run_id not in by_id]
            if missing:
                raise ValueError(f"Manual selection contains unavailable runs: {missing}")
            selected = [by_id[run_id] for run_id in selection["run_ids"]]
        elif mode == "random":
            size = int(selection["size"])
            if size > len(candidates):
                raise ValueError(f"Requested {size} runs but only {len(candidates)} are eligible")
            stable = sorted(candidates, key=lambda item: item["id"])
            selected = random.Random(int(selection.get("seed", 0))).sample(stable, size)
        elif mode == "sequential":
            size = int(selection["size"])
            if size > len(candidates):
                raise ValueError(f"Requested {size} runs but only {len(candidates)} are eligible")
            selected = self._ordered(candidates, selection.get("order", "newest"))[:size]
        else:
            selected = []
            seed = int(selection.get("seed", 0))
            for index, quota in enumerate(selection["quotas"]):
                group = [
                    run for run in all_runs
                    if run["source_type"] == quota["source_type"]
                    and run["outcome"] == quota["outcome"]
                ]
                count = int(quota["count"])
                if count > len(group):
                    raise ValueError(
                        f"Quota {quota['source_type']}/{quota['outcome']} requests "
                        f"{count}, only {len(group)} available"
                    )
                order = quota.get("order", "random")
                if order == "random":
                    group = random.Random(seed + index).sample(
                        sorted(group, key=lambda item: item["id"]), count
                    )
                else:
                    group = self._ordered(group, order)[:count]
                selected.extend(group)
            if not selected:
                raise ValueError("Rule quotas selected zero runs")
        if len({run["id"] for run in selected}) != len(selected):
            raise ValueError("Selection resolved to duplicate runs")
        categories: dict[str, int] = {}
        for run in selected:
            key = f"{run['source_type']}:{run['outcome']}"
            categories[key] = categories.get(key, 0) + 1
        return {
            "task_id": task_id,
            "eligible_count": len(all_runs),
            "selected_count": len(selected),
            "action_count": sum(run["training_action_count"] for run in selected),
            "chunk_count": sum(run["training_chunk_count"] for run in selected),
            "categories": categories,
            "run_ids": [run["id"] for run in selected],
            "runs": selected,
        }

    def is_test(self, run_id: str) -> bool:
        # Resolve first so labels cannot be attached to nonexistent records.
        self.run_service.get_run(run_id)
        return self.labels.is_test(run_id)

    def set_test(self, run_id: str, is_test: bool) -> dict[str, Any]:
        if type(is_test) is not bool:
            raise TypeError("is_test must be a boolean")
        run = self.run_service.get_run(run_id)
        self.labels.set_test(run_id, is_test)
        return {
            "run_id": run_id,
            "task_id": run.get("task_id"),
            "is_test": is_test,
            "excluded_from_training_packages": is_test,
            "excluded_from_default_batch_evaluation": is_test,
        }

    @staticmethod
    def _artifact(path: Path) -> dict[str, Any]:
        if path.is_symlink():
            raise ValueError(f"Symlink dataset artifacts are not allowed: {path}")
        resolved = path.resolve(strict=True)
        if not resolved.is_file():
            raise ValueError(f"Unsafe or missing dataset artifact: {path}")
        return {
            "path": str(resolved),
            "size": resolved.stat().st_size,
            "sha256": _sha256(resolved),
        }

    def _member(
        self,
        run: dict[str, Any],
        *,
        split_seed: int,
        validation_fraction: float,
    ) -> dict[str, Any]:
        output = Path(run["output_dir"])
        manifest = output / "run.json"
        if not manifest.is_file():
            manifest = output / "session.json"
        trajectory = Path(run["trajectory"])
        observations = trajectory.with_name("trajectory_observations.npz")
        root_id = str(run.get("root_session_id") or run.get("parent_session_id") or run["id"])
        return {
            "run_id": run["id"],
            "root_run_id": root_id,
            "parent_run_id": run.get("parent_session_id"),
            "source_type": run["source_type"],
            "outcome": run["outcome"],
            "resume_step": int(run.get("resume_step") or 0),
            "end_step": int(run["action_count"]),
            "action_count": run["training_action_count"],
            "chunk_count": run["training_chunk_count"],
            "split": _split(root_id, split_seed, validation_fraction),
            "artifacts": {
                "manifest": self._artifact(manifest),
                "trajectory": self._artifact(trajectory),
                "observations": self._artifact(observations),
            },
        }

    @staticmethod
    def _ensure_train_split(members: list[dict[str, Any]]) -> None:
        if members and all(member["split"] == "validation" for member in members):
            first_root = members[0]["root_run_id"]
            for member in members:
                if member["root_run_id"] == first_root:
                    member["split"] = "train"

    def write_evaluation_selection(
        self,
        path: Path,
        *,
        task_id: str,
        run_ids: list[str],
        split_seed: int,
        validation_fraction: float,
        success_consecutive_steps: int,
    ) -> dict[str, Any]:
        """Write a prepare_dataset-compatible transient selection manifest."""
        if not run_ids or len(run_ids) != len(set(run_ids)):
            raise ValueError("run_ids must be a non-empty unique list")
        available = {
            item["id"]: item for item in self.list_runs(task_id=task_id, eligible=True)
        }
        missing = [run_id for run_id in run_ids if run_id not in available]
        if missing:
            raise ValueError(f"Unavailable evaluation runs: {missing}")
        members = [
            self._member(
                available[run_id], split_seed=split_seed,
                validation_fraction=validation_fraction,
            )
            for run_id in run_ids
        ]
        self._ensure_train_split(members)
        selection = {
            "mode": "manual", "run_ids": run_ids,
            "source_types": sorted(SOURCE_TYPES), "outcomes": sorted(OUTCOMES),
            "seed": split_seed, "order": "newest", "quotas": [],
        }
        immutable = {
            "task_id": task_id,
            "selection": selection,
            "validation_fraction": validation_fraction,
            "split_seed": split_seed,
            "success_consecutive_steps": success_consecutive_steps,
            "members": members,
        }
        payload = {
            "schema_version": 1,
            "id": f"trajectory-evaluation-{uuid.uuid4().hex[:12]}",
            "project_id": self.ui_config.project_id,
            **immutable,
            "dataset_sha256": _stable_hash(immutable),
        }
        atomic_write_json(path, payload)
        return payload

    def create(
        self,
        *,
        name: str,
        task_id: str,
        selection: dict[str, Any],
        validation_fraction: float = 0.2,
        split_seed: int = 7,
        success_consecutive_steps: int = 5,
        parent_dataset_id: str | None = None,
    ) -> dict[str, Any]:
        if not isinstance(name, str) or not name.strip() or len(name.strip()) > 100:
            raise ValueError("name must contain 1..100 characters")
        if not 0 <= validation_fraction <= 0.9:
            raise ValueError("validation_fraction must be in [0, 0.9]")
        if type(split_seed) is not int:
            raise ValueError("split_seed must be an integer")
        if type(success_consecutive_steps) is not int or not 1 <= success_consecutive_steps <= 100:
            raise ValueError("success_consecutive_steps must be in [1, 100]")
        if parent_dataset_id is not None:
            self.get(parent_dataset_id)
        resolved = self.preview(task_id, selection)
        dataset_id = f"ds_{datetime.now().strftime('%Y%m%d_%H%M%S')}_{uuid.uuid4().hex[:8]}"
        directory = self.root / dataset_id
        with self.lock:
            directory.mkdir(parents=False, exist_ok=False)
            try:
                members = [
                    self._member(
                        run, split_seed=split_seed,
                        validation_fraction=validation_fraction,
                    )
                    for run in resolved["runs"]
                ]
                # Freeze the tiny-dataset fallback here. Downstream prepare
                # consumes the exact split stored in this immutable manifest.
                self._ensure_train_split(members)
                immutable = {
                    "task_id": task_id,
                    "selection": selection,
                    "validation_fraction": validation_fraction,
                    "split_seed": split_seed,
                    "success_consecutive_steps": success_consecutive_steps,
                    "members": members,
                }
                now = _utc_now()
                payload = {
                    "schema_version": 1,
                    "id": dataset_id,
                    "project_id": self.ui_config.project_id,
                    "name": name.strip(),
                    "task_id": task_id,
                    "status": "FROZEN",
                    "integrity_status": "HEALTHY",
                    "integrity_error": None,
                    "annotation_status": "NOT_STARTED",
                    "annotation_id": None,
                    "annotation_config": None,
                    "pending_annotation_id": None,
                    "pending_annotation_config": None,
                    "annotation_history": [],
                    "parent_dataset_id": parent_dataset_id,
                    "created_at": now,
                    "updated_at": now,
                    "selection": selection,
                    "validation_fraction": validation_fraction,
                    "split_seed": split_seed,
                    "success_consecutive_steps": success_consecutive_steps,
                    "member_count": len(members),
                    "action_count": sum(member["action_count"] for member in members),
                    "chunk_count": sum(member["chunk_count"] for member in members),
                    "categories": resolved["categories"],
                    "members": members,
                    "dataset_sha256": _stable_hash(immutable),
                }
                manifest_path = directory / "dataset.json"
                atomic_write_json(manifest_path, payload)
                self.repository.upsert(payload, manifest_path)
                return self._public(payload)
            except BaseException:
                shutil.rmtree(directory, ignore_errors=True)
                raise

    def derive(self, dataset_id: str, **kwargs: Any) -> dict[str, Any]:
        parent = self.get(dataset_id)
        task_id = kwargs.pop("task_id", parent["task_id"])
        if task_id != parent["task_id"]:
            raise ValueError("A derived dataset must keep the parent task")
        return self.create(task_id=task_id, parent_dataset_id=dataset_id, **kwargs)

    def delete_unannotated(
        self, dataset_id: str, confirm_dataset_id: str
    ) -> dict[str, Any]:
        """Remove an unused frozen manifest without touching referenced runs."""
        return self._delete_dataset(
            dataset_id, confirm_dataset_id, require_unannotated=True
        )

    def delete_dataset(
        self, dataset_id: str, confirm_dataset_id: str
    ) -> dict[str, Any]:
        """Remove a non-active dataset version while preserving shared sources."""
        return self._delete_dataset(
            dataset_id, confirm_dataset_id, require_unannotated=False
        )

    def _delete_dataset(
        self, dataset_id: str, confirm_dataset_id: str, *, require_unannotated: bool
    ) -> dict[str, Any]:
        if confirm_dataset_id != dataset_id:
            raise ValueError("confirm_dataset_id must exactly match dataset_id")
        with self.lock:
            manifest_path, payload = self._load(dataset_id)
            annotation_status = payload.get("annotation_status")
            if annotation_status == "RUNNING":
                raise ConflictError(
                    "Dataset annotation is active and cannot be deleted",
                    code="DATASET_ANNOTATION_ACTIVE",
                    context={"dataset_id": dataset_id},
                )
            if require_unannotated and annotation_status != "NOT_STARTED":
                raise ConflictError(
                    "Only an unannotated frozen dataset can be canceled",
                    code="DATASET_ALREADY_USED",
                    context={
                        "dataset_id": dataset_id,
                        "annotation_status": payload.get("annotation_status"),
                    },
                )
            children = [
                item["id"] for item in self.list()
                if item.get("parent_dataset_id") == dataset_id
            ]
            if children:
                raise ConflictError(
                    "Dataset has derived versions and cannot be canceled",
                    code="DATASET_HAS_DERIVATIONS",
                    context={"dataset_id": dataset_id, "children": children},
                )
            directory = manifest_path.parent
            root = self.root.resolve()
            if directory.is_symlink() or directory.resolve().parent != root:
                raise ValueError("Dataset directory is outside the managed root")
            shutil.rmtree(directory)
            self.repository.delete(dataset_id)
        return {
            "deleted": dataset_id,
            "annotation_status": annotation_status,
            "source_runs_deleted": False,
            "shared_cache_deleted": False,
        }

    def _manifest_path(self, dataset_id: str) -> Path:
        if not dataset_id or Path(dataset_id).name != dataset_id or ".." in dataset_id:
            raise ValueError("Invalid dataset id")
        path = self.root / dataset_id / "dataset.json"
        if not path.is_file() or path.is_symlink():
            raise KeyError(dataset_id)
        return path

    def _load(self, dataset_id: str) -> tuple[Path, dict[str, Any]]:
        path = self._manifest_path(dataset_id)
        payload = json.loads(path.read_text(encoding="utf-8"))
        if payload.get("schema_version") != 1 or payload.get("id") != dataset_id:
            raise ValueError(f"Invalid training dataset manifest: {path}")
        return path, payload

    @staticmethod
    def _immutable_payload(payload: dict[str, Any]) -> dict[str, Any]:
        return {
            "task_id": payload.get("task_id"),
            "selection": payload.get("selection"),
            "validation_fraction": payload.get("validation_fraction"),
            "split_seed": payload.get("split_seed"),
            "success_consecutive_steps": payload.get("success_consecutive_steps"),
            "members": payload.get("members"),
        }

    @classmethod
    def _immutable_error(cls, payload: dict[str, Any]) -> str | None:
        actual = _stable_hash(cls._immutable_payload(payload))
        expected = payload.get("dataset_sha256")
        if actual != expected:
            return f"Immutable dataset manifest hash changed: expected {expected}, got {actual}"
        return None

    @staticmethod
    def _public(payload: dict[str, Any]) -> dict[str, Any]:
        result = dict(payload)
        result["evaluation_versions"] = list(payload.get("evaluation_versions", []))
        known = {item["id"] for item in result["evaluation_versions"]}
        for previous in payload.get("annotation_history", []):
            identifier = previous.get("annotation_id")
            if identifier and identifier not in known:
                result["evaluation_versions"].append({"id": identifier, "evaluator": "rynnvalue",
                    "status": previous["status"], "parameters": previous.get("config") or {},
                    "created_at": previous.get("completed_at"), "completed_at": previous.get("completed_at"),
                    "error": None, "legacy": True})
                known.add(identifier)
        result.setdefault("reward_version_id", None)
        result.setdefault("robometer_version_id", None)
        result["members"] = [
            {
                key: member.get(key) for key in (
                    "run_id", "root_run_id", "parent_run_id", "source_type",
                    "outcome", "resume_step", "end_step", "action_count",
                    "chunk_count", "split",
                )
            }
            for member in payload.get("members", [])
        ]
        return result

    def members_page(self, dataset_id: str, *, page: int = 1, page_size: int = 5) -> dict[str, Any]:
        if type(page) is not int or page < 1 or type(page_size) is not int or not 1 <= page_size <= 50:
            raise ValueError("page must be positive and page_size must be in [1, 50]")
        dataset = self.get(dataset_id)
        members = dataset["members"]
        pages = max(1, math.ceil(len(members) / page_size))
        page = min(page, pages)
        items = []
        for member in members[(page - 1) * page_size:page * page_size]:
            run = self.run_service.get_run(member["run_id"])
            items.append({**run, **member, "id": member["run_id"]})
        return {"items": items, "total": len(members), "page": page,
                "page_size": page_size, "pages": pages}

    def versions(self, dataset_id: str) -> list[dict[str, Any]]:
        return self.get(dataset_id, quick_verify=False).get("evaluation_versions", [])

    def get_version(self, dataset_id: str, version_id: str) -> dict[str, Any]:
        dataset = self.get(dataset_id, quick_verify=False)
        version = next((item for item in dataset.get("evaluation_versions", [])
                        if item["id"] == version_id), None)
        if version is None:
            raise KeyError(f"Unknown dataset evaluation version: {version_id}")
        directory = self.root / dataset_id / "annotations" / version_id
        path = directory / "version.json"
        if directory.is_symlink() or path.is_symlink():
            raise ValueError("Symlink evaluation versions are not allowed")
        if not path.is_file():
            return {**version, "dataset_id": dataset_id, "work_dir": str(directory / "work"),
                    "complete": False, "legacy": True,
                    "prepared_manifest_path": str(directory / "work" / "dataset_manifest.json"),
                    "reward_manifest_path": str(directory / "work" / "rewards" / "reward_manifest.json")}
        value = json.loads(path.read_text(encoding="utf-8"))
        if value.get("id") != version_id or value.get("dataset_id") != dataset_id:
            raise ValueError("Evaluation version identity mismatch")
        if value.get("dataset_sha256") != dataset["dataset_sha256"]:
            raise ValueError("Evaluation version dataset hash mismatch")
        return {**value, "status": version["status"]}

    def update_version(self, dataset_id: str, version: dict[str, Any]) -> dict[str, Any]:
        """Publish a lightweight summary; successful artifacts are immutable."""
        with self.lock:
            path, payload = self._load(dataset_id)
            versions = payload.setdefault("evaluation_versions", [])
            existing = next((item for item in versions if item["id"] == version["id"]), None)
            if existing is not None and existing.get("status") == "READY":
                return self._public(payload)
            summary = {key: version.get(key) for key in (
                "id", "evaluator", "status", "parameters", "created_at", "completed_at", "error",
            )}
            if existing == summary:
                return self._public(payload)
            if existing is None:
                versions.append(summary)
            else:
                existing.update(summary)
            if version["status"] == "READY":
                key = "robometer_version_id" if version["evaluator"] == "robometer" else "reward_version_id"
                payload[key] = version["id"]
                if key == "reward_version_id":
                    payload.update(annotation_id=version["id"], annotation_status="READY",
                                   annotation_config=version.get("parameters"))
            payload["updated_at"] = _utc_now()
            atomic_write_json(path, payload)
            self.repository.upsert(payload, path)
            return self._public(payload)

    def activate_version(self, dataset_id: str, version_id: str) -> dict[str, Any]:
        with self.lock:
            version = self.validate_version(dataset_id, version_id)
            if version.get("status") != "READY" or not version.get("complete"):
                raise ConflictError("Evaluation version is not ready", code="REWARD_VERSION_NOT_READY")
            path, payload = self._load(dataset_id)
            key = "robometer_version_id" if version["evaluator"] == "robometer" else "reward_version_id"
            payload[key] = version_id
            if key == "reward_version_id":
                payload.update(annotation_id=version_id, annotation_status="READY",
                               annotation_config=version.get("parameters"))
            payload["updated_at"] = _utc_now()
            atomic_write_json(path, payload)
            self.repository.upsert(payload, path)
            return self._public(payload)

    def validate_version(self, dataset_id: str, version_id: str) -> dict[str, Any]:
        """Full artifact verification only on explicit activation/training, never lists."""
        version = self.get_version(dataset_id, version_id)
        if not version.get("complete") or version.get("status") != "READY":
            raise ConflictError("Evaluation version is not ready", code="REWARD_VERSION_NOT_READY")
        pairs = [("config_path", "config_sha256")]
        if version["evaluator"] == "robometer":
            pairs.append(("robometer_manifest_path", "robometer_manifest_sha256"))
            value_path = version["robometer_manifest_path"]
        else:
            pairs.extend([("prepared_manifest_path", "prepared_manifest_sha256"),
                          ("reward_manifest_path", "reward_manifest_sha256")])
            value_path = version["reward_manifest_path"]
        for path_key, hash_key in pairs:
            current = Path(version[path_key])
            if current.is_symlink() or not current.is_file() or _sha256(current) != version[hash_key]:
                raise ValueError(f"Evaluation version integrity failed: {path_key}")
        manifest = json.loads(Path(value_path).read_text(encoding="utf-8"))
        for entry in manifest.get("episodes", []):
            current = Path(entry.get("reward_path") or entry["annotation_path"])
            expected = entry.get("reward_sha256") or entry.get("values_sha256") or entry["annotation_sha256"]
            if current.is_symlink() or not current.is_file() or _sha256(current) != expected:
                raise ValueError(f"Evaluation version integrity failed: {entry['run_id']}")
        return version

    def get(self, dataset_id: str, *, quick_verify: bool = True) -> dict[str, Any]:
        path, payload = self._load(dataset_id)
        if quick_verify and payload.get("integrity_status") != "BROKEN":
            immutable_error = self._immutable_error(payload)
            missing = []
            for member in payload.get("members", []):
                for artifact in member.get("artifacts", {}).values():
                    if not Path(artifact["path"]).is_file():
                        missing.append(artifact["path"])
            if immutable_error or missing:
                payload["integrity_status"] = "BROKEN"
                payload["integrity_error"] = immutable_error or f"Missing artifacts: {missing[:3]}"
                payload["updated_at"] = _utc_now()
                atomic_write_json(path, payload)
                self.repository.upsert(payload, path)
        return self._public(payload)

    def list(self, task_id: str | None = None) -> list[dict[str, Any]]:
        values = []
        for path in sorted(self.root.glob("*/dataset.json")):
            try:
                value = self.get(path.parent.name)
            except Exception:
                continue
            if task_id is None or value["task_id"] == task_id:
                values.append(value)
        return sorted(values, key=lambda item: item["created_at"], reverse=True)

    def verify(self, dataset_id: str) -> dict[str, Any]:
        path, payload = self._load(dataset_id)
        error = self._immutable_error(payload)
        try:
            if error is not None:
                raise ValueError(error)
            for member in payload["members"]:
                for name, artifact in member["artifacts"].items():
                    current = Path(artifact["path"])
                    if not current.is_file():
                        raise FileNotFoundError(f"{member['run_id']}:{name}: {current}")
                    if current.stat().st_size != artifact["size"]:
                        raise ValueError(f"Size changed: {member['run_id']}:{name}")
                    if _sha256(current) != artifact["sha256"]:
                        raise ValueError(f"SHA256 changed: {member['run_id']}:{name}")
        except Exception as exc:
            error = f"{type(exc).__name__}: {exc}"
        payload["integrity_status"] = "HEALTHY" if error is None else "BROKEN"
        payload["integrity_error"] = error
        payload["last_verified_at"] = _utc_now()
        payload["updated_at"] = _utc_now()
        atomic_write_json(path, payload)
        self.repository.upsert(payload, path)
        return self._public(payload)

    def update_annotation(
        self, dataset_id: str, status: str, *, annotation_id: str | None = None,
        annotation_config: dict[str, Any] | None = None,
    ) -> dict[str, Any]:
        path, payload = self._load(dataset_id)
        history = payload.setdefault("annotation_history", [])
        active_id = payload.get("annotation_id")
        pending_id = annotation_id or payload.get("pending_annotation_id")
        if status == "RUNNING":
            payload["annotation_status"] = "RUNNING"
            payload["pending_annotation_id"] = annotation_id
            payload["pending_annotation_config"] = annotation_config
        elif status == "READY":
            payload["annotation_status"] = "READY"
            payload["annotation_id"] = pending_id
            payload["annotation_config"] = (
                annotation_config or payload.get("pending_annotation_config")
            )
            payload["pending_annotation_id"] = None
            payload["pending_annotation_config"] = None
        else:
            # Preserve the last successful reward version if a later
            # re-annotation is canceled or fails.
            payload["annotation_status"] = "READY" if active_id else status
            payload["pending_annotation_id"] = None
            payload["pending_annotation_config"] = None
        if pending_id and status != "RUNNING" and not any(
            item.get("annotation_id") == pending_id and item.get("status") == status
            for item in history
        ):
            history.append({
                "annotation_id": pending_id,
                "status": status,
                "config": annotation_config,
                "completed_at": _utc_now(),
            })
        if status != "RUNNING":
            payload["last_annotation_id"] = pending_id
            payload["last_annotation_status"] = status
        payload["updated_at"] = _utc_now()
        atomic_write_json(path, payload)
        self.repository.upsert(payload, path)
        return self._public(payload)

    def references_for_run(self, run_id: str) -> list[dict[str, Any]]:
        return self.repository.references_for_run(run_id)

    def mark_broken_for_run(self, run_id: str) -> list[str]:
        changed = []
        for reference in self.references_for_run(run_id):
            try:
                path, payload = self._load(reference["id"])
                payload["integrity_status"] = "BROKEN"
                payload["integrity_error"] = f"Referenced run was deleted: {run_id}"
                payload["updated_at"] = _utc_now()
                atomic_write_json(path, payload)
                self.repository.upsert(payload, path)
                changed.append(reference["id"])
            except Exception:
                # The source deletion has already happened. Keep the API
                # operation recoverable; quick/full verification will still
                # identify the missing artifact on the next dataset read.
                continue
        return changed

    def require_ready_for_annotation(self, dataset_id: str) -> dict[str, Any]:
        value = self.verify(dataset_id)
        if value["integrity_status"] != "HEALTHY":
            raise ConflictError(
                "Dataset integrity verification failed",
                code="DATASET_BROKEN",
                context={"dataset_id": dataset_id, "error": value.get("integrity_error")},
            )
        if value["annotation_status"] == "RUNNING":
            raise ConflictError("Dataset annotation is already running", code="ANNOTATION_RUNNING")
        return value

    def require_ready_for_training(self, dataset_id: str) -> dict[str, Any]:
        value = self.verify(dataset_id)
        if value["integrity_status"] != "HEALTHY":
            raise ConflictError("Dataset is broken", code="DATASET_BROKEN")
        if value["annotation_status"] != "READY" or not value.get("annotation_id"):
            raise ConflictError("Dataset annotation is not ready", code="ANNOTATION_NOT_READY")
        return value

    def _index_existing(self) -> None:
        for path in self.root.glob("*/dataset.json"):
            try:
                payload = json.loads(path.read_text(encoding="utf-8"))
                self.repository.upsert(payload, path)
            except Exception:
                continue
