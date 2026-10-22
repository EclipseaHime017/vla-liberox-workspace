"""Queue isolated critic probes against immutable dataset rewards."""
from __future__ import annotations

import copy
import hashlib
import json
import re
import shutil
import uuid
from datetime import datetime
from pathlib import Path
from typing import Any

import yaml

from ..storage.files import atomic_write_json, atomic_write_yaml
from ..storage.paths import storage_path
from .offline_job_service import _sha256, _stable_hash


ARTIFACTS = frozenset({
    "qv.png", "training.png", "qv.csv", "qv.npz", "summary.json", "critic.pt",
    "effective_config.yaml",
})
REWARD_SOURCES = ("final", "rynnvalue", "stage", "sparse")


def _read_json(path: Path, *, limit: int = 2_000_000) -> dict[str, Any]:
    if path.is_symlink() or not path.is_file() or path.stat().st_size > limit:
        raise ValueError(f"Missing, unsafe, or oversized diagnostic metadata: {path.name}")
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise ValueError(f"Diagnostic metadata must be an object: {path.name}")
    return value


def _sealed_bytes(path: str | Path, expected: str) -> bytes:
    path = Path(path)
    if path.is_symlink() or not path.is_file():
        raise ValueError(f"Missing or unsafe sealed diagnostic input: {path.name}")
    value = path.read_bytes()
    if hashlib.sha256(value).hexdigest() != expected:
        raise ValueError(f"Saved diagnostic input changed while copying: {path.name}")
    return value


class CriticDiagnosticService:
    def __init__(self, jobs):
        self.jobs = jobs
        self.datasets = jobs.datasets

    def defaults(self, dataset_id: str | None = None) -> dict[str, Any]:
        sources = []
        default = None
        if dataset_id:
            dataset = self.datasets.get(dataset_id, quick_verify=False)
            for source in REWARD_SOURCES:
                identifier = dataset.get("evaluation_version_ids", {}).get(source)
                if not identifier:
                    continue
                version = self.datasets.get_version(dataset_id, identifier)
                if version.get("complete") and version.get("status") == "READY" and not version.get("legacy"):
                    sources.append(source)
                    if identifier == dataset.get("reward_version_id"):
                        default = source
        return {"train_steps": 3000, "batch_size": 32, "reward_sources": sources,
                "default_reward_source": default or next(iter(sources), None)}

    @staticmethod
    def _copy_rewards(version: dict, destination: Path) -> Path:
        manifest = json.loads(_sealed_bytes(version["reward_manifest_path"], version["reward_manifest_sha256"]))
        destination.mkdir(parents=True)
        copied = {}
        for index, entry in enumerate(manifest["episodes"]):
            for path_key, hash_key in (("reward_path", "reward_sha256"),
                                       ("annotation_path", "annotation_sha256")):
                if not entry.get(path_key):
                    continue
                original = storage_path(entry[path_key])
                expected = entry.get(hash_key)
                if not expected or original.is_symlink() or not original.is_file():
                    raise ValueError(f"Missing or unsafe saved reward: {entry.get('run_id')}")
                identity = (str(original.resolve()), expected)
                target = copied.get(identity)
                if target is None:
                    target = destination / f"{index:06d}_{path_key}.npz"
                    shutil.copyfile(original, target)
                    if _sha256(target) != expected:
                        raise ValueError(f"Saved reward changed while copying: {entry.get('run_id')}")
                    copied[identity] = target
                entry[path_key] = str(target.resolve())
        if manifest.get("stage_annotations_path"):
            original = Path(manifest["stage_annotations_path"])
            annotations = _read_json(original, limit=50_000_000)
            if _stable_hash(annotations) != manifest.get("stage_annotations_sha256"):
                raise ValueError("Saved Stage annotations failed integrity validation")
            target = destination / "stage_annotations.json"
            atomic_write_json(target, annotations)
            manifest["stage_annotations_path"] = str(target.resolve())
        path = destination / "reward_manifest.json"
        atomic_write_json(path, manifest)
        return path

    def start(self, parameters: dict[str, Any]) -> dict[str, Any]:
        from ..domain.critic_diagnostic import CriticDiagnosticRequest
        from ..policies.snapshots import validate_snapshot
        from .task_registry import freeze_task

        parameters = CriticDiagnosticRequest.model_validate(parameters).model_dump()
        with self.jobs.lock:
            dataset = self.datasets.require_ready_for_training(parameters["dataset_id"])
            if dataset["task_id"] != parameters["task_id"]:
                raise ValueError("The rollout task must match the selected dataset task")
            source = parameters["reward_source"]
            version_id = dataset.get("evaluation_version_ids", {}).get(source)
            if not version_id:
                raise ValueError("The selected dataset has no saved reward version for this source")
            version = self.datasets.validate_version(dataset["id"], version_id)
            if version.get("evaluator") != source or version.get("legacy"):
                raise ValueError("The selected saved reward version is unsupported")
            preview = self.jobs.preview_evaluation({
                "task_id": parameters["task_id"], "policy_id": parameters["policy_id"],
                "trials": 1, "init_state_indices": [parameters["init_state_index"]],
                "base_seed": parameters["seed"], "seed_count": 1, "schedule_seed": 0,
                "max_steps": parameters["max_steps"], "open_loop_steps": parameters["open_loop_steps"],
                "realtime": False,
            })
            policy = copy.deepcopy(validate_snapshot(preview["policy_snapshot"]))
            if not re.fullmatch(r"[0-9a-f]{64}", str(policy.get("content_sha256", ""))):
                raise ValueError("The selected model must have a verified immutable identity")
            job_id = f"critic_{datetime.now().strftime('%Y%m%d_%H%M%S')}_{uuid.uuid4().hex[:8]}"
            directory = self.jobs.jobs_root / job_id
            directory.mkdir(parents=True, exist_ok=False)
            published = False
            try:
                frozen = freeze_task(self.jobs.manager.catalog, parameters["task_id"], directory / "task_snapshot")
                bddl, init = frozen.paths(parameters["task_id"])
                task = {**preview["task_snapshot"], "bddl_path": str(bddl), "init_path": str(init),
                        "task_revision_sha256": frozen.manifest["task_revision_sha256"]}
                train = directory / "train"
                work = train / "work"
                work.mkdir(parents=True)
                prepared = json.loads(_sealed_bytes(version["prepared_manifest_path"], version["prepared_manifest_sha256"]))
                if prepared.get("source_dataset_sha256") != dataset["dataset_sha256"]:
                    raise ValueError("Saved rewards belong to a different frozen dataset")
                shutil.copyfile(version["prepared_manifest_path"], work / "dataset_manifest.json")
                if _sha256(work / "dataset_manifest.json") != version["prepared_manifest_sha256"]:
                    raise ValueError("Prepared dataset changed while copying")
                reward_manifest = self._copy_rewards(version, train / "rewards")
                sealed = yaml.safe_load(_sealed_bytes(version["config_path"], version["config_sha256"]))
                raw = self.jobs._load_base_config("iql", policy["family"])
                for key in ("data", "reward"):
                    raw[key] = copy.deepcopy(sealed[key])
                raw["model"] = copy.deepcopy(policy["model_config"])
                raw["model"]["contract"] = copy.deepcopy(policy["model_contract"])
                io = policy["model_contract"]["io"]
                raw["data"].update(action_horizon=io["replay_horizon"],
                                   action_dim=io["action_dim"], proprio_dim=io["proprio_dim"])
                if raw["data"].get("selection_manifest"):
                    selection, _ = self.datasets._load(dataset["id"])
                    shutil.copyfile(selection, train / "dataset.json")
                    raw["data"]["selection_manifest"] = str((train / "dataset.json").resolve())
                raw["reward"].update(manifest_path=str(reward_manifest.resolve()),
                    manifest_sha256=_sha256(reward_manifest), version_id=version_id)
                raw["paths"].update(work_dir=str(work.resolve()), output_dir=str((train / "output").resolve()),
                                    annotation_cache=str((train / "cache").resolve()))
                training_config = train / "config.yaml"
                atomic_write_yaml(training_config, raw)
                effective = {"schema_version": 1, "job_id": job_id, "dataset_id": dataset["id"],
                    "dataset_name": dataset["name"], "training_config": str(training_config.resolve()),
                    "policy_snapshot": policy, "task_snapshot": task,
                    "rollout": {
                        "max_steps": parameters["max_steps"], "open_loop_steps": preview["config"]["open_loop_steps"],
                        "seed": parameters["seed"], "init_state_index": parameters["init_state_index"],
                        "control_hz": preview["config"]["control_hz"],
                        "success_consecutive_steps": sealed["data"]["success_consecutive_steps"],
                    },
                    "critic": {"train_steps": parameters["train_steps"], "batch_size": parameters["batch_size"],
                               "seed": parameters["seed"]},
                    "artifacts_dir": str((directory / "artifacts").resolve())}
                config_path = directory / "effective_config.yaml"
                atomic_write_yaml(config_path, effective)
                script = Path(__file__).resolve().parents[3] / "scripts" / "run_critic_diagnostic.py"
                stages = [{"id": phase, "label": label, "environment": self.jobs.ui_config.train_environment,
                    "argv": ["python", str(script), "--config", str(config_path), "--phase", phase]}
                    for phase, label in (("rollout", "生成独立策略轨迹"), ("fit", "拟合 IQL Critic 并生成图表"))]
                job = self.jobs._new_job(kind="critic_diagnostic", dataset_id=dataset["id"], stages=stages,
                    config_path=config_path, output_path=directory / "artifacts", queued=True,
                    parameters={**parameters, "reward_version_id": version_id,
                                "policy_content_sha256": policy["content_sha256"], "requires_gpu": True})
                published = True
                return job
            finally:
                if not published and directory.exists():
                    shutil.rmtree(directory)

    def _owned_job(self, job_id: str) -> tuple[Path, dict]:
        path, job = self.jobs._load_job(job_id)
        directory = self.jobs.jobs_root / job_id
        if (job.get("kind") != "critic_diagnostic" or directory.is_symlink()
                or directory.resolve().parent != self.jobs.jobs_root.resolve()):
            raise ValueError("This is not an owned critic diagnostic job")
        return path, job

    def list(self) -> dict[str, list]:
        return {"jobs": [job for job in self.jobs.list() if job.get("kind") == "critic_diagnostic"]}

    def artifact(self, job_id: str, filename: str) -> Path:
        self._owned_job(job_id)
        if filename not in ARTIFACTS:
            raise ValueError("Unknown critic diagnostic artifact")
        directory = self.jobs.jobs_root / job_id
        path = directory / filename if filename == "effective_config.yaml" else directory / "artifacts" / filename
        if (path.is_symlink() or path.parent.is_symlink() or not path.is_file()
                or not path.resolve().is_relative_to(directory.resolve())):
            raise FileNotFoundError(f"Diagnostic artifact is missing or unsafe: {filename}")
        return path

    def get(self, job_id: str) -> dict[str, Any]:
        self._owned_job(job_id)
        result = self.jobs.get(job_id)
        result["artifacts"] = {}
        for filename in sorted(ARTIFACTS):
            try:
                self.artifact(job_id, filename)
            except FileNotFoundError:
                continue
            result["artifacts"][filename] = f"/api/tools/critic/{job_id}/artifacts/{filename}"
        for filename, key in (("summary.json", "diagnostic_summary"), ("progress.json", "critic_progress")):
            path = self.jobs.jobs_root / job_id / "artifacts" / filename
            if path.exists():
                try:
                    if path.parent.is_symlink():
                        continue
                    result[key] = _read_json(path)
                except (ValueError, OSError, json.JSONDecodeError):
                    continue
        return result

    def stop(self, job_id: str) -> dict:
        self._owned_job(job_id)
        return self.jobs.stop(job_id)

    def delete(self, job_id: str, confirmation: str) -> dict:
        self._owned_job(job_id)
        return self.jobs.delete_job(job_id, confirmation)
