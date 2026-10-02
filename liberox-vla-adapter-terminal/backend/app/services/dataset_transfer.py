"""Export frozen members as ordinary runs with self-contained saved evaluations."""
from __future__ import annotations

import copy
import json
import hashlib
import re
import shutil
import tempfile
import threading
from pathlib import Path

from ..storage.paths import storage_path, storage_lease
from types import SimpleNamespace

from .dataset_export_rewards import ExportRewards, checked
from .inherited_reward_inputs import offline_module
from .training_dataset_service import TrainingDatasetService
from ..storage.files import atomic_write_json
from .trajectory_reward_snapshot import _hash
from .dataset_stage_annotations import DatasetStageAnnotations


class _ReadOnlyDatasets(TrainingDatasetService):
    def __init__(self, project_root: Path, dataset_id: str):
        self.root = project_root / "datasets"
        self.run_service = self
        self.primary_dataset = dataset_id
        self.lock = threading.RLock()
        self.stage_labels = DatasetStageAnnotations(self, read_only=True)

    def get(self, dataset_id, *, quick_verify=False):
        _, payload = self._load(dataset_id)
        error = self._immutable_error(payload)
        if error:
            raise ValueError(error)
        return self._public(payload)

    def get_run(self, run_id):
        primary = self.root / self.primary_dataset / "dataset.json"
        for path in [primary, *sorted(self.root.glob("*/dataset.json"))]:
            try:
                payload = json.loads(path.read_text())
            except (OSError, ValueError):
                if path == primary:
                    raise
                continue
            member = next((item for item in payload["members"] if item["run_id"] == run_id), None)
            if member:
                artifacts = member["artifacts"]
                manifest = storage_path(artifacts["manifest"]["path"])
                return {**json.loads(manifest.read_text()), "id": run_id,
                        "trajectory": artifacts["trajectory"]["path"], "output_dir": str(manifest.parent)}
        raise KeyError(run_id)


class _ExportContext:
    read_only = True

    def __init__(self, datasets, base_config, offline_root, scratch):
        self.datasets, self.base_config_path = datasets, base_config
        self.ui_config = SimpleNamespace(offline_rl_root=offline_root)
        self.jobs_root, self.cache_root = scratch / "jobs", scratch / "cache"

    def _load_base_config(self, *args):
        config = offline_module(self.ui_config.offline_rl_root, "config")
        return copy.deepcopy(config.load_train_config(self.base_config_path).raw)

    def _effective_config(self, dataset):
        raw = self._load_base_config()
        raw["data"].update(project_id=dataset["project_id"], task_ids=[dataset["task_id"]],
            selection_manifest=str(self.datasets.root / dataset["id"] / "dataset.json"),
            **{key: dataset[key] for key in ("validation_fraction", "split_seed", "success_consecutive_steps")})
        raw["data"]["include_post_success"] = dataset.get("include_post_success", True)
        raw["paths"].update(work_dir=str(self.jobs_root / "prepare"), annotation_cache=str(self.cache_root))
        return raw


def export_dataset(project_root: Path, dataset_id: str, destination: Path, base_config: Path,
                   offline_root: Path, *, progress=lambda **_: None) -> Path:
    with storage_lease(project_root.resolve().parents[1]):
        return _export_dataset(project_root, dataset_id, destination, base_config, offline_root,
                               progress=progress)


def _export_dataset(project_root: Path, dataset_id: str, destination: Path, base_config: Path,
                    offline_root: Path, *, progress) -> Path:
    datasets = _ReadOnlyDatasets(project_root, dataset_id)
    selection_path, frozen = datasets._load(dataset_id)
    dataset = datasets.get(dataset_id)
    labels = datasets.stage_labels.load(frozen)["annotations"]
    destination = destination.absolute()
    if destination.exists() or destination.is_symlink():
        raise ValueError(f"Export destination already exists: {destination}")
    if destination.resolve().is_relative_to(project_root.resolve().parents[1]):
        raise ValueError("Export must be outside dataset-root to avoid duplicate run IDs")
    selection_hash = _hash(selection_path)
    members = frozen["members"]
    if not members or len({m["run_id"] for m in members}) != len(members):
        raise ValueError("Dataset must contain unique, nonempty members")
    task = re.sub(r"[^A-Za-z0-9_.-]+", "_", dataset["task_id"]).strip("._")
    task = task[:160] + "_" + hashlib.sha256(dataset["task_id"].encode()).hexdigest()[:8]
    destination.parent.mkdir(parents=True, exist_ok=True)
    progress(stage="验证数据与评价", total_runs=len(members), completed_runs=0)
    with tempfile.TemporaryDirectory(prefix=".dataset-export-", dir=destination.parent) as directory:
        scratch = Path(directory)
        staging = scratch / "payload"
        staging.mkdir()
        context = _ExportContext(datasets, base_config, offline_root, scratch)
        rewards = ExportRewards(context, dataset)
        exported = []
        for number, member in enumerate(members):
            run_id = member["run_id"]
            if not re.fullmatch(r"[A-Za-z0-9_-]+", run_id):
                raise ValueError(f"Unsafe run ID: {run_id}")
            artifacts = member["artifacts"]
            manifest = checked(artifacts["manifest"]["path"], artifacts["manifest"]["sha256"])
            source = manifest.parent
            if source.is_symlink():
                raise ValueError(f"Symlink run directory: {source}")
            for artifact in artifacts.values():
                path = checked(artifact["path"], artifact["sha256"])
                if not path.resolve().is_relative_to(source.resolve()):
                    raise ValueError(f"Run artifact is outside its directory: {path}")
            target = staging / "runs" / task / run_id
            progress(stage="复制轨迹", current_run=run_id)
            # Copy all recording files, not the old ZIP's video/CSV whitelist.
            # Streaming copy/hash avoids decompressing multi-GB observation archives.
            originals = _copy_run(source, target, progress)
            for artifact in artifacts.values():
                checked(target / storage_path(artifact["path"]).relative_to(source), artifact["sha256"])
            copied_episode = target / storage_path(artifacts["trajectory"]["path"]).relative_to(source).parent
            progress(stage="绑定已有评价", current_file="")
            evaluations = rewards.publish(member, copied_episode)
            label = labels[run_id]
            if label.get("error"):
                raise ValueError(f"Invalid dataset keyframes for {run_id}: {label['error']}")
            stage_path = copied_episode / "stage_annotation.json"
            if label.get("annotation") is not None:
                atomic_write_json(stage_path, label["annotation"])
            elif stage_path.exists():
                stage_path.unlink()  # Export copy only; never leak unrelated global labels.
            for path, signature in originals:
                if _signature(path) != signature:
                    raise ValueError(f"Source changed during export: {path}")
            exported.append({"run_id": run_id, "path": str(target.relative_to(staging)),
                "split": member["split"], "root_run_id": member["root_run_id"],
                "parent_run_id": member.get("parent_run_id"), "evaluations": evaluations})
            progress(completed_runs=number + 1)
        if _hash(selection_path) != selection_hash:
            raise ValueError("Dataset selection or active evaluation changed during export; retry")
        # Selection is provenance only: server training scans the ordinary runs tree.
        atomic_write_json(staging / "dataset.json", frozen)
        progress(stage="校验导出文件", current_file="")
        files = {str(p.relative_to(staging)): {"sha256": _hash(p), "size": p.stat().st_size}
                 for p in sorted(staging.rglob("*")) if p.is_file()}
        atomic_write_json(staging / "export.json", {"schema_version": 1, "complete": True,
            "kind": "dataset_runs_export", "dataset_id": dataset_id, "dataset_name": dataset["name"],
            "task_id": dataset["task_id"], "dataset_sha256": dataset["dataset_sha256"],
            "runs_root": "runs", "members": exported, "files": files})
        staging.rename(destination)
    return destination


def _signature(path: Path) -> tuple:
    stat = path.lstat()
    return stat.st_ino, stat.st_size, stat.st_mtime_ns, stat.st_ctime_ns, stat.st_mode


def _copy_run(source: Path, target: Path, progress) -> list:
    signatures = [(source, _signature(source))]
    target.mkdir(parents=True)
    for path in sorted(source.rglob("*")):
        if path.is_symlink():
            raise ValueError(f"Symlinks cannot be exported: {path}")
        output = target / path.relative_to(source)
        signature = _signature(path)
        if path.is_dir():
            output.mkdir(exist_ok=True)
            signatures.append((path, signature))
            continue
        if not path.is_file():
            raise ValueError(f"Not a regular recording file: {path}")
        progress(current_file=str(path.relative_to(source)))
        shutil.copyfile(path, output)
        if _hash(path) != _hash(output) or _signature(path) != signature:
            raise ValueError(f"Recording changed while copying: {path}")
        signatures.append((path, signature))
    return signatures
