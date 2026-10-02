"""Dataset-owned keyframes, separate from immutable selections and reward snapshots."""
from __future__ import annotations

import copy
import hashlib
import json
from pathlib import Path

from ..storage.files import atomic_write_json
from ..storage.paths import storage_path


def _read(path: Path) -> dict:
    if path.is_symlink():
        raise ValueError("Symlink Stage annotations are not allowed")
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise ValueError("Invalid Stage annotation metadata")
    return value


def _digest(value: dict) -> str:
    return hashlib.sha256(json.dumps(value, sort_keys=True, separators=(",", ":")).encode()).hexdigest()


class DatasetStageAnnotations:
    """Copy once, including missing labels; never silently follow a later global edit.

    Callers hold the dataset service lock. The export reader uses persist=False
    to resolve legacy labels without modifying source data.
    """

    def __init__(self, datasets, *, read_only: bool = False):
        self.datasets = datasets
        self.read_only = read_only

    def _path(self, dataset: dict) -> Path:
        return self.datasets.root / dataset["id"] / "stage_annotations.json"

    @staticmethod
    def _global(member: dict) -> dict:
        path = storage_path(member["artifacts"]["trajectory"]["path"]).with_name("stage_annotation.json")
        try:
            annotation = _read(path) if path.exists() else None
            return {"annotation": annotation, "origin": "global_copy" if annotation else "missing", "error": None}
        except (OSError, ValueError) as exc:
            return {"annotation": None, "origin": "global_copy", "error": str(exc)}

    def _historical(self, dataset: dict) -> dict | None:
        ids = self.datasets.current_evaluation_ids(dataset)
        # Final is the current fused recipe; Stage is the older standalone recipe.
        for source in ("final", "stage"):
            if source not in ids:
                continue
            try:
                version = self.datasets.get_version(dataset["id"], ids[source])
                manifest_path = storage_path(version["reward_manifest_path"])
                manifest = _read(manifest_path)
                if (version.get("reward_manifest_sha256") and hashlib.sha256(manifest_path.read_bytes()).hexdigest()
                        != version["reward_manifest_sha256"]):
                    raise ValueError("已有奖励的 manifest 哈希不匹配")
                if not manifest.get("stage_annotations_path"):
                    recipe = manifest.get("reward_config") or version.get("parameters", {})
                    if (source == "final" and recipe.get("alpha") == 0
                            and recipe.get("fusion_mode", "additive") == "additive"):
                        continue  # Alpha-zero Final does not need Stage labels.
                    raise ValueError("已有奖励缺少关键帧快照，请显式继承全局标注或重新标记")
                snapshot = _read(storage_path(manifest["stage_annotations_path"]))
                if _digest(snapshot) != manifest.get("stage_annotations_sha256"):
                    raise ValueError("已有奖励的关键帧快照哈希不匹配")
                if not isinstance(snapshot.get("annotations"), dict):
                    raise ValueError("Invalid Stage snapshot annotations")
                entries = {item["run_id"]: item for item in manifest["episodes"]}
                result = {}
                for member in dataset["members"]:
                    run_id = member["run_id"]
                    annotation = snapshot["annotations"].get(run_id)
                    if not isinstance(annotation, dict):
                        raise ValueError(f"已有奖励的关键帧快照缺少 {run_id}")
                    if (annotation.get("run_id") != run_id or annotation.get("trajectory_sha256")
                            != member["artifacts"]["trajectory"]["sha256"]):
                        raise ValueError(f"关键帧快照与冻结轨迹不匹配: {run_id}")
                    expected = entries.get(run_id, {}).get("stage_annotation_sha256")
                    if expected and expected != annotation.get("annotation_sha256"):
                        raise ValueError(f"关键帧快照与奖励不匹配: {run_id}")
                    result[run_id] = {"annotation": annotation, "origin": "reward_snapshot",
                                      "source_version_id": ids[source], "error": None}
                return result
            except (OSError, ValueError, KeyError, TypeError) as exc:
                # A broken historical snapshot must not be replaced by newer globals.
                return {member["run_id"]: {"annotation": None, "origin": "reward_snapshot",
                        "error": str(exc)} for member in dataset["members"]}
        return None

    def load(self, dataset: dict, *, persist: bool = True) -> dict:
        persist = persist and not self.read_only
        path = self._path(dataset)
        if path.exists() or path.is_symlink():
            value = _read(path)
            if (value.get("schema_version") != 1 or value.get("dataset_id") != dataset["id"]
                    or value.get("dataset_sha256") != dataset["dataset_sha256"]
                    or not isinstance(value.get("annotations"), dict)
                    or set(value["annotations"]) != {m["run_id"] for m in dataset["members"]}
                    or any(not isinstance(entry, dict) for entry in value["annotations"].values())):
                raise ValueError("Dataset Stage annotation identity mismatch")
            return value
        entries = self._historical(dataset)
        if entries is None:
            parent_entries = {}
            if dataset.get("parent_dataset_id"):
                try:
                    _, parent = self.datasets._load(dataset["parent_dataset_id"])
                except KeyError:
                    # Legacy derived datasets may outlive their parent.
                    return self._missing_parent(dataset, persist=persist)
                parent_entries = self.load(parent, persist=persist)["annotations"]
            entries = {}
            for member in dataset["members"]:
                run_id = member["run_id"]
                if run_id in parent_entries:
                    entries[run_id] = {**copy.deepcopy(parent_entries[run_id]), "origin": "parent_copy"}
                else:
                    entries[run_id] = self._global(member)
        value = {"schema_version": 1, "dataset_id": dataset["id"],
                 "dataset_sha256": dataset["dataset_sha256"], "annotations": entries}
        if persist:
            atomic_write_json(path, value)
        return value

    def _missing_parent(self, dataset: dict, *, persist: bool) -> dict:
        value = {"schema_version": 1, "dataset_id": dataset["id"],
                 "dataset_sha256": dataset["dataset_sha256"], "annotations": {
                     m["run_id"]: {"annotation": None, "origin": "parent_copy",
                         "error": "父数据集已不存在，请显式继承全局标注或重新标记"}
                     for m in dataset["members"]}}
        if persist:
            atomic_write_json(self._path(dataset), value)
        return value

    def member(self, dataset_id: str, run_id: str) -> tuple[dict, dict, dict]:
        _, dataset = self.datasets._load(dataset_id)
        member = next((item for item in dataset["members"] if item["run_id"] == run_id), None)
        if member is None:
            raise ValueError("当前轨迹不属于所选标注数据集，请明确选择其他标注目标")
        value = self.load(dataset)
        return dataset, member, value["annotations"][run_id]

    def save(self, dataset: dict, run_id: str, annotation: dict, *, origin: str = "dataset") -> None:
        value = self.load(dataset)
        value["annotations"][run_id] = {"annotation": annotation, "origin": origin, "error": None}
        atomic_write_json(self._path(dataset), value)
