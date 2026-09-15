"""Bind per-trajectory global results for training without running evaluators."""
from __future__ import annotations

import copy
import hashlib
import json
import shutil
import uuid
from pathlib import Path

from ..core.exceptions import ConflictError
from ..storage.files import atomic_write_json, atomic_write_yaml
from .trajectory_reward_snapshot import (
    _hash, _refresh_snapshot_identity, read_reward_snapshot, snapshot_path,
)


def _json(path: Path) -> dict:
    if path.is_symlink():
        raise ValueError("Symlink reward metadata is not allowed")
    return json.loads(path.read_text(encoding="utf-8"))


def _saved_metadata(jobs, run_id: str, metadata: dict) -> tuple[dict, dict]:
    """Read original chunk semantics, including old sidecars lacking a full episode."""
    if metadata.get("episode", {}).get("trajectory_path") and metadata.get("prepared"):
        return metadata["episode"], metadata["prepared"]
    candidates = list(jobs.datasets.root.glob("*/annotations/*/work/dataset_manifest.json"))
    project = getattr(jobs, "project_root", jobs.datasets.root.parent)
    candidates += list((project / "trajectory-evaluations").glob("*/work/rynnvalue/dataset_manifest.json"))
    if metadata.get("prepared_manifest_path"):
        candidates.insert(0, Path(metadata["prepared_manifest_path"]))
    for path in candidates:
        try:
            prepared = _json(path)
            rewards = _json(path.parent / "rewards" / "reward_manifest.json")
            entry = next(item for item in rewards["episodes"] if item["run_id"] == run_id)
            episode = next(item for item in prepared["episodes"] if item["run_id"] == run_id)
            if (entry.get("reward_sha256", entry.get("annotation_sha256")) == metadata["values_sha256"]
                    and episode["trajectory_sha256"] == metadata["trajectory_sha256"]
                    and episode["observations_sha256"] == metadata["observations_sha256"]):
                return episode, {key: value for key, value in prepared.items() if key != "episodes"}
        except (OSError, ValueError, KeyError, StopIteration):
            continue
    raise ValueError("旧全局评价缺少可验证的 chunk 元数据，请用已有模型输出重新生成该类型评价")


def global_members(jobs, dataset: dict, source: str, *, validate: bool = False) -> tuple[list, list]:
    _, frozen = jobs.datasets._load(dataset["id"])
    records, missing = [], []
    for member in frozen["members"]:
        try:
            run = jobs.datasets.run_service.get_run(member["run_id"])
        except KeyError:
            missing.append({"run_id": member["run_id"],
                            "error": "轨迹记录不存在或尚未加载，请检查部署设备的数据目录与会话索引"})
            continue
        try:
            path = snapshot_path(run, source)
            snapshot = read_reward_snapshot(run, source)
            if snapshot is None and path and path.is_file() and validate:
                snapshot = _refresh_snapshot_identity(run, source)
            if snapshot:
                metadata = snapshot["metadata"]
                values = path.with_name(metadata["values_file"])
            elif path and path.exists():
                raise ValueError("全局评价的源数据已变化或评价损坏")
            elif source == "rynnvalue" and Path(run["trajectory"]).with_name("rynnvalue_evaluation.json").exists():
                # Native Rynn sidecars predate source-specific reward snapshots.
                path = Path(run["trajectory"]).with_name("rynnvalue_evaluation.json")
                metadata = _json(path)
                values = path.with_name("rynnvalue_evaluation.npz")
                if metadata.get("schema_version") not in (5, 6):
                    raise ValueError("全局 RynnValue 评价需要升级")
            else:
                metadata, values = _first_saved_result(jobs, member["run_id"], source)
            if (metadata.get("run_id") != member["run_id"] or values.is_symlink()
                    or _hash(values) != metadata["values_sha256"]):
                raise ValueError("全局评价身份或数组哈希不匹配")
            for name in ("trajectory", "observations"):
                if member["artifacts"][name]["sha256"] != metadata[f"{name}_sha256"]:
                    raise ValueError("全局评价与冻结数据源不匹配")
            episode, header = _saved_metadata(jobs, member["run_id"], metadata)
            if episode.get("source_manifest_sha256") != member["artifacts"]["manifest"]["sha256"]:
                raise ValueError("全局评价与冻结轨迹的任务清单不匹配")
            recipe = metadata.get("reward_config") or {}
            if recipe.get("source", "rynnvalue") != source:
                raise ValueError("全局评价类型不匹配")
            records.append((member, metadata, values, episode, header))
        except (OSError, ValueError, KeyError, TypeError) as exc:
            missing.append({"run_id": member["run_id"], "error": str(exc)})
    return records, missing


def _first_saved_result(jobs, run_id: str, source: str) -> tuple[dict, Path]:
    """Same lazy legacy-global fallback as detail, without publishing in GET."""
    candidates = []
    for dataset in jobs.datasets.list():
        if not any(item["run_id"] == run_id for item in dataset["members"]):
            continue
        for saved in dataset.get("evaluation_versions", []):
            if saved.get("status") == "READY" and saved.get("evaluator") == source:
                candidates.append((saved.get("completed_at") or saved.get("created_at") or "",
                                   dataset["id"], saved["id"]))
    for _, owner, identifier in sorted(candidates):
        try:
            version = jobs.datasets.get_version(owner, identifier)
            prepared = _json(Path(version["prepared_manifest_path"]))
            index = _json(Path(version["reward_manifest_path"]))
            episode = next(item for item in prepared["episodes"] if item["run_id"] == run_id)
            entry = next(item for item in index["episodes"] if item["run_id"] == run_id)
            return {"run_id": run_id, "source": source, "evaluation_id": identifier,
                "trajectory_sha256": episode["trajectory_sha256"],
                "observations_sha256": episode["observations_sha256"],
                "values_sha256": entry["reward_sha256"], "entry": entry, "episode": episode,
                "prepared": {key: value for key, value in prepared.items() if key != "episodes"},
                "reward_config": index["reward_config"], "annotation_config": index.get("annotation_config", {})
            }, Path(entry["reward_path"])
        except (KeyError, ValueError, OSError, StopIteration):
            continue
    raise ValueError("缺少全局评价")


def bind_global_rewards(jobs, dataset: dict, source: str) -> dict:
    records, missing = global_members(jobs, dataset, source, validate=True)
    if missing or not records:
        raise ConflictError(f"缺少可用的 {source} 评价：{missing}", code="REWARD_VERSION_NOT_READY")
    raw = jobs._effective_config(dataset)
    episodes, entries = [], []
    for member, metadata, values, saved, header in records:
        for field in ("action_horizon", "action_dim", "proprio_dim", "control_hz"):
            if field in header and header[field] != raw["data"][field]:
                raise ValueError(f"Global reward {field} differs from training: {member['run_id']}")
        episode = copy.deepcopy(saved)
        if int(episode["recorded_action_count"]) != int(member["end_step"]):
            raise ValueError(f"Global reward recording length changed: {member['run_id']}")
        if int(episode.get("resume_step") or 0) != int(member.get("resume_step") or 0):
            raise ValueError(f"Global reward branch boundary changed: {member['run_id']}")
        for field in ("split", "root_run_id", "parent_run_id"):
            episode[field] = member[field]
        for name, field in (("trajectory", "trajectory_path"), ("observations", "observations_path"),
                            ("manifest", "source_manifest")):
            episode[field] = member["artifacts"][name]["path"]
        episodes.append(episode)
        entries.append({**metadata.get("entry", {}), "run_id": member["run_id"],
            "reward_path": str(values), "annotation_path": str(values),
            "reward_sha256": metadata["values_sha256"], "annotation_sha256": metadata["values_sha256"],
            "saved_reward_config": metadata["reward_config"],
            "saved_annotation_config": metadata.get("annotation_config", {}),
            "annotator": metadata.get("annotator", {}), "official_outputs": metadata.get("official_outputs", {}),
            "source_values_sha256": metadata["values_sha256"], "source_evaluated_at": metadata.get("evaluated_at"),
            "source_evaluation_id": metadata.get("evaluation_id"), "source_origin": "global"})
    identifier = "global_" + uuid.uuid4().hex
    work = jobs.jobs_root / identifier / "work"
    work.mkdir(parents=True, exist_ok=False)
    for entry in entries:
        destination = work / f"{entry['reward_sha256']}.npz"
        shutil.copyfile(entry["reward_path"], destination)
        if _hash(destination) != entry["reward_sha256"]:
            raise ValueError("Global reward changed while taking training snapshot")
        entry.update(reward_path=str(destination), annotation_path=str(destination))
    digest = hashlib.sha256(json.dumps(episodes, sort_keys=True).encode()).hexdigest()
    prepared = {**records[0][4], "episodes": episodes, "dataset_sha256": digest,
        "source_dataset_id": dataset["id"], "source_dataset_sha256": dataset["dataset_sha256"],
        "episode_count": len(episodes), "success_count": sum(bool(ep["success"]) for ep in episodes)}
    raw["reward"].update(entries[0]["saved_reward_config"])
    raw["reward"].update(entries[0]["saved_annotation_config"])
    raw["reward"].update(source=source, rynnvalue=source == "rynnvalue",
                          manifest_path=None, manifest_sha256=None, version_id=None)
    raw["data"]["stage_annotations_manifest"] = None
    raw["paths"]["work_dir"] = str(work)
    index = {"schema_version": 1, "kind": "derived_iql_reward", "complete": True,
        "binding_kind": "global_trajectory_snapshots", "version_id": identifier,
        "dataset_sha256": digest, "reward_config": entries[0]["saved_reward_config"],
        "annotation_config": entries[0]["saved_annotation_config"], "episodes": entries}
    prepared_path, reward_path = work / "dataset_manifest.json", work / "reward_manifest.json"
    config_path = work.parent / "effective_config.yaml"
    atomic_write_json(prepared_path, prepared)
    atomic_write_json(reward_path, index)
    atomic_write_yaml(config_path, raw)
    return {"id": identifier, "evaluator": source, "status": "READY", "complete": True,
        "origin": "global", "work_dir": str(work), "config_path": str(config_path),
        "config_sha256": _hash(config_path), "prepared_manifest_path": str(prepared_path),
        "prepared_manifest_sha256": _hash(prepared_path), "reward_manifest_path": str(reward_path),
        "reward_manifest_sha256": _hash(reward_path)}
