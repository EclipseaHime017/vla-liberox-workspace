"""Resolve saved evaluations and bind them to exported copies, never source runs."""
from __future__ import annotations

import copy
import json
import shutil
from pathlib import Path

import numpy as np

from ..storage.files import atomic_write_json
from ..storage.paths import storage_path
from .global_reward_binding import global_members
from .inherited_reward_inputs import offline_module, validate_global_values
from .trajectory_reward_snapshot import SOURCES, _hash, _stat, has_implicit_global, snapshot_path


def read_json(path: Path) -> dict:
    if path.is_symlink() or not path.is_file():
        raise ValueError(f"Missing or symlink evaluation: {path}")
    return json.loads(path.read_text(encoding="utf-8"))


def checked(path: str | Path, digest: str) -> Path:
    path = storage_path(path)
    if path.is_symlink() or not path.is_file() or _hash(path) != digest:
        raise ValueError(f"Evaluation hash mismatch: {path}")
    return path


def entry_for(payload: dict, run_id: str) -> dict:
    matches = [item for item in payload.get("episodes", []) if item.get("run_id") == run_id]
    if len(matches) != 1:
        raise ValueError(f"Evaluation must contain exactly one entry for {run_id}")
    return matches[0]


class ExportRewards:
    def __init__(self, context, dataset: dict):
        self.context, self.dataset = context, dataset
        self.versions: dict[tuple[str, str], dict] = {}
        self.history: list | None = None

    def _version(self, owner: str, identifier: str) -> dict:
        key = (owner, identifier)
        if key not in self.versions:
            candidate = self.context.datasets.get_version(owner, identifier)
            if candidate.get("legacy") and candidate.get("status") == "READY":
                # Older completed results predate version.json. Validate their
                # manifests/arrays below and seal only the exported copy.
                for prefix in ("prepared_manifest", "reward_manifest"):
                    path = storage_path(candidate[f"{prefix}_path"])
                    read_json(path)
                    candidate[f"{prefix}_sha256"] = _hash(path)
                self.versions[key] = candidate
            else:
                self.versions[key] = self.context.datasets.validate_version(owner, identifier)
        return self.versions[key]

    def _first_global_version(self, run_id: str, source: str) -> dict | None:
        # Legacy global detail falls back to the first saved dataset evaluation.
        if self.history is None:
            self.history = self.context.datasets.list()
        choices = [(v.get("completed_at") or v.get("created_at") or "", ds["id"], v["id"])
                   for ds in self.history if any(m["run_id"] == run_id for m in ds["members"])
                   for v in ds.get("evaluation_versions", [])
                   if v.get("evaluator") == source and v.get("status") == "READY"]
        if not choices:
            return None
        _, owner, identifier = min(choices)
        return self._version(owner, identifier)

    def _reward_version(self, version: dict, run_id: str) -> tuple[dict, Path]:
        manifest = read_json(checked(version["reward_manifest_path"], version["reward_manifest_sha256"]))
        prepared = read_json(checked(version["prepared_manifest_path"], version["prepared_manifest_sha256"]))
        if manifest.get("complete") is not True or manifest.get("dataset_sha256") != prepared.get("dataset_sha256"):
            raise ValueError("Incomplete or mismatched reward manifest")
        entry, episode = entry_for(manifest, run_id), entry_for(prepared, run_id)
        metadata = {"run_id": run_id, "evaluation_id": version["id"],
            "evaluated_at": version.get("completed_at"), "episode": episode,
            "prepared": {k: v for k, v in prepared.items() if k != "episodes"},
            "trajectory_sha256": episode["trajectory_sha256"],
            "observations_sha256": episode["observations_sha256"],
            "values_sha256": entry["reward_sha256"], "entry": copy.deepcopy(entry),
            "reward_config": entry.get("saved_reward_config") or manifest["reward_config"],
            "annotation_config": entry.get("saved_annotation_config") or manifest.get("annotation_config", {}),
            "annotator": entry.get("annotator") or manifest.get("annotator", {}),
            "official_outputs": entry.get("official_outputs", {})}
        if version.get("annotation_manifest_path"):
            annotation = read_json(storage_path(version["annotation_manifest_path"]))
            core = offline_module(self.context.ui_config.offline_rl_root, "io")
            if core.stable_hash(annotation) != manifest.get("annotation_manifest_sha256"):
                raise ValueError("Official annotation manifest hash mismatch")
            metadata["official_outputs"] = entry_for(annotation, run_id).get("official_outputs", {})
        if manifest.get("stage_annotations_path"):
            snapshot = read_json(storage_path(manifest["stage_annotations_path"]))
            core = offline_module(self.context.ui_config.offline_rl_root, "io")
            if core.stable_hash(snapshot) != manifest["stage_annotations_sha256"]:
                raise ValueError("Stage keyframe snapshot hash mismatch")
            metadata["stage_annotations_snapshot"] = snapshot
            metadata["stage_annotations_sha256"] = manifest["stage_annotations_sha256"]
        return metadata, checked(entry["reward_path"], entry["reward_sha256"])

    def reward(self, member: dict, source: str) -> tuple[dict, Path] | None:
        run_id = member["run_id"]
        selected = self.dataset["evaluation_version_ids"].get(source)
        origin = "dataset" if selected else "global"
        if selected:
            result = self._reward_version(self._version(self.dataset["id"], selected), run_id)
        else:
            run = self.context.datasets.get_run(run_id)
            sidecar = snapshot_path(run, source)
            native = storage_path(run["trajectory"]).with_name("rynnvalue_evaluation.json")
            if (sidecar and sidecar.exists()) or (source == "rynnvalue" and native.exists()):
                records, errors = global_members(self.context, self.dataset, source, validate=True, run_ids={run_id})
                if errors or len(records) != 1:
                    raise ValueError(f"Invalid {source} evaluation for {run_id}: {errors}")
                _, meta, values, episode, header = records[0]
                result = ({**meta, "episode": episode, "prepared": header}, values)
            else:
                if has_implicit_global(run, source):
                    return None  # Keep environment outcomes/keyframes, not an unrelated historical recipe.
                version = self._first_global_version(run_id, source)
                if version is None:
                    return None  # Labels are copied, not evaluated or turned into new rewards.
                result = self._reward_version(version, run_id)
        metadata, values = result
        metadata = copy.deepcopy(metadata)
        for name in ("trajectory", "observations"):
            if metadata[f"{name}_sha256"] != member["artifacts"][name]["sha256"]:
                raise ValueError(f"{source} {name} differs from frozen member {run_id}")
        if metadata["episode"]["source_manifest_sha256"] != member["artifacts"]["manifest"]["sha256"]:
            raise ValueError(f"{source} prompt manifest differs from frozen member {run_id}")
        if metadata["reward_config"].get("source", "rynnvalue") != source:
            raise ValueError(f"Reward source mismatch: {run_id}/{source}")
        checked(values, metadata["values_sha256"])
        validate_global_values(values, metadata["episode"], source, self.context.ui_config.offline_rl_root)
        metadata.update(source=source, origin=origin)
        return metadata, values

    def robometer(self, member: dict) -> tuple[dict, Path] | None:
        from .robometer_evaluation_service import ARRAY_KEYS, SCHEMA_VERSION

        run_id = member["run_id"]
        selected = self.dataset["evaluation_version_ids"].get("robometer")
        trajectory = storage_path(member["artifacts"]["trajectory"]["path"])
        native = trajectory.with_name("robometer_evaluation.json")
        version = self._version(self.dataset["id"], selected) if selected else None
        if version is None and not native.exists():
            version = self._first_global_version(run_id, "robometer")
        if version:
            manifest = read_json(checked(version["robometer_manifest_path"], version["robometer_manifest_sha256"]))
            if manifest.get("complete") is not True:
                raise ValueError("Robometer manifest is incomplete")
            item = entry_for(manifest, run_id)
            values = storage_path(item["annotation_path"])
            metadata = {**item, "schema_version": manifest["schema_version"],
                "evaluated_at": version.get("completed_at"), "evaluation_id": version["id"],
                **{key: manifest.get(key) for key in ("annotator", "evaluation_config", "inference_config")}}
        elif native.exists():
            metadata = read_json(native)
            name = metadata.get("values_file", "robometer_evaluation.npz")
            if Path(name).name != name:
                raise ValueError("Unsafe Robometer values path")
            values = native.with_name(name)
        else:
            return None
        if metadata.get("schema_version") != SCHEMA_VERSION or metadata.get("run_id") != run_id:
            raise ValueError(f"Invalid Robometer identity: {run_id}")
        for name in ("trajectory", "observations", "manifest"):
            if metadata.get(f"{name}_sha256") != member["artifacts"][name]["sha256"]:
                raise ValueError(f"Robometer {name} hash mismatch: {run_id}")
        checked(values, metadata["values_sha256"])
        with np.load(values, allow_pickle=False) as arrays:
            if not ARRAY_KEYS <= set(arrays.files) or any(
                    arrays[k].shape != (metadata["sample_count"],) or not np.isfinite(arrays[k]).all()
                    for k in ARRAY_KEYS):
                raise ValueError(f"Invalid Robometer outputs: {run_id}")
        return {**metadata, "origin": "dataset" if selected else "global"}, values

    def publish(self, member: dict, episode: Path) -> dict:
        results = {}
        for source in (*SOURCES, "robometer"):
            result = self.robometer(member) if source == "robometer" else self.reward(member, source)
            if result is None:
                results[source] = {"status": "NOT_EVALUATED"}
                continue
            metadata, values = result
            name = ("robometer_evaluation.npz" if source == "robometer" else
                    f"trajectory_reward.{metadata['values_sha256']}.npz")
            shutil.copyfile(values, episode / name)
            checked(episode / name, metadata["values_sha256"])
            metadata.update(values_file=name)
            if source != "robometer":
                metadata.update(schema_version=1, observations_fingerprint=list(_stat(episode / "trajectory_observations.npz")))
                # Retain independent official output arrays referenced by newer rewards.
                entry = metadata.get("entry", {})
                if entry.get("official_annotation_path"):
                    original = storage_path(entry["official_annotation_path"])
                    if original.exists() or metadata["origin"] == "dataset":
                        checked(original, entry["official_annotation_sha256"])
                        extra = f"official_annotation.{_hash(original)}.npz"
                        shutil.copyfile(original, episode / extra)
                        entry["official_annotation_path"] = extra
                    else:
                        # Global snapshots outlive the job that produced them; the
                        # combined values retain the complete model heads unchanged.
                        metadata["original_official_annotation_path"] = entry.pop("official_annotation_path")
                        entry.pop("official_annotation_sha256", None)
                atomic_write_json(episode / f"trajectory_reward.{source}.json", metadata)
                if source == "rynnvalue":
                    # Native sidecar remains consumable by the GUI and annotation cache.
                    from .trajectory_evaluation_service import EVALUATION_SCHEMA_VERSION
                    shutil.copyfile(episode / name, episode / "rynnvalue_evaluation.npz")
                    with np.load(values, allow_pickle=False) as arrays:
                        count = len(arrays["boundary_steps"])
                    atomic_write_json(episode / "rynnvalue_evaluation.json", {**metadata,
                        "schema_version": EVALUATION_SCHEMA_VERSION, "boundary_count": count,
                        "values_file": "rynnvalue_evaluation.npz"})
            else:
                atomic_write_json(episode / "robometer_evaluation.json", metadata)
            results[source] = {"status": "READY", "origin": metadata["origin"],
                "evaluation_id": metadata.get("evaluation_id"), "values_sha256": metadata["values_sha256"]}
        return results
