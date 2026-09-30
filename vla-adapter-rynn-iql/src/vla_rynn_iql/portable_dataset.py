"""Self-contained, content-verified PC → server training snapshots.

Original recordings/sidecars are copied byte-for-byte. Portable references use
bundle:// paths; only a new run's private manifests are resolved to local paths.
No evaluator, database, web server or absolute path from the PC is needed to train.
"""
from __future__ import annotations

import copy
import hashlib
import json
import os
import re
import shutil
import tempfile
from pathlib import Path

from .storage_paths import storage_path

import yaml

from .io import atomic_json, sha256_file, stable_hash


BUNDLE_NAME = "training_bundle.json"
PREFIX = "bundle://"


def task_slug(task_id: str) -> str:
    name = re.sub(r"[^A-Za-z0-9_.-]+", "__", task_id)
    if not name or name in {".", ".."}:
        raise ValueError("Invalid task ID")
    return name


def contained(root: Path, relative: str) -> Path:
    value = Path(relative)
    if value.is_absolute() or ".." in value.parts or not value.parts:
        raise ValueError(f"Unsafe bundle path: {relative}")
    path = root / value
    if not path.resolve().is_relative_to(root.resolve()):
        raise ValueError(f"Bundle path escapes its root: {relative}")
    if any(part.is_symlink() for part in (path, *path.parents) if part != root.parent):
        raise ValueError(f"Bundle symlinks are not allowed: {relative}")
    return path


def map_references(value, resolve):
    if isinstance(value, dict):
        return {key: map_references(item, resolve) for key, item in value.items()}
    if isinstance(value, list):
        return [map_references(item, resolve) for item in value]
    return resolve(value) if isinstance(value, str) else value


def read_bundle(directory: Path, *, verify: bool = False) -> dict:
    directory = directory.expanduser().resolve()
    payload = json.loads((directory / BUNDLE_NAME).read_text())
    if payload.get("schema_version") != 1 or payload.get("kind") != "portable_training_dataset":
        raise ValueError("Unsupported training bundle")
    expected = payload.get("bundle_sha256")
    if stable_hash({k: v for k, v in payload.items() if k != "bundle_sha256"}) != expected:
        raise ValueError("Training bundle manifest hash mismatch")
    for name, info in payload["files"].items():
        path = contained(directory, name)
        if not path.is_file() or path.stat().st_size != info["size"]:
            raise ValueError(f"Missing or truncated transfer file: {name}")
        if verify and sha256_file(path) != info["sha256"]:
            raise ValueError(f"Transfer file hash mismatch: {name}")
    return payload


def discover_bundles(root: Path) -> list[dict]:
    """Only lightweight manifests on the interactive selection screen."""
    result = []
    for path in sorted(root.rglob(BUNDLE_NAME)):
        try:
            data = json.loads(path.read_text())
            if data.get("kind") != "portable_training_dataset":
                continue
            result.append({"path": str(path.parent), "id": data["dataset"]["id"],
                "name": data["dataset"].get("name", data["dataset"]["id"]),
                "task_id": data["dataset"]["task_id"], "members": len(data["dataset"]["members"]),
                "rewards": list(data["rewards"])})
        except (OSError, KeyError, ValueError):
            continue
    return result


def export_bundle(destination: Path, frozen: dict, selection_path: Path,
                  prepared_path: Path, versions: dict[str, dict], *, notes: dict | None = None) -> Path:
    """Publish atomically; destination must be new. Evaluation arrays never change."""
    destination = destination.expanduser().resolve()
    if destination.exists():
        raise FileExistsError(f"Refusing to replace a training bundle: {destination}")
    source_roots = [selection_path.parent.resolve(), *[
        storage_path(member["artifacts"]["manifest"]["path"]).resolve().parent for member in frozen["members"]]]
    if any(destination.is_relative_to(root) for root in source_roots):
        raise ValueError("Export destination must be outside all source run/dataset directories")
    destination.parent.mkdir(parents=True, exist_ok=True)
    temporary = Path(tempfile.mkdtemp(prefix=f".{destination.name}.", dir=destination.parent))
    copies: dict[str, str] = {}

    def copy_file(source: Path, relative: str | None = None, *, expected: str | None = None) -> str:
        source = storage_path(source)
        if source.is_symlink() or not source.is_file():
            raise ValueError(f"Missing or unsafe export source: {source}")
        key = str(source)
        if key in copies:
            if expected and sha256_file(contained(temporary, copies[key][len(PREFIX):])) != expected:
                raise ValueError(f"Copied source differs from its frozen hash: {source}")
            return copies[key]
        relative = relative or f"values/{sha256_file(source)}{source.suffix}"
        target = contained(temporary, relative)
        target.parent.mkdir(parents=True, exist_ok=True)
        before = sha256_file(source)
        if expected and before != expected:
            raise ValueError(f"Source differs from its frozen hash: {source}")
        if target.exists() and sha256_file(target) != before:
            raise ValueError(f"Export destination collision: {relative}")
        if not target.exists():
            shutil.copy2(source, target)
        if sha256_file(target) != before or sha256_file(source) != before:
            raise ValueError(f"Source changed during export: {source}")
        copies[key] = PREFIX + relative
        return copies[key]

    def verified_bytes(source: Path, expected: str | None = None) -> bytes:
        content = storage_path(source).read_bytes()
        if expected and hashlib.sha256(content).hexdigest() != expected:
            raise ValueError(f"Sealed evaluation metadata hash mismatch: {source}")
        return content

    def prepared(source: Path, relative: str, expected: str | None = None) -> str:
        payload = json.loads(verified_bytes(source, expected))
        for episode in payload["episodes"]:
            for field in ("trajectory_path", "observations_path", "source_manifest"):
                hash_key = "source_manifest_sha256" if field == "source_manifest" else field.replace("_path", "_sha256")
                episode[field] = copy_file(Path(episode[field]), expected=episode[hash_key])
        atomic_json(temporary / relative, payload)
        return PREFIX + relative

    try:
        for member in frozen["members"]:
            for artifact in member["artifacts"].values():
                if sha256_file(storage_path(artifact["path"])) != artifact["sha256"]:
                    raise ValueError(f"Frozen source changed: {member['run_id']}")
            run_root = storage_path(member["artifacts"]["manifest"]["path"]).parent
            run_id = member["run_id"]
            if not re.fullmatch(r"[\w.-]+", run_id) or run_id in {".", ".."}:
                raise ValueError("Unsafe run ID")
            for source in sorted(run_root.rglob("*")):
                if source.is_symlink():
                    raise ValueError(f"Symlink source is not exportable: {source}")
                if source.is_file():
                    copy_file(source, f"runs/{run_id}/{source.relative_to(run_root).as_posix()}")
        # Retain the complete source dataset archive, including inactive results.
        # These bytes are provenance; runnable references are described separately.
        for source in sorted(selection_path.parent.rglob("*")):
            if source.is_symlink():
                raise ValueError(f"Symlink dataset artifact: {source}")
            if source.is_file():
                copy_file(source, f"dataset/{source.relative_to(selection_path.parent).as_posix()}")
        action_manifest = prepared(prepared_path, "prepared/dataset_manifest.json")
        # Revalidate the frozen source expectations against the actual copied bytes.
        for member in frozen["members"]:
            for artifact in member["artifacts"].values():
                copy_file(storage_path(artifact["path"]), expected=artifact["sha256"])
        rewards = {}
        for name, version in versions.items():
            if name not in {"sparse", "stage", "rynnvalue", "final"}:
                raise ValueError(f"Unsupported training reward: {name}")
            raw = yaml.safe_load(verified_bytes(Path(version["config_path"]), version.get("config_sha256")))
            index_path = Path(version["reward_manifest_path"])
            index_bytes = verified_bytes(index_path, version.get("reward_manifest_sha256"))
            index = json.loads(index_bytes)
            if not index.get("complete"):
                raise ValueError(f"Incomplete {name} evaluation")
            for entry in index["episodes"]:
                for key in ("reward_path", "annotation_path", "official_annotation_path"):
                    if entry.get(key):
                        original = storage_path(entry[key])
                        hash_key = key.replace("_path", "_sha256")
                        if entry.get(hash_key) and sha256_file(original) != entry[hash_key]:
                            raise ValueError(f"Corrupted {name} values: {entry['run_id']}")
                        entry[key] = copy_file(original, expected=entry.get(hash_key))
            if index.get("stage_annotations_path"):
                snapshot = Path(index["stage_annotations_path"])
                snapshot_bytes = snapshot.read_bytes()
                if stable_hash(json.loads(snapshot_bytes)) != index.get("stage_annotations_sha256"):
                    raise ValueError("Stage snapshot hash mismatch")
                index["stage_annotations_path"] = copy_file(snapshot,
                    expected=hashlib.sha256(snapshot_bytes).hexdigest())
            directory = f"rewards/{name}"
            atomic_json(temporary / directory / "reward_manifest.json", index)
            rewards[name] = {"prepared": prepared(Path(version["prepared_manifest_path"]),
                                                   f"{directory}/dataset_manifest.json",
                                                   version.get("prepared_manifest_sha256")),
                "manifest": PREFIX + f"{directory}/reward_manifest.json",
                "reward": raw["reward"], "data": raw["data"], "version_id": version["id"],
                "origin": version.get("origin", "dataset"),
                "source_manifest_sha256": hashlib.sha256(index_bytes).hexdigest()}
        files = {path.relative_to(temporary).as_posix(): {"size": path.stat().st_size,
                 "sha256": sha256_file(path)} for path in sorted(temporary.rglob("*")) if path.is_file()}
        payload = {"schema_version": 1, "kind": "portable_training_dataset",
                   "dataset": copy.deepcopy(frozen), "prepared": action_manifest,
                   "rewards": rewards, "files": files, "notes": notes or {}}
        payload["bundle_sha256"] = stable_hash(payload)
        atomic_json(temporary / BUNDLE_NAME, payload)
        read_bundle(temporary, verify=True)
        os.replace(temporary, destination)
    finally:
        if temporary.exists():
            shutil.rmtree(temporary)
    return destination


def materialize_bundle(directory: Path, work: Path, raw: dict, source: str) -> dict:
    """Create local execution manifests, never edit the transferred snapshot."""
    from .methods import training_method
    directory = directory.expanduser().resolve()
    if work.resolve().is_relative_to(directory):
        raise ValueError("Execution manifests must be outside the transferred dataset")
    bundle = read_bundle(directory, verify=True)
    raw = copy.deepcopy(raw)
    needs_reward = training_method(raw).requires_rewards
    saved = bundle["rewards"].get(source)
    if needs_reward and saved is None:
        raise ValueError(f"Dataset has no complete {source} reward; generate/export it on the PC first")

    def resolve(value):
        if not value.startswith(PREFIX):
            return value
        relative = value[len(PREFIX):]
        if relative not in bundle["files"]:
            raise ValueError(f"Unverified bundle reference: {value}")
        return str(contained(directory, relative))

    def document(reference):
        if not isinstance(reference, str) or not reference.startswith(PREFIX):
            raise ValueError("Runnable manifest references must use bundle:// paths")
        value = json.loads(Path(resolve(reference)).read_text())
        critical = {"trajectory_path", "observations_path", "source_manifest", "reward_path",
                    "annotation_path", "official_annotation_path", "stage_annotations_path"}
        def check(item):
            if isinstance(item, list):
                for child in item:
                    check(child)
            elif isinstance(item, dict):
                for key, child in item.items():
                    if key in critical and child is not None and (
                            not isinstance(child, str) or not child.startswith(PREFIX)):
                        raise ValueError(f"Unportable runnable reference: {key}")
                    check(child)
        check(value)
        return map_references(value, resolve)

    work.mkdir(parents=True, exist_ok=False)
    manifest = document(saved["prepared"] if needs_reward else bundle["prepared"])
    members = {item["run_id"] for item in bundle["dataset"]["members"]}
    if {item["run_id"] for item in manifest["episodes"]} != members:
        raise ValueError("Transferred prepared members differ from the frozen dataset")
    for key in ("action_horizon", "action_dim", "proprio_dim", "control_hz"):
        if raw["data"][key] != manifest[key]:
            raise ValueError(f"Transferred {key} conflicts with the server model/data configuration")
    atomic_json(work / "dataset_manifest.json", manifest)
    raw["paths"]["work_dir"] = str(work)
    raw["paths"]["dataset_sources"] = [str(directory / "runs")]
    dataset = bundle["dataset"]
    for key in ("validation_fraction", "split_seed", "success_consecutive_steps"):
        raw["data"][key] = dataset[key]
    raw["data"].update(project_id=dataset.get("project_id", raw["data"]["project_id"]),
                       task_ids=[dataset["task_id"]], selection_manifest=None,
                       stage_annotations_manifest=None)
    if needs_reward:
        reduction = {key: raw["reward"][key] for key in ("gamma", "accumulate_primitive_steps")}
        raw["reward"] = {**saved["reward"], **reduction}
        index = document(saved["manifest"])
        # Relocation changes manifest bytes, not reward semantics. This stable
        # identity permits server resume across run directories and GPU counts.
        index["portable_source_sha256"] = saved["source_manifest_sha256"]
        path = work / "reward_manifest.json"
        atomic_json(path, index)
        raw["reward"].update(manifest_path=str(path), manifest_sha256=sha256_file(path),
                              version_id=saved["version_id"])
        raw["reward"]["source"] = source
        if source == "final" and raw["reward"].get("fusion_mode") == "multiplicative":
            raw["reward"]["accumulate_primitive_steps"] = False
    atomic_json(work / "transfer_provenance.json", {
        "bundle": str(directory), "bundle_sha256": bundle["bundle_sha256"],
        "source_dataset_sha256": dataset["dataset_sha256"],
        "source_reward_manifest_sha256": saved["source_manifest_sha256"] if needs_reward else None})
    return raw
