#!/usr/bin/env python3
"""Seal a dataset-local evaluation without modifying any source sidecars."""

from __future__ import annotations

import argparse
import hashlib
import json
import shutil
from datetime import datetime, timezone
from pathlib import Path


def digest(path: Path) -> str:
    value = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            value.update(chunk)
    return value.hexdigest()


def write(path: Path, value: dict) -> None:
    temporary = path.with_suffix(".tmp")
    temporary.write_text(json.dumps(value, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    temporary.replace(path)


def seal(version_path: Path) -> dict:
    version = json.loads(version_path.read_text(encoding="utf-8"))
    if version.get("complete"):
        raise ValueError("A sealed evaluation version cannot be overwritten")
    work = Path(version["work_dir"])
    config_path = version_path.parent / "effective_config.yaml"
    shutil.copyfile(Path(version["config_path"]), config_path)
    version.update(config_path=str(config_path.resolve()), config_sha256=digest(config_path))
    source = version["evaluator"]
    manifest_path = (work / "robometer" / "robometer_manifest.json" if source == "robometer"
                     else work / "rewards" / "reward_manifest.json")
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    if manifest.get("complete") is not True:
        raise ValueError("Evaluation did not complete")
    expected = set(version["run_ids"])
    if {entry["run_id"] for entry in manifest["episodes"]} != expected:
        raise ValueError("Evaluation membership does not match the frozen dataset")
    # Freeze references to global official-output caches before publishing.
    official_dir = work / "official_outputs"
    official_dir.mkdir(exist_ok=True)
    for entry in manifest["episodes"]:
        value_path = Path(entry.get("reward_path") or entry["annotation_path"])
        expected_hash = entry.get("reward_sha256") or entry.get("values_sha256") or entry["annotation_sha256"]
        if value_path.is_symlink() or digest(value_path) != expected_hash:
            raise ValueError(f"Evaluation values changed for {entry['run_id']}")
        if entry.get("official_annotation_path"):
            original = Path(entry["official_annotation_path"])
            copied = official_dir / f"{entry['run_id']}.npz"
            shutil.copyfile(original, copied)
            entry["official_annotation_path"] = str(copied.resolve())
            entry["official_annotation_sha256"] = digest(copied)
    write(manifest_path, manifest)
    annotation_path = work / "annotations" / "annotation_manifest.json"
    if annotation_path.is_file():
        annotations = json.loads(annotation_path.read_text(encoding="utf-8"))
        for entry in annotations["episodes"]:
            original = Path(entry["annotation_path"])
            copied = official_dir / f"{entry['run_id']}.npz"
            if not copied.exists():
                shutil.copyfile(original, copied)
            if digest(copied) != entry["annotation_sha256"]:
                raise ValueError("Official output hash mismatch")
            entry["annotation_path"] = str(copied.resolve())
        write(annotation_path, annotations)
        version["annotation_manifest_path"] = str(annotation_path.resolve())
        manifest["annotation_manifest_sha256"] = hashlib.sha256(
            json.dumps(annotations, sort_keys=True, separators=(",", ":"),
                       default=str).encode("utf-8")
        ).hexdigest()
        write(manifest_path, manifest)
    prepared_path = work / "dataset_manifest.json"
    if source != "robometer":
        prepared = json.loads(prepared_path.read_text(encoding="utf-8"))
        if (prepared.get("source_dataset_sha256") != version["dataset_sha256"]
                or manifest.get("dataset_sha256") != prepared.get("dataset_sha256")):
            raise ValueError("Prepared data / reward dataset mismatch")
        version.update(prepared_manifest_path=str(prepared_path.resolve()),
                       prepared_manifest_sha256=digest(prepared_path),
                       reward_manifest_path=str(manifest_path.resolve()),
                       reward_manifest_sha256=digest(manifest_path))
        version["derivation_implementation_sha256"] = manifest.get("derivation_implementation_sha256")
        version["reward_config"] = manifest.get("reward_config")
    else:
        version.update(robometer_manifest_path=str(manifest_path.resolve()),
                       robometer_manifest_sha256=digest(manifest_path))
    version.update(complete=True, status="READY", completed_at=datetime.now(timezone.utc).isoformat())
    write(version_path, version)
    return version


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--version", type=Path, required=True)
    seal(parser.parse_args().version)
