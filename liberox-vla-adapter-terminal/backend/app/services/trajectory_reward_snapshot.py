"""A trajectory's first successful reward, independently stored from packages.

Automatic dataset evaluations fill this snapshot once. Only an explicit manual
evaluation may replace it. Labels and the original trajectory are never written.
"""

from __future__ import annotations

import hashlib
import json
import os
import shutil
import tempfile
import threading
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import numpy as np

from ..storage.files import atomic_write_json


SIDECAR = "trajectory_reward.json"
SOURCES = ("sparse", "stage", "rynnvalue")
_LOCK = threading.RLock()
_CACHE: dict[str, tuple[tuple, dict[str, Any] | None]] = {}
_RYNN_CACHE: dict[str, tuple[tuple, bool]] = {}
_IDENTITY_CHECKS: dict[str, tuple] = {}


def _hash(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _stat(path: Path) -> tuple:
    try:
        value = path.stat()
        return value.st_size, value.st_mtime_ns, value.st_ctime_ns, value.st_ino
    except OSError:
        return ()


def _trajectory(run: dict) -> Path | None:
    value = run.get("trajectory")
    if not value:
        return None
    path = Path(value)
    return path if path.is_file() and not path.is_symlink() else None


def snapshot_path(run: dict, source: str | None = None) -> Path | None:
    trajectory = _trajectory(run)
    if trajectory is None:
        return None
    legacy = trajectory.with_name(SIDECAR)
    if source is not None:
        if source not in SOURCES:
            raise ValueError(f"Unsupported reward source: {source}")
        dedicated = trajectory.with_name(f"trajectory_reward.{source}.json")
        if dedicated.exists():
            return dedicated
        try:
            if json.loads(legacy.read_text()).get("source") == source:
                return legacy
        except (OSError, ValueError):
            pass
        return dedicated
    if legacy.exists():
        return legacy
    candidates = [trajectory.with_name(f"trajectory_reward.{item}.json") for item in SOURCES]
    existing = [path for path in candidates if path.exists()]
    return min(existing, key=lambda path: path.stat().st_mtime_ns) if existing else legacy


def _identity_fingerprint(trajectory: Path, sidecar: Path) -> tuple:
    values = sidecar
    try:
        name = json.loads(sidecar.read_text())["values_file"]
        if Path(name).name == name:
            values = sidecar.with_name(name)
    except (OSError, ValueError, KeyError):
        pass
    return tuple(_stat(path) for path in (
        trajectory, sidecar, values,
        trajectory.with_name("trajectory_observations.npz")))


def needs_snapshot_validation(run: dict, source: str | None = None) -> bool:
    """Cheap signal for a cold/copied snapshot awaiting background validation."""
    trajectory = _trajectory(run)
    sidecar = snapshot_path(run, source)
    if trajectory is None or sidecar is None or not sidecar.is_file():
        return False
    return (_IDENTITY_CHECKS.get(str(sidecar)) != _identity_fingerprint(trajectory, sidecar)
            and read_reward_snapshot(run, source) is None)


def _refresh_snapshot_identity(run: dict, source: str | None = None) -> dict[str, Any] | None:
    """Revalidate moved/touched observations off the request path, once per stat."""
    trajectory = _trajectory(run)
    if trajectory is None:
        return None
    sidecar = snapshot_path(run, source)
    if _IDENTITY_CHECKS.get(str(sidecar)) == _identity_fingerprint(trajectory, sidecar):
        return None
    try:
        original = sidecar.read_bytes()
        payload = json.loads(original)
        values_name = payload["values_file"]
        if Path(values_name).name != values_name:
            return None
        values = trajectory.with_name(values_name)
        observations = trajectory.with_name("trajectory_observations.npz")
        if (payload.get("schema_version") != 1 or payload.get("run_id") != run["id"]
                or sidecar.is_symlink() or values.is_symlink() or observations.is_symlink()
                or _hash(trajectory) != payload["trajectory_sha256"]
                or _hash(values) != payload["values_sha256"]
                or _hash(observations) != payload["observations_sha256"]):
            return None
        with _LOCK:
            if sidecar.read_bytes() != original:
                return read_reward_snapshot(run, source)
            payload["observations_fingerprint"] = list(_stat(observations))
            atomic_write_json(sidecar, payload)
            _CACHE.pop(str(sidecar), None)
        return read_reward_snapshot(run, source)
    except (OSError, KeyError, ValueError, TypeError):
        return None
    finally:
        _IDENTITY_CHECKS[str(sidecar)] = _identity_fingerprint(trajectory, sidecar)


def read_reward_snapshot(run: dict, source: str | None = None) -> dict[str, Any] | None:
    trajectory = _trajectory(run)
    if trajectory is None:
        return None
    sidecar = snapshot_path(run, source)
    if sidecar.is_symlink() or not sidecar.is_file():
        return None
    try:
        payload = json.loads(sidecar.read_text(encoding="utf-8"))
        name = payload["values_file"]
        if Path(name).name != name:
            return None
        values = trajectory.with_name(name)
        observations = trajectory.with_name("trajectory_observations.npz")
        fingerprint = (_stat(sidecar), _stat(trajectory), _stat(values), _stat(observations))
        cached = _CACHE.get(str(sidecar))
        if cached and cached[0] == fingerprint:
            return cached[1]
        if (payload.get("schema_version") != 1 or payload.get("run_id") != str(run.get("id"))
                or (source is not None and payload.get("source") != source)
                or values.is_symlink() or payload["trajectory_sha256"] != _hash(trajectory)
                or payload["values_sha256"] != _hash(values)
                or list(_stat(observations)) != payload.get("observations_fingerprint")):
            result = None
        else:
            with np.load(values, allow_pickle=False) as archive:
                arrays = {key: archive[key] for key in archive.files}
            if "final_reward" not in arrays or not np.isfinite(arrays["final_reward"]).all():
                result = None
            else:
                result = {"metadata": payload, "arrays": arrays}
        if len(_CACHE) > 128:
            _CACHE.clear()
        _CACHE[str(sidecar)] = (fingerprint, result)
        return result
    except (OSError, KeyError, ValueError, TypeError):
        return None


def _existing_rynn(trajectory: Path, run_id: str) -> bool:
    """Existing valid global Rynn output predates this automatic initializer."""
    from .trajectory_evaluation_service import (
        COMPATIBLE_EVALUATION_SCHEMA_VERSIONS, OFFICIAL_ARRAY_KEYS, PBRS_ARRAY_KEYS,
    )
    sidecar = trajectory.with_name("rynnvalue_evaluation.json")
    values = trajectory.with_name("rynnvalue_evaluation.npz")
    observations = trajectory.with_name("trajectory_observations.npz")
    fingerprint = (run_id, *( _stat(path) for path in (sidecar, values, trajectory, observations)))
    cached = _RYNN_CACHE.get(str(sidecar))
    if cached and cached[0] == fingerprint:
        return cached[1]
    try:
        payload = json.loads(sidecar.read_text(encoding="utf-8"))
        valid = (payload.get("schema_version") in COMPATIBLE_EVALUATION_SCHEMA_VERSIONS
                and payload.get("run_id") == run_id
                and not sidecar.is_symlink() and not values.is_symlink() and not observations.is_symlink()
                and payload.get("trajectory_sha256") == _hash(trajectory)
                and payload.get("values_sha256") == _hash(values)
                and payload.get("observations_sha256") == _hash(observations))
        if valid:
            with np.load(values, allow_pickle=False) as archive, np.load(trajectory, allow_pickle=False) as source:
                required = {"boundary_steps"} | OFFICIAL_ARRAY_KEYS | PBRS_ARRAY_KEYS
                boundaries = archive["boundary_steps"]
                valid = (required.issubset(archive.files) and len(boundaries) >= 2
                         and int(boundaries[0]) == 0
                         and int(boundaries[-1]) == len(source["env_action"]))
    except (OSError, ValueError, TypeError, KeyError):
        valid = False
    if len(_RYNN_CACHE) > 128:
        _RYNN_CACHE.clear()
    _RYNN_CACHE[str(sidecar)] = (fingerprint, valid)
    return valid


def bind_reward_snapshot(prepared_path: Path, reward_manifest_path: Path, *,
                         overwrite: bool = False, origin: str = "dataset",
                         evaluation_id: str | None = None, run_ids: list[str] | None = None,
                         evaluated_at: str | None = None) -> dict[str, Any]:
    prepared = json.loads(Path(prepared_path).read_text(encoding="utf-8"))
    rewards = json.loads(Path(reward_manifest_path).read_text(encoding="utf-8"))
    if rewards.get("complete") is not True or rewards.get("dataset_sha256") != prepared.get("dataset_sha256"):
        raise ValueError("Reward snapshot input is incomplete or mismatched")
    recipe = rewards.get("reward_config") or {}
    source = recipe.get("source", "rynnvalue")
    if source not in {"sparse", "stage", "rynnvalue"}:
        raise ValueError("Unsupported trajectory reward source")
    members = {entry["run_id"]: entry for entry in prepared["episodes"]}
    selected = set(run_ids) if run_ids is not None else None
    checked = []
    bound, skipped = [], []
    # Validate every candidate before replacing any ready trajectory result.
    for entry in rewards["episodes"]:
        run_id = str(entry["run_id"])
        if selected is not None and run_id not in selected:
            continue
        episode = members[run_id]
        trajectory = Path(episode["trajectory_path"])
        # Dataset regeneration must not reread a multi-GB observation archive
        # merely to preserve an already published first result.
        run = {"id": run_id, "trajectory": str(trajectory)}
        if not overwrite and (read_reward_snapshot(run, source) is not None
                              or (source == "rynnvalue" and _existing_rynn(trajectory, run_id))):
            skipped.append(run_id)
            continue
        observations = Path(episode["observations_path"])
        values = Path(entry.get("reward_path") or entry["annotation_path"])
        expected = entry.get("reward_sha256") or entry["annotation_sha256"]
        if (trajectory.is_symlink() or observations.is_symlink() or values.is_symlink()
                or _hash(trajectory) != episode["trajectory_sha256"]
                or _hash(observations) != episode["observations_sha256"] or _hash(values) != expected):
            raise ValueError(f"Reward snapshot source hash mismatch: {run_id}")
        with np.load(values, allow_pickle=False) as archive:
            final = archive["final_reward"]
            if not np.isfinite(final).all():
                raise ValueError(f"Invalid reward values: {run_id}")
        checked.append((run_id, episode, entry, trajectory, observations, values, expected))
    with _LOCK:
        for run_id, episode, entry, trajectory, observations, values, digest in checked:
            sidecar = trajectory.with_name(f"trajectory_reward.{source}.json")
            run = {"id": run_id, "trajectory": str(trajectory)}
            if not overwrite and (read_reward_snapshot(run, source) is not None
                                  or (source == "rynnvalue" and _existing_rynn(trajectory, run_id))):
                skipped.append(run_id)
                continue
            # Content-addressed arrays plus an atomic JSON pointer keep readers
            # and a previous successful snapshot safe during manual replacement.
            destination = trajectory.with_name(f"trajectory_reward.{digest}.npz")
            descriptor, temporary_name = tempfile.mkstemp(prefix=".trajectory_reward.", dir=trajectory.parent)
            os.close(descriptor)
            temporary = Path(temporary_name)
            try:
                shutil.copyfile(values, temporary)
                if _hash(temporary) != digest:
                    raise ValueError(f"Reward values changed during publication: {run_id}")
                os.replace(temporary, destination)
            finally:
                temporary.unlink(missing_ok=True)
            payload = {
                "schema_version": 1, "run_id": run_id, "source": source,
                "reward_config": recipe, "trajectory_sha256": episode["trajectory_sha256"],
                "observations_sha256": episode["observations_sha256"],
                "observations_fingerprint": list(_stat(observations)),
                "evaluated_at": evaluated_at or datetime.now(timezone.utc).isoformat(),
                "origin": origin, "evaluation_id": evaluation_id,
                "values_file": destination.name, "values_sha256": digest,
                "entry": {**entry, "annotator": entry.get("annotator") or rewards.get("annotator") or {},
                          "official_outputs": entry.get("official_outputs") or {}},
                "episode": episode,
                "prepared": {key: value for key, value in prepared.items() if key != "episodes"},
                "prepared_manifest_path": str(Path(prepared_path).resolve()),
                "annotator": entry.get("annotator") or rewards.get("annotator") or {},
                "official_outputs": entry.get("official_outputs") or {},
                "annotation_config": rewards.get("annotation_config") or {},
                "derivation_implementation_sha256": rewards.get("derivation_implementation_sha256"),
            }
            atomic_write_json(sidecar, payload)
            _CACHE.pop(str(sidecar), None)
            bound.append(run_id)
    return {"bound": bound, "skipped": skipped, "count": len(bound)}


def ensure_first_reward_snapshot(run: dict, datasets: Any, source: str | None = None) -> dict[str, Any] | None:
    """One-time lazy initialization for packages evaluated before this feature."""
    if source is None:
        results = [ensure_first_reward_snapshot(run, datasets, item) for item in SOURCES]
        return next((item for item in results if item is not None), None)
    existing = read_reward_snapshot(run, source)
    trajectory = _trajectory(run)
    if existing is None and trajectory is not None and snapshot_path(run, source).is_file():
        # Preserve the first result's identity. A changed source is not an
        # invitation to silently substitute some other dataset's evaluation.
        return _refresh_snapshot_identity(run, source)
    if existing is not None or trajectory is None or (source == "rynnvalue" and _existing_rynn(trajectory, str(run["id"]))):
        return existing
    candidates = []
    for dataset in datasets.list():
        if not any(member["run_id"] == run.get("id") for member in dataset.get("members", [])):
            continue
        for summary in dataset.get("evaluation_versions", []):
            if summary.get("status") == "READY" and summary.get("evaluator") == source:
                candidates.append((summary.get("completed_at") or summary.get("created_at") or "",
                                   dataset["id"], summary["id"]))
    for timestamp, dataset_id, evaluation_id in sorted(candidates):
        try:
            version = datasets.get_version(dataset_id, evaluation_id)
            bind_reward_snapshot(Path(version["prepared_manifest_path"]), Path(version["reward_manifest_path"]),
                run_ids=[run["id"]], evaluation_id=evaluation_id, evaluated_at=timestamp or None)
            current = read_reward_snapshot(run, source)
            if current is not None:
                return current
        except (OSError, KeyError, ValueError, TypeError):
            continue
    return None
