"""Durable official Robometer single-trajectory outputs."""

from __future__ import annotations

import hashlib
import json
import os
import shutil
import tempfile
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import numpy as np

from ..storage.files import atomic_write_json


SIDECAR_NAME = "robometer_evaluation.json"
VALUES_NAME = "robometer_evaluation.npz"
SCHEMA_VERSION = 1
ARRAY_KEYS = frozenset({
    "observation_steps", "time_seconds", "progress_pred", "success_probs",
})


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


class RobometerEvaluationService:
    """Validates and binds Robometer output without touching RynnValue files."""

    def __init__(self, run_service: Any, project_root: Path):
        self.run_service = run_service
        self.root = project_root.resolve()
        # Strict validation touches multi-GB observation archives.  Keep it out
        # of list endpoints and reuse it until one of the relevant files
        # changes on disk.
        self._validation_cache: dict[str, tuple[tuple[Any, ...], dict[str, Any] | None]] = {}

    @staticmethod
    def _fingerprint(paths: tuple[Path, ...]) -> tuple[Any, ...]:
        values: list[Any] = []
        for path in paths:
            try:
                stat = path.stat()
                values.extend((str(path), stat.st_ino, stat.st_size, stat.st_mtime_ns, stat.st_ctime_ns))
            except OSError:
                values.extend((str(path), None))
        return tuple(values)

    @staticmethod
    def _episode_dir(run: dict[str, Any]) -> Path:
        raw = Path(str(run.get("trajectory") or ""))
        if raw.is_symlink():
            raise ValueError(f"Symlink trajectories cannot be evaluated: {run.get('id')}")
        trajectory = raw.resolve()
        if not trajectory.is_file() or trajectory.name != "trajectory.npz":
            raise FileNotFoundError(f"Trajectory is unavailable for {run.get('id')}")
        return trajectory.parent

    @staticmethod
    def _manifest_path(run: dict[str, Any], episode: Path) -> Path:
        candidates = [
            Path(str(run.get("output_dir") or "")) / "run.json",
            episode.parents[1] / "run.json",
            episode.parents[1] / "session.json",
        ]
        for path in candidates:
            if path.is_file() and not path.is_symlink():
                return path.resolve()
        raise FileNotFoundError(f"Run manifest is unavailable for {run.get('id')}")

    def _load_status(self, run: dict[str, Any]) -> dict[str, Any] | None:
        """Read only the small sidecar for catalog/list presentation."""
        episode = self._episode_dir(run)
        sidecar, values = episode / SIDECAR_NAME, episode / VALUES_NAME
        if any(path.is_symlink() or not path.is_file() for path in (sidecar, values)):
            return None
        try:
            payload = json.loads(sidecar.read_text(encoding="utf-8"))
            if payload.get("schema_version") != SCHEMA_VERSION or payload.get("run_id") != run.get("id"):
                return None
            if type(payload.get("sample_count")) is not int or payload["sample_count"] < 1:
                return None
            return payload
        except (OSError, ValueError, json.JSONDecodeError):
            return None

    def _load(self, run: dict[str, Any]) -> dict[str, Any] | None:
        episode = self._episode_dir(run)
        sidecar, values = episode / SIDECAR_NAME, episode / VALUES_NAME
        observations = episode / "trajectory_observations.npz"
        manifest = self._manifest_path(run, episode)
        paths = (sidecar, values, observations, manifest, episode / "trajectory.npz")
        fingerprint = self._fingerprint(paths)
        cache_key = str(run.get("id"))
        cached = self._validation_cache.get(cache_key)
        if cached is not None and cached[0] == fingerprint:
            return cached[1]
        payload = self._load_status(run)
        if payload is None or any(path.is_symlink() or not path.is_file() for path in paths):
            self._validation_cache[cache_key] = (fingerprint, None)
            return None
        try:
            if (
                payload.get("trajectory_sha256") != _sha256(episode / "trajectory.npz")
                or payload.get("observations_sha256") != _sha256(observations)
                or payload.get("manifest_sha256") != _sha256(manifest)
                or payload.get("values_sha256") != _sha256(values)
            ):
                self._validation_cache[cache_key] = (fingerprint, None)
                return None
            with np.load(values, allow_pickle=False) as arrays:
                if not ARRAY_KEYS.issubset(arrays.files):
                    payload = None
                else:
                    lengths = {len(arrays[key]) for key in ARRAY_KEYS}
                    finite = all(np.isfinite(arrays[key]).all() for key in ARRAY_KEYS)
                    steps = arrays["observation_steps"].astype(int)
            if payload is not None:
                with np.load(observations, allow_pickle=False) as source:
                    observation_count = len(source["agentview_image"])
                if lengths != {int(payload.get("sample_count", -1))} or not finite:
                    payload = None
                if not len(steps) or int(steps[0]) != 0 or int(steps[-1]) != observation_count - 1:
                    payload = None
        except (OSError, ValueError, KeyError, json.JSONDecodeError):
            payload = None
        self._validation_cache[cache_key] = (fingerprint, payload)
        return payload

    @staticmethod
    def _public(payload: dict[str, Any] | None) -> dict[str, Any]:
        if payload is None:
            return {"status": "NOT_EVALUATED"}
        annotator = payload.get("annotator") or {}
        return {
            "status": "READY",
            "evaluated_at": payload.get("evaluated_at"),
            "model": annotator.get("model"),
            "revision": annotator.get("revision"),
            "sample_count": payload.get("sample_count", 0),
            "source_key": payload.get("source_key"),
        }

    def status(self, run: dict[str, Any]) -> dict[str, Any]:
        try:
            return self._public(self._load_status(run))
        except (OSError, ValueError, FileNotFoundError):
            return {"status": "NOT_EVALUATED"}

    def exists(self, run: dict[str, Any]) -> bool:
        try:
            return self._load(run) is not None
        except (OSError, ValueError, FileNotFoundError):
            return False

    def seed_version_cache(self, run_ids: list[str], values_dir: Path,
                           inference_config: dict[str, Any]) -> dict[str, int]:
        """Copy compatible global outputs into a new version, never alter sidecars."""
        values_dir.mkdir(parents=True, exist_ok=True)
        restored = 0
        for run_id in run_ids:
            run = self.run_service.get_run(run_id)
            payload = self._load(run)
            if payload is None:
                continue
            previous = payload.get("inference_config")
            if previous is None:
                # Legacy v1 always ran BF16 with the configured official commit.
                model, evaluation = payload.get("annotator", {}), payload.get("evaluation_config", {})
                previous = {
                    "model": {"checkpoint": model.get("model"), "revision": model.get("revision"),
                              "robometer_commit": model.get("robometer_commit"), "dtype": "bfloat16"},
                    "evaluation": {key: evaluation.get(key) for key in ("control_hz", "fps", "prefix_frames")},
                }
            if previous != inference_config:
                continue
            target = values_dir / f"{run_id}.npz"
            metadata = values_dir / f"{run_id}.json"
            # A copied dataset-local version is preferable to a global sidecar.
            if target.exists() or metadata.exists():
                continue
            shutil.copyfile(self._episode_dir(run) / VALUES_NAME, target)
            atomic_write_json(metadata, {
                **{key: payload.get(key) for key in (
                    "schema_version", "run_id", "source_key", "values_sha256", "trajectory_sha256",
                    "observations_sha256", "manifest_sha256", "sample_count")},
                "annotation_path": str(target.resolve()), "inference_config": previous,
            })
            restored += 1
        return {"restored": restored, "skipped": len(run_ids) - restored}

    def bind(self, manifest_path: Path, *, overwrite: bool,
             run_ids: list[str] | None = None) -> dict[str, Any]:
        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
        if manifest.get("schema_version") != SCHEMA_VERSION or manifest.get("complete") is not True:
            raise ValueError("Robometer manifest is incomplete or incompatible")
        bound: list[str] = []
        skipped: list[str] = []
        for item in manifest.get("episodes") or []:
            run_id = str(item["run_id"])
            if run_ids is not None and run_id not in run_ids:
                continue
            run = self.run_service.get_run(run_id)
            episode = self._episode_dir(run)
            trajectory = episode / "trajectory.npz"
            observations = episode / "trajectory_observations.npz"
            source_manifest = self._manifest_path(run, episode)
            if item.get("trajectory_sha256") != _sha256(trajectory):
                raise ValueError(f"Robometer trajectory hash mismatch: {run_id}")
            if item.get("observations_sha256") != _sha256(observations):
                raise ValueError(f"Robometer observation hash mismatch: {run_id}")
            if item.get("manifest_sha256") != _sha256(source_manifest):
                raise ValueError(f"Robometer prompt manifest hash mismatch: {run_id}")
            if not overwrite and self._load(run) is not None:
                skipped.append(run_id)
                continue
            source = Path(str(item["annotation_path"])).resolve()
            if source.is_symlink() or not source.is_file():
                raise FileNotFoundError(source)
            if item.get("values_sha256") != _sha256(source):
                raise ValueError(f"Robometer values hash mismatch: {run_id}")
            with np.load(source, allow_pickle=False) as arrays:
                missing = sorted(ARRAY_KEYS - set(arrays.files))
                if missing:
                    raise ValueError(f"Robometer output is missing {missing}: {run_id}")
                lengths = {len(arrays[key]) for key in ARRAY_KEYS}
                if lengths != {int(item.get("sample_count", -1))}:
                    raise ValueError(f"Robometer output length mismatch: {run_id}")
                if any(not np.isfinite(arrays[key]).all() for key in ARRAY_KEYS):
                    raise ValueError(f"Robometer output contains NaN/Inf: {run_id}")
            destination = episode / VALUES_NAME
            descriptor, temporary_name = tempfile.mkstemp(
                prefix=f".{VALUES_NAME}.", suffix=".tmp", dir=episode,
            )
            os.close(descriptor)
            temporary = Path(temporary_name)
            try:
                shutil.copyfile(source, temporary)
                os.replace(temporary, destination)
            finally:
                temporary.unlink(missing_ok=True)
            payload = {
                "schema_version": SCHEMA_VERSION,
                "run_id": run_id,
                "evaluated_at": _utc_now(),
                "trajectory_sha256": _sha256(trajectory),
                "observations_sha256": _sha256(observations),
                "manifest_sha256": item.get("manifest_sha256"),
                "source_key": item.get("source_key"),
                "values_file": VALUES_NAME,
                "values_sha256": _sha256(destination),
                "sample_count": int(item["sample_count"]),
                "annotator": manifest.get("annotator") or {},
                "evaluation_config": manifest.get("evaluation_config") or {},
                "inference_config": manifest.get("inference_config"),
            }
            atomic_write_json(episode / SIDECAR_NAME, payload)
            self._validation_cache.pop(run_id, None)
            bound.append(run_id)
        return {"bound": bound, "skipped": skipped, "count": len(bound)}

    def detail(self, run: dict[str, Any]) -> dict[str, Any] | None:
        payload = self._load(run)
        if payload is None:
            return None
        episode = self._episode_dir(run)
        with np.load(episode / VALUES_NAME, allow_pickle=False) as arrays:
            return {
                **self._public(payload),
                "observation_steps": arrays["observation_steps"].astype(int).tolist(),
                "time_seconds": arrays["time_seconds"].astype(float).tolist(),
                "progress_pred": arrays["progress_pred"].astype(float).tolist(),
                "success_probs": arrays["success_probs"].astype(float).tolist(),
                "evaluation_config": payload.get("evaluation_config") or {},
            }
