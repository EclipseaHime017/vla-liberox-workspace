"""Durable RynnValue annotations bound to individual trajectory episodes."""

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


SIDECAR_NAME = "rynnvalue_evaluation.json"
VALUES_NAME = "rynnvalue_evaluation.npz"
EVALUATION_SCHEMA_VERSION = 5
OFFICIAL_ARRAY_KEYS = frozenset({
    "absolute_temporal_distance_seconds",
    "absolute_value_entropy_nats",
    "absolute_value_logits",
    "relative_temporal_distance_seconds",
    "relative_value_logits",
})
PBRS_ARRAY_KEYS = frozenset({"pbrs_shaping_reward", "pbrs_chunk_reward"})


def _utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


class TrajectoryEvaluationService:
    """Owns the small sidecar that makes reward evaluation dataset-independent.

    The original trajectory and observations remain untouched.  The value
    arrays are copied next to them so a run keeps its evaluation when frozen
    datasets or the global annotation cache are removed.
    """

    def __init__(self, run_service: Any, project_root: Path):
        self.run_service = run_service
        self.root = project_root.resolve()

    @staticmethod
    def _episode_dir(run: dict[str, Any]) -> Path:
        raw = Path(str(run.get("trajectory") or ""))
        if raw.is_symlink():
            raise ValueError(f"Symlink trajectories cannot be evaluated: {run.get('id')}")
        trajectory = raw.resolve()
        if not trajectory.is_file() or trajectory.name != "trajectory.npz":
            raise FileNotFoundError(f"Trajectory is unavailable for {run.get('id')}")
        return trajectory.parent

    def _load(self, run: dict[str, Any]) -> dict[str, Any] | None:
        episode = self._episode_dir(run)
        sidecar = episode / SIDECAR_NAME
        values = episode / VALUES_NAME
        if not sidecar.is_file() or sidecar.is_symlink() or not values.is_file() or values.is_symlink():
            return None
        try:
            payload = json.loads(sidecar.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError):
            return None
        trajectory = episode / "trajectory.npz"
        if (
            payload.get("schema_version") != EVALUATION_SCHEMA_VERSION
            or payload.get("run_id") != run.get("id")
            or payload.get("trajectory_sha256") != _sha256(trajectory)
            or payload.get("values_sha256") != _sha256(values)
        ):
            return None
        try:
            with np.load(values, allow_pickle=False) as arrays:
                if not ({"boundary_steps"} | OFFICIAL_ARRAY_KEYS | PBRS_ARRAY_KEYS).issubset(
                    arrays.files
                ):
                    return None
                boundaries = arrays["boundary_steps"]
            with np.load(trajectory, allow_pickle=False) as source:
                recorded_action_count = len(source["env_action"])
            if (
                len(boundaries) < 2
                or int(boundaries[0]) != 0
                or int(boundaries[-1]) != recorded_action_count
            ):
                # Older annotations stopped at the first confirmed terminal and
                # therefore hid the recorded post-success trajectory tail.
                return None
        except (OSError, ValueError):
            return None
        return payload

    @staticmethod
    def _public(payload: dict[str, Any] | None) -> dict[str, Any]:
        if payload is None:
            return {"status": "NOT_EVALUATED"}
        return {
            "status": "READY",
            "evaluated_at": payload.get("evaluated_at"),
            "model": payload.get("annotator", {}).get("model")
            or payload.get("annotator", {}).get("model_name"),
            "revision": payload.get("annotator", {}).get("resolved_revision")
            or payload.get("annotator", {}).get("requested_revision")
            or payload.get("annotator", {}).get("revision"),
            "boundary_count": payload.get("boundary_count", 0),
            "source_key": payload.get("source_key"),
        }

    def status(self, run: dict[str, Any]) -> dict[str, Any]:
        try:
            return self._public(self._load(run))
        except (FileNotFoundError, OSError):
            return {"status": "NOT_EVALUATED"}

    def exists(self, run: dict[str, Any]) -> bool:
        """Only the complete v5 macro-action reward sidecar counts as evaluated."""
        try:
            return self._load(run) is not None
        except (FileNotFoundError, OSError, ValueError):
            return False

    def status_for_id(self, run_id: str) -> dict[str, Any]:
        return self.status(self.run_service.get_run(run_id))

    def unevaluated_ids(self, runs: list[dict[str, Any]]) -> list[str]:
        return [run["id"] for run in runs if self.status(run)["status"] != "READY"]

    def seed_cache(self, run_ids: list[str], cache_root: Path) -> dict[str, int]:
        """Restore content-cache entries from durable per-trajectory sidecars."""
        cache_root.mkdir(parents=True, exist_ok=True)
        restored = 0
        skipped = 0
        for run_id in run_ids:
            run = self.run_service.get_run(run_id)
            payload = self._load(run)
            if payload is None or not payload.get("source_key"):
                skipped += 1
                continue
            episode = self._episode_dir(run)
            source = episode / VALUES_NAME
            source_key = str(payload["source_key"])
            destination = cache_root / f"{source_key}.npz"
            metadata = cache_root / f"{source_key}.json"
            digest = _sha256(source)
            if not destination.is_file() or _sha256(destination) != digest:
                descriptor, temporary_name = tempfile.mkstemp(
                    prefix=f".{source_key}.", suffix=".tmp", dir=cache_root,
                )
                os.close(descriptor)
                temporary = Path(temporary_name)
                try:
                    shutil.copyfile(source, temporary)
                    os.replace(temporary, destination)
                finally:
                    temporary.unlink(missing_ok=True)
            cache_payload = {
                "schema_version": EVALUATION_SCHEMA_VERSION,
                "run_id": run_id,
                "source_key": source_key,
                "annotation_path": str(destination.resolve()),
                "annotation_sha256": digest,
                "environment_success": payload.get("environment_success"),
                "annotator": payload.get("annotator") or {},
                "official_outputs": payload.get("official_outputs") or {},
                "pbrs_reward": payload.get("pbrs_reward") or {},
            }
            atomic_write_json(metadata, cache_payload)
            restored += 1
        return {"restored": restored, "skipped": skipped}

    def bind(
        self,
        prepared_path: Path,
        reward_path: Path,
        *,
        overwrite: bool,
    ) -> dict[str, Any]:
        prepared = json.loads(prepared_path.read_text(encoding="utf-8"))
        rewards = json.loads(reward_path.read_text(encoding="utf-8"))
        if rewards.get("complete") is not True:
            raise ValueError("Reward manifest is incomplete")
        if rewards.get("schema_version") != EVALUATION_SCHEMA_VERSION:
            raise ValueError(
                "Reward manifest does not contain the complete official RynnValue output set"
            )
        if rewards.get("dataset_sha256") != prepared.get("dataset_sha256"):
            raise ValueError("Reward manifest does not match the prepared trajectories")
        prepared_by_id = {item["run_id"]: item for item in prepared.get("episodes", [])}
        bound: list[str] = []
        skipped: list[str] = []
        for reward in rewards.get("episodes", []):
            run_id = str(reward["run_id"])
            episode = prepared_by_id.get(run_id)
            if episode is None:
                raise ValueError(f"Reward run is missing from prepared manifest: {run_id}")
            raw_trajectory = Path(episode["trajectory_path"])
            if raw_trajectory.is_symlink():
                raise ValueError(f"Symlink trajectories cannot be evaluated: {run_id}")
            trajectory = raw_trajectory.resolve()
            episode_dir = trajectory.parent
            sidecar = episode_dir / SIDECAR_NAME
            destination = episode_dir / VALUES_NAME
            if sidecar.exists() and not overwrite:
                try:
                    run = self.run_service.get_run(run_id)
                    if self._load(run) is not None:
                        skipped.append(run_id)
                        continue
                except Exception:
                    pass
            source = Path(reward["annotation_path"]).resolve()
            if not source.is_file() or source.is_symlink():
                raise FileNotFoundError(source)
            if _sha256(source) != reward["annotation_sha256"]:
                raise ValueError(f"RynnValue annotation hash mismatch: {run_id}")
            required = {"boundary_steps"} | OFFICIAL_ARRAY_KEYS | PBRS_ARRAY_KEYS
            with np.load(source, allow_pickle=False) as arrays:
                missing = sorted(required - set(arrays.files))
                if missing:
                    raise ValueError(
                        f"RynnValue annotation is missing official outputs for {run_id}: {missing}"
                    )
                boundary_count = int(len(arrays["boundary_steps"]))
            descriptor, temporary_name = tempfile.mkstemp(
                prefix=f".{VALUES_NAME}.", suffix=".tmp", dir=episode_dir,
            )
            os.close(descriptor)
            temporary = Path(temporary_name)
            try:
                shutil.copyfile(source, temporary)
                os.replace(temporary, destination)
            finally:
                temporary.unlink(missing_ok=True)
            payload = {
                "schema_version": EVALUATION_SCHEMA_VERSION,
                "run_id": run_id,
                "evaluated_at": _utc_now(),
                "trajectory_sha256": _sha256(trajectory),
                "observations_sha256": episode.get("observations_sha256"),
                "source_key": reward.get("source_key"),
                "values_file": VALUES_NAME,
                "values_sha256": _sha256(destination),
                "boundary_count": boundary_count,
                "annotator": rewards.get("annotator") or reward.get("annotator") or {},
                "reward_config": rewards.get("reward_config") or {},
                "official_outputs": reward.get("official_outputs") or {},
                "pbrs_reward": reward.get("pbrs_reward") or {},
                "environment_success": reward.get("environment_success"),
            }
            atomic_write_json(sidecar, payload)
            bound.append(run_id)
        return {"bound": bound, "skipped": skipped, "count": len(bound)}

    def detail(self, run_id: str) -> dict[str, Any]:
        run = self.run_service.get_run(run_id)
        episode = self._episode_dir(run)
        trajectory_path = episode / "trajectory.npz"
        with np.load(trajectory_path, allow_pickle=False) as source:
            time_seconds = source["time_seconds"].astype(float).tolist()
            action_time = time_seconds[: len(source["env_action"])]
            env_action = source["env_action"].astype(float).tolist()
            raw_action = source["raw_action"].astype(float).tolist()
            eef_position = source["eef_position"].astype(float).tolist()
            eef_axis_angle = source["eef_axis_angle"].astype(float).tolist()
            gripper_qpos = source["gripper_qpos"].astype(float).tolist()
        artifacts = run.get("artifacts") or {}
        public_artifacts = {
            name: f"/api/sessions/{run_id}/artifacts/{name}"
            for name in artifacts
            if name.endswith((".mp4", ".png"))
        }
        payload = self._load(run)
        evaluation: dict[str, Any] | None = None
        if payload is not None:
            with np.load(episode / VALUES_NAME, allow_pickle=False) as values:
                boundary_steps = values["boundary_steps"].astype(int)
                chunk_return = values["pbrs_chunk_reward"].astype(float)
                if len(boundary_steps) != len(chunk_return) + 1:
                    raise ValueError(
                        "RynnValue boundary/chunk-return length mismatch: "
                        f"{len(boundary_steps)} boundaries for {len(chunk_return)} chunks"
                    )
                chunk_lengths = np.diff(boundary_steps).astype(int)
                reward_config = payload.get("reward_config") or {}
                absolute_distance = values[
                    "absolute_temporal_distance_seconds"
                ].astype(float)
                if absolute_distance.ndim != 2 or absolute_distance.shape[1] != 1:
                    raise ValueError(
                        "PBRS detail requires exactly one RynnValue absolute temporal-distance head"
                    )
                pbrs_shaping = values["pbrs_shaping_reward"].astype(float)
                if len(pbrs_shaping) != len(chunk_return):
                    raise ValueError(
                        "RynnValue shape reward component length mismatch: "
                        f"{len(pbrs_shaping)} for {len(chunk_return)} chunks"
                    )
                evaluation = {
                    **self._public(payload),
                    "boundary_steps": boundary_steps.tolist(),
                    "official_outputs": {
                        "absolute_temporal_distance_seconds": values[
                            "absolute_temporal_distance_seconds"
                        ].astype(float).tolist(),
                        "absolute_value_entropy_nats": values[
                            "absolute_value_entropy_nats"
                        ].astype(float).tolist(),
                        "absolute_value_logits": values[
                            "absolute_value_logits"
                        ].astype(float).tolist(),
                        "relative_temporal_distance_seconds": values[
                            "relative_temporal_distance_seconds"
                        ].astype(float).tolist(),
                        "relative_value_logits": values[
                            "relative_value_logits"
                        ].astype(float).tolist(),
                        "analysis": (payload.get("official_outputs") or {}).get("analysis"),
                        "inference_method": (payload.get("official_outputs") or {}).get(
                            "inference_method"
                        ),
                        "prefix_image_slots": (payload.get("official_outputs") or {}).get(
                            "prefix_image_slots"
                        ),
                        "absolute_slot": (payload.get("official_outputs") or {}).get(
                            "absolute_slot"
                        ),
                        "relative_slot": (payload.get("official_outputs") or {}).get(
                            "relative_slot"
                        ),
                    },
                    "pbrs_reward": {
                        "shape_reward": pbrs_shaping.tolist(),
                        "final_reward": chunk_return.tolist(),
                        "chunk_start_steps": boundary_steps[:-1].tolist(),
                        "chunk_end_steps": boundary_steps[1:].tolist(),
                        "chunk_lengths": chunk_lengths.tolist(),
                        "description": (payload.get("pbrs_reward") or {}).get("description"),
                    },
                    "reward_config": reward_config,
                }
        return {
            "run": run,
            "artifacts": public_artifacts,
            "series": {
                "time_seconds": time_seconds,
                "action_time_seconds": action_time,
                "env_action": env_action,
                "raw_action": raw_action,
                "eef_position": eef_position,
                "eef_axis_angle": eef_axis_angle,
                "gripper_qpos": gripper_qpos,
            },
            "evaluation": evaluation,
        }
